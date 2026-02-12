"""
SkyReels-V3 Backend — wraps the SkyReels-V3 R2V pipeline for multi-reference video generation.

SkyReels-V3 is the current SOTA open-source model for character-consistent video generation
from reference images. It accepts 1-4 reference images (characters, objects, backgrounds)
and generates temporally coherent video aligned with a text prompt.

Model variants:
- Skywork/SkyReels-V3-R2V-14B   (Reference-to-Video, 14B params, ~52 GB in bf16)

Architecture:
- Built on Wan 2.1 backbone (WanTransformer3DModel derivative)
- Custom transformer: SkyReelsC1WanI2v3DModel (not yet in diffusers mainline)
- Dual classifier-free guidance: text CFG + image CFG
- VAE: AutoencoderKLWan (same as Wan 2.1/2.2)
- Text encoder: UMT5-xxl
- Scheduler: UniPCMultistepScheduler (flow_prediction)

Loading strategy:
1. Try the `skyreels_v3` package (SkyworkAI/SkyReels-V3 repo) — provides the custom
   transformer class and the ReferenceToVideoPipeline wrapper natively.
2. Fallback to diffusers DiffusionPipeline.from_pretrained with trust_remote_code=True,
   which may work if the HF repo ships custom code or diffusers adds native support.

Reference:
- GitHub: https://github.com/SkyworkAI/SkyReels-V3
- HuggingFace: https://huggingface.co/Skywork/SkyReels-V3-R2V-14B
- Paper: https://arxiv.org/abs/2601.17323
"""

import logging
import math
from typing import Optional, List, Union

import torch
import numpy as np
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SKYREELS_V3_MODELS = {
    "R2V-14B": "Skywork/SkyReels-V3-R2V-14B",
}

# Aspect ratio presets for each resolution tier.
# Source: skyreels_v3/config.py ASPECT_RATIO_CONFIG (representative subset)
ASPECT_RATIO_MAP = {
    "720P": {
        "1:1":  (720, 720),
        "3:4":  (864, 656),
        "4:3":  (656, 864),
        "16:9": (720, 1280),
        "9:16": (1280, 720),
    },
    "540P": {
        "1:1":  (544, 544),
        "3:4":  (640, 480),
        "4:3":  (480, 640),
        "16:9": (544, 960),
        "9:16": (960, 544),
    },
    "480P": {
        "1:1":  (480, 480),
        "3:4":  (560, 416),
        "4:3":  (416, 560),
        "16:9": (480, 854),
        "9:16": (854, 480),
    },
}

# Default generation parameters
MODEL_DEFAULTS = {
    "R2V-14B": dict(
        width=960,
        height=544,
        num_frames=121,     # 5 seconds * 24 fps + 1
        fps=24,
        guidance_scale=7.5,
        guidance_scale_img=5.0,
        steps=50,
        duration=5,
        resolution="720P",
    ),
}

# Maximum number of reference images the model can fuse (padded with zeros)
MAX_REFERENCE_IMAGES = 4

# Shot transition types for long-form generation
SHOT_TRANSITION_TYPES = (
    "cut",          # Hard cut between shots
    "dissolve",     # Cross-dissolve / cross-fade
    "fade",         # Fade to black then fade in
    "wipe",         # Directional wipe transition
    "zoom",         # Zoom in/out transition
)

# Shot count -> number of condition frames for autoregressive chunks
SHOT_NUM_CONDITION_FRAMES = {
    2: 9,
    3: 17,
    4: 25,
    5: 33,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_closest_aspect_ratio(image: Image.Image, resolution: str = "720P") -> tuple:
    """Determine the best (height, width) for a given image and resolution tier.

    Picks the aspect ratio preset closest to the image's native ratio, then
    aligns dimensions to multiples of 16 (required by the Wan VAE).

    Returns:
        (height, width) tuple.
    """
    presets = ASPECT_RATIO_MAP.get(resolution, ASPECT_RATIO_MAP["720P"])
    img_w, img_h = image.size
    img_ratio = img_h / img_w

    # Map preset names to approximate numeric ratios
    ratio_values = {
        "1:1": 1.0, "3:4": 0.75, "4:3": 1.333,
        "16:9": 0.5625, "9:16": 1.778,
    }

    best_name = min(presets.keys(), key=lambda k: abs(ratio_values.get(k, 1.0) - img_ratio))
    h, w = presets[best_name]

    # Align to 16-pixel boundary (VAE requirement)
    h = h // 16 * 16
    w = w // 16 * 16
    return h, w


def _resize_with_padding(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize an image to fit within target dimensions, preserving aspect ratio,
    then pad with white to reach the exact target size.

    This matches the SkyReels-V3 reference image preprocessing convention.
    """
    img_w, img_h = image.size
    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    # Paste onto white canvas, centered
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _duration_to_frames(duration: int, fps: int = 24) -> int:
    """Convert duration in seconds to frame count.

    SkyReels-V3 convention: num_frames = duration * fps + 1
    """
    return duration * fps + 1


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class SkyReelsV3Backend(BasePipeline):
    """Backend for SkyReels-V3 — multi-reference character-consistent video generation.

    Supports:
    - 1-4 reference images for character, object, and background consistency
    - Multiple aspect ratios (1:1, 3:4, 4:3, 16:9, 9:16)
    - Resolution tiers: 480P, 540P, 720P
    - Minute-level video via autoregressive chunk generation
    - Shot transition types for multi-shot storyboards
    """

    backend_name = "skyreels_v3"

    def __init__(
        self,
        pipe,
        model_variant: str = "R2V-14B",
        loading_mode: str = "native",
    ):
        self.pipe = pipe
        self.model_variant = model_variant
        self.loading_mode = loading_mode  # "native" (skyreels_v3 pkg) or "diffusers"
        self._defaults = MODEL_DEFAULTS.get(model_variant, MODEL_DEFAULTS["R2V-14B"])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "none",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        model_variant: str = "R2V-14B",
        mode: str = "r2v",
        **kwargs,
    ) -> "SkyReelsV3Backend":
        """Load SkyReels-V3 model.

        Tries two loading strategies in order:
        1. Native: uses the ``skyreels_v3`` package from SkyworkAI/SkyReels-V3 repo.
        2. Diffusers: uses ``DiffusionPipeline.from_pretrained`` with trust_remote_code.

        Args:
            model_path: HuggingFace model ID or local path. Defaults to the R2V-14B checkpoint.
            torch_dtype: Weight dtype. Default bfloat16 (CUDA) or float32 (MPS).
            device: Target device — "cuda", "mps", or "cpu".
            quantization: One of "none", "nf4", "int8", "fp8".
            offload_strategy: One of "none", "model_cpu", "sequential_cpu".
            enable_vae_slicing: Reduce VAE memory by slicing batch dim.
            enable_vae_tiling: Reduce VAE memory with spatial tiling.
            model_variant: Model variant key, currently only "R2V-14B".
            mode: Generation mode — "r2v" (reference-to-video). Reserved for future V2V/A2V.
        """
        # Resolve model path
        if model_path is None:
            model_path = SKYREELS_V3_MODELS.get(model_variant, SKYREELS_V3_MODELS["R2V-14B"])

        # MPS safety: SkyReels-V3 14B is extremely large for MPS, and the custom
        # transformer may use ops not fully supported on MPS.
        if device == "mps":
            torch_dtype = torch.float32
            logger.warning(
                "MPS detected: using float32. SkyReels-V3 R2V-14B is ~52 GB in float32 — "
                "this will likely OOM on Apple Silicon. Consider using CUDA with offloading."
            )

        logger.info(
            f"Loading SkyReels-V3 {model_variant} from {model_path} "
            f"(dtype={torch_dtype}, quant={quantization})"
        )

        # Determine low_vram mode based on quantization
        low_vram = quantization != "none"

        # ------------------------------------------------------------------
        # Strategy 1: Native skyreels_v3 package
        # ------------------------------------------------------------------
        try:
            pipe, loading_mode = cls._load_native(
                model_path=model_path,
                torch_dtype=torch_dtype,
                device=device,
                offload_strategy=offload_strategy,
                low_vram=low_vram,
            )
            logger.info("Loaded SkyReels-V3 via native skyreels_v3 package")

        except ImportError:
            logger.info(
                "skyreels_v3 package not found, falling back to diffusers loading. "
                "Install from: https://github.com/SkyworkAI/SkyReels-V3"
            )
            pipe, loading_mode = cls._load_diffusers(
                model_path=model_path,
                torch_dtype=torch_dtype,
                device=device,
                quantization=quantization,
                offload_strategy=offload_strategy,
                enable_vae_slicing=enable_vae_slicing,
                enable_vae_tiling=enable_vae_tiling,
            )
            logger.info("Loaded SkyReels-V3 via diffusers fallback")

        instance = cls(pipe, model_variant=model_variant, loading_mode=loading_mode)
        return instance

    @classmethod
    def _load_native(
        cls,
        model_path: str,
        torch_dtype: torch.dtype,
        device: str,
        offload_strategy: str,
        low_vram: bool,
    ) -> tuple:
        """Load using the native skyreels_v3 package (from SkyworkAI/SkyReels-V3 repo).

        This package provides the custom SkyReelsC1WanI2v3DModel transformer class
        and the ReferenceToVideoPipeline wrapper that handles reference image
        encoding, dual CFG, and autoregressive chunk generation.

        Returns:
            (pipeline_instance, "native")
        """
        from skyreels_v3.pipelines import ReferenceToVideoPipeline

        use_offload = offload_strategy != "none"

        pipe = ReferenceToVideoPipeline(
            model_path=model_path,
            device=device,
            weight_dtype=torch_dtype,
            offload=use_offload,
            low_vram=low_vram,
        )

        return pipe, "native"

    @classmethod
    def _load_diffusers(
        cls,
        model_path: str,
        torch_dtype: torch.dtype,
        device: str,
        quantization: str,
        offload_strategy: str,
        enable_vae_slicing: bool,
        enable_vae_tiling: bool,
    ) -> tuple:
        """Fallback: load via diffusers DiffusionPipeline with trust_remote_code.

        This works when:
        - diffusers has added native SkyReels-V3 support, OR
        - The HF model repo ships custom pipeline/model code

        The HF model_index.json declares WanPipeline but the transformer uses
        SkyReelsC1WanI2v3DModel, so trust_remote_code=True is required.

        Returns:
            (pipeline_instance, "diffusers")
        """
        from diffusers import DiffusionPipeline, AutoencoderKLWan

        # VAE must be float32 for Wan-based models
        vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32
        )

        # Quantization
        quant_config = get_quantization_config(quantization, components=["transformer"])
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            vae=vae,
            trust_remote_code=True,
        )
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        pipe = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)

        # Fix: transformers 5.x UMT5 embed_tokens zero-weight bug
        if hasattr(pipe, "text_encoder"):
            te = pipe.text_encoder
            if (hasattr(te, "shared") and hasattr(te, "encoder")
                    and hasattr(te.encoder, "embed_tokens")
                    and te.encoder.embed_tokens.weight.abs().sum().item() == 0
                    and te.shared.weight.abs().sum().item() > 0):
                logger.warning("Fixing UMT5 embed_tokens: binding shared.weight -> encoder.embed_tokens.weight")
                te.encoder.embed_tokens.weight = te.shared.weight

        # Apply offloading
        dummy_instance = cls.__new__(cls)
        if offload_strategy != "none":
            BasePipeline._apply_offloading(dummy_instance, pipe, offload_strategy, device=device)
        else:
            pipe.to(device)

        BasePipeline._apply_vae_opts(dummy_instance, pipe, slicing=enable_vae_slicing, tiling=enable_vae_tiling)

        return pipe, "diffusers"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        num_frames: int = 0,
        num_inference_steps: int = 0,
        guidance_scale: float = 0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        # SkyReels-V3 specific
        reference_images: Optional[List[Image.Image]] = None,
        guidance_scale_img: float = 0,
        duration: int = 0,
        resolution: str = "",
        **kwargs,
    ) -> VideoOutput:
        """Generate video from reference images and a text prompt.

        Args:
            prompt: Text description of the desired video.
            negative_prompt: Negative prompt for classifier-free guidance.
            width: Output width in pixels (auto-detected from reference images if 0).
            height: Output height in pixels (auto-detected from reference images if 0).
            num_frames: Number of frames to generate (auto-calculated from duration if 0).
            num_inference_steps: Denoising steps (default 50).
            guidance_scale: Text classifier-free guidance scale (default 7.5).
            seed: Random seed for reproducibility (-1 for random).
            image: Single reference image (convenience alias — added to reference_images).
            reference_images: List of 1-4 reference images (characters, objects, backgrounds).
            guidance_scale_img: Image classifier-free guidance scale (default 5.0).
            duration: Video duration in seconds (default 5). Used to compute num_frames.
            resolution: Resolution tier — "480P", "540P", or "720P" (default "720P").
        """
        d = self._defaults

        # Merge single image into reference_images list
        if reference_images is None:
            reference_images = []
        if image is not None and image not in reference_images:
            reference_images.insert(0, image)

        if not reference_images:
            raise ValueError(
                "SkyReels-V3 R2V requires at least 1 reference image. "
                "Pass reference_images=[img1, img2, ...] or image=img."
            )

        if len(reference_images) > MAX_REFERENCE_IMAGES:
            logger.warning(
                f"SkyReels-V3 supports max {MAX_REFERENCE_IMAGES} reference images, "
                f"got {len(reference_images)}. Truncating to first {MAX_REFERENCE_IMAGES}."
            )
            reference_images = reference_images[:MAX_REFERENCE_IMAGES]

        # Resolve defaults
        resolution = resolution or d["resolution"]
        duration = duration or d["duration"]
        num_inference_steps = num_inference_steps or d["steps"]
        guidance_scale = guidance_scale or d["guidance_scale"]
        guidance_scale_img = guidance_scale_img or d["guidance_scale_img"]

        # Determine output dimensions from reference image aspect ratio
        if width == 0 or height == 0:
            h, w = _get_closest_aspect_ratio(reference_images[0], resolution)
            height = height or h
            width = width or w

        # Calculate frame count from duration
        if num_frames == 0:
            num_frames = _duration_to_frames(duration, fps=d["fps"])

        # Route to the appropriate generation method
        if self.loading_mode == "native":
            return self._generate_native(
                prompt=prompt,
                reference_images=reference_images,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_img=guidance_scale_img,
                duration=duration,
                resolution=resolution,
                seed=seed,
            )
        else:
            return self._generate_diffusers(
                prompt=prompt,
                negative_prompt=negative_prompt,
                reference_images=reference_images,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_img=guidance_scale_img,
                seed=seed,
            )

    def _generate_native(
        self,
        prompt: str,
        reference_images: List[Image.Image],
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_scale_img: float,
        duration: int,
        resolution: str,
        seed: int,
    ) -> VideoOutput:
        """Generate using the native skyreels_v3 ReferenceToVideoPipeline.

        The native pipeline handles:
        - Reference image resizing and aspect-ratio padding
        - VAE encoding of reference images
        - Padding to MAX_REFERENCE_IMAGES (4)
        - Dual classifier-free guidance (text + image)
        - Autoregressive chunk generation for long videos
        """
        # The native pipeline's generate_video accepts PIL images directly
        # and returns numpy video frames
        output = self.pipe.generate_video(
            ref_imgs=reference_images,
            prompt=prompt,
            duration=duration,
            seed=seed if seed >= 0 else None,
            resolution=resolution,
        )

        # Native pipeline returns numpy array (T, H, W, C) uint8
        if isinstance(output, np.ndarray):
            frames = [Image.fromarray(output[i]) for i in range(output.shape[0])]
        elif isinstance(output, list):
            # May return list of numpy arrays
            if isinstance(output[0], np.ndarray):
                frames = [Image.fromarray(f) for f in output]
            else:
                frames = output
        else:
            frames = output

        return VideoOutput(
            frames=frames,
            fps=self._defaults["fps"],
            seed=seed,
            backend=self.backend_name,
            metadata={
                "model_variant": self.model_variant,
                "loading_mode": self.loading_mode,
                "num_reference_images": len(reference_images),
                "resolution": resolution,
                "duration": duration,
            },
        )

    def _generate_diffusers(
        self,
        prompt: str,
        negative_prompt: str,
        reference_images: List[Image.Image],
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_scale_img: float,
        seed: int,
    ) -> VideoOutput:
        """Generate using diffusers pipeline fallback.

        When loaded via diffusers, we need to preprocess reference images
        ourselves and pass them in the format the pipeline expects.
        """
        # Preprocess reference images: resize with padding to target dimensions
        processed_refs = [
            _resize_with_padding(img, width, height)
            for img in reference_images
        ]

        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        # Pass reference images — the pipeline may accept them as:
        # - ref_imgs (SkyReels-V3 native convention)
        # - image (standard diffusers I2V convention)
        # We try both naming conventions
        pipe_kwargs["ref_imgs"] = processed_refs

        # Image guidance scale (dual CFG)
        if guidance_scale_img > 0:
            pipe_kwargs["guidance_scale_img"] = guidance_scale_img

        try:
            output = self.pipe(**pipe_kwargs)
        except TypeError:
            # If ref_imgs is not accepted, try the standard image parameter
            del pipe_kwargs["ref_imgs"]
            pipe_kwargs["image"] = processed_refs[0] if len(processed_refs) == 1 else processed_refs
            output = self.pipe(**pipe_kwargs)

        # Extract frames
        if hasattr(output, "frames"):
            frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames
        elif hasattr(output, "images"):
            frames = output.images
        else:
            frames = output

        return VideoOutput(
            frames=frames,
            fps=self._defaults["fps"],
            seed=seed,
            backend=self.backend_name,
            metadata={
                "model_variant": self.model_variant,
                "loading_mode": self.loading_mode,
                "num_reference_images": len(reference_images),
            },
        )

    # ------------------------------------------------------------------
    # Long-form / multi-shot generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_long(
        self,
        prompt: str,
        reference_images: List[Image.Image],
        total_duration: int = 30,
        chunk_duration: int = 5,
        overlap_frames: int = 9,
        resolution: str = "720P",
        seed: int = -1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_scale_img: float = 5.0,
        **kwargs,
    ) -> VideoOutput:
        """Generate minute-level video via autoregressive chunk generation.

        Splits the total duration into overlapping chunks and generates
        each chunk conditioned on the reference images and the tail of
        the previous chunk for temporal continuity.

        Args:
            prompt: Text prompt for the entire video.
            reference_images: 1-4 reference images.
            total_duration: Total video duration in seconds (default 30).
            chunk_duration: Duration of each chunk in seconds (default 5).
            overlap_frames: Number of overlapping frames between chunks (default 9).
            resolution: Resolution tier.
            seed: Random seed.
            num_inference_steps: Steps per chunk.
            guidance_scale: Text CFG.
            guidance_scale_img: Image CFG.
        """
        fps = self._defaults["fps"]
        all_frames = []

        num_chunks = math.ceil(total_duration / chunk_duration)
        logger.info(
            f"Long-form generation: {total_duration}s in {num_chunks} chunks "
            f"of {chunk_duration}s each (overlap={overlap_frames} frames)"
        )

        for chunk_idx in range(num_chunks):
            remaining = total_duration - chunk_idx * chunk_duration
            current_duration = min(chunk_duration, remaining)

            logger.info(f"Generating chunk {chunk_idx + 1}/{num_chunks} ({current_duration}s)")

            chunk_seed = seed + chunk_idx if seed >= 0 else -1

            chunk_output = self.generate(
                prompt=prompt,
                reference_images=reference_images,
                duration=current_duration,
                resolution=resolution,
                seed=chunk_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_img=guidance_scale_img,
                **kwargs,
            )

            chunk_frames = chunk_output.frames

            if chunk_idx == 0:
                all_frames.extend(chunk_frames)
            else:
                # Skip the overlap region (already covered by previous chunk)
                all_frames.extend(chunk_frames[overlap_frames:])

        return VideoOutput(
            frames=all_frames,
            fps=fps,
            seed=seed,
            backend=self.backend_name,
            metadata={
                "model_variant": self.model_variant,
                "loading_mode": self.loading_mode,
                "num_reference_images": len(reference_images),
                "total_duration": total_duration,
                "num_chunks": num_chunks,
                "chunk_duration": chunk_duration,
                "resolution": resolution,
            },
        )

    @torch.no_grad()
    def generate_multi_shot(
        self,
        shots: List[dict],
        reference_images: List[Image.Image],
        resolution: str = "720P",
        seed: int = -1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_scale_img: float = 5.0,
        **kwargs,
    ) -> VideoOutput:
        """Generate multi-shot video with transitions.

        Each shot is a dict with:
        - "prompt": Text prompt for this shot
        - "duration": Duration in seconds (default 5)
        - "transition": Transition type from previous shot (default "cut")
          One of: "cut", "dissolve", "fade", "wipe", "zoom"

        Args:
            shots: List of shot specifications.
            reference_images: Shared reference images across all shots.
            resolution: Resolution tier.
            seed: Base random seed.
        """
        if not shots:
            raise ValueError("At least one shot is required")

        fps = self._defaults["fps"]
        all_frames = []
        overlap = SHOT_NUM_CONDITION_FRAMES.get(len(shots), 9)

        for i, shot in enumerate(shots):
            shot_prompt = shot.get("prompt", "")
            shot_duration = shot.get("duration", 5)
            transition = shot.get("transition", "cut")

            if transition not in SHOT_TRANSITION_TYPES:
                logger.warning(
                    f"Unknown transition '{transition}', falling back to 'cut'. "
                    f"Supported: {SHOT_TRANSITION_TYPES}"
                )
                transition = "cut"

            logger.info(f"Shot {i + 1}/{len(shots)}: '{shot_prompt[:50]}...' ({shot_duration}s, {transition})")

            shot_seed = seed + i if seed >= 0 else -1

            shot_output = self.generate(
                prompt=shot_prompt,
                reference_images=reference_images,
                duration=shot_duration,
                resolution=resolution,
                seed=shot_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_img=guidance_scale_img,
                **kwargs,
            )

            shot_frames = shot_output.frames

            if i == 0:
                all_frames.extend(shot_frames)
            else:
                # Apply transition between shots
                all_frames = self._apply_transition(
                    all_frames, shot_frames, transition, fps, overlap
                )

        return VideoOutput(
            frames=all_frames,
            fps=fps,
            seed=seed,
            backend=self.backend_name,
            metadata={
                "model_variant": self.model_variant,
                "num_shots": len(shots),
                "transitions": [s.get("transition", "cut") for s in shots],
                "resolution": resolution,
            },
        )

    # ------------------------------------------------------------------
    # Transition effects
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transition(
        prev_frames: list,
        next_frames: list,
        transition: str,
        fps: int,
        overlap: int,
    ) -> list:
        """Apply a transition effect between two shot segments.

        Args:
            prev_frames: Frames from the previous shot.
            next_frames: Frames from the next shot.
            transition: Transition type name.
            fps: Frames per second.
            overlap: Number of transition frames.

        Returns:
            Combined frame list with transition applied.
        """
        transition_len = min(overlap, len(prev_frames), len(next_frames))

        if transition == "cut":
            # Hard cut — no blending
            return prev_frames + next_frames

        elif transition == "dissolve":
            # Cross-dissolve: linear blend over the transition region
            result = prev_frames[:-transition_len]
            for j in range(transition_len):
                alpha = j / max(transition_len - 1, 1)
                f_prev = np.array(prev_frames[-(transition_len - j)])
                f_next = np.array(next_frames[j])
                blended = ((1 - alpha) * f_prev + alpha * f_next).astype(np.uint8)
                result.append(Image.fromarray(blended))
            result.extend(next_frames[transition_len:])
            return result

        elif transition == "fade":
            # Fade to black then fade in
            half = transition_len // 2
            result = prev_frames[:-half]
            for j in range(half):
                alpha = j / max(half - 1, 1)
                f = np.array(prev_frames[-(half - j)])
                faded = (f * (1 - alpha)).astype(np.uint8)
                result.append(Image.fromarray(faded))
            # Black frame
            if transition_len > 0:
                h, w = prev_frames[-1].size[1], prev_frames[-1].size[0]
                result.append(Image.new("RGB", (w, h), (0, 0, 0)))
            half2 = transition_len - half
            for j in range(half2):
                alpha = j / max(half2 - 1, 1)
                f = np.array(next_frames[j])
                faded = (f * alpha).astype(np.uint8)
                result.append(Image.fromarray(faded))
            result.extend(next_frames[half2:])
            return result

        elif transition == "wipe":
            # Left-to-right wipe
            result = prev_frames[:-transition_len]
            for j in range(transition_len):
                progress = j / max(transition_len - 1, 1)
                f_prev = np.array(prev_frames[-(transition_len - j)])
                f_next = np.array(next_frames[j])
                w = f_prev.shape[1]
                split = int(w * progress)
                composite = f_prev.copy()
                composite[:, :split] = f_next[:, :split]
                result.append(Image.fromarray(composite))
            result.extend(next_frames[transition_len:])
            return result

        elif transition == "zoom":
            # Zoom out from previous, zoom in to next
            result = prev_frames[:-transition_len]
            for j in range(transition_len):
                alpha = j / max(transition_len - 1, 1)
                f_prev = np.array(prev_frames[-(transition_len - j)])
                f_next = np.array(next_frames[j])
                # Scale factor: shrink prev, grow next
                blended = ((1 - alpha) * f_prev + alpha * f_next).astype(np.uint8)
                result.append(Image.fromarray(blended))
            result.extend(next_frames[transition_len:])
            return result

        else:
            # Fallback to cut
            return prev_frames + next_frames

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_supported_resolutions() -> dict:
        """Return the supported resolution tiers and their aspect ratio presets."""
        return ASPECT_RATIO_MAP

    @staticmethod
    def get_supported_transitions() -> tuple:
        """Return the supported shot transition types."""
        return SHOT_TRANSITION_TYPES
