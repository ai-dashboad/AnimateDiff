"""
Wan VACE Backend — wraps diffusers WanVACEPipeline for controllable video generation.

Wan VACE (Video All-in-one Creation and Editing) supports:
- Reference-to-Video: Generate consistent video from a reference image (character/subject/scene)
- Video Continuation: Extend a video by generating new frames from the last frames
- Inpainting: Mask-based editing of video regions
- First-Last-Frame: Interpolate motion between two keyframes

Model variants (Wan 2.1 VACE, uses WanVACETransformer3DModel):
- Wan-AI/Wan2.1-VACE-1.3B-diffusers   (1.3B params, ~8 GB VRAM, MPS compatible)
- Wan-AI/Wan2.1-VACE-14B-diffusers    (14B params, ~24 GB VRAM with offloading)

VACE Masking Convention:
- Black mask (0):   Condition on this frame/region — the model preserves it
- White mask (255): Generate new content for this frame/region
- Gray (128,128,128) placeholder frames: Neutral filler for frames to be generated

NOTE: While the model IDs say "Wan2.1", the WanVACEPipeline in diffusers 0.36+
      supports dual-transformer / MoE mode via transformer_2 and boundary_ratio,
      so this backend is forward-compatible with any future Wan 2.2 VACE weights.
"""

import logging
from typing import Optional, List, Union
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

# Available VACE model checkpoints
VACE_MODELS = {
    "1.3B": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "14B": "Wan-AI/Wan2.1-VACE-14B-diffusers",
}

# Defaults per model variant
MODEL_DEFAULTS = {
    "1.3B": dict(width=832, height=480, num_frames=81, fps=16, guidance_scale=5.0, steps=30),
    "14B": dict(width=1280, height=720, num_frames=81, fps=16, guidance_scale=5.0, steps=50),
}

# Supported VACE modes
VACE_MODES = ("reference_to_video", "continuation", "inpaint", "first_last_frame")


class Wan22VACEBackend(BasePipeline):
    """Backend for Wan VACE — controllable video generation with masks and reference images."""

    backend_name = "wan22_vace"

    def __init__(self, pipe, model_variant: str = "1.3B"):
        self.pipe = pipe
        self.model_variant = model_variant
        self._defaults = MODEL_DEFAULTS.get(model_variant, MODEL_DEFAULTS["1.3B"])

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
        model_variant: str = "1.3B",
        lora_paths: Optional[List[str]] = None,
        lora_scales: Optional[List[float]] = None,
        **kwargs,
    ) -> "Wan22VACEBackend":
        from diffusers import WanVACEPipeline, AutoencoderKLWan

        # Resolve model path
        if model_path is None:
            model_path = VACE_MODELS.get(model_variant, VACE_MODELS["1.3B"])

        # MPS safety: use float32 (float16/bfloat16 not fully supported for Wan on MPS)
        if device == "mps":
            torch_dtype = torch.float32
            logger.info("MPS detected: using float32 (float16/bfloat16 not fully supported for Wan on MPS)")
            # 14B is very large for MPS without offloading; warn but allow attempt
            if model_variant == "14B":
                logger.warning(
                    "Wan VACE 14B is ~56 GB in float32 — this will likely OOM on MPS. "
                    "Consider using 1.3B variant or CUDA with model_cpu offloading."
                )

        logger.info(f"Loading Wan VACE {model_variant} from {model_path} (dtype={torch_dtype}, quant={quantization})")

        # VAE must always be float32 for Wan
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)

        # Quantization config
        quant_config = get_quantization_config(quantization, components=["transformer"])
        load_kwargs = dict(torch_dtype=torch_dtype, vae=vae)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        pipe = WanVACEPipeline.from_pretrained(model_path, **load_kwargs)

        # Fix: transformers 5.x UMT5 embed_tokens zero-weight bug
        te = pipe.text_encoder
        if (hasattr(te, "shared") and hasattr(te, "encoder")
                and hasattr(te.encoder, "embed_tokens")
                and te.encoder.embed_tokens.weight.abs().sum().item() == 0
                and te.shared.weight.abs().sum().item() > 0):
            logger.warning("Fixing UMT5 embed_tokens: binding shared.weight -> encoder.embed_tokens.weight")
            te.encoder.embed_tokens.weight = te.shared.weight

        instance = cls(pipe, model_variant=model_variant)

        # Load LoRAs if provided
        if lora_paths:
            instance._load_loras(lora_paths, lora_scales or [1.0] * len(lora_paths))

        # Apply offloading
        if offload_strategy != "none":
            instance._apply_offloading(pipe, offload_strategy, device=device)
        else:
            pipe.to(device)

        instance._apply_vae_opts(pipe, slicing=enable_vae_slicing, tiling=enable_vae_tiling)

        return instance

    def _load_loras(self, lora_paths: List[str], lora_scales: List[float]):
        """Load LoRA weights into the VACE transformer."""
        lora_names = []
        for i, (path, scale) in enumerate(zip(lora_paths, lora_scales)):
            adapter_name = f"lora_{i}"
            logger.info(f"Loading LoRA: {path} (scale={scale})")

            # Handle both repo IDs and local paths
            if "/" in path and not path.startswith("/") and not path.startswith("."):
                parts = path.rsplit("/", 1)
                if len(parts) == 2 and "." in parts[1]:
                    self.pipe.load_lora_weights(parts[0], weight_name=parts[1], adapter_name=adapter_name)
                else:
                    self.pipe.load_lora_weights(path, adapter_name=adapter_name)
            else:
                self.pipe.load_lora_weights(path, adapter_name=adapter_name)

            lora_names.append(adapter_name)

        if lora_names:
            scales = lora_scales[:len(lora_names)]
            self.pipe.set_adapters(lora_names, adapter_weights=scales)
            logger.info(f"Activated LoRAs: {lora_names} with scales {scales}")

    # -------------------------------------------------------------------------
    # Conditioning preparation helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _prepare_reference_to_video(
        image: Image.Image,
        width: int,
        height: int,
        num_frames: int,
    ) -> tuple:
        """Prepare video frames and mask for reference-to-video (image-to-video) mode.

        The first frame is the reference image (conditioned, black mask).
        Remaining frames are gray placeholders (generated, white mask).

        Returns:
            (video_frames, mask_frames) — both lists of PIL Images.
        """
        ref = image.resize((width, height))
        gray = Image.new("RGB", (width, height), (128, 128, 128))
        mask_black = Image.new("L", (width, height), 0)
        mask_white = Image.new("L", (width, height), 255)

        video = [ref] + [gray] * (num_frames - 1)
        mask = [mask_black] + [mask_white] * (num_frames - 1)
        return video, mask

    @staticmethod
    def _prepare_continuation(
        source_frames: List[Image.Image],
        width: int,
        height: int,
        num_frames: int,
        overlap_frames: int = 4,
    ) -> tuple:
        """Prepare video frames and mask for continuation (video extension) mode.

        The first `overlap_frames` are the last frames from the source video
        (conditioned, black mask). The rest are gray placeholders to generate.

        Args:
            source_frames: Frames from the existing video to continue from.
            overlap_frames: Number of source frames to condition on (typically 1-8).

        Returns:
            (video_frames, mask_frames) — both lists of PIL Images.
        """
        gray = Image.new("RGB", (width, height), (128, 128, 128))
        mask_black = Image.new("L", (width, height), 0)
        mask_white = Image.new("L", (width, height), 255)

        # Take the last N frames from the source video as conditioning
        overlap = min(overlap_frames, len(source_frames))
        cond_frames = [f.resize((width, height)) for f in source_frames[-overlap:]]

        gen_count = num_frames - overlap
        if gen_count <= 0:
            raise ValueError(
                f"num_frames ({num_frames}) must be greater than overlap_frames ({overlap}). "
                f"Increase num_frames or decrease overlap."
            )

        video = cond_frames + [gray] * gen_count
        mask = [mask_black] * overlap + [mask_white] * gen_count
        return video, mask

    @staticmethod
    def _prepare_inpaint(
        source_frames: List[Image.Image],
        mask_frames: List[Image.Image],
        width: int,
        height: int,
    ) -> tuple:
        """Prepare video frames and mask for inpainting mode.

        Each source frame is paired with its corresponding mask.
        Black mask = preserve, white mask = generate/inpaint.

        Args:
            source_frames: Original video frames.
            mask_frames: Per-frame masks (L mode, 0=keep 255=inpaint).

        Returns:
            (video_frames, mask_frames) — both lists of PIL Images, resized.
        """
        if len(source_frames) != len(mask_frames):
            raise ValueError(
                f"source_frames ({len(source_frames)}) and mask_frames ({len(mask_frames)}) "
                f"must have the same length"
            )

        video = [f.resize((width, height)) for f in source_frames]
        masks = [m.convert("L").resize((width, height)) for m in mask_frames]
        return video, masks

    @staticmethod
    def _prepare_first_last_frame(
        first_image: Image.Image,
        last_image: Image.Image,
        width: int,
        height: int,
        num_frames: int,
    ) -> tuple:
        """Prepare video frames and mask for first-last-frame interpolation.

        First and last frames are conditioned (black mask).
        Intermediate frames are gray placeholders (white mask).

        Returns:
            (video_frames, mask_frames) — both lists of PIL Images.
        """
        first = first_image.resize((width, height))
        last = last_image.resize((width, height))
        gray = Image.new("RGB", (width, height), (128, 128, 128))
        mask_black = Image.new("L", (width, height), 0)
        mask_white = Image.new("L", (width, height), 255)

        video = [first] + [gray] * (num_frames - 2) + [last]
        mask = [mask_black] + [mask_white] * (num_frames - 2) + [mask_black]
        return video, mask

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

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
        # VACE-specific parameters
        mode: str = "reference_to_video",
        video: Optional[List[Image.Image]] = None,
        mask: Optional[List[Image.Image]] = None,
        reference_images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        conditioning_scale: float = 1.0,
        last_image: Optional[Image.Image] = None,
        source_frames: Optional[List[Image.Image]] = None,
        mask_frames: Optional[List[Image.Image]] = None,
        overlap_frames: int = 4,
        **kwargs,
    ) -> VideoOutput:
        """Generate video using VACE conditioning.

        There are two ways to call this:

        1. **High-level mode** (recommended): Set ``mode`` to one of the supported
           modes and pass the appropriate inputs (``image``, ``source_frames``,
           ``mask_frames``, ``last_image``). The backend will prepare the
           conditioning video and mask automatically.

        2. **Low-level / raw**: Pass ``video`` and ``mask`` directly as lists of
           PIL Images (per the VACE convention: black mask = condition,
           white mask = generate). This bypasses mode-based preparation.

        Args:
            prompt: Text prompt for generation.
            negative_prompt: Negative prompt.
            mode: One of "reference_to_video", "continuation", "inpaint",
                  "first_last_frame". Ignored when ``video`` and ``mask`` are
                  provided directly.
            image: Reference image (for reference_to_video / first_last_frame).
            last_image: Last frame image (for first_last_frame mode).
            source_frames: Source video frames (for continuation / inpaint).
            mask_frames: Per-frame masks for inpainting (L-mode PIL Images).
            overlap_frames: Number of overlap frames for continuation (default 4).
            reference_images: Additional reference images for character/subject
                              consistency (passed to the pipeline as-is).
            conditioning_scale: Strength of VACE conditioning (0.0-1.0).
            video: Raw VACE video conditioning (list of PIL Images). Overrides
                   mode-based preparation.
            mask: Raw VACE mask conditioning (list of PIL Images). Overrides
                  mode-based preparation.
        """
        d = self._defaults
        width = width or d["width"]
        height = height or d["height"]
        num_frames = num_frames or d["num_frames"]
        num_inference_steps = num_inference_steps or d["steps"]
        guidance_scale = guidance_scale or d["guidance_scale"]

        # --- Build conditioning video and mask from high-level mode ---
        if video is None or mask is None:
            if mode not in VACE_MODES:
                raise ValueError(f"Unknown VACE mode: {mode}. Supported: {VACE_MODES}")

            if mode == "reference_to_video":
                if image is None:
                    raise ValueError("reference_to_video mode requires 'image' (reference image)")
                video, mask = self._prepare_reference_to_video(image, width, height, num_frames)

            elif mode == "continuation":
                frames = source_frames
                if frames is None:
                    raise ValueError("continuation mode requires 'source_frames' (list of PIL Images)")
                video, mask = self._prepare_continuation(frames, width, height, num_frames, overlap_frames)

            elif mode == "inpaint":
                if source_frames is None or mask_frames is None:
                    raise ValueError("inpaint mode requires both 'source_frames' and 'mask_frames'")
                # num_frames is determined by the source for inpainting
                num_frames = len(source_frames)
                video, mask = self._prepare_inpaint(source_frames, mask_frames, width, height)

            elif mode == "first_last_frame":
                if image is None or last_image is None:
                    raise ValueError("first_last_frame mode requires both 'image' (first frame) and 'last_image'")
                video, mask = self._prepare_first_last_frame(image, last_image, width, height, num_frames)

        else:
            # Raw mode — user provided video + mask directly
            mode = "raw"

        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            video=video,
            mask=mask,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            conditioning_scale=conditioning_scale,
            generator=generator,
            output_type="pil",
        )

        # Reference images for subject/character consistency
        if reference_images is not None:
            pipe_kwargs["reference_images"] = reference_images

        output = self.pipe(**pipe_kwargs)
        frames = output.frames[0]

        return VideoOutput(
            frames=frames,
            fps=d["fps"],
            seed=seed,
            backend=self.backend_name,
            metadata={
                "model_variant": self.model_variant,
                "mode": mode,
                "conditioning_scale": conditioning_scale,
            },
        )

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def load_video_frames(path: str) -> List[Image.Image]:
        """Load video frames from an MP4 file or a directory of images.

        Useful for loading source_frames for continuation or inpainting modes.
        """
        from diffusers.utils import load_video

        path = Path(path)
        if path.suffix in (".mp4", ".avi", ".mov", ".webm", ".gif"):
            return load_video(str(path))
        elif path.is_dir():
            frame_files = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
            if not frame_files:
                raise ValueError(f"No PNG/JPG frames found in directory: {path}")
            return [Image.open(f).convert("RGB") for f in frame_files]
        else:
            raise ValueError(f"Unsupported video format: {path}. Use MP4/AVI/MOV/WEBM/GIF or a directory of images.")

    @staticmethod
    def create_region_mask(
        width: int,
        height: int,
        num_frames: int,
        bbox: Optional[tuple] = None,
    ) -> List[Image.Image]:
        """Create a uniform region mask for inpainting.

        If ``bbox`` is provided as (x1, y1, x2, y2) in pixel coordinates,
        only the bounding box region is white (inpaint); the rest is black (keep).
        If ``bbox`` is None, the entire frame is white (full regeneration).

        Returns:
            List of L-mode PIL Images.
        """
        masks = []
        for _ in range(num_frames):
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                m = Image.new("L", (width, height), 0)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(m)
                draw.rectangle([x1, y1, x2, y2], fill=255)
                masks.append(m)
            else:
                masks.append(Image.new("L", (width, height), 255))
        return masks

    @staticmethod
    def aspect_ratio_resize(
        image: Image.Image,
        max_area: int = 480 * 832,
        mod_value: int = 16,
    ) -> tuple:
        """Resize an image to fit within max_area while respecting aspect ratio
        and alignment to mod_value.

        Returns:
            (resized_image, height, width)
        """
        aspect_ratio = image.height / image.width
        h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        resized = image.resize((w, h))
        return resized, h, w
