"""
LTX-2 Backend — wraps diffusers LTX2Pipeline for joint audio-video generation.

LTX-2 is the first open-source model for synchronized audio-video generation,
built on an asymmetric dual-stream DiT architecture:
  - 14B-parameter video stream
  - 5B-parameter audio stream
  - 19B total parameters

Model variants:
- Lightricks/LTX-2                    (19B, dev — full precision, bfloat16)
- rootonchair/LTX-2-19b-distilled     (19B, distilled — 8 steps, CFG=1.0)

Key capabilities:
- Joint text-to-audio-video generation (single forward pass)
- Image-to-audio-video generation (condition on input image)
- Video-only fallback (generates video without audio stream)
- Two-stage pipeline: base generation + latent upsampler for 2x spatial upscale
- LoRA support (distilled LoRA for fast Stage 2 refinement)

Constraints:
- Width & height must be divisible by 32
- num_frames must be divisible by 8, plus 1 (e.g. 121 = 15*8 + 1)
- CUDA only (19B model requires GPU; FP8 quantization available)
- Minimum ~16GB VRAM with model_cpu offloading (FP8), ~24GB recommended

Audio output:
- Raw waveform tensor decoded by the vocoder component
- Default sample rate: 24000 Hz
- Synchronized with video at generation time (dual-stream attention)
"""

import logging
import os
from typing import Optional, List

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────────────────

LTX2_MODELS = {
    "19b": "Lightricks/LTX-2",
    "19b-distilled": "rootonchair/LTX-2-19b-distilled",
}

# Default alias
LTX2_MODELS["default"] = LTX2_MODELS["19b"]

# Per-variant generation defaults
MODEL_DEFAULTS = {
    "19b": dict(
        width=768,
        height=512,
        num_frames=121,
        fps=24,
        guidance_scale=4.0,
        steps=40,
        sigmas=None,
    ),
    "19b-distilled": dict(
        width=768,
        height=512,
        num_frames=121,
        fps=24,
        guidance_scale=1.0,
        steps=8,
        sigmas=None,  # populated at load time from DISTILLED_SIGMA_VALUES
    ),
}

MODEL_DEFAULTS["default"] = MODEL_DEFAULTS["19b"]

# Standard negative prompt recommended by Lightricks
DEFAULT_NEGATIVE_PROMPT = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, "
    "disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, "
    "weird hand, ugly, transition, static."
)


def _align_resolution(width: int, height: int, num_frames: int):
    """Ensure width/height are divisible by 32 and num_frames is 8k+1.

    Rounds up to the nearest valid value if constraints are not met.
    """
    if width % 32 != 0:
        width = ((width + 31) // 32) * 32
        logger.warning(f"Width adjusted to {width} (must be divisible by 32)")
    if height % 32 != 0:
        height = ((height + 31) // 32) * 32
        logger.warning(f"Height adjusted to {height} (must be divisible by 32)")
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1
        logger.warning(f"num_frames adjusted to {num_frames} (must be 8k+1)")
    return width, height, num_frames


class LTX2Backend(BasePipeline):
    """LTX-2 joint audio-video generation backend.

    Supports three generation modes:
    - "text_to_av": Joint text-to-audio-video (default)
    - "image_to_av": Image-conditioned audio-video generation
    - "text_to_video": Video-only generation (no audio stream)
    """

    backend_name = "ltx2"

    def __init__(
        self,
        pipe,
        model_variant: str = "19b",
        mode: str = "text_to_av",
        upsampler_pipe=None,
    ):
        self.pipe = pipe
        self.model_variant = model_variant
        self.mode = mode
        self.upsampler_pipe = upsampler_pipe
        self._defaults = MODEL_DEFAULTS.get(model_variant, MODEL_DEFAULTS["default"])

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "model_cpu",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        model_variant: str = "19b",
        mode: str = "text_to_av",
        enable_upsampler: bool = False,
        lora_paths: Optional[List[str]] = None,
        lora_scales: Optional[List[float]] = None,
        **kwargs,
    ) -> "LTX2Backend":
        """Load LTX-2 pipeline from HuggingFace Hub or local path.

        Args:
            model_path: HuggingFace model ID or local path. Auto-resolved from model_variant if None.
            torch_dtype: Model precision. LTX-2 is trained in bfloat16 — use that unless quantizing.
            device: Target device. Must be "cuda" for the 19B model.
            quantization: One of "none", "nf4", "int8", "fp8". FP8 recommended for 16GB GPUs.
            offload_strategy: "none", "model_cpu", or "sequential_cpu".
                model_cpu is recommended for 24GB GPUs, sequential_cpu for 16GB.
            enable_vae_slicing: Slice VAE decoding to reduce peak VRAM.
            enable_vae_tiling: Tile VAE decoding (needed for Stage 2 / high-res).
            model_variant: One of "19b", "19b-distilled", "default".
            mode: Generation mode — "text_to_av", "image_to_av", or "text_to_video".
            enable_upsampler: Load the 2x latent upsampler for two-stage generation.
            lora_paths: Optional list of LoRA weight paths to load.
            lora_scales: Optional list of LoRA scales (one per path).

        Returns:
            LTX2Backend instance ready for generation.
        """
        # Resolve model path
        if model_path is None:
            model_path = LTX2_MODELS.get(model_variant, LTX2_MODELS["default"])

        # MPS is not supported for the 19B model
        if device == "mps":
            logger.error(
                "LTX-2 (19B params) requires CUDA. Apple Silicon / MPS is not supported. "
                "Consider using the 'ltx' backend (LTX-Video v1) for MPS."
            )
            raise RuntimeError("LTX-2 requires CUDA. MPS is not supported for the 19B dual-stream model.")

        logger.info(
            f"Loading LTX-2 {model_variant} from {model_path} "
            f"(mode={mode}, dtype={torch_dtype}, quant={quantization})"
        )

        # Choose pipeline class based on mode
        if mode in ("image_to_av", "i2v"):
            from diffusers import LTX2ImageToVideoPipeline as PipelineClass
            mode = "image_to_av"
        else:
            from diffusers import LTX2Pipeline as PipelineClass

        # Quantization — quantize the transformer (the main 19B component)
        quant_config = get_quantization_config(quantization, components=["transformer"])

        load_kwargs = dict(torch_dtype=torch_dtype)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        pipe = PipelineClass.from_pretrained(model_path, **load_kwargs)

        # Load distilled sigma values for the distilled variant
        defaults = dict(MODEL_DEFAULTS.get(model_variant, MODEL_DEFAULTS["default"]))
        if model_variant == "19b-distilled":
            try:
                from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
                defaults["sigmas"] = DISTILLED_SIGMA_VALUES
                logger.info(f"Loaded distilled sigma schedule ({len(DISTILLED_SIGMA_VALUES)} values)")
            except ImportError:
                logger.warning("DISTILLED_SIGMA_VALUES not found in diffusers; using default step count")

        # Load the latent upsampler if requested (for two-stage generation)
        upsampler_pipe = None
        if enable_upsampler:
            upsampler_pipe = cls._load_upsampler(model_path, pipe, torch_dtype, device, offload_strategy)

        instance = cls(pipe, model_variant=model_variant, mode=mode, upsampler_pipe=upsampler_pipe)
        instance._defaults = defaults

        # Load LoRAs if provided
        if lora_paths:
            instance._load_loras(lora_paths, lora_scales or [1.0] * len(lora_paths))

        # Apply offloading strategy
        if offload_strategy != "none":
            instance._apply_offloading(pipe, offload_strategy, device=device)
        else:
            pipe.to(device)

        instance._apply_vae_opts(pipe, slicing=enable_vae_slicing, tiling=enable_vae_tiling)

        return instance

    @classmethod
    def _load_upsampler(cls, model_path, pipe, torch_dtype, device, offload_strategy):
        """Load the LTX-2 latent upsampler for two-stage generation."""
        try:
            from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        except ImportError:
            logger.warning(
                "LTX2LatentUpsamplePipeline not available in this diffusers version. "
                "Skipping upsampler — single-stage generation only."
            )
            return None

        logger.info(f"Loading LTX-2 latent upsampler from {model_path}/latent_upsampler")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            model_path, subfolder="latent_upsampler", torch_dtype=torch_dtype,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)

        if offload_strategy == "model_cpu":
            upsample_pipe.enable_model_cpu_offload()
        elif offload_strategy == "sequential_cpu":
            upsample_pipe.enable_sequential_cpu_offload()
        else:
            upsample_pipe.to(device)

        return upsample_pipe

    def _load_loras(self, lora_paths: List[str], lora_scales: List[float]):
        """Load LoRA weights into the pipeline."""
        adapter_names = []
        for i, (path, scale) in enumerate(zip(lora_paths, lora_scales)):
            adapter_name = f"lora_{i}"
            logger.info(f"Loading LoRA: {path} (scale={scale}, adapter={adapter_name})")

            # Handle both HuggingFace repo IDs and local file paths
            if "/" in path and not path.startswith("/") and not path.startswith("."):
                parts = path.rsplit("/", 1)
                if len(parts) == 2 and "." in parts[1]:
                    self.pipe.load_lora_weights(parts[0], weight_name=parts[1], adapter_name=adapter_name)
                else:
                    self.pipe.load_lora_weights(path, adapter_name=adapter_name)
            else:
                self.pipe.load_lora_weights(path, adapter_name=adapter_name)

            adapter_names.append(adapter_name)

        if adapter_names:
            scales = lora_scales[:len(adapter_names)]
            self.pipe.set_adapters(adapter_names, adapter_weights=scales)
            logger.info(f"Activated LoRAs: {adapter_names} with scales {scales}")

    def _get_audio_sample_rate(self) -> int:
        """Get the audio sample rate from the vocoder config."""
        if hasattr(self.pipe, "vocoder") and self.pipe.vocoder is not None:
            config = getattr(self.pipe.vocoder, "config", None)
            if config is not None and hasattr(config, "output_sampling_rate"):
                return config.output_sampling_rate
            if config is not None and hasattr(config, "sampling_rate"):
                return config.sampling_rate
        # Default LTX-2 audio sample rate
        return 24000

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
        # LTX-2 specific parameters
        frame_rate: float = 0,
        enable_audio: bool = True,
        enable_upscale: bool = False,
        sigmas: Optional[List[float]] = None,
        guidance_rescale: float = 0.0,
        max_sequence_length: int = 1024,
        **kwargs,
    ) -> VideoOutput:
        """Generate synchronized audio and video from a text prompt.

        Args:
            prompt: Text description of the desired video content.
            negative_prompt: What to avoid in generation. Uses recommended default if empty.
            width: Video width in pixels (must be divisible by 32). Default: 768.
            height: Video height in pixels (must be divisible by 32). Default: 512.
            num_frames: Number of frames to generate (must be 8k+1). Default: 121.
            num_inference_steps: Denoising steps. Default: 40 (dev) or 8 (distilled).
            guidance_scale: CFG scale. Default: 4.0 (dev) or 1.0 (distilled).
            seed: Random seed for reproducibility. -1 for random.
            image: Optional input image for image-to-video mode.
            frame_rate: Video frame rate in FPS. Default: 24.0.
            enable_audio: Whether to generate audio alongside video. Default: True.
            enable_upscale: Whether to run the 2x latent upsampler (Stage 2). Default: False.
            sigmas: Custom sigma schedule. Auto-populated for distilled variant.
            guidance_rescale: Guidance rescale factor for CFG. Default: 0.0.
            max_sequence_length: Maximum prompt token length. Default: 1024.

        Returns:
            VideoOutput with frames, optional audio tensor, and metadata.
        """
        d = self._defaults

        # Apply defaults
        width = width or d["width"]
        height = height or d["height"]
        num_frames = num_frames or d["num_frames"]
        num_inference_steps = num_inference_steps or d["steps"]
        guidance_scale = guidance_scale or d["guidance_scale"]
        frame_rate = frame_rate or d["fps"]
        sigmas = sigmas or d.get("sigmas")

        # Enforce LTX-2 alignment constraints
        width, height, num_frames = _align_resolution(width, height, num_frames)

        # Use default negative prompt if none provided
        if not negative_prompt:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        # ── Build pipeline kwargs ────────────────────────────────────────
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=float(frame_rate),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )

        if sigmas is not None:
            pipe_kwargs["sigmas"] = sigmas

        # Image-to-video mode
        if image is not None and self.mode == "image_to_av":
            pipe_kwargs["image"] = image
        elif image is not None and self.mode != "image_to_av":
            logger.warning(
                "Image provided but pipeline is in text_to_av mode. "
                "Load with mode='image_to_av' to use image conditioning. Ignoring image."
            )

        # ── Two-stage vs. single-stage generation ────────────────────────
        if enable_upscale and self.upsampler_pipe is not None:
            return self._generate_two_stage(pipe_kwargs, frame_rate, seed, enable_audio)
        else:
            return self._generate_single_stage(pipe_kwargs, frame_rate, seed, enable_audio)

    def _generate_single_stage(
        self, pipe_kwargs: dict, frame_rate: float, seed: int, enable_audio: bool,
    ) -> VideoOutput:
        """Single-stage generation: generate and decode in one pass."""
        pipe_kwargs["output_type"] = "pil"
        pipe_kwargs["return_dict"] = False

        result = self.pipe(**pipe_kwargs)

        # LTX2Pipeline returns (video, audio) tuple when return_dict=False
        video_output, audio_output = self._unpack_output(result)

        # Extract frames from the video output
        frames = self._extract_frames(video_output)

        # Process audio
        audio_tensor = None
        audio_sr = 0
        if enable_audio and audio_output is not None:
            audio_tensor = self._process_audio(audio_output)
            audio_sr = self._get_audio_sample_rate()

        return VideoOutput(
            frames=frames,
            fps=int(frame_rate),
            seed=seed,
            backend=self.backend_name,
            audio=audio_tensor,
            audio_sample_rate=audio_sr,
            metadata={
                "model_variant": self.model_variant,
                "mode": self.mode,
                "stage": "single",
            },
        )

    def _generate_two_stage(
        self, pipe_kwargs: dict, frame_rate: float, seed: int, enable_audio: bool,
    ) -> VideoOutput:
        """Two-stage generation: base generation in latent space + upsampled decode.

        Stage 1: Generate latents at base resolution
        Stage 2: Upsample latents 2x and refine with distilled LoRA
        """
        logger.info("Two-stage generation: Stage 1 — base latent generation")

        # Stage 1: generate in latent space
        stage1_kwargs = dict(pipe_kwargs)
        stage1_kwargs["output_type"] = "latent"
        stage1_kwargs["return_dict"] = False

        result = self.pipe(**stage1_kwargs)
        video_latent, audio_latent = self._unpack_output(result)

        # Stage 2: upsample video latents
        logger.info("Two-stage generation: Stage 2 — latent upsampling")
        upscaled_video_latent = self.upsampler_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        # Stage 2 refinement: load distilled LoRA and re-denoise
        try:
            from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
            from diffusers import FlowMatchEulerDiscreteScheduler

            # Load Stage 2 distilled LoRA if not already loaded
            if not hasattr(self, "_stage2_lora_loaded") or not self._stage2_lora_loaded:
                model_path = LTX2_MODELS.get(self.model_variant, LTX2_MODELS["default"])
                self.pipe.load_lora_weights(
                    model_path,
                    adapter_name="stage_2_distilled",
                    weight_name="ltx-2-19b-distilled-lora-384.safetensors",
                )
                self.pipe.set_adapters("stage_2_distilled", 1.0)
                self._stage2_lora_loaded = True

            # Enable VAE tiling for high-res decode
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()

            # Switch scheduler for Stage 2
            original_scheduler = self.pipe.scheduler
            new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )
            self.pipe.scheduler = new_scheduler

            # Stage 2 inference
            stage2_kwargs = dict(
                latents=upscaled_video_latent,
                audio_latents=audio_latent if enable_audio else None,
                prompt=pipe_kwargs["prompt"],
                negative_prompt=pipe_kwargs.get("negative_prompt"),
                num_inference_steps=3,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                guidance_scale=1.0,
                output_type="pil",
                return_dict=False,
            )

            if "generator" in pipe_kwargs:
                stage2_kwargs["generator"] = pipe_kwargs["generator"]

            result = self.pipe(**stage2_kwargs)
            video_output, audio_output = self._unpack_output(result)

            # Restore original scheduler
            self.pipe.scheduler = original_scheduler

        except ImportError:
            logger.warning(
                "Stage 2 distilled utilities not available. "
                "Decoding Stage 1 latents directly (lower quality)."
            )
            # Fallback: just decode the upscaled latents via VAE
            stage2_kwargs = dict(
                latents=upscaled_video_latent,
                audio_latents=audio_latent if enable_audio else None,
                prompt=pipe_kwargs["prompt"],
                negative_prompt=pipe_kwargs.get("negative_prompt"),
                num_inference_steps=pipe_kwargs.get("num_inference_steps", 3),
                guidance_scale=1.0,
                output_type="pil",
                return_dict=False,
            )
            result = self.pipe(**stage2_kwargs)
            video_output, audio_output = self._unpack_output(result)

        frames = self._extract_frames(video_output)

        audio_tensor = None
        audio_sr = 0
        if enable_audio and audio_output is not None:
            audio_tensor = self._process_audio(audio_output)
            audio_sr = self._get_audio_sample_rate()

        return VideoOutput(
            frames=frames,
            fps=int(frame_rate),
            seed=seed,
            backend=self.backend_name,
            audio=audio_tensor,
            audio_sample_rate=audio_sr,
            metadata={
                "model_variant": self.model_variant,
                "mode": self.mode,
                "stage": "two-stage",
            },
        )

    @staticmethod
    def _unpack_output(result):
        """Unpack pipeline output into (video, audio) regardless of format.

        LTX2Pipeline returns:
        - return_dict=False: tuple of (video, audio)
        - return_dict=True: LTX2PipelineOutput with .frames and .audio
        """
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]
            else:
                return result[0], None
        # Named output object
        video = getattr(result, "frames", None) or getattr(result, "images", None)
        audio = getattr(result, "audio", None)
        return video, audio

    @staticmethod
    def _extract_frames(video_output):
        """Extract a flat list of PIL frames from the pipeline video output.

        Handles both nested list format (List[List[PIL.Image]]) and
        numpy/tensor formats.
        """
        if isinstance(video_output, list):
            # Nested list: video_output[batch][frame]
            if len(video_output) > 0 and isinstance(video_output[0], list):
                return video_output[0]
            # Flat list of PIL images
            if len(video_output) > 0 and isinstance(video_output[0], Image.Image):
                return video_output
            # Single batch of numpy arrays
            return video_output
        # numpy or tensor — need to convert
        import numpy as np
        if hasattr(video_output, "numpy"):
            video_output = video_output.cpu().numpy()
        if isinstance(video_output, np.ndarray):
            # Shape: (batch, frames, H, W, C) or (frames, H, W, C)
            if video_output.ndim == 5:
                video_output = video_output[0]
            # Convert each frame to PIL
            frames = []
            for i in range(video_output.shape[0]):
                frame = video_output[i]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                frames.append(Image.fromarray(frame))
            return frames
        return video_output

    @staticmethod
    def _process_audio(audio_output) -> Optional[torch.Tensor]:
        """Convert audio output to a clean tensor for saving.

        LTX-2 audio output may be a nested list, numpy array, or tensor.
        Returns a 1D float32 tensor suitable for WAV encoding.
        """
        if audio_output is None:
            return None

        import numpy as np

        if isinstance(audio_output, torch.Tensor):
            audio = audio_output.float().cpu()
        elif isinstance(audio_output, np.ndarray):
            audio = torch.from_numpy(audio_output).float()
        elif isinstance(audio_output, list):
            # Nested list: audio[batch] -> array
            if len(audio_output) > 0:
                first = audio_output[0]
                if isinstance(first, torch.Tensor):
                    audio = first.float().cpu()
                elif isinstance(first, np.ndarray):
                    audio = torch.from_numpy(first).float()
                else:
                    return None
            else:
                return None
        else:
            return None

        # Flatten batch dimension if present
        if audio.ndim >= 3:
            audio = audio[0]
        # Squeeze to 1D if mono
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        return audio

    def save(self, output: VideoOutput, path: str, fps: int = 0):
        """Save VideoOutput to file with embedded audio.

        For MP4 output with audio, uses ffmpeg to mux the audio and video
        streams. For GIF output, audio is silently dropped.

        Alternatively, the diffusers encode_video utility can be used directly
        for maximum compatibility with LTX-2 outputs.
        """
        fps = fps or output.fps or self._defaults["fps"]

        if path.endswith(".mp4") and output.audio is not None and output.audio_sample_rate > 0:
            # Use diffusers encode_video if available (handles LTX-2 format natively)
            try:
                from diffusers.pipelines.ltx2.export_utils import encode_video
                import numpy as np

                # Convert frames to numpy array if needed
                if isinstance(output.frames[0], Image.Image):
                    video_np = np.stack([np.array(f) for f in output.frames]) / 255.0
                else:
                    video_np = output.frames
                    if isinstance(video_np, torch.Tensor):
                        video_np = video_np.cpu().numpy()

                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                encode_video(
                    video_np,
                    fps=float(fps),
                    audio=output.audio.float().cpu(),
                    audio_sample_rate=output.audio_sample_rate,
                    output_path=path,
                )
                logger.info(f"Saved audio-video to {path} (via diffusers encode_video)")
                return
            except ImportError:
                logger.info("diffusers encode_video not available, falling back to ffmpeg")

        # Fallback to base class (ffmpeg with audio muxing support)
        super().save(output, path, fps)
