"""
Base Pipeline — abstract interface for all video generation backends.

All backends (Wan, HunyuanVideo, CogVideoX, LTX, AnimateDiff) inherit from
BasePipeline and implement the same load/generate/save interface.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VideoOutput:
    """Standardized output from any video generation backend."""
    frames: list  # List[PIL.Image] — the generated video frames
    fps: int = 8
    seed: int = -1
    backend: str = ""
    metadata: dict = field(default_factory=dict)


class BasePipeline(ABC):
    """Abstract base class for unified video generation."""

    backend_name: str = "base"

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "none",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        **kwargs,
    ) -> "BasePipeline":
        """Load model and return a pipeline instance."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> VideoOutput:
        """Generate video frames from a text prompt."""
        ...

    def save(self, output: VideoOutput, path: str, fps: int = 8):
        """Save VideoOutput to a GIF or MP4 file.

        For MP4, uses ffmpeg with H.264 encoding for high quality.
        Falls back to diffusers export_to_video (OpenCV/MPEG-4) if ffmpeg
        is unavailable.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if path.endswith(".mp4"):
            self._save_mp4_ffmpeg(output.frames, path, fps)
        else:
            from diffusers.utils import export_to_gif
            export_to_gif(output.frames, path)
        logger.info(f"Saved video to {path}")

    @staticmethod
    def _save_mp4_ffmpeg(frames: list, path: str, fps: int):
        """Save frames to MP4 using ffmpeg pipe (H.264, CRF 18)."""
        import subprocess
        import shutil

        if not shutil.which("ffmpeg"):
            from diffusers.utils import export_to_video
            logger.warning("ffmpeg not found, falling back to OpenCV export (lower quality)")
            export_to_video(frames, path, fps=fps)
            return

        w, h = frames[0].size
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-an",
            path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        import numpy as np
        for frame in frames:
            proc.stdin.write(np.array(frame).tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().decode()[-200:]
            logger.warning(f"ffmpeg encode failed: {err}")
            from diffusers.utils import export_to_video
            export_to_video(frames, path, fps=fps)

    def _make_generator(self, seed: int, device: str) -> Optional[torch.Generator]:
        if seed >= 0:
            return torch.Generator(device=device).manual_seed(seed)
        return None

    def _apply_offloading(self, pipe, strategy: str, device: str = "cuda"):
        """Apply memory offloading strategy to a diffusers pipeline.

        Note: CPU offloading only works with CUDA. For MPS, we skip offloading
        and move the full pipeline to the device instead.
        """
        # CPU offloading requires CUDA — skip for MPS/CPU and just move to device
        if device != "cuda" and device != "cpu":
            logger.info(f"Offloading not supported on {device}, moving pipeline to {device}")
            pipe.to(device)
            return

        if strategy == "model_cpu" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif strategy == "sequential_cpu" and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        else:
            logger.warning(f"Offload strategy '{strategy}' not available, moving to {device}")
            pipe.to(device)

    def _apply_vae_opts(self, pipe, slicing: bool = True, tiling: bool = False):
        """Apply VAE memory optimizations."""
        if slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    def _apply_quantization(self, quantization: str):
        """Return a PipelineQuantizationConfig or None."""
        if quantization == "none":
            return None

        try:
            from diffusers import PipelineQuantizationConfig
        except ImportError:
            logger.warning("PipelineQuantizationConfig not available; skipping quantization. Upgrade diffusers>=0.31")
            return None

        if quantization == "nf4":
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer"],
            )
        elif quantization == "int8":
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=["transformer"],
            )
        elif quantization == "fp8":
            return PipelineQuantizationConfig(
                quant_backend="torchao",
                quant_kwargs={"quant_type": "float8_e4m3fn"},
                components_to_quantize=["transformer"],
            )
        else:
            logger.warning(f"Unknown quantization: {quantization}, skipping")
            return None
