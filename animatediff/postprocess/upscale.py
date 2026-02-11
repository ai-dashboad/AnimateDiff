"""
Video Upscaler — Real-ESRGAN anime super-resolution for video frames.

Models:
- realesr-animevideov3: 10MB, designed for anime video (default)
- RealESRGAN_x4plus_anime_6B: 17MB, anime images
- RealESRGAN_x4plus: 64MB, general purpose

Backends:
- python: Uses realesrgan pip package (requires basicsr, may not compile on Python 3.14)
- ncnn:   Uses realesrgan-ncnn-vulkan binary (standalone, no Python deps)
- auto:   Tries python first, falls back to ncnn

MPS/Apple Silicon: Python backend uses half=False + tiling. ncnn uses Vulkan GPU natively.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Model download URLs (Python backend)
MODEL_URLS = {
    "animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

# ncnn model name mapping
NCNN_MODEL_NAMES = {
    "animevideov3": "realesr-animevideov3",
    "anime_6B": "realesrgan-x4plus-anime",
    "general": "realesrgan-x4plus",
}

# Default ncnn binary location (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NCNN_DIR = PROJECT_ROOT / "tools" / "realesrgan-ncnn-vulkan"


def _find_ncnn_binary() -> str | None:
    """Find realesrgan-ncnn-vulkan binary."""
    # Check project tools/ directory
    candidate = NCNN_DIR / "realesrgan-ncnn-vulkan"
    if candidate.is_file() and os.access(str(candidate), os.X_OK):
        return str(candidate)
    # Check PATH
    which = shutil.which("realesrgan-ncnn-vulkan")
    if which:
        return which
    return None


def _find_ncnn_models_dir() -> str | None:
    """Find ncnn models directory."""
    candidate = NCNN_DIR / "models"
    if candidate.is_dir():
        return str(candidate)
    return None


class VideoUpscaler:
    """Upscale video frames using Real-ESRGAN."""

    def __init__(
        self,
        model_name: Literal["animevideov3", "anime_6B", "general"] = "animevideov3",
        scale: int = 2,
        tile: int = 0,
        device: str = "cpu",
        backend: Literal["auto", "python", "ncnn"] = "auto",
    ):
        self.model_name = model_name
        self.scale = scale
        self.tile = tile
        self.device = device
        self.backend = backend
        self._upsampler = None
        self._resolved_backend = None

    def _resolve_backend(self) -> str:
        """Determine which backend to use."""
        if self._resolved_backend:
            return self._resolved_backend

        if self.backend == "python":
            self._resolved_backend = "python"
        elif self.backend == "ncnn":
            self._resolved_backend = "ncnn"
        else:  # auto
            try:
                import realesrgan  # noqa: F401
                import basicsr  # noqa: F401
                self._resolved_backend = "python"
                logger.info("Upscale backend: python (realesrgan pip package)")
            except ImportError:
                if _find_ncnn_binary():
                    self._resolved_backend = "ncnn"
                    logger.info("Upscale backend: ncnn-vulkan (Python realesrgan unavailable)")
                else:
                    raise ImportError(
                        "No upscale backend available. Either:\n"
                        "  1. pip install realesrgan (requires basicsr)\n"
                        "  2. Place realesrgan-ncnn-vulkan binary in tools/realesrgan-ncnn-vulkan/"
                    )
        return self._resolved_backend

    def _ensure_loaded(self):
        """Lazy-load the Python upscaler model."""
        if self._upsampler is not None:
            return

        from realesrgan import RealESRGANer
        from basicsr.utils.download_util import load_file_from_url

        model_url = MODEL_URLS.get(self.model_name, MODEL_URLS["animevideov3"])
        model_path = load_file_from_url(url=model_url, model_dir="weights/realesrgan", progress=True)

        # Select model architecture
        if self.model_name == "animevideov3":
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=16, upscale=4, act_type="prelu",
            )
        elif self.model_name == "anime_6B":
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4,
            )
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )

        # MPS requires fp32; CUDA can use fp16
        half = self.device == "cuda"

        # Determine gpu_id
        gpu_id = None
        if self.device == "cuda":
            gpu_id = 0
        elif self.device == "mps":
            gpu_id = None  # Real-ESRGAN uses torch.device, not gpu_id for MPS

        # MPS tiling is recommended for memory
        tile = self.tile
        if self.device == "mps" and tile == 0:
            tile = 256

        self._upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half,
            gpu_id=gpu_id,
        )

        # For MPS, manually move model to device
        if self.device == "mps":
            import torch
            self._upsampler.model.to(torch.device("mps"))

        logger.info(f"Loaded Real-ESRGAN model: {self.model_name} (device={self.device}, tile={tile})")

    def _upscale_frame_python(self, frame: Image.Image) -> Image.Image:
        """Upscale a single frame using Python backend."""
        self._ensure_loaded()
        import cv2

        img = np.array(frame)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output, _ = self._upsampler.enhance(img_bgr, outscale=self.scale)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)

    def _upscale_frames_ncnn(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Upscale frames using realesrgan-ncnn-vulkan binary (batch directory mode)."""
        binary = _find_ncnn_binary()
        if not binary:
            raise RuntimeError("realesrgan-ncnn-vulkan binary not found")

        models_dir = _find_ncnn_models_dir()
        if not models_dir:
            raise RuntimeError(
                f"ncnn models not found. Expected at: {NCNN_DIR / 'models'}"
            )

        ncnn_model_name = NCNN_MODEL_NAMES.get(self.model_name, "realesr-animevideov3")

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "input")
            out_dir = os.path.join(tmpdir, "output")
            os.makedirs(in_dir)
            os.makedirs(out_dir)

            # Save input frames as PNG
            for i, frame in enumerate(frames):
                frame.save(os.path.join(in_dir, f"{i:06d}.png"))

            # Run ncnn binary
            cmd = [
                binary,
                "-i", in_dir,
                "-o", out_dir,
                "-s", str(self.scale),
                "-n", ncnn_model_name,
                "-m", models_dir,
                "-f", "png",
            ]
            logger.info(f"Running ncnn upscale: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"realesrgan-ncnn-vulkan failed (exit {result.returncode}): "
                    f"{result.stderr[:500]}"
                )

            # Load output frames
            upscaled = []
            for i in range(len(frames)):
                out_path = os.path.join(out_dir, f"{i:06d}.png")
                if not os.path.exists(out_path):
                    raise RuntimeError(f"ncnn output missing: {out_path}")
                upscaled.append(Image.open(out_path).convert("RGB"))

            return upscaled

    def upscale_frame(self, frame: Image.Image) -> Image.Image:
        """Upscale a single frame."""
        backend = self._resolve_backend()
        if backend == "python":
            return self._upscale_frame_python(frame)
        else:
            # ncnn: batch of 1
            return self._upscale_frames_ncnn([frame])[0]

    def upscale_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Upscale all frames in a video."""
        backend = self._resolve_backend()
        logger.info(f"Upscaling {len(frames)} frames with {self.model_name} ({self.scale}x, backend={backend})")

        if backend == "ncnn":
            return self._upscale_frames_ncnn(frames)

        # Python backend: frame-by-frame
        result = []
        for i, frame in enumerate(frames):
            result.append(self._upscale_frame_python(frame))
            if (i + 1) % 10 == 0:
                logger.info(f"  Upscaled {i+1}/{len(frames)} frames")
        return result
