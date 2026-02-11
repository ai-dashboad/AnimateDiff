"""
Frame Interpolation — RIFE-based frame interpolation for smoother video output.

Supports:
- Practical-RIFE (PyTorch, best quality)
- rife-ncnn-vulkan-python (cross-platform, works on macOS)
- Fallback: simple frame blending

Typical use: 2x interpolation (16fps → 32fps) or 4x (16fps → 64fps)
"""

import logging
from typing import List, Optional, Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FrameInterpolator:
    """Frame interpolation using RIFE or fallback blending."""

    def __init__(self, backend: Literal["auto", "rife", "ncnn", "blend"] = "auto", device: str = "cpu"):
        self.device = device
        self.backend = self._resolve_backend(backend)
        self._model = None

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend

        # Try PyTorch RIFE first
        try:
            import torch
            # Check if practical-rife model files exist or torch is available
            return "rife"
        except ImportError:
            pass

        # Try NCNN variant (good for macOS)
        try:
            import rife_ncnn_vulkan_python
            return "ncnn"
        except ImportError:
            pass

        logger.info("No RIFE backend available, using simple frame blending")
        return "blend"

    def interpolate(
        self,
        frames: List[Image.Image],
        multiplier: int = 2,
        scale: float = 1.0,
    ) -> List[Image.Image]:
        """Interpolate frames to increase frame rate.

        Args:
            frames: Input PIL Image frames.
            multiplier: Frame rate multiplier (2 = double fps, 4 = quadruple).
            scale: Spatial scale for RIFE inference (0.5 for half-res, faster).

        Returns:
            Interpolated frame list.
        """
        if multiplier <= 1:
            return frames
        if len(frames) < 2:
            return frames

        logger.info(f"Interpolating {len(frames)} frames with {multiplier}x ({self.backend})")

        if self.backend == "rife":
            return self._interpolate_rife(frames, multiplier, scale)
        elif self.backend == "ncnn":
            return self._interpolate_ncnn(frames, multiplier)
        else:
            return self._interpolate_blend(frames, multiplier)

    def _interpolate_rife(
        self, frames: List[Image.Image], multiplier: int, scale: float
    ) -> List[Image.Image]:
        """Interpolate using Practical-RIFE PyTorch model."""
        import torch
        import torch.nn.functional as F

        if self._model is None:
            self._model = self._load_rife_model()

        result = []
        for i in range(len(frames) - 1):
            img0 = self._pil_to_tensor(frames[i])
            img1 = self._pil_to_tensor(frames[i + 1])

            result.append(frames[i])

            # Generate intermediate frames
            for j in range(1, multiplier):
                timestep = j / multiplier
                with torch.no_grad():
                    mid = self._model.inference(img0, img1, timestep=timestep, scale=scale)
                result.append(self._tensor_to_pil(mid))

        result.append(frames[-1])
        return result

    def _load_rife_model(self):
        """Load RIFE model. Tries multiple locations."""
        import torch

        # Try importing from practical-rife
        try:
            import sys
            import importlib

            # Look for common RIFE model locations
            for model_dir in ["train_log", "rife_model", "models/rife"]:
                try:
                    spec = importlib.util.find_spec(f"{model_dir}.RIFE_HDv3")
                    if spec:
                        module = importlib.import_module(f"{model_dir}.RIFE_HDv3")
                        model = module.Model()
                        model.load_model(model_dir, -1)
                        model.eval()
                        if self.device != "cpu":
                            model.device()
                        logger.info(f"Loaded RIFE model from {model_dir}")
                        return model
                except (ImportError, FileNotFoundError):
                    continue
        except Exception as e:
            logger.warning(f"Could not load RIFE PyTorch model: {e}")

        # Fallback: use a simple optical flow interpolation
        logger.warning("RIFE model not found, falling back to blend mode")
        self.backend = "blend"
        return None

    def _interpolate_ncnn(self, frames: List[Image.Image], multiplier: int) -> List[Image.Image]:
        """Interpolate using rife-ncnn-vulkan-python."""
        try:
            from rife_ncnn_vulkan_python import Rife

            rife = Rife(gpuid=0, model="rife-v4.6")

            result = []
            for i in range(len(frames) - 1):
                result.append(frames[i])
                img0 = np.array(frames[i])
                img1 = np.array(frames[i + 1])

                for j in range(1, multiplier):
                    timestep = j / multiplier
                    mid = rife.process(img0, img1, timestep=timestep)
                    result.append(Image.fromarray(mid))

            result.append(frames[-1])
            return result
        except Exception as e:
            logger.warning(f"NCNN interpolation failed: {e}, falling back to blend")
            return self._interpolate_blend(frames, multiplier)

    def _interpolate_blend(self, frames: List[Image.Image], multiplier: int) -> List[Image.Image]:
        """Simple alpha blending interpolation (fallback)."""
        result = []
        for i in range(len(frames) - 1):
            result.append(frames[i])
            arr0 = np.array(frames[i], dtype=np.float32)
            arr1 = np.array(frames[i + 1], dtype=np.float32)

            for j in range(1, multiplier):
                alpha = j / multiplier
                blended = (arr0 * (1 - alpha) + arr1 * alpha).astype(np.uint8)
                result.append(Image.fromarray(blended))

        result.append(frames[-1])
        return result

    def _pil_to_tensor(self, img: Image.Image):
        """Convert PIL Image to RIFE-format tensor [1, 3, H, W] float32 0-1."""
        import torch
        arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _tensor_to_pil(self, tensor) -> Image.Image:
        """Convert RIFE output tensor to PIL Image."""
        arr = (tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
