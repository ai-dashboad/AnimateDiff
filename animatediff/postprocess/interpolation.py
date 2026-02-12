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

    def __init__(self, backend: Literal["auto", "rife", "ncnn", "minterpolate", "blend"] = "auto", device: str = "cpu"):
        self.device = device
        self.backend = self._resolve_backend(backend)
        self._model = None

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend

        # Try PyTorch RIFE — need torch + model code + pretrained weights
        try:
            import torch
            import importlib
            # Only use rife backend if actual weight files exist
            for model_dir in ["train_log", "rife_model", "models/rife"]:
                try:
                    spec = importlib.util.find_spec(f"{model_dir}.RIFE_HDv3")
                    if spec:
                        return "rife"
                except (ImportError, ModuleNotFoundError, ValueError):
                    continue
        except ImportError:
            pass

        # Try NCNN variant (good for macOS)
        try:
            import rife_ncnn_vulkan_python
            return "ncnn"
        except ImportError:
            pass

        # Try ffmpeg minterpolate (motion-compensated, decent quality)
        try:
            import subprocess
            r = subprocess.run(
                ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5,
            )
            if "minterpolate" in r.stdout:
                return "minterpolate"
        except (FileNotFoundError, subprocess.TimeoutExpired):
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
        elif self.backend == "minterpolate":
            return self._interpolate_minterpolate(frames, multiplier)
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
        """Load RIFE model. Tries pip package then local directories."""
        import torch

        # Try the xhluca/rife pip package first
        try:
            from rife.RIFE_HDv2 import Model as RifeModel
            model = RifeModel()
            model.eval()
            model.device()
            logger.info("Loaded RIFE HDv2 from pip package (no pretrained weights — inference only)")
            return model
        except Exception as e:
            logger.debug(f"rife pip package load failed: {e}")

        # Try loading from local model directories with weights
        try:
            import importlib
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

        # Fallback to minterpolate if available, then blend
        logger.warning("RIFE model not found, falling back to minterpolate/blend")
        try:
            import subprocess
            r = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5)
            if "minterpolate" in r.stdout:
                self.backend = "minterpolate"
                return None
        except Exception:
            pass
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

    def _interpolate_minterpolate(self, frames: List[Image.Image], multiplier: int) -> List[Image.Image]:
        """Interpolate using ffmpeg minterpolate filter (motion-compensated)."""
        import subprocess
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_file = os.path.join(tmpdir, "output.mp4")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)
            os.makedirs(output_dir)

            # Write input frames as PNGs
            for i, frame in enumerate(frames):
                frame.save(os.path.join(input_dir, f"{i:06d}.png"))

            input_fps = 24  # assume 24fps input
            output_fps = input_fps * multiplier

            # Run ffmpeg minterpolate
            cmd = [
                "ffmpeg", "-y", "-framerate", str(input_fps),
                "-i", os.path.join(input_dir, "%06d.png"),
                "-vf", f"minterpolate=fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                "-pix_fmt", "rgb24", output_file,
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"minterpolate failed: {e}, falling back to blend")
                return self._interpolate_blend(frames, multiplier)

            # Extract frames back from video
            cmd2 = [
                "ffmpeg", "-y", "-i", output_file,
                "-pix_fmt", "rgb24",
                os.path.join(output_dir, "%06d.png"),
            ]
            try:
                subprocess.run(cmd2, capture_output=True, timeout=120, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Frame extraction failed: {e}, falling back to blend")
                return self._interpolate_blend(frames, multiplier)

            # Read output frames
            result = []
            png_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".png"))
            for png_file in png_files:
                img = Image.open(os.path.join(output_dir, png_file)).convert("RGB")
                result.append(img)

            if not result:
                logger.warning("minterpolate produced no frames, falling back to blend")
                return self._interpolate_blend(frames, multiplier)

            logger.info(f"minterpolate: {len(frames)} -> {len(result)} frames")
            return result

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
