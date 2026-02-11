"""
Video Upscaler — Real-ESRGAN anime super-resolution for video frames.

Models:
- realesr-animevideov3: 10MB, designed for anime video (default)
- RealESRGAN_x4plus_anime_6B: 17MB, anime images
- RealESRGAN_x4plus: 64MB, general purpose

MPS/Apple Silicon: works with half=False and tiling.
"""

import logging
from typing import List, Optional, Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Model download URLs
MODEL_URLS = {
    "animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


class VideoUpscaler:
    """Upscale video frames using Real-ESRGAN."""

    def __init__(
        self,
        model_name: Literal["animevideov3", "anime_6B", "general"] = "animevideov3",
        scale: int = 2,
        tile: int = 0,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.scale = scale
        self.tile = tile
        self.device = device
        self._upsampler = None

    def _ensure_loaded(self):
        """Lazy-load the upscaler model."""
        if self._upsampler is not None:
            return

        try:
            from realesrgan import RealESRGANer
            from basicsr.utils.download_util import load_file_from_url
        except ImportError:
            raise ImportError(
                "Real-ESRGAN not installed. Install with: pip install realesrgan"
            )

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

    def upscale_frame(self, frame: Image.Image) -> Image.Image:
        """Upscale a single frame."""
        self._ensure_loaded()
        import cv2

        img = np.array(frame)
        # Real-ESRGAN expects BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        output, _ = self._upsampler.enhance(img_bgr, outscale=self.scale)

        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)

    def upscale_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Upscale all frames in a video."""
        logger.info(f"Upscaling {len(frames)} frames with {self.model_name} ({self.scale}x)")
        result = []
        for i, frame in enumerate(frames):
            result.append(self.upscale_frame(frame))
            if (i + 1) % 10 == 0:
                logger.info(f"  Upscaled {i+1}/{len(frames)} frames")
        return result
