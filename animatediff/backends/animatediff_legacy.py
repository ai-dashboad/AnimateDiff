"""
AnimateDiff Legacy Backend — wraps existing V2/SDXL/Lightning/Legacy pipelines.

Provides backward compatibility with all existing AnimateDiff features:
- SD1.5 text-to-video with motion modules
- SDXL text-to-video
- Lightning ultra-fast inference (1-8 steps)
- SparseCtrl (legacy pipeline only)
- FreeInit, FreeNoise, IP-Adapter, Prompt Travel (V2)
"""

import logging
from typing import Optional

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput

logger = logging.getLogger(__name__)


class AnimateDiffBackend(BasePipeline):
    """Wraps existing AnimateDiff pipelines (V2/SDXL/Lightning/Legacy)."""

    backend_name = "animatediff"

    def __init__(self, pipe, pipeline_type: str = "v2"):
        self.pipe = pipe
        self.pipeline_type = pipeline_type

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "none",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        pipeline_type: str = "v2",
        scheduler: str = "ddim",
        motion_adapter: Optional[str] = None,
        lightning_steps: int = 4,
        **kwargs,
    ) -> "AnimateDiffBackend":
        if model_path is None:
            model_path = "runwayml/stable-diffusion-v1-5"

        if pipeline_type == "sdxl":
            from animatediff.pipelines.pipeline_sdxl import AnimateDiffSDXL
            pipe = AnimateDiffSDXL.from_pretrained(
                model_path=model_path or "stabilityai/stable-diffusion-xl-base-1.0",
                motion_adapter_path=motion_adapter or "guoyww/animatediff-motion-adapter-sdxl-beta",
                torch_dtype=torch_dtype,
                device=device,
                scheduler=scheduler,
            )
        elif pipeline_type == "lightning":
            from animatediff.pipelines.pipeline_lightning import AnimateDiffLightning
            pipe = AnimateDiffLightning.from_pretrained(
                model_path=model_path or "emilianJR/epiCRealism",
                num_steps=lightning_steps,
                torch_dtype=torch_dtype,
                device=device,
            )
        else:
            from animatediff.pipelines.pipeline_v2 import AnimateDiffV2Pipeline
            pipe = AnimateDiffV2Pipeline.from_pretrained(
                model_path=model_path,
                motion_adapter_path=motion_adapter or "guoyww/animatediff-motion-adapter-v1-5-3",
                torch_dtype=torch_dtype,
                device=device,
                scheduler=scheduler,
                enable_vae_slicing=enable_vae_slicing,
                enable_vae_tiling=enable_vae_tiling,
            )

        return cls(pipe, pipeline_type=pipeline_type)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "bad quality, worst quality",
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> VideoOutput:
        output = self.pipe.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            **kwargs,
        )
        frames = output.frames[0]

        return VideoOutput(
            frames=frames,
            seed=seed,
            backend=self.backend_name,
            metadata={"pipeline_type": self.pipeline_type},
        )

    def save(self, output: VideoOutput, path: str, fps: int = 8):
        """Use existing pipeline save or default."""
        if hasattr(self.pipe, "save"):
            # The V2/SDXL/Lightning pipelines have their own save method
            # but they expect their native output format, not VideoOutput.
            # Use base class save instead.
            pass
        super().save(output, path, fps=fps)
