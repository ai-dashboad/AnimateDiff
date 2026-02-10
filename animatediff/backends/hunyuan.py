"""
HunyuanVideo Backend — wraps diffusers HunyuanVideoPipeline.

Model: Tencent-Hunyuan/HunyuanVideo (8.3B parameters)

Features:
- High-quality text-to-video generation
- Dual-stream to single-stream transformer architecture
- 3D causal VAE for temporal coherence
- BitsAndBytes NF4 quantization (runs on 8GB VRAM)
"""

import logging
from typing import Optional

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

HUNYUAN_MODELS = {
    "default": "hunyuanvideo-community/HunyuanVideo",
}


class HunyuanBackend(BasePipeline):
    backend_name = "hunyuan"

    def __init__(self, pipe, model_variant: str = "default"):
        self.pipe = pipe
        self.model_variant = model_variant

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "model_cpu",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
        model_variant: str = "default",
        **kwargs,
    ) -> "HunyuanBackend":
        from diffusers import HunyuanVideoPipeline

        if model_path is None:
            model_path = HUNYUAN_MODELS.get(model_variant, HUNYUAN_MODELS["default"])

        logger.info(f"Loading HunyuanVideo from {model_path} (dtype={torch_dtype}, quant={quantization})")

        quant_config = get_quantization_config(quantization, components=["transformer"])

        load_kwargs = dict(torch_dtype=torch_dtype)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        pipe = HunyuanVideoPipeline.from_pretrained(model_path, **load_kwargs)

        instance = cls(pipe, model_variant=model_variant)

        if offload_strategy != "none":
            instance._apply_offloading(pipe, offload_strategy, device=device)
        else:
            pipe.to(device)

        instance._apply_vae_opts(pipe, slicing=enable_vae_slicing, tiling=enable_vae_tiling)

        return instance

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 720,
        height: int = 480,
        num_frames: int = 33,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> VideoOutput:
        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        output = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )
        frames = output.frames[0]

        return VideoOutput(
            frames=frames,
            seed=seed,
            backend=self.backend_name,
        )
