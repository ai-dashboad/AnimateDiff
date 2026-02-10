"""
LTX-Video Backend — wraps diffusers LTXPipeline.

Features:
- First DiT-based model capable of real-time 30 FPS generation
- Only 8 diffusion steps needed (no classifier-free guidance)
- Audio+video joint generation (LTX-2)
- Extremely efficient on consumer GPUs
"""

import logging
from typing import Optional

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

LTX_MODELS = {
    "default": "Lightricks/LTX-Video",
}


class LTXBackend(BasePipeline):
    backend_name = "ltx"

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
        offload_strategy: str = "none",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        model_variant: str = "default",
        **kwargs,
    ) -> "LTXBackend":
        from diffusers import LTXPipeline

        if model_path is None:
            model_path = LTX_MODELS.get(model_variant, LTX_MODELS["default"])

        logger.info(f"Loading LTX-Video from {model_path} (dtype={torch_dtype}, quant={quantization})")

        quant_config = get_quantization_config(quantization, components=["transformer"])

        load_kwargs = dict(torch_dtype=torch_dtype)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        pipe = LTXPipeline.from_pretrained(model_path, **load_kwargs)

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
        width: int = 768,
        height: int = 512,
        num_frames: int = 65,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> VideoOutput:
        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
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
