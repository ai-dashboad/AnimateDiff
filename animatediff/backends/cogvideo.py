"""
CogVideoX Backend — wraps diffusers CogVideoXPipeline.

Model variants:
- THUDM/CogVideoX-2b  (6 GB VRAM with FP8, lightest option)
- THUDM/CogVideoX-5b  (12+ GB VRAM, better quality)

Features:
- Text-to-video with 3D causal VAE
- Adaptive LayerNorm for text-video alignment
- 3D full attention for motion capture
- LoRA support built-in
"""

import logging
from typing import Optional

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

COGVIDEO_MODELS = {
    "2B": "THUDM/CogVideoX-2b",
    "5B": "THUDM/CogVideoX-5b",
}


class CogVideoBackend(BasePipeline):
    backend_name = "cogvideo"

    def __init__(self, pipe, model_variant: str = "2B"):
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
        model_variant: str = "2B",
        **kwargs,
    ) -> "CogVideoBackend":
        from diffusers import CogVideoXPipeline

        if model_path is None:
            model_path = COGVIDEO_MODELS.get(model_variant, COGVIDEO_MODELS["2B"])

        logger.info(f"Loading CogVideoX-{model_variant} from {model_path} (dtype={torch_dtype}, quant={quantization})")

        quant_config = get_quantization_config(quantization, components=["transformer"])

        load_kwargs = dict(torch_dtype=torch_dtype)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        pipe = CogVideoXPipeline.from_pretrained(model_path, **load_kwargs)

        instance = cls(pipe, model_variant=model_variant)

        if offload_strategy != "none":
            instance._apply_offloading(pipe, offload_strategy)
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
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
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
            metadata={"model_variant": self.model_variant},
        )
