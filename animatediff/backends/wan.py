"""
Wan 2.1 Backend — wraps diffusers WanPipeline.

Model variants:
- Wan-AI/Wan2.1-T2V-1.3B  (6-8 GB VRAM, consumer-friendly)
- Wan-AI/Wan2.1-T2V-14B   (20+ GB VRAM, highest quality)

Features:
- Text-to-Video and Image-to-Video
- BitsAndBytes NF4 quantization for low VRAM
- 3D causal VAE for temporal coherence
- Apache 2.0 license
"""

import logging
from typing import Optional

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

WAN_MODELS = {
    "1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "14B": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
}

WAN_I2V_MODELS = {
    "14B": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
}


class WanBackend(BasePipeline):
    backend_name = "wan"

    def __init__(self, pipe, model_variant: str = "1.3B"):
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
        model_variant: str = "1.3B",
        mode: str = "t2v",
        **kwargs,
    ) -> "WanBackend":
        from diffusers import WanPipeline, WanImageToVideoPipeline

        if model_path is None:
            if mode == "i2v":
                model_path = WAN_I2V_MODELS.get(model_variant, WAN_I2V_MODELS["14B"])
            else:
                model_path = WAN_MODELS.get(model_variant, WAN_MODELS["1.3B"])

        logger.info(f"Loading Wan {model_variant} from {model_path} (dtype={torch_dtype}, quant={quantization})")

        quant_config = get_quantization_config(quantization, components=["transformer"])

        load_kwargs = dict(torch_dtype=torch_dtype)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        PipelineClass = WanImageToVideoPipeline if mode == "i2v" else WanPipeline
        pipe = PipelineClass.from_pretrained(model_path, **load_kwargs)

        instance = cls(pipe, model_variant=model_variant)

        # Apply offloading (must be before .to(device) for cpu offload)
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
        width: int = 480,
        height: int = 320,
        num_frames: int = 33,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> VideoOutput:
        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        pipe_kwargs = dict(
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

        # Image-to-Video
        if image is not None and hasattr(self.pipe, "image"):
            pipe_kwargs["image"] = image

        output = self.pipe(**pipe_kwargs)
        frames = output.frames[0]

        return VideoOutput(
            frames=frames,
            seed=seed,
            backend=self.backend_name,
            metadata={"model_variant": self.model_variant},
        )
