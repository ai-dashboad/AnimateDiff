"""
Wan 2.2 Backend — wraps diffusers WanPipeline for Wan 2.2 models.

Model variants (MoE dual-path architecture):
- Wan-AI/Wan2.2-T2V-A14B-Diffusers   (27B total / 14B active, CUDA only — FP8 MoE)
- Wan-AI/Wan2.2-I2V-A14B-Diffusers   (27B total / 14B active, CUDA only — FP8 MoE)
- Wan-AI/Wan2.2-TI2V-5B-Diffusers    (5B dense, works on MPS — unified T2V+I2V)

Key differences from Wan 2.1:
- Two-stage denoising: high-noise transformer + low-noise transformer (MoE)
- guidance_scale_2 parameter for the second transformer
- Dual-transformer LoRA loading (load_into_transformer_2=True)
- TI2V-5B: dense model, accepts optional image input, 24 fps, 121 frames

NOTE: A14B models use FP8 (Float8_e4m3fn) internally in the MoE experts.
      MPS does NOT support FP8 — only TI2V-5B works on Apple Silicon.
"""

import logging
from typing import Optional, List

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

WAN22_T2V_MODELS = {
    "A14B": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "5B": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}

WAN22_I2V_MODELS = {
    "A14B": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "5B": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",  # TI2V-5B handles both T2V and I2V
}

# TI2V-5B defaults differ from A14B
MODEL_DEFAULTS = {
    "A14B": dict(width=1280, height=720, num_frames=81, fps=16, guidance_scale=4.0, guidance_scale_2=3.0, steps=40),
    "5B": dict(width=1280, height=704, num_frames=121, fps=24, guidance_scale=5.0, guidance_scale_2=None, steps=50),
}


class Wan22Backend(BasePipeline):
    backend_name = "wan22"

    def __init__(self, pipe, model_variant: str = "5B", lora_names: Optional[List[str]] = None):
        self.pipe = pipe
        self.model_variant = model_variant
        self.lora_names = lora_names or []
        self._defaults = MODEL_DEFAULTS.get(model_variant, MODEL_DEFAULTS["5B"])

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "none",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        model_variant: str = "5B",
        mode: str = "t2v",
        lora_paths: Optional[List[str]] = None,
        lora_scales: Optional[List[float]] = None,
        **kwargs,
    ) -> "Wan22Backend":
        from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan

        # Resolve model path
        if model_path is None:
            if mode == "i2v" and model_variant != "5B":
                model_path = WAN22_I2V_MODELS.get(model_variant, WAN22_I2V_MODELS["A14B"])
            else:
                model_path = WAN22_T2V_MODELS.get(model_variant, WAN22_T2V_MODELS["5B"])

        # MPS safety: A14B uses FP8 internally, which MPS doesn't support
        if device == "mps" and model_variant == "A14B":
            logger.warning("Wan 2.2 A14B uses FP8 MoE experts — not supported on MPS. Falling back to TI2V-5B.")
            model_variant = "5B"
            model_path = WAN22_T2V_MODELS["5B"]

        # MPS requires float32 for Wan models
        if device == "mps":
            torch_dtype = torch.float32
            logger.info("MPS detected: using float32 (float16/bfloat16 not fully supported for Wan on MPS)")

        logger.info(f"Loading Wan 2.2 {model_variant} from {model_path} (dtype={torch_dtype}, quant={quantization})")

        # VAE must always be float32 for Wan
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)

        # Quantization config — for A14B, quantize both transformers
        components = ["transformer", "transformer_2"] if model_variant == "A14B" else ["transformer"]
        quant_config = get_quantization_config(quantization, components=components)

        load_kwargs = dict(torch_dtype=torch_dtype, vae=vae)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        # Choose pipeline class
        if mode == "i2v":
            PipelineClass = WanImageToVideoPipeline
        else:
            PipelineClass = WanPipeline

        pipe = PipelineClass.from_pretrained(model_path, **load_kwargs)

        # Fix: transformers 5.x UMT5 embed_tokens zero-weight bug (same as Wan 2.1)
        te = pipe.text_encoder
        if (hasattr(te, "shared") and hasattr(te, "encoder")
                and hasattr(te.encoder, "embed_tokens")
                and te.encoder.embed_tokens.weight.abs().sum().item() == 0
                and te.shared.weight.abs().sum().item() > 0):
            logger.warning("Fixing UMT5 embed_tokens: binding shared.weight -> encoder.embed_tokens.weight")
            te.encoder.embed_tokens.weight = te.shared.weight

        instance = cls(pipe, model_variant=model_variant)

        # Load LoRAs if provided
        if lora_paths:
            instance._load_loras(lora_paths, lora_scales or [1.0] * len(lora_paths))

        # Apply offloading
        if offload_strategy != "none":
            instance._apply_offloading(pipe, offload_strategy, device=device)
        else:
            pipe.to(device)

        instance._apply_vae_opts(pipe, slicing=enable_vae_slicing, tiling=enable_vae_tiling)

        return instance

    def _load_loras(self, lora_paths: List[str], lora_scales: List[float]):
        """Load LoRA weights. For A14B, supports dual-transformer LoRA loading."""
        for i, (path, scale) in enumerate(zip(lora_paths, lora_scales)):
            adapter_name = f"lora_{i}"

            # Detect if this is a dual-transformer LoRA (by filename convention)
            is_low_noise = "_LOW" in path or "_low" in path or "transformer_2" in path

            load_kwargs = dict(adapter_name=adapter_name)
            if is_low_noise and self.model_variant == "A14B":
                load_kwargs["load_into_transformer_2"] = True
                logger.info(f"Loading LoRA into transformer_2 (low-noise): {path} (scale={scale})")
            else:
                logger.info(f"Loading LoRA into transformer (high-noise): {path} (scale={scale})")

            # Handle both repo IDs and local paths
            if "/" in path and not path.startswith("/") and not path.startswith("."):
                # Looks like a HuggingFace repo ID — split off weight_name
                parts = path.rsplit("/", 1)
                if len(parts) == 2 and "." in parts[1]:
                    self.pipe.load_lora_weights(parts[0], weight_name=parts[1], **load_kwargs)
                else:
                    self.pipe.load_lora_weights(path, **load_kwargs)
            else:
                self.pipe.load_lora_weights(path, **load_kwargs)

            self.lora_names.append(adapter_name)

        if self.lora_names:
            scales = lora_scales[:len(self.lora_names)]
            self.pipe.set_adapters(self.lora_names, adapter_weights=scales)
            logger.info(f"Activated LoRAs: {self.lora_names} with scales {scales}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        num_frames: int = 0,
        num_inference_steps: int = 0,
        guidance_scale: float = 0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        guidance_scale_2: Optional[float] = None,
        **kwargs,
    ) -> VideoOutput:
        d = self._defaults
        width = width or d["width"]
        height = height or d["height"]
        num_frames = num_frames or d["num_frames"]
        num_inference_steps = num_inference_steps or d["steps"]
        guidance_scale = guidance_scale or d["guidance_scale"]

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

        # A14B MoE: separate guidance for the second transformer
        gs2 = guidance_scale_2 if guidance_scale_2 is not None else d.get("guidance_scale_2")
        if gs2 is not None and self.model_variant == "A14B":
            pipe_kwargs["guidance_scale_2"] = gs2

        # Image-to-video (TI2V-5B accepts image as optional input)
        if image is not None:
            pipe_kwargs["image"] = image

        output = self.pipe(**pipe_kwargs)
        frames = output.frames[0]

        return VideoOutput(
            frames=frames,
            fps=d["fps"],
            seed=seed,
            backend=self.backend_name,
            metadata={
                "model_variant": self.model_variant,
                "loras": self.lora_names,
            },
        )
