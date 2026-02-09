"""
AnimateDiff SDXL Pipeline — wraps diffusers AnimateDiffSDXLPipeline.

Generates higher quality videos using Stable Diffusion XL as the base model.
Supports SDXL-specific features like dual text encoders.
"""

import torch
from typing import Optional, Union, Dict

from diffusers import (
    AnimateDiffSDXLPipeline,
    MotionAdapter,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import export_to_gif, export_to_video

SDXL_SCHEDULER_MAP = {
    "ddim": lambda config: DDIMScheduler.from_config(
        config, clip_sample=False, timestep_spacing="linspace",
        beta_schedule="linear", steps_offset=1,
    ),
    "euler": lambda config: EulerDiscreteScheduler.from_config(config, beta_schedule="linear"),
    "dpm++": lambda config: DPMSolverMultistepScheduler.from_config(
        config, beta_schedule="linear", algorithm_type="dpmsolver++",
    ),
    "dpm++-karras": lambda config: DPMSolverMultistepScheduler.from_config(
        config, beta_schedule="linear", algorithm_type="dpmsolver++", use_karras_sigmas=True,
    ),
}


class AnimateDiffSDXL:
    """High-level wrapper around diffusers AnimateDiffSDXLPipeline."""

    def __init__(self, pipe: AnimateDiffSDXLPipeline):
        self.pipe = pipe

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        motion_adapter_path: str = "guoyww/animatediff-motion-adapter-sdxl-beta",
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        scheduler: str = "ddim",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
    ) -> "AnimateDiffSDXL":
        adapter = MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch_dtype)

        scheduler_cls = SDXL_SCHEDULER_MAP.get(scheduler)
        sched_instance = None
        if scheduler_cls:
            sched_config = DDIMScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            ).config
            sched_instance = scheduler_cls(sched_config)

        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            model_path,
            motion_adapter=adapter,
            torch_dtype=torch_dtype,
            variant="fp16",
        )

        if sched_instance is not None:
            pipe.scheduler = sched_instance

        if enable_vae_slicing:
            pipe.enable_vae_slicing()
        if enable_vae_tiling:
            pipe.enable_vae_tiling()

        pipe.to(device)
        return cls(pipe)

    def load_lora(self, lora_path: str, adapter_name: str = "default", scale: float = 1.0):
        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], [scale])

    def set_scheduler(self, name: str):
        if name in SDXL_SCHEDULER_MAP:
            self.pipe.scheduler = SDXL_SCHEDULER_MAP[name](self.pipe.scheduler.config)
        else:
            raise ValueError(f"Unknown scheduler: {name}. Choose from: {list(SDXL_SCHEDULER_MAP.keys())}")

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, Dict[int, str]],
        negative_prompt: str = "low quality, worst quality, blurry",
        num_frames: int = 16,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 8.0,
        seed: int = -1,
        output_type: str = "pil",
        decode_chunk_size: int = 4,
    ):
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type=output_type,
            decode_chunk_size=decode_chunk_size,
        )
        return output

    def save(self, output, path: str, fps: int = 8):
        frames = output.frames[0]
        if path.endswith(".gif"):
            export_to_gif(frames, path)
        elif path.endswith(".mp4"):
            export_to_video(frames, path, fps=fps)
        else:
            export_to_gif(frames, path)
