"""
AnimateDiff V2 Pipeline — wraps the official diffusers AnimateDiffPipeline.

Supports:
- SD1.5 text-to-video with MotionAdapter
- FreeInit for temporal consistency
- FreeNoise for long video generation (>16 frames)
- Prompt Travel (frame-varying prompts via FreeNoise)
- IP-Adapter for image-conditioned generation
- LoRA loading (DreamBooth, motion LoRAs)
- VAE slicing/tiling for memory optimization
"""

import torch
from typing import Optional, Union, List, Dict, Callable

from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
)
from diffusers.utils import export_to_gif, export_to_video

SCHEDULER_MAP = {
    "ddim": lambda config: DDIMScheduler.from_config(
        config, clip_sample=False, timestep_spacing="linspace",
        beta_schedule="linear", steps_offset=1,
    ),
    "euler": lambda config: EulerDiscreteScheduler.from_config(config, beta_schedule="linear"),
    "euler-a": lambda config: EulerAncestralDiscreteScheduler.from_config(config, beta_schedule="linear"),
    "dpm++": lambda config: DPMSolverMultistepScheduler.from_config(
        config, beta_schedule="linear", algorithm_type="dpmsolver++",
    ),
    "dpm++-karras": lambda config: DPMSolverMultistepScheduler.from_config(
        config, beta_schedule="linear", algorithm_type="dpmsolver++", use_karras_sigmas=True,
    ),
    "pndm": lambda config: PNDMScheduler.from_config(config, beta_schedule="linear"),
}


class AnimateDiffV2Pipeline:
    """High-level wrapper around diffusers AnimateDiffPipeline with all features exposed."""

    def __init__(self, pipe: AnimateDiffPipeline):
        self.pipe = pipe

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        motion_adapter_path: str = "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        scheduler: str = "ddim",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
    ) -> "AnimateDiffV2Pipeline":
        adapter = MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch_dtype)

        pipe = AnimateDiffPipeline.from_pretrained(
            model_path,
            motion_adapter=adapter,
            torch_dtype=torch_dtype,
        )

        if scheduler in SCHEDULER_MAP:
            pipe.scheduler = SCHEDULER_MAP[scheduler](pipe.scheduler.config)

        if enable_vae_slicing:
            pipe.enable_vae_slicing()
        if enable_vae_tiling:
            pipe.enable_vae_tiling()

        pipe.to(device)
        return cls(pipe)

    def load_lora(self, lora_path: str, adapter_name: str = "default", scale: float = 1.0):
        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], [scale])

    def load_ip_adapter(
        self,
        repo_id: str = "h94/IP-Adapter",
        subfolder: str = "models",
        weight_name: str = "ip-adapter_sd15.bin",
        scale: float = 0.6,
    ):
        self.pipe.load_ip_adapter(repo_id, subfolder=subfolder, weight_name=weight_name)
        self.pipe.set_ip_adapter_scale(scale)

    def enable_free_init(
        self,
        num_iters: int = 3,
        method: str = "butterworth",
        use_fast_sampling: bool = True,
    ):
        self.pipe.enable_free_init(
            num_iters=num_iters, method=method, use_fast_sampling=use_fast_sampling,
        )

    def disable_free_init(self):
        self.pipe.disable_free_init()

    def enable_free_noise(
        self,
        context_length: int = 16,
        context_stride: int = 4,
        weighting_scheme: str = "pyramid",
    ):
        self.pipe.enable_free_noise(
            context_length=context_length,
            context_stride=context_stride,
            weighting_scheme=weighting_scheme,
        )

    def disable_free_noise(self):
        self.pipe.disable_free_noise()

    def set_scheduler(self, name: str):
        if name in SCHEDULER_MAP:
            self.pipe.scheduler = SCHEDULER_MAP[name](self.pipe.scheduler.config)
        else:
            raise ValueError(f"Unknown scheduler: {name}. Choose from: {list(SCHEDULER_MAP.keys())}")

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, Dict[int, str]],
        negative_prompt: str = "bad quality, worst quality, low resolution",
        num_frames: int = 16,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        ip_adapter_image=None,
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
            ip_adapter_image=ip_adapter_image,
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
