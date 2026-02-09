"""
AnimateDiff-Lightning Pipeline — ultra-fast 1-8 step video generation.

Based on ByteDance's cross-model diffusion distillation (arXiv:2403.12706).
Produces high-quality animations in as few as 2-4 denoising steps.
"""

import torch
from typing import Optional, Union, Dict

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif, export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


LIGHTNING_REPO = "ByteDance/AnimateDiff-Lightning"
VALID_STEPS = {1, 2, 4, 8}


class AnimateDiffLightning:
    """Ultra-fast AnimateDiff inference using distilled motion modules."""

    def __init__(self, pipe: AnimateDiffPipeline, num_steps: int):
        self.pipe = pipe
        self.num_steps = num_steps

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "emilianJR/epiCRealism",
        num_steps: int = 4,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        enable_vae_slicing: bool = True,
    ) -> "AnimateDiffLightning":
        if num_steps not in VALID_STEPS:
            raise ValueError(f"num_steps must be one of {VALID_STEPS}, got {num_steps}")

        ckpt = f"animatediff_lightning_{num_steps}step_diffusers.safetensors"

        adapter = MotionAdapter().to(device, torch_dtype)
        adapter.load_state_dict(
            load_file(hf_hub_download(LIGHTNING_REPO, ckpt), device=device)
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            model_path,
            motion_adapter=adapter,
            torch_dtype=torch_dtype,
        ).to(device)

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear",
        )

        if enable_vae_slicing:
            pipe.enable_vae_slicing()

        return cls(pipe, num_steps)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, Dict[int, str]],
        negative_prompt: str = "",
        num_frames: int = 16,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 1.0,
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
            num_inference_steps=self.num_steps,
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
