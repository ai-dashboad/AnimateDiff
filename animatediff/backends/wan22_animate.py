"""
Wan 2.2 Animate Backend — character animation and replacement.

Uses WanAnimatePipeline from diffusers for:
- Motion Imitation: transfer gestures/expressions from a reference video to a character image
- Role-Play (replacement): swap a character in a video while preserving the scene

Model: Wan-AI/Wan2.2-Animate-14B-Diffusers (14B, WanAnimateTransformer3DModel)

IMPORTANT: pose_video and face_video inputs must be PREPROCESSED:
- pose_video: skeletal keypoint sequences extracted via preprocessing scripts
- face_video: facial feature sequences extracted via preprocessing scripts
Preprocessing scripts are in the original Wan2.2 repo (not yet in diffusers).
"""

import logging
from typing import Optional, List, Union
from pathlib import Path

import torch
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput

logger = logging.getLogger(__name__)

WAN22_ANIMATE_MODEL = "Wan-AI/Wan2.2-Animate-14B-Diffusers"


class Wan22AnimateBackend(BasePipeline):
    backend_name = "wan22_animate"

    def __init__(self, pipe, preprocess_fn=None):
        self.pipe = pipe
        self.preprocess_fn = preprocess_fn

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
        **kwargs,
    ) -> "Wan22AnimateBackend":
        from diffusers import WanAnimatePipeline, AutoencoderKLWan
        from animatediff.core.quantization import get_quantization_config

        model_path = model_path or WAN22_ANIMATE_MODEL

        # MPS not supported (likely FP8 dependencies in the animate transformer)
        if device == "mps":
            logger.warning("Wan2.2-Animate may not work on MPS due to FP8 dependencies. Attempting with float32.")
            torch_dtype = torch.float32

        logger.info(f"Loading Wan2.2-Animate from {model_path}")

        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)

        quant_config = get_quantization_config(quantization, components=["transformer"])
        load_kwargs = dict(torch_dtype=torch_dtype, vae=vae)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        pipe = WanAnimatePipeline.from_pretrained(model_path, **load_kwargs)

        instance = cls(pipe)

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
        width: int = 1280,
        height: int = 720,
        num_frames: int = 0,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        pose_video: Optional[List] = None,
        face_video: Optional[List] = None,
        background_video: Optional[List] = None,
        mask_video: Optional[List] = None,
        mode: str = "animate",
        segment_frame_length: int = 77,
        **kwargs,
    ) -> VideoOutput:
        """Generate character animation or replacement video.

        Args:
            image: Character reference image.
            pose_video: Preprocessed skeletal keypoint video frames.
            face_video: Preprocessed facial feature video frames.
            background_video: (replace mode) Original background video.
            mask_video: (replace mode) Mask video (white=generate, black=preserve).
            mode: "animate" or "replace".
            segment_frame_length: Frames per inference segment (should be 4N+1).
        """
        if image is None:
            raise ValueError("Wan2.2-Animate requires a character reference image (--image)")
        if pose_video is None:
            raise ValueError("Wan2.2-Animate requires preprocessed pose_video")
        if face_video is None:
            raise ValueError("Wan2.2-Animate requires preprocessed face_video")

        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        pipe_kwargs = dict(
            image=image,
            pose_video=pose_video,
            face_video=face_video,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            segment_frame_length=segment_frame_length,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            mode=mode,
            output_type="pil",
        )

        # Replacement mode needs background and mask
        if mode == "replace":
            if background_video is not None:
                pipe_kwargs["background_video"] = background_video
            if mask_video is not None:
                pipe_kwargs["mask_video"] = mask_video

        output = self.pipe(**pipe_kwargs)
        frames = output.frames[0]

        return VideoOutput(
            frames=frames,
            fps=30,
            seed=seed,
            backend=self.backend_name,
            metadata={"mode": mode},
        )

    @staticmethod
    def load_preprocessed_video(path: str) -> List:
        """Load preprocessed pose/face video from a file or directory.

        The preprocessing outputs from the Wan2.2 repo are typically
        stored as directories of numbered frames or as .pt tensor files.
        """
        from diffusers.utils import load_video

        path = Path(path)
        if path.suffix in (".mp4", ".avi", ".mov", ".webm"):
            return load_video(str(path))
        elif path.suffix == ".pt":
            return torch.load(str(path), weights_only=True)
        elif path.is_dir():
            # Directory of frame images
            frames = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
            return [Image.open(f).convert("RGB") for f in frames]
        else:
            raise ValueError(f"Unsupported preprocessed video format: {path}")
