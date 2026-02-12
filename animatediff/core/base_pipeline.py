"""
Base Pipeline — abstract interface for all video generation backends.

All backends (Wan, HunyuanVideo, CogVideoX, LTX, AnimateDiff) inherit from
BasePipeline and implement the same load/generate/save interface.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VideoOutput:
    """Standardized output from any video generation backend."""
    frames: list  # List[PIL.Image] — the generated video frames
    fps: int = 8
    seed: int = -1
    backend: str = ""
    metadata: dict = field(default_factory=dict)
    audio: Optional[torch.Tensor] = None  # Raw audio waveform tensor (1D or 2D)
    audio_sample_rate: int = 0  # Audio sample rate in Hz (e.g. 24000)


class BasePipeline(ABC):
    """Abstract base class for unified video generation."""

    backend_name: str = "base"

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "none",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        **kwargs,
    ) -> "BasePipeline":
        """Load model and return a pipeline instance."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> VideoOutput:
        """Generate video frames from a text prompt."""
        ...

    def save(self, output: VideoOutput, path: str, fps: int = 8):
        """Save VideoOutput to a GIF or MP4 file.

        For MP4, uses ffmpeg with H.264 encoding for high quality.
        If the output contains audio, muxes audio into the MP4.
        Falls back to diffusers export_to_video (OpenCV/MPEG-4) if ffmpeg
        is unavailable.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if path.endswith(".mp4"):
            self._save_mp4_ffmpeg(
                output.frames, path, fps,
                audio=output.audio,
                audio_sample_rate=output.audio_sample_rate,
            )
        else:
            from diffusers.utils import export_to_gif
            export_to_gif(output.frames, path)
        logger.info(f"Saved video to {path}")

    @staticmethod
    def _save_mp4_ffmpeg(
        frames: list,
        path: str,
        fps: int,
        audio: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 0,
    ):
        """Save frames to MP4 using ffmpeg pipe (H.264, CRF 18).

        If audio tensor and sample_rate are provided, muxes audio into the MP4
        using AAC encoding.
        """
        import subprocess
        import shutil

        if not shutil.which("ffmpeg"):
            from diffusers.utils import export_to_video
            logger.warning("ffmpeg not found, falling back to OpenCV export (lower quality)")
            export_to_video(frames, path, fps=fps)
            return

        import numpy as np

        has_audio = audio is not None and audio_sample_rate > 0

        w, h = frames[0].size
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
        ]

        # If we have audio, add a second input from a WAV pipe
        if has_audio:
            cmd += [
                "-f", "wav",
                "-i", "pipe:1",
            ]

        cmd += [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ]

        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        else:
            cmd += ["-an"]

        cmd.append(path)

        if has_audio:
            # Two-pass: write video-only first, then mux audio
            # Simpler approach: write audio to temp WAV, then combine
            import tempfile
            import soundfile as sf

            audio_np = audio.float().cpu().numpy()
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]  # (1, samples)
            # soundfile expects (samples, channels)
            audio_np = audio_np.T if audio_np.shape[0] < audio_np.shape[1] else audio_np

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name
                sf.write(tmp_wav_path, audio_np, audio_sample_rate)

            # Build ffmpeg command with file-based audio input
            cmd_with_audio = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps),
                "-i", "pipe:0",
                "-i", tmp_wav_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                path,
            ]

            proc = subprocess.Popen(cmd_with_audio, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(np.array(frame).tobytes())
            proc.stdin.close()
            proc.wait()

            # Clean up temp file
            try:
                os.unlink(tmp_wav_path)
            except OSError:
                pass

            if proc.returncode != 0:
                err = proc.stderr.read().decode()[-300:]
                logger.warning(f"ffmpeg encode with audio failed: {err}")
                # Fallback: save without audio
                BasePipeline._save_mp4_ffmpeg(frames, path, fps)
        else:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(np.array(frame).tobytes())
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read().decode()[-200:]
                logger.warning(f"ffmpeg encode failed: {err}")
                from diffusers.utils import export_to_video
                export_to_video(frames, path, fps=fps)

    def _make_generator(self, seed: int, device: str) -> Optional[torch.Generator]:
        if seed >= 0:
            return torch.Generator(device=device).manual_seed(seed)
        return None

    def _apply_offloading(self, pipe, strategy: str, device: str = "cuda"):
        """Apply memory offloading strategy to a diffusers pipeline.

        Note: CPU offloading only works with CUDA. For MPS, we skip offloading
        and move the full pipeline to the device instead.
        """
        # CPU offloading requires CUDA — skip for MPS/CPU and just move to device
        if device != "cuda" and device != "cpu":
            logger.info(f"Offloading not supported on {device}, moving pipeline to {device}")
            pipe.to(device)
            return

        if strategy == "model_cpu" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif strategy == "sequential_cpu" and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        else:
            logger.warning(f"Offload strategy '{strategy}' not available, moving to {device}")
            pipe.to(device)

    def _apply_vae_opts(self, pipe, slicing: bool = True, tiling: bool = False):
        """Apply VAE memory optimizations."""
        if slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    def _apply_quantization(self, quantization: str):
        """Return a PipelineQuantizationConfig or None."""
        if quantization == "none":
            return None

        try:
            from diffusers import PipelineQuantizationConfig
        except ImportError:
            logger.warning("PipelineQuantizationConfig not available; skipping quantization. Upgrade diffusers>=0.31")
            return None

        if quantization == "nf4":
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer"],
            )
        elif quantization == "int8":
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=["transformer"],
            )
        elif quantization == "fp8":
            return PipelineQuantizationConfig(
                quant_backend="torchao",
                quant_kwargs={"quant_type": "float8_e4m3fn"},
                components_to_quantize=["transformer"],
            )
        else:
            logger.warning(f"Unknown quantization: {quantization}, skipping")
            return None

    def _apply_compile(
        self,
        pipe,
        compile_mode: str = "reduce-overhead",
        compile_backend: str = "inductor",
        compile_components: Optional[List[str]] = None,
        warmup: bool = True,
        warmup_width: int = 512,
        warmup_height: int = 512,
        warmup_num_frames: int = 16,
    ):
        """Apply torch.compile to a diffusers pipeline for faster inference.

        Compiles the compute-heavy components (transformer/unet + VAE) using
        torch.compile with the specified mode. Optionally runs a warmup pass
        to trigger compilation so subsequent calls are fast.

        This method is safe to call on any device -- it will skip compilation
        on MPS or CPU where torch.compile is not beneficial.

        Args:
            pipe: The diffusers pipeline instance to compile.
            compile_mode: torch.compile mode. One of:
                "reduce-overhead" -- CUDA graphs, best for repeated inference
                "max-autotune" -- Triton autotuning, best throughput (H100/4090)
                "max-autotune-no-cudagraphs" -- autotuning without CUDA graphs
                "default" -- minimal compilation
            compile_backend: Compiler backend ("inductor", "cudagraphs", "eager").
            compile_components: Which pipeline components to compile. Default:
                auto-detect (transformer + vae, or unet + vae).
            warmup: Whether to run a warmup inference to trigger compilation.
            warmup_width: Width for warmup inference.
            warmup_height: Height for warmup inference.
            warmup_num_frames: Frame count for warmup inference.
        """
        from animatediff.core.compile import PipelineCompiler

        compiler = PipelineCompiler(
            mode=compile_mode,
            backend=compile_backend,
        )

        if not compiler.is_compile_available():
            logger.info(
                "torch.compile not available on this device "
                "(requires CUDA with compute >= 8.0). Skipping compilation."
            )
            return

        # Log expected speedup
        speedup_info = compiler.estimate_speedup(self.backend_name)
        logger.info(f"Expected speedup: {speedup_info}")

        # Compile pipeline components
        result = compiler.compile_pipeline(pipe, components=compile_components)

        if result.compiled_components and warmup:
            warmup_time = compiler.warmup(
                pipe,
                width=warmup_width,
                height=warmup_height,
                num_frames=warmup_num_frames,
            )
            logger.info(
                f"Compilation complete: {result.compiled_components} "
                f"(compile={result.compile_time_s:.1f}s, warmup={warmup_time:.1f}s)"
            )

    def _apply_fp8(
        self,
        pipe,
        fp8_scheme: Optional[str] = None,
        fp8_components: Optional[List[str]] = None,
    ):
        """Apply post-load FP8 quantization to a pipeline.

        Converts pipeline components to FP8 using torchao for reduced memory
        and faster inference on Ada/Hopper GPUs. Safe to call on any GPU --
        will skip if FP8 is not supported.

        This is for post-load optimization. For load-time FP8, use
        quantization="fp8" in the load() method instead.

        Args:
            pipe: The diffusers pipeline instance.
            fp8_scheme: FP8 scheme to use. Options:
                "float8wo" -- weight-only (broadest compatibility)
                "float8dq" -- dynamic activation + weight (better quality)
                "float8dq_e4m3_row" -- row-wise dynamic (best quality, Hopper)
                None -- auto-select best for hardware.
            fp8_components: Components to quantize. Default: transformers only
                (VAE is precision-sensitive and should not be quantized).
        """
        from animatediff.core.fp8_inference import FP8InferenceOptimizer

        optimizer = FP8InferenceOptimizer(scheme=fp8_scheme)

        if not optimizer.is_available:
            logger.info(
                "FP8 not available on this device "
                "(requires CUDA with compute >= 8.9 + torchao). Skipping."
            )
            return

        result = optimizer.optimize_pipeline(pipe, components=fp8_components)

        if result.optimized_components:
            logger.info(
                f"FP8 optimization applied: {result.optimized_components} "
                f"(scheme={result.scheme}, time={result.optimization_time_s:.1f}s)"
            )
            if result.memory_saved_mb > 0:
                logger.info(f"  Memory saved: {result.memory_saved_mb:.0f} MB")
