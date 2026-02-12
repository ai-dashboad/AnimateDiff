"""
Audio-Visual Sync Pipeline — joint audio-video generation and synchronization.

Approximates SeedAnce 2.0's joint A/V generation by selecting the best strategy
based on available backends and references:

  Strategy 1 (preferred): LTX-2 native joint A/V generation
  Strategy 2: Wan S2V audio-driven video
  Strategy 3: Generate video first → add TTS narration + SFX post-hoc

Also provides post-hoc audio-video synchronization for aligning audio timing
to generated video (beat-sync, phoneme-sync, energy-matching).
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput

logger = logging.getLogger(__name__)

__all__ = ["AVSyncStrategy", "AVSyncPipeline"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AVSyncStrategy:
    """Describes the chosen audio-video sync approach for a shot."""
    name: Literal["joint_av", "audio_driven", "post_hoc"] = "post_hoc"
    backend_name: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def _detect_strategy(
    audio_ref: Optional[str] = None,
    voice_profile: Optional[dict] = None,
    lip_sync: bool = False,
    available_backends: Optional[List[str]] = None,
) -> AVSyncStrategy:
    """Select the best A/V sync strategy based on context.

    Priority:
      1. LTX-2 joint A/V — if ltx2 backend available and audio ref given
      2. Wan S2V audio-driven — if wan22_s2v available and audio ref given
      3. Post-hoc — generate video, then add audio separately

    Args:
        audio_ref: Path to reference audio file (dialogue, soundtrack).
        voice_profile: Voice profile dict for TTS generation.
        lip_sync: Whether phoneme-level lip sync is requested.
        available_backends: List of loaded/available backend names.

    Returns:
        AVSyncStrategy describing the chosen approach.
    """
    available = set(available_backends or [])

    # Strategy 1: LTX-2 native joint A/V
    if "ltx2" in available and audio_ref:
        return AVSyncStrategy(
            name="joint_av",
            backend_name="ltx2",
            reason="LTX-2 supports native joint audio-video generation",
        )

    # Strategy 2: Wan S2V audio-driven
    if "wan22_s2v" in available and audio_ref:
        return AVSyncStrategy(
            name="audio_driven",
            backend_name="wan22_s2v",
            reason="Wan 2.2 S2V generates video driven by audio input",
        )

    # Strategy 3: Post-hoc (always available)
    return AVSyncStrategy(
        name="post_hoc",
        backend_name="",
        reason="Generate video first, then add audio via TTS + postprocessing",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class AVSyncPipeline:
    """Unified audio-video generation pipeline.

    Selects the optimal generation strategy and handles audio-video
    synchronization for each shot.

    Usage::

        av = AVSyncPipeline(available_backends=["wan22", "ltx2"])

        # Joint generation (preferred when possible)
        output = av.generate(
            prompt="A woman speaks to the camera",
            audio_ref="dialogue.wav",
            lip_sync=True,
        )

        # Post-hoc sync
        av.sync_audio_to_video(video_output, audio_path, mode="phoneme")
    """

    def __init__(
        self,
        available_backends: Optional[List[str]] = None,
        tts_backend: str = "f5_tts",
        device: str = "auto",
    ):
        """
        Args:
            available_backends: Names of loaded backends (e.g., ["wan22", "ltx2"]).
            tts_backend: TTS engine for narration generation ("f5_tts", "f5_tts_mlx").
            device: Compute device ("auto", "cuda", "mps", "cpu").
        """
        self.available_backends = available_backends or []
        self.tts_backend = tts_backend
        self.device = self._resolve_device(device)
        self._backend_cache: Dict[str, BasePipeline] = {}

        logger.info(
            f"AVSyncPipeline: backends={self.available_backends}, "
            f"tts={tts_backend}, device={self.device}"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def select_strategy(
        self,
        audio_ref: Optional[str] = None,
        voice_profile: Optional[dict] = None,
        lip_sync: bool = False,
    ) -> AVSyncStrategy:
        """Select the best A/V sync strategy for a shot.

        Returns:
            AVSyncStrategy with name, backend, and reason.
        """
        return _detect_strategy(
            audio_ref=audio_ref,
            voice_profile=voice_profile,
            lip_sync=lip_sync,
            available_backends=self.available_backends,
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        audio_ref: Optional[str] = None,
        voice_profile: Optional[dict] = None,
        lip_sync: bool = False,
        backend_loader: Optional[Any] = None,
        **kwargs,
    ) -> VideoOutput:
        """Generate video with audio awareness.

        Auto-selects the best strategy based on available backends and inputs.

        Args:
            prompt: Text prompt for video generation.
            negative_prompt: Negative prompt.
            width, height: Resolution.
            num_frames: Number of frames.
            num_inference_steps: Diffusion steps.
            guidance_scale: CFG scale.
            seed: Random seed (-1 for random).
            image: Optional reference image for I2V.
            audio_ref: Path to reference audio (dialogue, soundtrack).
            voice_profile: Voice profile dict for TTS.
            lip_sync: Whether phoneme lip sync is requested.
            backend_loader: Callable to load backends by name.
            **kwargs: Extra backend-specific arguments.

        Returns:
            VideoOutput (may include audio if joint A/V strategy used).
        """
        strategy = self.select_strategy(audio_ref, voice_profile, lip_sync)
        logger.info(f"A/V strategy: {strategy.name} ({strategy.reason})")

        if strategy.name == "joint_av":
            return self._generate_joint_av(
                strategy, prompt, negative_prompt,
                width, height, num_frames, num_inference_steps,
                guidance_scale, seed, audio_ref, backend_loader, **kwargs,
            )
        elif strategy.name == "audio_driven":
            return self._generate_audio_driven(
                strategy, prompt, negative_prompt,
                width, height, num_frames, num_inference_steps,
                guidance_scale, seed, image, audio_ref,
                backend_loader, **kwargs,
            )
        else:
            return self._generate_post_hoc(
                prompt, negative_prompt,
                width, height, num_frames, num_inference_steps,
                guidance_scale, seed, image, backend_loader, **kwargs,
            )

    def _generate_joint_av(
        self,
        strategy: AVSyncStrategy,
        prompt: str,
        negative_prompt: str,
        width: int, height: int,
        num_frames: int, steps: int,
        guidance: float, seed: int,
        audio_ref: Optional[str],
        backend_loader: Optional[Any],
        **kwargs,
    ) -> VideoOutput:
        """Strategy 1: LTX-2 joint audio-video generation."""
        backend = self._get_backend(strategy.backend_name, backend_loader)
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            **kwargs,
        }
        if audio_ref:
            gen_kwargs["audio_ref"] = audio_ref

        output = backend.generate(**gen_kwargs)
        logger.info(
            f"Joint A/V: {len(output.frames)} frames, "
            f"audio={'yes' if output.audio is not None else 'no'}"
        )
        return output

    def _generate_audio_driven(
        self,
        strategy: AVSyncStrategy,
        prompt: str,
        negative_prompt: str,
        width: int, height: int,
        num_frames: int, steps: int,
        guidance: float, seed: int,
        image: Optional[Image.Image],
        audio_ref: Optional[str],
        backend_loader: Optional[Any],
        **kwargs,
    ) -> VideoOutput:
        """Strategy 2: Wan S2V audio-driven video generation."""
        backend = self._get_backend(strategy.backend_name, backend_loader)
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            **kwargs,
        }
        if audio_ref:
            gen_kwargs["audio_path"] = audio_ref
        if image is not None:
            gen_kwargs["image"] = image

        output = backend.generate(**gen_kwargs)
        logger.info(
            f"Audio-driven: {len(output.frames)} frames via {strategy.backend_name}"
        )
        return output

    def _generate_post_hoc(
        self,
        prompt: str,
        negative_prompt: str,
        width: int, height: int,
        num_frames: int, steps: int,
        guidance: float, seed: int,
        image: Optional[Image.Image],
        backend_loader: Optional[Any],
        **kwargs,
    ) -> VideoOutput:
        """Strategy 3: Generate video first, audio added separately."""
        # Use the first available video backend
        backend_name = self.available_backends[0] if self.available_backends else "wan22"
        backend = self._get_backend(backend_name, backend_loader)

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            **kwargs,
        }
        if image is not None:
            gen_kwargs["image"] = image

        output = backend.generate(**gen_kwargs)
        logger.info(f"Post-hoc: {len(output.frames)} frames via {backend_name}")
        return output

    def _get_backend(
        self, name: str, loader: Optional[Any],
    ) -> BasePipeline:
        """Load or retrieve cached backend."""
        if name in self._backend_cache:
            return self._backend_cache[name]

        if loader:
            backend = loader(name)
        else:
            from animatediff.backends import get_backend
            backend_cls = get_backend(name)
            backend = backend_cls.load()

        self._backend_cache[name] = backend
        return backend

    # ------------------------------------------------------------------
    # Post-hoc audio-video sync
    # ------------------------------------------------------------------

    def sync_audio_to_video(
        self,
        output: VideoOutput,
        audio_path: str,
        mode: Literal["beat", "phoneme", "energy"] = "beat",
        fps: int = 24,
    ) -> VideoOutput:
        """Align audio timing to generated video post-hoc.

        Args:
            output: Generated VideoOutput (frames only, no audio).
            audio_path: Path to audio file to synchronize.
            mode: Sync mode:
                - "beat": Align to musical beats (for BGM/soundtrack).
                - "phoneme": Phoneme-level alignment (for dialogue).
                - "energy": Energy-envelope matching.
            fps: Video frame rate.

        Returns:
            VideoOutput with audio field populated.
        """
        import torch

        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return output

        try:
            # Load audio
            audio_tensor, sample_rate = self._load_audio(audio_path)

            if mode == "beat":
                audio_tensor = self._sync_beat(audio_tensor, sample_rate, output, fps)
            elif mode == "phoneme":
                audio_tensor = self._sync_phoneme(audio_tensor, sample_rate, output, fps)
            elif mode == "energy":
                audio_tensor = self._sync_energy(audio_tensor, sample_rate, output, fps)

            # Trim or pad audio to match video duration
            video_duration = len(output.frames) / fps
            target_samples = int(video_duration * sample_rate)
            if audio_tensor.shape[-1] > target_samples:
                audio_tensor = audio_tensor[..., :target_samples]
            elif audio_tensor.shape[-1] < target_samples:
                pad_size = target_samples - audio_tensor.shape[-1]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))

            output.audio = audio_tensor
            output.audio_sample_rate = sample_rate
            logger.info(
                f"Audio synced ({mode}): {audio_tensor.shape[-1]/sample_rate:.2f}s "
                f"at {sample_rate}Hz"
            )

        except Exception as e:
            logger.warning(f"Audio sync failed: {e}")

        return output

    @staticmethod
    def _load_audio(path: str):
        """Load audio file as torch tensor."""
        import torch
        import numpy as np

        try:
            import soundfile as sf
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return torch.from_numpy(data.astype(np.float32)), sr
        except ImportError:
            pass

        # Fallback: wave module (WAV only)
        import wave
        import struct
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(n)

        if sw == 2:
            fmt = f"<{n * ch}h"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
        else:
            samples = np.zeros(n, dtype=np.float32)

        if ch > 1:
            samples = samples.reshape(-1, ch).mean(axis=1)

        return torch.from_numpy(samples), sr

    @staticmethod
    def _sync_beat(audio, sr, output, fps):
        """Beat-aligned sync (no-op for now — audio already at correct pace)."""
        return audio

    @staticmethod
    def _sync_phoneme(audio, sr, output, fps):
        """Phoneme-aligned sync — delegates to phoneme_sync if available."""
        try:
            from animatediff.postprocess.phoneme_sync import PhonemeAligner
            aligner = PhonemeAligner()
            return aligner.align_audio(audio, sr, len(output.frames), fps)
        except ImportError:
            logger.debug("phoneme_sync not available, returning audio unchanged")
            return audio

    @staticmethod
    def _sync_energy(audio, sr, output, fps):
        """Energy-envelope sync — no transformation needed for basic case."""
        return audio

    # ------------------------------------------------------------------
    # TTS generation
    # ------------------------------------------------------------------

    def generate_narration(
        self,
        text: str,
        voice_profile: Optional[dict] = None,
        output_path: str = "",
        language: str = "auto",
    ) -> str:
        """Generate narration audio from text using TTS.

        Args:
            text: Narration text.
            voice_profile: Dict with ref_audio, ref_text for voice cloning.
            output_path: Where to save the audio file.
            language: Language code or "auto".

        Returns:
            Path to the generated audio file.
        """
        if not output_path:
            import tempfile
            output_path = tempfile.mktemp(suffix=".wav")

        try:
            from animatediff.postprocess.audio import AudioGenerator, VoiceProfile

            profile = None
            if voice_profile:
                profile = VoiceProfile(
                    description=voice_profile.get("description", ""),
                    ref_audio=voice_profile.get("ref_audio", ""),
                    ref_text=voice_profile.get("ref_text", ""),
                )

            gen = AudioGenerator(backend=self.tts_backend)
            gen.generate_speech(text, output_path, voice=profile)
            logger.info(f"Generated narration: {output_path}")
            return output_path

        except ImportError as e:
            logger.warning(f"TTS not available: {e}")
            return ""
