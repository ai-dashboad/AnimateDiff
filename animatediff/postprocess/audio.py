"""
Audio Generator — TTS voice generation and background music for video.

Supports:
- F5-TTS: zero-shot voice cloning (pip install f5-tts)
- F5-TTS-MLX: native Apple Silicon variant (pip install f5-tts-mlx)
- CosyVoice2: alternative TTS (requires git clone)
- BGM/SFX from audio files
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Voice profile for per-character TTS."""
    name: str  # e.g., "narrator", "moDoctor", "hanLi"
    ref_audio: Optional[str] = None  # path to reference voice sample
    ref_text: Optional[str] = None  # transcript of reference audio


class AudioGenerator:
    """Generate voice, BGM, and sound effects for videos."""

    def __init__(
        self,
        tts_engine: str = "auto",
        device: str = "cpu",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        """
        Args:
            tts_engine: "f5", "f5_mlx", "cosyvoice", or "auto"
            device: torch device
            ref_audio: path to reference voice audio for cloning
            ref_text: transcript of the reference audio
        """
        self.device = device
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.tts_engine = self._resolve_engine(tts_engine)
        self._tts = None

    def _resolve_engine(self, engine: str) -> str:
        if engine != "auto":
            return engine

        # On macOS, prefer MLX variant
        import platform
        if platform.system() == "Darwin":
            try:
                import f5_tts_mlx
                return "f5_mlx"
            except ImportError:
                pass

        try:
            import f5_tts
            return "f5"
        except ImportError:
            pass

        logger.warning("No TTS engine available. Install with: pip install f5-tts")
        return "none"

    def generate_speech(
        self,
        text: str,
        output_path: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize.
            output_path: Path to save the output WAV file.
            ref_audio: Reference audio for voice cloning (overrides default).
            ref_text: Transcript of reference audio (overrides default).
            seed: Random seed for reproducibility.

        Returns:
            Path to the generated audio file.
        """
        ref_audio = ref_audio or self.ref_audio
        ref_text = ref_text or self.ref_text

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.tts_engine == "f5":
            return self._generate_f5(text, output_path, ref_audio, ref_text, seed)
        elif self.tts_engine == "f5_mlx":
            return self._generate_f5_mlx(text, output_path, ref_audio, ref_text)
        elif self.tts_engine == "cosyvoice":
            return self._generate_cosyvoice(text, output_path, ref_audio, ref_text)
        else:
            logger.warning("No TTS engine available, skipping speech generation")
            return ""

    def _generate_f5(self, text, output_path, ref_audio, ref_text, seed):
        """Generate speech using F5-TTS (PyTorch)."""
        from f5_tts.api import F5TTS

        if self._tts is None:
            self._tts = F5TTS()
            logger.info("Loaded F5-TTS model")

        self._tts.infer(
            ref_file=ref_audio or "",
            ref_text=ref_text or "",
            gen_text=text,
            file_wave=output_path,
            seed=seed,
        )
        logger.info(f"Generated speech: {output_path}")
        return output_path

    def _generate_f5_mlx(self, text, output_path, ref_audio, ref_text):
        """Generate speech using F5-TTS-MLX (Apple Silicon native)."""
        from f5_tts_mlx.generate import generate

        kwargs = dict(generation_text=text, output_path=output_path)
        if ref_audio:
            kwargs["ref_audio_path"] = ref_audio
        if ref_text:
            kwargs["ref_audio_text"] = ref_text

        audio = generate(**kwargs)

        # generate() saves to output_path if provided, but also returns audio
        if audio is not None and not os.path.exists(output_path):
            import soundfile as sf
            sf.write(output_path, audio, samplerate=24000)

        logger.info(f"Generated speech (MLX): {output_path}")
        return output_path

    def _generate_cosyvoice(self, text, output_path, ref_audio, ref_text):
        """Generate speech using CosyVoice2."""
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            import torchaudio

            if self._tts is None:
                self._tts = AutoModel(model_dir="pretrained_models/CosyVoice2-0.5B")
                logger.info("Loaded CosyVoice2 model")

            for _, result in enumerate(self._tts.inference_zero_shot(
                text, ref_text or "", ref_audio or ""
            )):
                torchaudio.save(output_path, result["tts_speech"], self._tts.sample_rate)
                break  # Just take the first result

            logger.info(f"Generated speech (CosyVoice): {output_path}")
            return output_path
        except ImportError:
            logger.error("CosyVoice not installed. Clone from: https://github.com/FunAudioLLM/CosyVoice")
            return ""

    def generate_narration(
        self,
        texts: List[str],
        output_dir: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        voice_profiles: Optional[Dict[str, "VoiceProfile"]] = None,
        voice_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate narration audio for multiple shots.

        Args:
            texts: List of narration texts (one per shot).
            output_dir: Directory to save audio files.
            ref_audio: Default reference voice audio.
            ref_text: Default reference voice transcript.
            voice_profiles: Optional dict of voice_id -> VoiceProfile for
                per-shot voice selection (e.g., different voices for
                narrator vs character dialogue).
            voice_ids: Optional list of voice_id per shot. If provided,
                looks up the corresponding VoiceProfile from voice_profiles.

        Returns:
            List of paths to generated audio files.
        """
        paths = []
        for i, text in enumerate(texts):
            if not text.strip():
                paths.append("")
                continue

            # Resolve per-shot voice
            shot_ref_audio = ref_audio
            shot_ref_text = ref_text
            if voice_profiles and voice_ids and i < len(voice_ids) and voice_ids[i]:
                profile = voice_profiles.get(voice_ids[i])
                if profile:
                    shot_ref_audio = profile.ref_audio
                    shot_ref_text = profile.ref_text

            path = os.path.join(output_dir, f"narration_{i:04d}.wav")
            result = self.generate_speech(text, path, shot_ref_audio, shot_ref_text)
            paths.append(result)
        return paths
