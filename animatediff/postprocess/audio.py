"""
Audio Generator — TTS voice generation and background music for video.

Supports:
- F5-TTS: zero-shot voice cloning (pip install f5-tts)
- F5-TTS-MLX: native Apple Silicon variant (pip install f5-tts-mlx)
- CosyVoice2: alternative TTS (requires git clone)
- BGM/SFX from audio files
- Per-character voice profiles with zero-shot voice cloning
- Audio post-processing: normalization, reverb, crossfading
"""

import logging
import os
import struct
import wave
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_SAMPLE_RATE = 24000  # F5-TTS native sample rate

# ---------------------------------------------------------------------------
# VoiceProfile — enhanced with description, language, speed
# ---------------------------------------------------------------------------


@dataclass
class VoiceProfile:
    """A character's voice profile for TTS generation.

    Supports zero-shot voice cloning when ``ref_audio`` and ``ref_text`` are
    provided.  Otherwise falls back to the default TTS voice.

    Attributes:
        name: Human-readable voice name (e.g. "narrator", "moDoctor").
        description: Natural-language description of the voice character.
        ref_audio: Path to a reference WAV file (mono, 24 kHz, 5-10 s).
        ref_text: Exact transcript of the reference audio.
        language: ISO 639-1 language code ("zh", "en", "ja").
        speed: Speaking-speed multiplier (1.0 = normal).
    """

    name: str
    description: str = ""
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    language: str = "zh"
    speed: float = 1.0

    # -- convenience constructors ------------------------------------------

    @classmethod
    def from_storyboard(cls, name: str, config: dict) -> "VoiceProfile":
        """Create a VoiceProfile from a storyboard ``voice_profiles`` entry.

        Example storyboard JSON::

            "voice_profiles": {
                "narrator": {
                    "description": "Male narrator, deep authoritative voice",
                    "ref_audio": "voices/narrator.wav",
                    "ref_text": "...",
                    "language": "zh",
                    "speed": 1.0
                }
            }

        Args:
            name: Voice profile key (e.g. "narrator").
            config: Dict parsed from the storyboard JSON entry.

        Returns:
            A populated VoiceProfile instance.
        """
        return cls(
            name=name,
            description=config.get("description", ""),
            ref_audio=config.get("ref_audio"),
            ref_text=config.get("ref_text"),
            language=config.get("language", "zh"),
            speed=config.get("speed", 1.0),
        )

    @classmethod
    def load_profiles_from_storyboard(
        cls, storyboard_data: dict,
    ) -> Dict[str, "VoiceProfile"]:
        """Load all voice profiles from a parsed storyboard dict.

        Args:
            storyboard_data: The full storyboard JSON parsed as a dict.

        Returns:
            Dict mapping voice_id -> VoiceProfile.
        """
        profiles: Dict[str, "VoiceProfile"] = {}
        vp_section = storyboard_data.get("voice_profiles", {})
        for voice_id, cfg in vp_section.items():
            profiles[voice_id] = cls.from_storyboard(voice_id, cfg)
        return profiles

    @property
    def has_reference(self) -> bool:
        """True when this profile has a usable reference audio."""
        return bool(self.ref_audio and os.path.isfile(self.ref_audio))


# ---------------------------------------------------------------------------
# AudioGenerator — core TTS engine wrapper
# ---------------------------------------------------------------------------


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
                import f5_tts_mlx  # noqa: F401

                return "f5_mlx"
            except ImportError:
                pass

        try:
            import f5_tts  # noqa: F401

            return "f5"
        except ImportError:
            pass

        logger.warning("No TTS engine available. Install with: pip install f5-tts")
        return "none"

    # ------------------------------------------------------------------
    # Low-level speech generation
    # ------------------------------------------------------------------

    def generate_speech(
        self,
        text: str,
        output_path: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        seed: Optional[int] = None,
        speed: float = 1.0,
    ) -> str:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize.
            output_path: Path to save the output WAV file.
            ref_audio: Reference audio for voice cloning (overrides default).
            ref_text: Transcript of reference audio (overrides default).
            seed: Random seed for reproducibility.
            speed: Speaking speed multiplier (1.0 = normal). Forwarded to the
                TTS engine when supported.

        Returns:
            Path to the generated audio file, or "" on failure.
        """
        ref_audio = ref_audio or self.ref_audio
        ref_text = ref_text or self.ref_text

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.tts_engine == "f5":
            return self._generate_f5(
                text, output_path, ref_audio, ref_text, seed, speed,
            )
        elif self.tts_engine == "f5_mlx":
            return self._generate_f5_mlx(
                text, output_path, ref_audio, ref_text, speed,
            )
        elif self.tts_engine == "cosyvoice":
            return self._generate_cosyvoice(text, output_path, ref_audio, ref_text)
        else:
            logger.warning("No TTS engine available, skipping speech generation")
            return ""

    def _generate_f5(self, text, output_path, ref_audio, ref_text, seed, speed=1.0):
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
            speed=speed,
        )
        logger.info(f"Generated speech: {output_path}")
        return output_path

    def _generate_f5_mlx(self, text, output_path, ref_audio, ref_text, speed=1.0):
        """Generate speech using F5-TTS-MLX (Apple Silicon native)."""
        from f5_tts_mlx.generate import generate

        kwargs: Dict[str, Any] = dict(
            generation_text=text,
            output_path=output_path,
            speed=speed,
        )
        if ref_audio:
            kwargs["ref_audio_path"] = ref_audio
        if ref_text:
            kwargs["ref_audio_text"] = ref_text

        audio = generate(**kwargs)

        # generate() saves to output_path if provided, but also returns audio
        if audio is not None and not os.path.exists(output_path):
            import soundfile as sf

            sf.write(output_path, audio, samplerate=_DEFAULT_SAMPLE_RATE)

        logger.info(f"Generated speech (MLX): {output_path}")
        return output_path

    def _generate_cosyvoice(self, text, output_path, ref_audio, ref_text):
        """Generate speech using CosyVoice2."""
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            import torchaudio

            if self._tts is None:
                self._tts = AutoModel(
                    model_dir="pretrained_models/CosyVoice2-0.5B",
                )
                logger.info("Loaded CosyVoice2 model")

            for _, result in enumerate(
                self._tts.inference_zero_shot(
                    text, ref_text or "", ref_audio or "",
                )
            ):
                torchaudio.save(
                    output_path, result["tts_speech"], self._tts.sample_rate,
                )
                break  # Just take the first result

            logger.info(f"Generated speech (CosyVoice): {output_path}")
            return output_path
        except ImportError:
            logger.error(
                "CosyVoice not installed. "
                "Clone from: https://github.com/FunAudioLLM/CosyVoice"
            )
            return ""

    # ------------------------------------------------------------------
    # Voice-profile-aware generation
    # ------------------------------------------------------------------

    def generate_narration_with_voice(
        self,
        text: str,
        voice_profile: VoiceProfile,
        output_path: str,
        seed: Optional[int] = None,
    ) -> str:
        """Generate narration using a specific voice profile.

        If the profile has reference audio the TTS engine uses zero-shot voice
        cloning; otherwise it falls back to the default voice.

        Args:
            text: The narration text to synthesize.
            voice_profile: A :class:`VoiceProfile` with optional ref audio.
            output_path: Where to save the resulting WAV.
            seed: Random seed for reproducible output.

        Returns:
            Path to the generated audio file, or "" on failure.
        """
        ref_audio = voice_profile.ref_audio if voice_profile.has_reference else None
        ref_text = voice_profile.ref_text if ref_audio else None
        speed = voice_profile.speed

        logger.info(
            f"Generating narration for voice '{voice_profile.name}' "
            f"(ref={'yes' if ref_audio else 'default'}, "
            f"speed={speed:.2f}): {text[:60]}..."
        )

        return self.generate_speech(
            text=text,
            output_path=output_path,
            ref_audio=ref_audio,
            ref_text=ref_text,
            seed=seed,
            speed=speed,
        )

    # ------------------------------------------------------------------
    # Batch narration (V3-compatible + enhanced)
    # ------------------------------------------------------------------

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

        This method is **backward-compatible** with V3.  When called without
        ``voice_profiles`` it behaves identically to the original version.

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
        paths: List[str] = []
        for i, text in enumerate(texts):
            if not text.strip():
                paths.append("")
                continue

            # Resolve per-shot voice profile
            profile: Optional[VoiceProfile] = None
            if voice_profiles and voice_ids and i < len(voice_ids) and voice_ids[i]:
                profile = voice_profiles.get(voice_ids[i])

            path = os.path.join(output_dir, f"narration_{i:04d}.wav")

            if profile is not None:
                result = self.generate_narration_with_voice(text, profile, path)
            else:
                # Fallback: use the default ref_audio / ref_text
                shot_ref_audio = ref_audio
                shot_ref_text = ref_text
                result = self.generate_speech(
                    text, path, shot_ref_audio, shot_ref_text,
                )

            paths.append(result)
        return paths


# ---------------------------------------------------------------------------
# Convenience wrapper (module-level function)
# ---------------------------------------------------------------------------


def generate_narration_with_voice(
    text: str,
    voice_profile: VoiceProfile,
    output_path: str,
    backend: str = "auto",
    seed: Optional[int] = None,
) -> str:
    """Module-level convenience: generate narration with a voice profile.

    Creates a one-shot :class:`AudioGenerator` and delegates.  Useful for
    quick scripting without manually managing engine lifetime.

    Args:
        text: Text to synthesize.
        voice_profile: Voice profile describing the target voice.
        output_path: Destination WAV path.
        backend: TTS backend — "f5_tts_mlx", "f5_tts", or "auto".
        seed: Optional random seed.

    Returns:
        Path to the generated audio file, or "" on failure.
    """
    gen = AudioGenerator(tts_engine=backend)
    return gen.generate_narration_with_voice(text, voice_profile, output_path, seed)


# ---------------------------------------------------------------------------
# Audio post-processing utilities
# ---------------------------------------------------------------------------
# These helpers work on WAV files using only the Python stdlib (``wave``,
# ``struct``) so that there are **zero** extra dependencies for basic usage.
# When ``numpy`` or ``soundfile`` are available they are used transparently
# for better performance.


def _read_wav(path: str) -> Tuple[bytes, int, int, int]:
    """Read a WAV file and return (raw_frames, n_channels, sample_width, rate)."""
    with wave.open(path, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
    return frames, params.nchannels, params.sampwidth, params.framerate


def _write_wav(
    path: str,
    frames: bytes,
    n_channels: int,
    sample_width: int,
    rate: int,
) -> str:
    """Write raw PCM frames to a WAV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return path


def _samples_from_bytes(raw: bytes, sample_width: int) -> List[float]:
    """Unpack PCM bytes to a list of float samples in [-1, 1]."""
    if sample_width == 2:
        fmt = f"<{len(raw) // 2}h"
        max_val = 32767.0
    elif sample_width == 4:
        fmt = f"<{len(raw) // 4}i"
        max_val = 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    ints = struct.unpack(fmt, raw)
    return [s / max_val for s in ints]


def _bytes_from_samples(samples: List[float], sample_width: int) -> bytes:
    """Pack float samples back to PCM bytes."""
    if sample_width == 2:
        fmt = "<h"
        max_val = 32767.0
    elif sample_width == 4:
        fmt = "<i"
        max_val = 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    parts = []
    for s in samples:
        clamped = max(-1.0, min(1.0, s))
        parts.append(struct.pack(fmt, int(clamped * max_val)))
    return b"".join(parts)


# -- public utilities -------------------------------------------------------


def normalize_audio(
    audio_path: str,
    target_db: float = -20.0,
    output_path: Optional[str] = None,
) -> str:
    """Normalize audio volume to *target_db* dBFS (peak normalization).

    Operates in-place when *output_path* is ``None``.

    Args:
        audio_path: Path to the source WAV file.
        target_db: Target peak level in dBFS (e.g. -20.0).
        output_path: Optional destination path; defaults to overwriting the
            source file.

    Returns:
        Path to the normalized audio file.
    """
    import math

    output_path = output_path or audio_path
    raw, nch, sw, rate = _read_wav(audio_path)
    samples = _samples_from_bytes(raw, sw)

    if not samples:
        logger.warning(f"normalize_audio: empty audio file {audio_path}")
        return audio_path

    peak = max(abs(s) for s in samples)
    if peak < 1e-8:
        logger.warning(f"normalize_audio: silence detected in {audio_path}")
        return audio_path

    current_db = 20.0 * math.log10(peak)
    gain_db = target_db - current_db
    gain_linear = 10.0 ** (gain_db / 20.0)

    normalized = [s * gain_linear for s in samples]
    out_bytes = _bytes_from_samples(normalized, sw)

    result = _write_wav(output_path, out_bytes, nch, sw, rate)
    logger.info(
        f"Normalized {audio_path} -> {output_path} "
        f"(peak {current_db:.1f} dB -> {target_db:.1f} dB)"
    )
    return result


def add_reverb(
    audio_path: str,
    room_size: float = 0.3,
    damping: float = 0.5,
    wet: float = 0.15,
    output_path: Optional[str] = None,
) -> str:
    """Add a simple algorithmic reverb for cinematic atmosphere.

    Uses a Schroeder reverb approximation with multiple comb filters and
    two all-pass filters.  No external dependencies required.

    Args:
        audio_path: Path to the source WAV file.
        room_size: Room size factor in [0, 1]. Larger values give longer decay.
        damping: High-frequency damping in [0, 1].
        wet: Wet/dry mix in [0, 1]. 0 = fully dry, 1 = fully wet.
        output_path: Optional destination path; defaults to overwriting.

    Returns:
        Path to the reverbed audio file.
    """
    output_path = output_path or audio_path
    raw, nch, sw, rate = _read_wav(audio_path)
    samples = _samples_from_bytes(raw, sw)

    if not samples:
        return audio_path

    # --- Schroeder reverb (mono simplification) --------------------------
    # Comb-filter delay lengths (in samples) scaled by room_size
    base_delays = [1557, 1617, 1491, 1422, 1277, 1356]
    scale = 0.5 + room_size * 1.5  # range [0.5, 2.0]
    comb_delays = [int(d * scale * rate / 44100) for d in base_delays]
    feedback = 0.7 + room_size * 0.2  # [0.7 .. 0.9]

    # All-pass delays
    ap_delays = [int(225 * scale * rate / 44100), int(556 * scale * rate / 44100)]

    n = len(samples)

    def _comb(inp: List[float], delay: int, fb: float, damp: float) -> List[float]:
        out = [0.0] * n
        buf = [0.0] * delay
        filt = 0.0
        idx = 0
        for i in range(n):
            rd = buf[idx]
            filt = rd * (1.0 - damp) + filt * damp
            buf[idx] = inp[i] + filt * fb
            out[i] = rd
            idx = (idx + 1) % delay
        return out

    def _allpass(inp: List[float], delay: int) -> List[float]:
        out = [0.0] * n
        buf = [0.0] * delay
        idx = 0
        g = 0.5
        for i in range(n):
            rd = buf[idx]
            out[i] = -inp[i] + rd
            buf[idx] = inp[i] + rd * g
            idx = (idx + 1) % delay
        return out

    # Sum comb filters
    reverb = [0.0] * n
    for delay in comb_delays:
        c = _comb(samples, max(delay, 1), feedback, damping)
        for i in range(n):
            reverb[i] += c[i]

    # Normalise comb sum
    num_combs = len(comb_delays)
    reverb = [r / num_combs for r in reverb]

    # Chain all-pass filters
    for delay in ap_delays:
        reverb = _allpass(reverb, max(delay, 1))

    # Mix wet / dry
    dry = 1.0 - wet
    mixed = [samples[i] * dry + reverb[i] * wet for i in range(n)]

    out_bytes = _bytes_from_samples(mixed, sw)
    result = _write_wav(output_path, out_bytes, nch, sw, rate)
    logger.info(
        f"Added reverb to {audio_path} -> {output_path} "
        f"(room={room_size:.2f}, wet={wet:.2f})"
    )
    return result


def crossfade_audio(
    audio_a: str,
    audio_b: str,
    overlap_ms: int = 500,
    output_path: Optional[str] = None,
) -> str:
    """Crossfade between two audio clips.

    The end of *audio_a* is faded out while the beginning of *audio_b* is
    faded in over *overlap_ms* milliseconds.  The result is a single
    seamlessly joined WAV.

    Args:
        audio_a: Path to the first audio clip.
        audio_b: Path to the second audio clip.
        overlap_ms: Crossfade duration in milliseconds.
        output_path: Destination path.  Defaults to ``audio_a`` stem +
            ``_xfade.wav``.

    Returns:
        Path to the crossfaded audio file.
    """
    if output_path is None:
        stem = Path(audio_a).stem
        output_path = str(Path(audio_a).with_name(f"{stem}_xfade.wav"))

    raw_a, nch_a, sw_a, rate_a = _read_wav(audio_a)
    raw_b, nch_b, sw_b, rate_b = _read_wav(audio_b)

    if sw_a != sw_b:
        raise ValueError(
            f"Sample width mismatch: {audio_a}={sw_a}, {audio_b}={sw_b}"
        )
    if nch_a != nch_b:
        raise ValueError(
            f"Channel count mismatch: {audio_a}={nch_a}, {audio_b}={nch_b}"
        )
    if rate_a != rate_b:
        raise ValueError(
            f"Sample rate mismatch: {audio_a}={rate_a}, {audio_b}={rate_b}"
        )

    samples_a = _samples_from_bytes(raw_a, sw_a)
    samples_b = _samples_from_bytes(raw_b, sw_b)

    overlap_samples = int(rate_a * overlap_ms / 1000) * nch_a
    overlap_samples = min(overlap_samples, len(samples_a), len(samples_b))

    if overlap_samples <= 0:
        # No overlap possible: just concatenate
        combined = samples_a + samples_b
    else:
        # Build: head_a + crossfade_region + tail_b
        head = samples_a[: len(samples_a) - overlap_samples]
        tail = samples_b[overlap_samples:]

        xfade: List[float] = []
        for j in range(overlap_samples):
            t = j / float(overlap_samples)  # 0 -> 1
            fade_out = 1.0 - t
            fade_in = t
            a_idx = len(samples_a) - overlap_samples + j
            mixed = samples_a[a_idx] * fade_out + samples_b[j] * fade_in
            xfade.append(mixed)

        combined = head + xfade + tail

    out_bytes = _bytes_from_samples(combined, sw_a)
    result = _write_wav(output_path, out_bytes, nch_a, sw_a, rate_a)
    logger.info(
        f"Crossfaded {audio_a} + {audio_b} -> {output_path} "
        f"(overlap={overlap_ms}ms)"
    )
    return result


def concatenate_audio(
    audio_paths: List[str],
    output_path: str,
    gap_ms: int = 0,
) -> str:
    """Concatenate multiple WAV files with an optional silent gap between each.

    All input files must share the same sample rate, channels, and sample
    width.

    Args:
        audio_paths: Ordered list of WAV file paths to concatenate.
        output_path: Destination WAV path.
        gap_ms: Duration of silence to insert between clips (ms).

    Returns:
        Path to the concatenated audio file.
    """
    valid = [p for p in audio_paths if p and os.path.isfile(p)]
    if not valid:
        logger.warning("concatenate_audio: no valid audio files provided")
        return ""

    raw0, nch, sw, rate = _read_wav(valid[0])
    all_frames = [raw0]

    gap_samples = int(rate * gap_ms / 1000) * nch
    silence = b"\x00" * (gap_samples * sw)

    for p in valid[1:]:
        raw, nc, s, r = _read_wav(p)
        if (nc, s, r) != (nch, sw, rate):
            logger.warning(
                f"concatenate_audio: skipping {p} (format mismatch: "
                f"ch={nc} sw={s} rate={r} vs ch={nch} sw={sw} rate={rate})"
            )
            continue
        if gap_ms > 0:
            all_frames.append(silence)
        all_frames.append(raw)

    combined = b"".join(all_frames)
    result = _write_wav(output_path, combined, nch, sw, rate)
    logger.info(
        f"Concatenated {len(valid)} clips -> {output_path} "
        f"(gap={gap_ms}ms)"
    )
    return result


def get_audio_duration(audio_path: str) -> float:
    """Return the duration of a WAV file in seconds.

    Args:
        audio_path: Path to a WAV file.

    Returns:
        Duration in seconds, or 0.0 if the file cannot be read.
    """
    try:
        with wave.open(audio_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception as exc:
        logger.warning(f"get_audio_duration: cannot read {audio_path}: {exc}")
        return 0.0


def prepare_reference_audio(
    input_path: str,
    output_path: Optional[str] = None,
    target_sr: int = _DEFAULT_SAMPLE_RATE,
    max_duration: float = 10.0,
) -> str:
    """Convert an audio file to the format expected by F5-TTS reference input.

    The output is mono, 16-bit PCM WAV at *target_sr* Hz, trimmed to at most
    *max_duration* seconds.  Uses ``ffmpeg`` when available for broad format
    support, falling back to a pure-Python approach for WAV inputs.

    Args:
        input_path: Path to any audio file (mp3, m4a, flac, wav, ...).
        output_path: Destination path.  Defaults to ``<stem>_ref.wav`` next to
            the input.
        target_sr: Target sample rate (F5-TTS expects 24000).
        max_duration: Maximum duration in seconds (F5-TTS works best with
            5-10 s clips).

    Returns:
        Path to the prepared reference WAV.
    """
    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).with_name(f"{stem}_ref.wav"))

    import shutil

    if shutil.which("ffmpeg"):
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", str(target_sr),
            "-sample_fmt", "s16",
            "-t", str(max_duration),
            output_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info(
                f"Prepared reference audio: {input_path} -> {output_path} "
                f"(mono, {target_sr}Hz, max {max_duration}s)"
            )
            return output_path
        else:
            logger.warning(
                f"ffmpeg failed ({result.returncode}), "
                f"attempting pure-Python fallback: {result.stderr[:200]}"
            )

    # Fallback: read as WAV, truncate, write
    try:
        raw, nch, sw, rate = _read_wav(input_path)
        samples = _samples_from_bytes(raw, sw)

        # Convert stereo to mono
        if nch == 2:
            mono = []
            for i in range(0, len(samples), 2):
                mono.append((samples[i] + samples[i + 1]) * 0.5)
            samples = mono
            nch = 1

        # Truncate
        max_samples = int(max_duration * rate)
        samples = samples[:max_samples]

        out_bytes = _bytes_from_samples(samples, 2)  # always 16-bit output
        _write_wav(output_path, out_bytes, 1, 2, rate)
        logger.info(
            f"Prepared reference audio (fallback): "
            f"{input_path} -> {output_path}"
        )
        return output_path
    except Exception as exc:
        logger.error(f"Failed to prepare reference audio: {exc}")
        return input_path  # return original as last resort
