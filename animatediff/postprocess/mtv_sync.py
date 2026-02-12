"""
MTV Sync -- multi-track audio separation and frame-level video synchronization.

Separates audio into speech / music / SFX streams and provides per-frame
synchronization data that the video pipeline can use to drive motion
intensity, shot transitions, and lip-sync alignment.

Backends:
- demucs:  Meta's Hybrid Transformer Demucs (neural source separation).
           ``pip install demucs`` or via torchaudio's HDEMUCS pipeline.
- basic:   Simple frequency-band separation using numpy FFT (always available).
- auto:    Try demucs -> fall back to basic.

Integration:
- Reuses :class:`~animatediff.postprocess.beat_sync.BeatAnalyzer` for
  music beat detection.
- :class:`SyncMap` is designed to be compatible with
  :class:`~animatediff.postprocess.beat_sync.ShotBeatAligner`.
"""

import logging
import math
import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

_DEMUCS_AVAILABLE = False
_DEMUCS_BACKEND = "none"  # "demucs_cli" | "torchaudio" | "none"

try:
    import demucs.separate  # noqa: F401

    _DEMUCS_AVAILABLE = True
    _DEMUCS_BACKEND = "demucs_cli"
except ImportError:
    try:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS  # noqa: F401

        _DEMUCS_AVAILABLE = True
        _DEMUCS_BACKEND = "torchaudio"
    except ImportError:
        pass

_LIBROSA_AVAILABLE = False
try:
    import librosa  # noqa: F401

    _LIBROSA_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AudioStreams:
    """Result of audio source separation.

    Each field is a path to a mono/stereo WAV file for that stream, or
    ``None`` when the stream could not be separated.
    """

    speech: Optional[str] = None  # path to separated speech WAV
    music: Optional[str] = None  # path to separated music WAV
    effects: Optional[str] = None  # path to separated SFX WAV
    original: str = ""  # path to original audio


@dataclass
class TimeSegment:
    """A labelled time range within an audio file."""

    start: float  # seconds
    end: float  # seconds
    label: str  # "speech", "silence", "music", etc.
    confidence: float = 1.0  # 0.0 - 1.0


@dataclass
class SyncMap:
    """Per-frame synchronization map for video generation.

    All lists have length ``total_frames``.  The map carries enough
    information for the video pipeline to:

    * modulate motion intensity per frame (``frame_energy``)
    * detect beat hits for transition timing (``frame_is_beat``)
    * identify speech regions for lip-sync (``frame_is_speech``)
    * suggest qualitative motion levels (``suggested_motion``)

    Compatible with :class:`~animatediff.postprocess.beat_sync.ShotBeatAligner`.
    """

    fps: int
    total_frames: int
    frame_energy: List[float] = field(default_factory=list)
    frame_is_beat: List[bool] = field(default_factory=list)
    frame_is_speech: List[bool] = field(default_factory=list)
    suggested_motion: List[str] = field(default_factory=list)

    # Auxiliary data -- useful for downstream consumers
    beat_times: List[float] = field(default_factory=list)
    speech_segments: List[TimeSegment] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def energy_at(self, frame: int) -> float:
        """Return energy for a given frame, clamped to valid range."""
        if not self.frame_energy:
            return 0.0
        idx = max(0, min(frame, len(self.frame_energy) - 1))
        return self.frame_energy[idx]

    def is_beat_at(self, frame: int) -> bool:
        """Return whether a frame coincides with a music beat."""
        if not self.frame_is_beat:
            return False
        idx = max(0, min(frame, len(self.frame_is_beat) - 1))
        return self.frame_is_beat[idx]

    def is_speech_at(self, frame: int) -> bool:
        """Return whether a frame coincides with speech."""
        if not self.frame_is_speech:
            return False
        idx = max(0, min(frame, len(self.frame_is_speech) - 1))
        return self.frame_is_speech[idx]

    def motion_at(self, frame: int) -> str:
        """Return suggested motion level at a frame."""
        if not self.suggested_motion:
            return "medium"
        idx = max(0, min(frame, len(self.suggested_motion) - 1))
        return self.suggested_motion[idx]

    def duration(self) -> float:
        """Total duration in seconds."""
        return self.total_frames / self.fps if self.fps > 0 else 0.0


# ---------------------------------------------------------------------------
# WAV I/O helpers (stdlib-only, mirrors audio.py patterns)
# ---------------------------------------------------------------------------


def _read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file as a mono float64 numpy array in [-1, 1].

    Returns:
        (samples, sample_rate)
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
        max_val = 32767.0
    elif sample_width == 4:
        dtype = np.int32
        max_val = 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64) / max_val

    # Convert to mono by averaging channels
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, rate


def _write_wav_mono(path: str, samples: np.ndarray, rate: int) -> str:
    """Write a mono float array to a 16-bit WAV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------------------
# AudioStreamAnalyzer
# ---------------------------------------------------------------------------


class AudioStreamAnalyzer:
    """Separates and analyzes audio into speech / SFX / BGM streams.

    Usage::

        analyzer = AudioStreamAnalyzer(backend="auto")
        streams = analyzer.separate("trailer_audio.wav")
        speech_segments = analyzer.get_speech_segments("trailer_audio.wav")
        energy = analyzer.get_energy_per_frame("trailer_audio.wav", fps=24)
    """

    def __init__(self, backend: str = "auto"):
        """
        Args:
            backend: Source separation backend.

                * ``"demucs"`` -- Meta's Demucs neural source separation.
                  Requires ``pip install demucs`` *or* torchaudio with the
                  HDEMUCS pipeline.
                * ``"basic"`` -- Simple frequency-band filtering using numpy
                  FFT.  Always available, no extra dependencies.
                * ``"auto"`` -- Try demucs first, fall back to basic.
        """
        self.backend = self._resolve_backend(backend)
        logger.info(f"AudioStreamAnalyzer initialized (backend={self.backend})")

        # Cache: audio_path -> AudioStreams
        self._cache: Dict[str, AudioStreams] = {}

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "demucs":
            if not _DEMUCS_AVAILABLE:
                raise ImportError(
                    "Demucs is not available. Install with:\n"
                    "    pip install demucs\n"
                    "Or install torchaudio for the built-in HDEMUCS pipeline."
                )
            return "demucs"
        if backend == "basic":
            return "basic"
        if backend == "auto":
            if _DEMUCS_AVAILABLE:
                logger.info(
                    f"Auto-selected demucs backend ({_DEMUCS_BACKEND})"
                )
                return "demucs"
            logger.info(
                "Demucs not available, using basic frequency-band separation"
            )
            return "basic"
        raise ValueError(
            f"Unknown backend: '{backend}'. Supported: demucs, basic, auto"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def separate(self, audio_path: str, output_dir: Optional[str] = None) -> AudioStreams:
        """Separate audio into speech, music, and effects streams.

        Args:
            audio_path: Path to the source audio file (WAV recommended;
                Demucs also supports MP3/FLAC/OGG).
            output_dir: Directory for separated WAV outputs.  Defaults to
                a ``_separated`` subdirectory next to the source file.

        Returns:
            :class:`AudioStreams` with paths to the separated files.
        """
        audio_path = os.path.abspath(audio_path)
        if audio_path in self._cache:
            return self._cache[audio_path]

        if output_dir is None:
            stem = Path(audio_path).stem
            output_dir = str(Path(audio_path).parent / f"{stem}_separated")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self.backend == "demucs":
            streams = self._separate_demucs(audio_path, output_dir)
        else:
            streams = self._separate_basic(audio_path, output_dir)

        self._cache[audio_path] = streams
        return streams

    def get_speech_segments(self, audio_path: str) -> List[TimeSegment]:
        """Detect speech segments with timestamps.

        Uses energy-based voice activity detection (VAD) on the speech
        stream.  If source separation produced a speech track, VAD runs
        on that; otherwise it runs on the original audio filtered to the
        speech frequency band.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Sorted list of :class:`TimeSegment` for speech regions.
        """
        # Try to get the separated speech track
        streams = self.separate(audio_path)
        vad_path = streams.speech or audio_path

        try:
            samples, sr = _read_wav_mono(vad_path)
        except Exception as exc:
            logger.warning(
                f"Cannot read {vad_path} for VAD, trying original: {exc}"
            )
            samples, sr = _read_wav_mono(audio_path)

        return self._energy_vad(samples, sr)

    def get_music_beats(self, audio_path: str) -> List[float]:
        """Detect music beats, delegating to :class:`BeatAnalyzer`.

        Falls back to a simple onset-based approach when librosa is not
        available.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Sorted list of beat timestamps in seconds.
        """
        # Prefer separated music track for cleaner beat detection
        streams = self.separate(audio_path)
        beat_path = streams.music or audio_path

        if _LIBROSA_AVAILABLE:
            from animatediff.postprocess.beat_sync import BeatAnalyzer

            try:
                analyzer = BeatAnalyzer(beat_path)
                return analyzer.get_beats()
            except Exception as exc:
                logger.warning(f"BeatAnalyzer failed on {beat_path}: {exc}")

        # Fallback: simple onset detection with numpy
        return self._simple_onset_beats(beat_path)

    def get_energy_per_frame(
        self, audio_path: str, fps: int = 24
    ) -> List[float]:
        """Compute per-frame audio energy for motion intensity matching.

        Returns RMS energy sampled at the given video frame rate,
        normalised to [0, 1].

        Args:
            audio_path: Path to the audio file.
            fps: Video frame rate.

        Returns:
            List of floats (one per video frame) in [0, 1].
        """
        try:
            samples, sr = _read_wav_mono(audio_path)
        except Exception as exc:
            logger.warning(f"Cannot read {audio_path} for energy: {exc}")
            return []

        hop = max(1, sr // fps)
        n_frames = len(samples) // hop

        if n_frames == 0:
            return []

        energy = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            end = min(start + hop, len(samples))
            chunk = samples[start:end]
            energy[i] = np.sqrt(np.mean(chunk ** 2))

        # Normalise to [0, 1]
        peak = energy.max()
        if peak > 1e-10:
            energy /= peak

        return energy.tolist()

    # ------------------------------------------------------------------
    # Demucs backend
    # ------------------------------------------------------------------

    def _separate_demucs(self, audio_path: str, output_dir: str) -> AudioStreams:
        """Separate using Meta's Demucs (neural source separation).

        Demucs outputs four stems: drums, bass, other, vocals.
        We map these to our three-stream model:
            speech  <- vocals
            music   <- bass + other (merged)
            effects <- drums
        """
        if _DEMUCS_BACKEND == "torchaudio":
            return self._separate_torchaudio(audio_path, output_dir)
        return self._separate_demucs_cli(audio_path, output_dir)

    def _separate_demucs_cli(self, audio_path: str, output_dir: str) -> AudioStreams:
        """Separate using the demucs Python API (CLI-style)."""
        import shlex

        import demucs.separate

        logger.info(f"Running Demucs separation on: {audio_path}")

        try:
            demucs.separate.main(shlex.split(
                f'-n htdemucs --two-stems vocals -o "{output_dir}" "{audio_path}"'
            ))
        except SystemExit:
            pass  # demucs.separate.main calls sys.exit(0) on success
        except Exception as exc:
            logger.warning(f"Demucs two-stem failed, trying full: {exc}")
            try:
                demucs.separate.main(shlex.split(
                    f'-n htdemucs -o "{output_dir}" "{audio_path}"'
                ))
            except SystemExit:
                pass

        # Locate output files
        stem = Path(audio_path).stem
        model_dir = Path(output_dir) / "htdemucs" / stem

        vocals_path = model_dir / "vocals.wav"
        drums_path = model_dir / "drums.wav"
        bass_path = model_dir / "bass.wav"
        other_path = model_dir / "other.wav"
        no_vocals_path = model_dir / "no_vocals.wav"

        speech_out = str(Path(output_dir) / "speech.wav")
        music_out = str(Path(output_dir) / "music.wav")
        effects_out = str(Path(output_dir) / "effects.wav")

        # Map demucs stems to our streams
        if vocals_path.exists():
            self._copy_or_convert(str(vocals_path), speech_out)
        elif no_vocals_path.exists():
            speech_out = None
        else:
            speech_out = None

        if no_vocals_path.exists():
            # Two-stem mode: no_vocals = everything except speech
            self._copy_or_convert(str(no_vocals_path), music_out)
            effects_out = None
        else:
            # Full separation: merge bass + other for music, drums for effects
            if bass_path.exists() and other_path.exists():
                self._merge_stems(
                    [str(bass_path), str(other_path)], music_out
                )
            elif other_path.exists():
                self._copy_or_convert(str(other_path), music_out)
            else:
                music_out = None

            if drums_path.exists():
                self._copy_or_convert(str(drums_path), effects_out)
            else:
                effects_out = None

        streams = AudioStreams(
            speech=speech_out if speech_out and os.path.isfile(speech_out) else None,
            music=music_out if music_out and os.path.isfile(music_out) else None,
            effects=effects_out if effects_out and os.path.isfile(effects_out) else None,
            original=audio_path,
        )
        logger.info(
            f"Demucs separation complete: "
            f"speech={'yes' if streams.speech else 'no'}, "
            f"music={'yes' if streams.music else 'no'}, "
            f"effects={'yes' if streams.effects else 'no'}"
        )
        return streams

    def _separate_torchaudio(self, audio_path: str, output_dir: str) -> AudioStreams:
        """Separate using torchaudio's built-in HDEMUCS pipeline."""
        import torch
        import torchaudio
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
        from torchaudio.transforms import Fade

        logger.info(f"Running torchaudio HDEMUCS on: {audio_path}")

        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model()

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model.to(device)
        model.eval()

        target_sr = bundle.sample_rate
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Process in segments to manage memory
        sources = self._separate_chunked(
            model, waveform.to(device), target_sr, segment=10.0, overlap=0.1
        )

        # model.sources order: drums, bass, other, vocals
        source_names = model.sources
        source_dict = dict(zip(source_names, sources))

        speech_out = str(Path(output_dir) / "speech.wav")
        music_out = str(Path(output_dir) / "music.wav")
        effects_out = str(Path(output_dir) / "effects.wav")

        # vocals -> speech
        if "vocals" in source_dict:
            torchaudio.save(
                speech_out, source_dict["vocals"].cpu(), target_sr
            )
        else:
            speech_out = None

        # bass + other -> music
        music_parts = []
        for key in ("bass", "other"):
            if key in source_dict:
                music_parts.append(source_dict[key])
        if music_parts:
            music_tensor = sum(music_parts)
            torchaudio.save(music_out, music_tensor.cpu(), target_sr)
        else:
            music_out = None

        # drums -> effects
        if "drums" in source_dict:
            torchaudio.save(
                effects_out, source_dict["drums"].cpu(), target_sr
            )
        else:
            effects_out = None

        streams = AudioStreams(
            speech=speech_out if speech_out and os.path.isfile(speech_out) else None,
            music=music_out if music_out and os.path.isfile(music_out) else None,
            effects=effects_out if effects_out and os.path.isfile(effects_out) else None,
            original=audio_path,
        )
        logger.info(
            f"Torchaudio HDEMUCS separation complete: "
            f"speech={'yes' if streams.speech else 'no'}, "
            f"music={'yes' if streams.music else 'no'}, "
            f"effects={'yes' if streams.effects else 'no'}"
        )
        return streams

    @staticmethod
    def _separate_chunked(
        model,
        mix: "torch.Tensor",
        sr: int,
        segment: float = 10.0,
        overlap: float = 0.1,
    ) -> List["torch.Tensor"]:
        """Run HDEMUCS on overlapping chunks to control memory usage.

        Args:
            model: An HDemucs model instance.
            mix: Input waveform tensor of shape (channels, samples).
            sr: Sample rate.
            segment: Chunk duration in seconds.
            overlap: Overlap fraction between chunks.

        Returns:
            List of separated source tensors, one per source.
        """
        import torch
        from torchaudio.transforms import Fade

        channels, length = mix.shape
        segment_samples = int(segment * sr)
        overlap_samples = int(overlap * segment_samples)
        stride = segment_samples - overlap_samples

        # Fade transforms for smooth overlap-add
        fade = Fade(
            fade_in_len=overlap_samples,
            fade_out_len=overlap_samples,
            fade_shape="linear",
        )

        n_sources = len(model.sources)
        final = torch.zeros(n_sources, channels, length, device=mix.device)

        offset = 0
        while offset < length:
            end = min(offset + segment_samples, length)
            chunk = mix[:, offset:end]

            # Pad if too short
            if chunk.shape[1] < segment_samples:
                pad_len = segment_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            with torch.no_grad():
                out = model(chunk[None])  # (1, n_sources, channels, time)
            out = out[0]  # (n_sources, channels, time)

            # Trim padding
            actual_len = end - offset
            out = out[:, :, :actual_len]

            # Apply fade for overlap-add
            out = fade(out)

            final[:, :, offset:end] += out

            offset += stride

        return [final[i] for i in range(n_sources)]

    # ------------------------------------------------------------------
    # Basic backend (numpy FFT, no external deps)
    # ------------------------------------------------------------------

    def _separate_basic(self, audio_path: str, output_dir: str) -> AudioStreams:
        """Separate using simple frequency-band filtering (numpy FFT).

        This is a rough approximation useful when Demucs is not available.

        Band allocation:
            speech:  300 Hz - 3000 Hz (voice fundamental + formants)
            music:   everything outside the speech band
            effects: high-frequency transients above 5000 Hz (percussive/SFX)
        """
        logger.info(f"Running basic frequency-band separation on: {audio_path}")

        try:
            samples, sr = _read_wav_mono(audio_path)
        except Exception as exc:
            logger.error(f"Cannot read {audio_path}: {exc}")
            return AudioStreams(original=audio_path)

        if len(samples) == 0:
            logger.warning(f"Empty audio file: {audio_path}")
            return AudioStreams(original=audio_path)

        # FFT
        spectrum = np.fft.rfft(samples)
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / sr)

        # Frequency band masks
        speech_mask = (freqs >= 300) & (freqs <= 3000)
        music_mask = ~speech_mask
        effects_mask = freqs >= 5000

        # Extract speech band
        speech_spec = np.zeros_like(spectrum)
        speech_spec[speech_mask] = spectrum[speech_mask]
        speech_samples = np.fft.irfft(speech_spec, n=len(samples))

        # Music: everything outside speech band (but below effects threshold
        # to avoid double-counting the high-frequency content that also goes
        # into effects)
        music_only_mask = music_mask & (freqs < 5000)
        music_spec = np.zeros_like(spectrum)
        music_spec[music_only_mask] = spectrum[music_only_mask]
        music_samples = np.fft.irfft(music_spec, n=len(samples))

        # Effects: high-frequency transients
        effects_spec = np.zeros_like(spectrum)
        effects_spec[effects_mask] = spectrum[effects_mask]
        effects_samples = np.fft.irfft(effects_spec, n=len(samples))

        # Write outputs
        speech_out = str(Path(output_dir) / "speech.wav")
        music_out = str(Path(output_dir) / "music.wav")
        effects_out = str(Path(output_dir) / "effects.wav")

        _write_wav_mono(speech_out, speech_samples, sr)
        _write_wav_mono(music_out, music_samples, sr)
        _write_wav_mono(effects_out, effects_samples, sr)

        streams = AudioStreams(
            speech=speech_out,
            music=music_out,
            effects=effects_out,
            original=audio_path,
        )
        logger.info("Basic frequency-band separation complete")
        return streams

    # ------------------------------------------------------------------
    # Speech / VAD helpers
    # ------------------------------------------------------------------

    def _energy_vad(
        self,
        samples: np.ndarray,
        sr: int,
        frame_ms: int = 30,
        energy_threshold: float = 0.02,
        min_speech_ms: int = 200,
        min_silence_ms: int = 300,
    ) -> List[TimeSegment]:
        """Energy-based voice activity detection.

        Splits audio into short frames, classifies each as speech or
        silence based on RMS energy, then merges adjacent frames into
        contiguous segments.

        Args:
            samples: Mono float samples in [-1, 1].
            sr: Sample rate.
            frame_ms: Analysis frame duration in milliseconds.
            energy_threshold: RMS threshold below which a frame is silence.
            min_speech_ms: Minimum speech segment duration (merge shorter).
            min_silence_ms: Minimum silence gap (bridge shorter gaps).

        Returns:
            List of :class:`TimeSegment` for detected speech regions.
        """
        frame_samples = int(sr * frame_ms / 1000)
        n_frames = len(samples) // frame_samples

        if n_frames == 0:
            return []

        # Compute per-frame RMS energy
        is_speech = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            start = i * frame_samples
            end = start + frame_samples
            rms = np.sqrt(np.mean(samples[start:end] ** 2))
            is_speech[i] = rms > energy_threshold

        # Merge segments: bridge short silence gaps
        min_silence_frames = max(1, int(min_silence_ms / frame_ms))
        merged = is_speech.copy()
        silence_count = 0
        for i in range(n_frames):
            if not merged[i]:
                silence_count += 1
            else:
                if 0 < silence_count < min_silence_frames:
                    # Bridge the gap
                    merged[i - silence_count : i] = True
                silence_count = 0

        # Remove short speech segments
        min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        segments: List[TimeSegment] = []
        in_speech = False
        seg_start = 0

        for i in range(n_frames):
            if merged[i] and not in_speech:
                seg_start = i
                in_speech = True
            elif not merged[i] and in_speech:
                seg_len = i - seg_start
                if seg_len >= min_speech_frames:
                    segments.append(TimeSegment(
                        start=seg_start * frame_ms / 1000.0,
                        end=i * frame_ms / 1000.0,
                        label="speech",
                        confidence=0.8,  # energy-based VAD is approximate
                    ))
                in_speech = False

        # Handle segment that extends to the end
        if in_speech:
            seg_len = n_frames - seg_start
            if seg_len >= min_speech_frames:
                segments.append(TimeSegment(
                    start=seg_start * frame_ms / 1000.0,
                    end=n_frames * frame_ms / 1000.0,
                    label="speech",
                    confidence=0.8,
                ))

        logger.info(f"VAD detected {len(segments)} speech segments")
        return segments

    # ------------------------------------------------------------------
    # Simple onset-based beat detection (fallback when no librosa)
    # ------------------------------------------------------------------

    def _simple_onset_beats(self, audio_path: str) -> List[float]:
        """Detect beats using a simple onset-strength approach (numpy only).

        Computes the spectral flux (frame-to-frame spectral change) and
        picks peaks as beat candidates.  Much less accurate than librosa
        but works with zero extra dependencies.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Sorted list of beat timestamps in seconds.
        """
        try:
            samples, sr = _read_wav_mono(audio_path)
        except Exception as exc:
            logger.warning(f"Cannot read {audio_path} for beat detection: {exc}")
            return []

        if len(samples) < sr:  # less than 1 second
            return []

        # Short-time spectral analysis
        hop = sr // 20  # 50 ms hops -> 20 fps analysis
        frame_len = hop * 2
        n_frames = (len(samples) - frame_len) // hop

        if n_frames < 2:
            return []

        # Compute magnitude spectra
        prev_mag = None
        onset_strength = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop
            frame = samples[start : start + frame_len]
            # Apply Hann window
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_len) / frame_len))
            windowed = frame * window
            mag = np.abs(np.fft.rfft(windowed))

            if prev_mag is not None:
                # Spectral flux: sum of positive differences
                diff = mag - prev_mag
                onset_strength[i] = np.sum(np.maximum(diff, 0))
            prev_mag = mag

        # Normalise
        os_max = onset_strength.max()
        if os_max > 1e-10:
            onset_strength /= os_max

        # Peak picking: find local maxima above a threshold
        threshold = 0.3
        beats: List[float] = []
        min_beat_gap = int(0.25 * 20)  # minimum 250ms between beats

        for i in range(1, n_frames - 1):
            if onset_strength[i] > threshold:
                if (onset_strength[i] > onset_strength[i - 1] and
                        onset_strength[i] >= onset_strength[i + 1]):
                    time = i * hop / sr
                    if not beats or (time - beats[-1]) > (min_beat_gap * hop / sr):
                        beats.append(time)

        logger.info(f"Simple onset detection found {len(beats)} beats")
        return beats

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_or_convert(src: str, dst: str) -> None:
        """Copy a WAV file, or convert to mono 16-bit if needed."""
        import shutil

        if src == dst:
            return
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            logger.warning(f"File copy failed ({src} -> {dst}): {exc}")

    @staticmethod
    def _merge_stems(stem_paths: List[str], output_path: str) -> None:
        """Sum multiple stem WAV files together into one output.

        All stems are read, summed sample-by-sample, and normalised to
        prevent clipping.
        """
        arrays = []
        sr_out = None

        for path in stem_paths:
            if not os.path.isfile(path):
                continue
            try:
                arr, sr = _read_wav_mono(path)
                arrays.append(arr)
                if sr_out is None:
                    sr_out = sr
            except Exception as exc:
                logger.warning(f"Cannot read stem {path}: {exc}")

        if not arrays or sr_out is None:
            return

        # Zero-pad to longest
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a))) for a in arrays]

        merged = sum(padded)  # type: ignore[arg-type]
        # Normalise to prevent clipping
        peak = np.max(np.abs(merged))
        if peak > 1.0:
            merged = merged / peak

        _write_wav_mono(output_path, merged, sr_out)


# ---------------------------------------------------------------------------
# VideoAudioSync
# ---------------------------------------------------------------------------


class VideoAudioSync:
    """Synchronise video generation parameters with audio features.

    Creates per-frame maps of audio energy, beat positions, and speech
    regions that drive motion intensity and transition timing in the
    video pipeline.

    Usage::

        analyzer = AudioStreamAnalyzer()
        sync = VideoAudioSync(analyzer)
        sync_map = sync.create_sync_map("audio.wav", fps=24, total_frames=240)
        adjusted = sync.align_transitions_to_audio([0, 72, 144], sync_map)
    """

    # Motion level thresholds (on normalised [0, 1] energy)
    _HIGH_ENERGY_THRESHOLD = 0.65
    _LOW_ENERGY_THRESHOLD = 0.25

    # How close (in seconds) a boundary must be to an audio feature to snap
    _SNAP_WINDOW_SECONDS = 0.5

    def __init__(self, analyzer: AudioStreamAnalyzer):
        """
        Args:
            analyzer: An :class:`AudioStreamAnalyzer` instance (handles
                source separation and feature extraction).
        """
        self.analyzer = analyzer

    def create_sync_map(
        self,
        audio_path: str,
        fps: int,
        total_frames: int,
    ) -> SyncMap:
        """Create a per-frame synchronisation map.

        Combines energy, beat, and speech data into a single frame-level
        structure that downstream video generators can query.

        Args:
            audio_path: Path to the audio file.
            fps: Target video frame rate.
            total_frames: Total number of video frames.

        Returns:
            A populated :class:`SyncMap`.
        """
        logger.info(
            f"Creating sync map: {audio_path} "
            f"({total_frames} frames @ {fps} fps)"
        )

        # -- Energy per frame -------------------------------------------------
        energy = self.analyzer.get_energy_per_frame(audio_path, fps=fps)
        # Pad or trim to match total_frames
        if len(energy) < total_frames:
            energy.extend([0.0] * (total_frames - len(energy)))
        energy = energy[:total_frames]

        # -- Beat detection ---------------------------------------------------
        beats = self.analyzer.get_music_beats(audio_path)
        beat_set = self._beats_to_frame_set(beats, fps)
        frame_is_beat = [i in beat_set for i in range(total_frames)]

        # -- Speech detection -------------------------------------------------
        speech_segments = self.analyzer.get_speech_segments(audio_path)
        frame_is_speech = self._segments_to_frame_flags(
            speech_segments, fps, total_frames
        )

        # -- Suggested motion level -------------------------------------------
        suggested_motion: List[str] = []
        for i in range(total_frames):
            e = energy[i]
            if e >= self._HIGH_ENERGY_THRESHOLD:
                suggested_motion.append("high")
            elif e <= self._LOW_ENERGY_THRESHOLD:
                suggested_motion.append("low")
            else:
                suggested_motion.append("medium")

        sync_map = SyncMap(
            fps=fps,
            total_frames=total_frames,
            frame_energy=energy,
            frame_is_beat=frame_is_beat,
            frame_is_speech=frame_is_speech,
            suggested_motion=suggested_motion,
            beat_times=beats,
            speech_segments=speech_segments,
        )

        # Summary statistics
        n_beats = sum(frame_is_beat)
        n_speech = sum(frame_is_speech)
        avg_energy = sum(energy) / len(energy) if energy else 0.0
        logger.info(
            f"Sync map ready: {total_frames} frames, "
            f"{n_beats} beat frames, {n_speech} speech frames, "
            f"avg energy={avg_energy:.3f}"
        )

        return sync_map

    def align_transitions_to_audio(
        self,
        shot_boundaries: List[int],
        sync_map: SyncMap,
    ) -> List[int]:
        """Adjust shot cut points to align with audio features.

        Each cut point is nudged to the nearest music beat or speech
        pause within a snap window.  Priority order:

        1. Snap to a music beat (cuts on beats feel rhythmic).
        2. Snap to a speech pause boundary (avoid cutting mid-sentence).
        3. Keep original position if no good candidate is nearby.

        Args:
            shot_boundaries: List of frame indices where shots begin.
                Frame 0 is always kept. Typically this would be something
                like ``[0, 72, 144, 216]``.
            sync_map: A :class:`SyncMap` for the same audio.

        Returns:
            Adjusted list of shot boundary frame indices.
        """
        if not shot_boundaries:
            return []

        snap_frames = int(self._SNAP_WINDOW_SECONDS * sync_map.fps)
        adjusted: List[int] = []

        for idx, boundary in enumerate(shot_boundaries):
            # Always keep the very first boundary at its original position
            if idx == 0:
                adjusted.append(boundary)
                continue

            best = boundary
            best_score = 0.0

            # Search within snap window
            lo = max(0, boundary - snap_frames)
            hi = min(sync_map.total_frames - 1, boundary + snap_frames)

            for candidate in range(lo, hi + 1):
                score = 0.0
                dist = abs(candidate - boundary)
                proximity = 1.0 - (dist / (snap_frames + 1))

                # Prefer beat frames (weight 2.0)
                if sync_map.is_beat_at(candidate):
                    score += 2.0 * proximity

                # Prefer speech pause boundaries: frame is non-speech but
                # the previous frame was speech (end of utterance)
                if candidate > 0:
                    was_speech = sync_map.is_speech_at(candidate - 1)
                    is_now_silence = not sync_map.is_speech_at(candidate)
                    if was_speech and is_now_silence:
                        score += 1.5 * proximity

                # Slightly prefer lower energy moments (less jarring cuts)
                e = sync_map.energy_at(candidate)
                score += (1.0 - e) * 0.3 * proximity

                if score > best_score:
                    best_score = score
                    best = candidate

            # Ensure monotonically increasing boundaries
            if adjusted and best <= adjusted[-1]:
                best = adjusted[-1] + 1

            adjusted.append(min(best, sync_map.total_frames - 1))

            if best != boundary:
                logger.debug(
                    f"Shot boundary {idx}: frame {boundary} -> {best} "
                    f"(score={best_score:.2f})"
                )

        return adjusted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _beats_to_frame_set(beats: List[float], fps: int) -> set:
        """Convert beat timestamps to a set of frame indices."""
        return {round(t * fps) for t in beats}

    @staticmethod
    def _segments_to_frame_flags(
        segments: List[TimeSegment],
        fps: int,
        total_frames: int,
    ) -> List[bool]:
        """Convert time segments to per-frame boolean flags."""
        flags = [False] * total_frames
        for seg in segments:
            start_frame = max(0, int(seg.start * fps))
            end_frame = min(total_frames, int(seg.end * fps))
            for i in range(start_frame, end_frame):
                flags[i] = True
        return flags


# ---------------------------------------------------------------------------
# CLI -- quick test / debug entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Separate audio into speech/music/SFX streams and "
            "generate a per-frame sync map."
        )
    )
    parser.add_argument(
        "audio", help="Path to audio file (WAV recommended)"
    )
    parser.add_argument(
        "--backend",
        choices=["demucs", "basic", "auto"],
        default="auto",
        help="Separation backend (default: auto)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Video frame rate for sync map (default: 24)",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=0,
        help="Total video frames (default: derived from audio duration)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for separated audio outputs",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # 1. Source separation
    analyzer = AudioStreamAnalyzer(backend=args.backend)
    streams = analyzer.separate(args.audio, output_dir=args.output_dir)

    print(f"\n{'=' * 60}")
    print("Source Separation Results")
    print(f"{'=' * 60}")
    print(f"  Original:  {streams.original}")
    print(f"  Speech:    {streams.speech or '(none)'}")
    print(f"  Music:     {streams.music or '(none)'}")
    print(f"  Effects:   {streams.effects or '(none)'}")

    # 2. Speech segments
    segments = analyzer.get_speech_segments(args.audio)
    print(f"\n{'=' * 60}")
    print(f"Speech Segments ({len(segments)})")
    print(f"{'=' * 60}")
    for seg in segments[:20]:
        print(
            f"  {seg.start:7.2f}s - {seg.end:7.2f}s  "
            f"({seg.end - seg.start:.2f}s)  [{seg.label}]"
        )
    if len(segments) > 20:
        print(f"  ... ({len(segments) - 20} more)")

    # 3. Beat detection
    beats = analyzer.get_music_beats(args.audio)
    print(f"\n{'=' * 60}")
    print(f"Music Beats ({len(beats)})")
    print(f"{'=' * 60}")
    for i, t in enumerate(beats[:20]):
        print(f"  Beat {i + 1:3d}: {t:7.3f}s")
    if len(beats) > 20:
        print(f"  ... ({len(beats) - 20} more)")

    # 4. Sync map
    if args.total_frames <= 0:
        # Derive from audio duration
        try:
            _, sr = _read_wav_mono(args.audio)
            with wave.open(args.audio, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
            args.total_frames = int(duration * args.fps)
        except Exception:
            args.total_frames = 240  # fallback: 10 seconds at 24fps

    sync = VideoAudioSync(analyzer)
    sync_map = sync.create_sync_map(
        args.audio, fps=args.fps, total_frames=args.total_frames
    )

    print(f"\n{'=' * 60}")
    print(f"Sync Map ({sync_map.total_frames} frames @ {sync_map.fps} fps)")
    print(f"{'=' * 60}")

    # Show first 48 frames (2 seconds at 24fps)
    show_n = min(48, sync_map.total_frames)
    print(f"\n{'Frame':>6}  {'Energy':>7}  {'Beat':>5}  {'Speech':>7}  {'Motion':>7}")
    print(f"{'-' * 6}  {'-' * 7}  {'-' * 5}  {'-' * 7}  {'-' * 7}")
    for i in range(show_n):
        print(
            f"{i:6d}  {sync_map.frame_energy[i]:7.3f}  "
            f"{'  *  ' if sync_map.frame_is_beat[i] else '  .  '}  "
            f"{'  yes  ' if sync_map.frame_is_speech[i] else '  .    '}  "
            f"{sync_map.suggested_motion[i]:>7}"
        )
    if sync_map.total_frames > show_n:
        print(f"  ... ({sync_map.total_frames - show_n} more frames)")

    # 5. Transition alignment demo
    n_shots = 4
    shot_dur = sync_map.total_frames // n_shots
    boundaries = [i * shot_dur for i in range(n_shots)]

    adjusted = sync.align_transitions_to_audio(boundaries, sync_map)

    print(f"\n{'=' * 60}")
    print(f"Transition Alignment ({n_shots} shots)")
    print(f"{'=' * 60}")
    print(f"{'Shot':>5}  {'Original':>9}  {'Adjusted':>9}  {'Delta':>7}")
    print(f"{'-' * 5}  {'-' * 9}  {'-' * 9}  {'-' * 7}")
    for i, (orig, adj) in enumerate(zip(boundaries, adjusted)):
        delta = adj - orig
        print(f"{i + 1:5d}  {orig:9d}  {adj:9d}  {delta:+7d}")

    print()
