"""
Phoneme-Level Lip Sync — align generated lip movements to phoneme timelines.

Layers on top of the existing lip_sync.py to provide phoneme-level accuracy:
  1. Extract phoneme timeline from audio (via whisper or forced alignment)
  2. Map phonemes to visemes (mouth shapes)
  3. Score existing lip sync output against phoneme timeline
  4. Re-align if needed (stretch/compress mouth animation frames)

Supported languages: zh (Chinese), en (English), ja (Japanese), ko (Korean).

Requires:
    pip install openai-whisper  (for phoneme extraction)
    OR pip install whisperx     (for word-level alignment)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ["PhonemeAligner", "PhonemeTimeline", "Viseme"]

# ---------------------------------------------------------------------------
# Phoneme → Viseme mapping
# ---------------------------------------------------------------------------

# Standard viseme categories (based on Microsoft/MPEG-4 viseme set)
VISEME_NAMES = [
    "silence",    # 0: Closed mouth (silence, pauses)
    "bilabial",   # 1: P, B, M — lips pressed together
    "labiodental", # 2: F, V — lower lip + upper teeth
    "dental",     # 3: TH — tongue between teeth
    "alveolar",   # 4: T, D, N, L — tongue on alveolar ridge
    "postalveolar", # 5: SH, CH, ZH, J — tongue behind ridge
    "velar",      # 6: K, G, NG — back of tongue raised
    "open_front", # 7: AH, AA — wide open mouth
    "open_mid",   # 8: EH, AE — medium open mouth
    "close_front", # 9: IY, IH — lips spread, slightly open
    "close_back", # 10: UW, UH — lips rounded, small opening
    "mid_central", # 11: ER, AX — neutral mouth position
    "diphthong",  # 12: OW, AW, AY — transitional
    "rounded",    # 13: OO, W — strongly rounded lips
]

# IPA-like phoneme → viseme index mapping (covers en, zh, ja, ko)
_PHONEME_TO_VISEME: Dict[str, int] = {
    # Silence
    "SIL": 0, "SP": 0, "": 0, " ": 0,
    # Bilabial (P, B, M)
    "P": 1, "B": 1, "M": 1, "p": 1, "b": 1, "m": 1,
    # Labiodental (F, V)
    "F": 2, "V": 2, "f": 2, "v": 2,
    # Dental (TH, DH)
    "TH": 3, "DH": 3,
    # Alveolar (T, D, N, L, S, Z)
    "T": 4, "D": 4, "N": 4, "L": 4, "S": 4, "Z": 4,
    "t": 4, "d": 4, "n": 4, "l": 4, "s": 4, "z": 4,
    # Postalveolar (SH, CH, ZH, JH)
    "SH": 5, "CH": 5, "ZH": 5, "JH": 5,
    "sh": 5, "ch": 5, "zh": 5, "j": 5, "q": 5, "x": 5,
    # Velar (K, G, NG, H)
    "K": 6, "G": 6, "NG": 6, "HH": 6,
    "k": 6, "g": 6, "h": 6, "ng": 6,
    # Open front (AH, AA)
    "AH": 7, "AA": 7, "AO": 7, "a": 7,
    # Open mid (EH, AE)
    "EH": 8, "AE": 8, "e": 8,
    # Close front (IY, IH)
    "IY": 9, "IH": 9, "i": 9, "ii": 9,
    # Close back (UW, UH)
    "UW": 10, "UH": 10, "u": 10, "uu": 10,
    # Mid central (ER, AX)
    "ER": 11, "AX": 11,
    # Diphthong
    "OW": 12, "AW": 12, "AY": 12, "EY": 12, "OY": 12,
    # Rounded
    "OO": 13, "W": 13, "R": 13, "w": 13, "r": 13, "o": 13,
}


def phoneme_to_viseme(phoneme: str) -> int:
    """Map a phoneme string to a viseme index (0-13)."""
    # Strip stress markers and clean
    clean = phoneme.strip().rstrip("0123456789")
    return _PHONEME_TO_VISEME.get(clean, _PHONEME_TO_VISEME.get(clean.upper(), 0))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Viseme:
    """A single viseme (mouth shape) with timing."""
    index: int = 0          # Viseme index (0-13)
    name: str = "silence"   # Viseme name
    start_time: float = 0.0  # Start time in seconds
    end_time: float = 0.0    # End time in seconds
    phoneme: str = ""        # Source phoneme

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def mouth_openness(self) -> float:
        """Approximate mouth openness for this viseme (0=closed, 1=wide open)."""
        openness_map = {
            0: 0.0,   # silence
            1: 0.05,  # bilabial (lips pressed)
            2: 0.15,  # labiodental
            3: 0.2,   # dental
            4: 0.25,  # alveolar
            5: 0.3,   # postalveolar
            6: 0.35,  # velar
            7: 1.0,   # open front (AH)
            8: 0.7,   # open mid
            9: 0.3,   # close front
            10: 0.25, # close back
            11: 0.4,  # mid central
            12: 0.6,  # diphthong
            13: 0.2,  # rounded
        }
        return openness_map.get(self.index, 0.3)


@dataclass
class PhonemeTimeline:
    """Timeline of phonemes/visemes extracted from audio."""
    visemes: List[Viseme] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0

    def viseme_at(self, time: float) -> Viseme:
        """Get the active viseme at a given time."""
        for v in self.visemes:
            if v.start_time <= time < v.end_time:
                return v
        return Viseme()  # silence

    def openness_curve(self, fps: int, num_frames: int) -> np.ndarray:
        """Generate per-frame mouth openness curve.

        Returns:
            numpy array of shape (num_frames,) with values in [0, 1].
        """
        curve = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            t = i / fps
            v = self.viseme_at(t)
            curve[i] = v.mouth_openness
        return curve


# ---------------------------------------------------------------------------
# Phoneme extraction
# ---------------------------------------------------------------------------

def _extract_phonemes_whisper(
    audio_path: str,
    language: str = "auto",
) -> PhonemeTimeline:
    """Extract word-level timestamps using Whisper, then map to phonemes.

    This is a simplified approach: Whisper gives word-level timing, and we
    estimate phoneme boundaries within each word proportionally.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper required for phoneme extraction.\n"
            "Install with: pip install openai-whisper"
        )

    logger.info(f"Extracting phonemes from {audio_path} via Whisper...")
    model = whisper.load_model("base")

    decode_opts = {}
    if language != "auto":
        decode_opts["language"] = language

    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        **decode_opts,
    )

    detected_lang = result.get("language", language)
    timeline = PhonemeTimeline(language=detected_lang)

    # Extract word-level timing from segments
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info.get("word", "").strip()
            start = word_info.get("start", 0.0)
            end = word_info.get("end", 0.0)
            if not word or end <= start:
                continue

            # Convert word to approximate phoneme sequence
            phonemes = _word_to_phonemes(word, detected_lang)
            if not phonemes:
                continue

            # Distribute timing evenly across phonemes
            ph_duration = (end - start) / len(phonemes)
            for j, ph in enumerate(phonemes):
                ph_start = start + j * ph_duration
                ph_end = ph_start + ph_duration
                vis_idx = phoneme_to_viseme(ph)
                timeline.visemes.append(Viseme(
                    index=vis_idx,
                    name=VISEME_NAMES[vis_idx],
                    start_time=ph_start,
                    end_time=ph_end,
                    phoneme=ph,
                ))

    if timeline.visemes:
        timeline.duration = timeline.visemes[-1].end_time
    logger.info(
        f"Phoneme extraction: {len(timeline.visemes)} visemes, "
        f"{timeline.duration:.2f}s, lang={detected_lang}"
    )
    return timeline


def _word_to_phonemes(word: str, language: str) -> List[str]:
    """Approximate phoneme decomposition of a word.

    For CJK languages, each character maps to ~1-2 phonemes.
    For English, uses a simple heuristic (not a full G2P model).
    """
    phonemes = []

    if language in ("zh", "chinese"):
        # Each Chinese character → initial + final
        for char in word:
            if '\u4e00' <= char <= '\u9fff':
                phonemes.extend(["sh", "a"])  # Simplified placeholder
            elif char.isascii() and char.isalpha():
                phonemes.append(char.lower())
    elif language in ("ja", "japanese"):
        # Each kana → 1 phoneme (simplified)
        for char in word:
            if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':
                phonemes.append("a")  # Simplified
            elif char.isascii():
                phonemes.append(char.lower())
    elif language in ("ko", "korean"):
        for char in word:
            if '\uac00' <= char <= '\ud7a3':
                phonemes.extend(["k", "a"])  # Simplified
            elif char.isascii():
                phonemes.append(char.lower())
    else:
        # English: simple character-level heuristic
        vowels = set("aeiouAEIOU")
        for char in word:
            if char.isalpha():
                if char.lower() in vowels:
                    phonemes.append("AH" if char.lower() in "ao" else "EH")
                else:
                    phonemes.append(char.upper())

    return phonemes


# ---------------------------------------------------------------------------
# Phoneme aligner
# ---------------------------------------------------------------------------

class PhonemeAligner:
    """Align lip sync output against a phoneme timeline for accuracy.

    Workflow:
      1. Extract phoneme timeline from audio
      2. Generate mouth openness curve from timeline
      3. Score existing lip sync quality against the curve
      4. Optionally re-align frames to match phoneme timing
    """

    def __init__(self, extraction_backend: str = "whisper"):
        """
        Args:
            extraction_backend: "whisper" (default) — uses Whisper for phoneme extraction.
        """
        self.extraction_backend = extraction_backend

    def extract_timeline(
        self,
        audio_path: str,
        language: str = "auto",
    ) -> PhonemeTimeline:
        """Extract phoneme timeline from audio.

        Returns:
            PhonemeTimeline with viseme sequence and timing.
        """
        if self.extraction_backend == "whisper":
            return _extract_phonemes_whisper(audio_path, language)
        else:
            raise ValueError(f"Unknown extraction backend: {self.extraction_backend}")

    def score_sync_quality(
        self,
        frames: List[Image.Image],
        timeline: PhonemeTimeline,
        fps: int = 24,
    ) -> float:
        """Score how well lip movements in frames match the phoneme timeline.

        Compares per-frame mouth region brightness variance (proxy for openness)
        against the expected viseme openness curve.

        Args:
            frames: Lip-synced video frames.
            timeline: Expected phoneme timeline.
            fps: Video frame rate.

        Returns:
            Sync score in [0, 1]. Higher = better match.
        """
        if not frames or not timeline.visemes:
            return 0.0

        expected = timeline.openness_curve(fps, len(frames))
        observed = self._estimate_mouth_openness(frames)

        # Normalize both to [0, 1]
        if observed.max() > 0:
            observed = observed / observed.max()

        # Correlation between expected and observed curves
        if np.std(expected) < 1e-6 or np.std(observed) < 1e-6:
            return 0.5

        correlation = float(np.corrcoef(expected, observed)[0, 1])
        # Map [-1, 1] → [0, 1]
        score = (correlation + 1.0) / 2.0

        logger.info(f"Phoneme sync score: {score:.3f} (correlation={correlation:.3f})")
        return score

    def align_audio(
        self,
        audio_tensor,
        sample_rate: int,
        num_frames: int,
        fps: int,
    ):
        """Align audio tensor to match video frame count (basic trim/pad).

        Returns the adjusted audio tensor.
        """
        video_duration = num_frames / fps
        target_samples = int(video_duration * sample_rate)

        if audio_tensor.shape[-1] > target_samples:
            return audio_tensor[..., :target_samples]
        elif audio_tensor.shape[-1] < target_samples:
            import torch
            pad_size = target_samples - audio_tensor.shape[-1]
            return torch.nn.functional.pad(audio_tensor, (0, pad_size))
        return audio_tensor

    def realign_frames(
        self,
        frames: List[Image.Image],
        timeline: PhonemeTimeline,
        fps: int = 24,
        strength: float = 0.5,
    ) -> List[Image.Image]:
        """Re-align frame timing to better match phoneme boundaries.

        Uses temporal warping: frames during speech get condensed around
        phoneme transitions, frames during silence get stretched.

        This is a subtle temporal adjustment — not frame regeneration.

        Args:
            frames: Original frames.
            timeline: Phoneme timeline from audio.
            fps: Video frame rate.
            strength: Adjustment strength (0 = no change, 1 = full realignment).

        Returns:
            Re-aligned frames (same count, potentially resampled).
        """
        if not frames or not timeline.visemes or strength <= 0:
            return list(frames)

        n = len(frames)
        expected = timeline.openness_curve(fps, n)

        # Build a time-warp map: frames where phonemes change get more weight
        warp_weights = np.ones(n, dtype=np.float32)
        for i in range(1, n):
            delta = abs(expected[i] - expected[i - 1])
            # Phoneme transitions get higher weight (more important to show)
            warp_weights[i] = 1.0 + delta * 3.0 * strength

        # Normalize warp weights to maintain total frame count
        warp_weights = warp_weights / warp_weights.mean()

        # Build source index mapping
        cumulative = np.cumsum(warp_weights)
        cumulative = cumulative / cumulative[-1] * (n - 1)

        # Resample frames using the warp map
        result = []
        for i in range(n):
            src_idx = min(n - 1, max(0, int(round(cumulative[i]))))
            result.append(frames[src_idx])

        return result

    @staticmethod
    def _estimate_mouth_openness(frames: List[Image.Image]) -> np.ndarray:
        """Estimate per-frame mouth openness from video frames.

        Uses the variance of the lower-center region of each frame as a
        proxy for mouth movement intensity.
        """
        openness = np.zeros(len(frames), dtype=np.float32)

        for i, frame in enumerate(frames):
            arr = np.array(frame.convert("L"), dtype=np.float32)
            h, w = arr.shape

            # Lower center region (approximate mouth area)
            y1 = int(h * 0.55)
            y2 = int(h * 0.85)
            x1 = int(w * 0.3)
            x2 = int(w * 0.7)

            region = arr[y1:y2, x1:x2]
            if region.size > 0:
                openness[i] = np.std(region) / 128.0  # Normalize

        return openness
