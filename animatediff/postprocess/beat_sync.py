"""
Beat Sync — detect beats in background music and align shot cuts to musical beats.

Supports:
- Beat and downbeat detection via librosa
- Per-frame energy curves for motion intensity matching
- Multiple alignment modes: snap_to_beat, snap_to_downbeat, energy_match
- Shot duration suggestions that respect musical structure

Requires:
    pip install librosa

Usage:
    analyzer = BeatAnalyzer("bgm.mp3")
    aligner = ShotBeatAligner(analyzer)
    adjusted_shots = aligner.align_shots(shots, mode="snap_to_beat")
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import guard — librosa is an optional heavyweight dependency
try:
    import librosa
    import numpy as np

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def _require_librosa() -> None:
    """Raise a clear error if librosa is not installed."""
    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for beat sync analysis but is not installed.\n"
            "Install it with:\n"
            "    pip install librosa\n"
            "Or for the full audio processing stack:\n"
            "    pip install librosa soundfile"
        )


@dataclass
class BeatInfo:
    """Summary of beat analysis results."""
    tempo: float  # BPM
    beat_times: list[float] = field(default_factory=list)  # seconds
    downbeat_times: list[float] = field(default_factory=list)  # seconds
    duration: float = 0.0  # total audio duration in seconds


class BeatAnalyzer:
    """Analyze music for beats, downbeats, and energy.

    Loads an audio file and computes rhythmic features using librosa.
    All timestamps are in seconds. Thread-safe after construction.
    """

    def __init__(self, audio_path: str, sr: int = 22050):
        """Load audio and compute beat features.

        Args:
            audio_path: Path to the audio file (WAV, MP3, FLAC, etc.).
            sr: Sample rate for analysis. 22050 is the librosa default and
                provides a good balance between accuracy and speed.
        """
        _require_librosa()

        self.audio_path = audio_path
        self.sr = sr

        logger.info(f"Loading audio for beat analysis: {audio_path}")
        self._y, self._sr = librosa.load(audio_path, sr=sr)
        self.duration = float(len(self._y) / self._sr)
        logger.info(
            f"Audio loaded: {self.duration:.1f}s at {self._sr}Hz "
            f"({len(self._y)} samples)"
        )

        # Pre-compute beat features
        self._tempo, self._beat_frames = librosa.beat.beat_track(
            y=self._y, sr=self._sr
        )
        self._beat_times = librosa.frames_to_time(
            self._beat_frames, sr=self._sr
        ).tolist()

        # Compute downbeats (bar-level) — use beat_track with tightness for
        # stability, then estimate bars assuming 4/4 time
        self._downbeat_times = self._estimate_downbeats()

        logger.info(
            f"Beat analysis complete: {self._tempo:.1f} BPM, "
            f"{len(self._beat_times)} beats, "
            f"{len(self._downbeat_times)} downbeats"
        )

    def _estimate_downbeats(self) -> list[float]:
        """Estimate downbeat positions assuming 4/4 time signature.

        Librosa's beat tracker returns all beats. In 4/4 time, every 4th beat
        is a downbeat (bar start). We use onset strength to pick the strongest
        beat in each group of 4 as the actual downbeat.
        """
        if len(self._beat_times) < 4:
            return list(self._beat_times)

        # Compute onset envelope for strength comparison
        onset_env = librosa.onset.onset_strength(y=self._y, sr=self._sr)
        onset_times = librosa.times_like(onset_env, sr=self._sr)

        def _onset_strength_at(t: float) -> float:
            """Get onset strength at a given time."""
            idx = int(np.argmin(np.abs(onset_times - t)))
            return float(onset_env[idx])

        downbeats = []
        for i in range(0, len(self._beat_times), 4):
            group = self._beat_times[i : i + 4]
            # Pick the beat with highest onset strength as the downbeat
            strongest = max(group, key=_onset_strength_at)
            downbeats.append(strongest)

        return downbeats

    def get_beats(self) -> list[float]:
        """Return beat timestamps in seconds.

        Returns:
            Sorted list of beat positions (seconds from audio start).
        """
        return list(self._beat_times)

    def get_downbeats(self) -> list[float]:
        """Return downbeat (bar start) timestamps in seconds.

        Returns:
            Sorted list of downbeat positions (seconds from audio start).
        """
        return list(self._downbeat_times)

    def get_energy_curve(self, fps: int = 24) -> list[float]:
        """Return per-frame energy values for motion intensity matching.

        Computes RMS energy at video frame rate, normalized to [0, 1].
        Higher values indicate louder, more energetic musical passages
        which can be used to drive faster cuts or more dynamic motion.

        Args:
            fps: Video frame rate to sample energy at.

        Returns:
            List of float energy values, one per video frame, in [0, 1].
        """
        _require_librosa()

        # Compute RMS energy with hop length matching video fps
        hop_length = max(1, self._sr // fps)
        rms = librosa.feature.rms(y=self._y, hop_length=hop_length)[0]

        # Normalize to [0, 1]
        rms_max = float(np.max(rms)) if len(rms) > 0 else 1.0
        if rms_max > 0:
            normalized = (rms / rms_max).tolist()
        else:
            normalized = [0.0] * len(rms)

        logger.debug(
            f"Energy curve: {len(normalized)} frames at {fps}fps, "
            f"peak energy at frame {np.argmax(rms) if len(rms) > 0 else 0}"
        )
        return normalized

    def get_tempo(self) -> float:
        """Return detected BPM.

        Returns:
            Tempo in beats per minute (float).
        """
        return float(self._tempo)

    def get_info(self) -> BeatInfo:
        """Return a summary of beat analysis results.

        Returns:
            BeatInfo dataclass with tempo, beats, downbeats, and duration.
        """
        return BeatInfo(
            tempo=self.get_tempo(),
            beat_times=self.get_beats(),
            downbeat_times=self.get_downbeats(),
            duration=self.duration,
        )


class ShotBeatAligner:
    """Align shot cut points to musical beats.

    Takes a BeatAnalyzer and adjusts shot durations so that cuts between
    shots land on rhythmically meaningful moments in the music. This creates
    a more professional, music-video-like feel.

    Usage:
        analyzer = BeatAnalyzer("bgm.mp3")
        aligner = ShotBeatAligner(analyzer)

        shots = [
            {"duration_seconds": 3.0, "num_frames": 72, "fps": 24},
            {"duration_seconds": 4.0, "num_frames": 96, "fps": 24},
        ]
        adjusted = aligner.align_shots(shots, mode="snap_to_beat")
    """

    # Minimum shot duration (seconds) to prevent degenerate ultra-short cuts
    MIN_SHOT_DURATION = 0.5
    # Maximum allowed adjustment ratio (prevent shots from being stretched
    # or compressed more than 2x)
    MAX_ADJUSTMENT_RATIO = 2.0

    def __init__(self, analyzer: BeatAnalyzer):
        """
        Args:
            analyzer: A BeatAnalyzer instance loaded with the target music.
        """
        self.analyzer = analyzer
        self._beats = analyzer.get_beats()
        self._downbeats = analyzer.get_downbeats()
        self._tempo = analyzer.get_tempo()

    def _find_nearest(self, t: float, candidates: list[float]) -> float:
        """Find the timestamp in candidates closest to t.

        Args:
            t: Target time in seconds.
            candidates: Sorted list of candidate timestamps.

        Returns:
            The candidate closest to t. Returns t if candidates is empty.
        """
        if not candidates:
            return t

        _require_librosa()
        arr = np.array(candidates)
        idx = int(np.argmin(np.abs(arr - t)))
        return float(candidates[idx])

    def align_shots(
        self,
        shots: list[dict],
        mode: str = "snap_to_beat",
        start_offset: float = 0.0,
    ) -> list[dict]:
        """Adjust shot durations so cuts land on beats.

        Each shot dict must contain: duration_seconds, num_frames, fps.
        Returns new shot dicts with adjusted durations. The original dicts
        are not modified.

        Modes:
        - "snap_to_beat": snap each shot end to the nearest beat.
        - "snap_to_downbeat": snap to the nearest downbeat (for stronger,
          more deliberate cuts).
        - "energy_match": adjust shot duration based on musical energy.
          High energy passages get shorter shots (faster cutting), low
          energy passages get longer shots (slower, more contemplative).

        Args:
            shots: List of shot dicts with duration_seconds, num_frames, fps.
            mode: Alignment mode (see above).
            start_offset: Time offset in seconds from audio start where the
                first shot begins. Useful when video does not start at the
                beginning of the music track.

        Returns:
            New list of shot dicts with adjusted duration_seconds and
            num_frames. Original shot list is not mutated.
        """
        if not shots:
            return []

        valid_modes = ("snap_to_beat", "snap_to_downbeat", "energy_match")
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown alignment mode: '{mode}'. "
                f"Supported modes: {valid_modes}"
            )

        logger.info(
            f"Aligning {len(shots)} shots to music "
            f"(mode={mode}, tempo={self._tempo:.1f} BPM)"
        )

        if mode == "snap_to_beat":
            return self._snap_to_targets(shots, self._beats, start_offset)
        elif mode == "snap_to_downbeat":
            return self._snap_to_targets(shots, self._downbeats, start_offset)
        elif mode == "energy_match":
            return self._energy_match(shots, start_offset)

        return shots  # unreachable, but satisfies type checker

    def _snap_to_targets(
        self,
        shots: list[dict],
        targets: list[float],
        start_offset: float,
    ) -> list[dict]:
        """Snap shot cut points to the nearest target timestamps.

        Walks through shots sequentially, accumulating time. At each cut
        point (end of a shot), finds the nearest beat/downbeat and adjusts
        the shot duration to land on it.

        Args:
            shots: List of shot dicts.
            targets: Sorted list of target timestamps (beats or downbeats).
            start_offset: Time offset for the first shot.

        Returns:
            Adjusted shot dicts.
        """
        if not targets:
            logger.warning("No beat targets found, returning shots unchanged")
            return [dict(s) for s in shots]

        adjusted = []
        current_time = start_offset

        for i, shot in enumerate(shots):
            fps = shot.get("fps", 24)
            original_duration = shot["duration_seconds"]
            desired_end = current_time + original_duration

            # Find nearest target for the cut point
            snapped_end = self._find_nearest(desired_end, targets)

            # Enforce minimum duration and max adjustment ratio
            new_duration = snapped_end - current_time
            new_duration = self._clamp_duration(
                new_duration, original_duration
            )

            # Recompute snapped end after clamping
            snapped_end = current_time + new_duration

            new_num_frames = max(1, round(new_duration * fps))

            adjusted_shot = dict(shot)
            adjusted_shot["duration_seconds"] = round(new_duration, 4)
            adjusted_shot["num_frames"] = new_num_frames
            adjusted.append(adjusted_shot)

            delta = new_duration - original_duration
            if abs(delta) > 0.01:
                logger.debug(
                    f"Shot {i}: {original_duration:.2f}s -> {new_duration:.2f}s "
                    f"(delta={delta:+.2f}s, snapped to {snapped_end:.3f}s)"
                )

            current_time = snapped_end

        total_original = sum(s["duration_seconds"] for s in shots)
        total_adjusted = sum(s["duration_seconds"] for s in adjusted)
        logger.info(
            f"Beat alignment complete: {total_original:.2f}s -> "
            f"{total_adjusted:.2f}s (delta={total_adjusted - total_original:+.2f}s)"
        )

        return adjusted

    def _energy_match(
        self,
        shots: list[dict],
        start_offset: float,
    ) -> list[dict]:
        """Adjust shot durations based on musical energy.

        High-energy sections get shorter shots (faster cutting rhythm),
        while low-energy sections get longer shots (slower, more contemplative).
        Cut points are still snapped to the nearest beat for rhythmic coherence.

        Args:
            shots: List of shot dicts.
            start_offset: Time offset for the first shot.

        Returns:
            Adjusted shot dicts.
        """
        _require_librosa()

        # Get energy at a reasonable analysis rate
        analysis_fps = 10  # 10 samples/sec is enough for energy envelope
        energy = self.analyzer.get_energy_curve(fps=analysis_fps)

        if not energy:
            logger.warning("Empty energy curve, falling back to snap_to_beat")
            return self._snap_to_targets(shots, self._beats, start_offset)

        adjusted = []
        current_time = start_offset

        for i, shot in enumerate(shots):
            fps = shot.get("fps", 24)
            original_duration = shot["duration_seconds"]

            # Sample average energy over this shot's time span
            start_frame = int(current_time * analysis_fps)
            end_frame = int((current_time + original_duration) * analysis_fps)
            start_frame = max(0, min(start_frame, len(energy) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(energy)))

            avg_energy = float(np.mean(energy[start_frame:end_frame]))

            # Map energy to duration scaling:
            #   energy=0.0 -> scale=1.3 (slower cuts for quiet parts)
            #   energy=0.5 -> scale=1.0 (normal)
            #   energy=1.0 -> scale=0.7 (faster cuts for loud parts)
            scale = 1.3 - 0.6 * avg_energy
            scaled_duration = original_duration * scale

            # Snap the end point to nearest beat
            desired_end = current_time + scaled_duration
            snapped_end = self._find_nearest(desired_end, self._beats)
            new_duration = snapped_end - current_time

            # Clamp to sane range
            new_duration = self._clamp_duration(new_duration, original_duration)
            snapped_end = current_time + new_duration

            new_num_frames = max(1, round(new_duration * fps))

            adjusted_shot = dict(shot)
            adjusted_shot["duration_seconds"] = round(new_duration, 4)
            adjusted_shot["num_frames"] = new_num_frames
            adjusted.append(adjusted_shot)

            logger.debug(
                f"Shot {i}: energy={avg_energy:.2f}, scale={scale:.2f}, "
                f"{original_duration:.2f}s -> {new_duration:.2f}s"
            )

            current_time = snapped_end

        return adjusted

    def _clamp_duration(
        self, new_duration: float, original_duration: float
    ) -> float:
        """Clamp a duration to prevent degenerate values.

        Ensures the adjusted duration is:
        - At least MIN_SHOT_DURATION seconds
        - No more than MAX_ADJUSTMENT_RATIO times the original
        - No less than 1/MAX_ADJUSTMENT_RATIO times the original

        Args:
            new_duration: The proposed new duration.
            original_duration: The original shot duration.

        Returns:
            Clamped duration.
        """
        min_dur = max(
            self.MIN_SHOT_DURATION,
            original_duration / self.MAX_ADJUSTMENT_RATIO,
        )
        max_dur = original_duration * self.MAX_ADJUSTMENT_RATIO
        return max(min_dur, min(new_duration, max_dur))

    def suggest_shot_durations(
        self,
        total_duration: float,
        num_shots: int,
        mode: str = "even",
        start_offset: float = 0.0,
    ) -> list[float]:
        """Suggest shot durations that align to beats.

        Given a total video duration and number of shots, proposes duration
        values that land on musically significant moments.

        Modes:
        - "even": distribute evenly, then snap cut points to nearest beats.
        - "musical": follow musical structure. Verse sections get longer shots,
          chorus/high-energy sections get shorter shots.
        - "accelerating": start with longer shots and progressively shorten
          toward the end, creating an accelerating rhythm that builds tension.

        Args:
            total_duration: Total video duration in seconds.
            num_shots: Number of shots to distribute.
            mode: Distribution mode (see above).
            start_offset: Time offset from audio start (seconds).

        Returns:
            List of suggested durations in seconds, one per shot.
        """
        if num_shots <= 0:
            return []
        if num_shots == 1:
            return [total_duration]

        valid_modes = ("even", "musical", "accelerating")
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown suggestion mode: '{mode}'. "
                f"Supported modes: {valid_modes}"
            )

        logger.info(
            f"Suggesting {num_shots} shot durations for {total_duration:.1f}s "
            f"(mode={mode})"
        )

        if mode == "even":
            return self._suggest_even(
                total_duration, num_shots, start_offset
            )
        elif mode == "musical":
            return self._suggest_musical(
                total_duration, num_shots, start_offset
            )
        elif mode == "accelerating":
            return self._suggest_accelerating(
                total_duration, num_shots, start_offset
            )

        return [total_duration / num_shots] * num_shots  # unreachable

    def _suggest_even(
        self,
        total_duration: float,
        num_shots: int,
        start_offset: float,
    ) -> list[float]:
        """Evenly distribute shots, then snap cut points to beats.

        Args:
            total_duration: Total duration in seconds.
            num_shots: Number of shots.
            start_offset: Time offset from audio start.

        Returns:
            List of durations.
        """
        base_duration = total_duration / num_shots

        # Build candidate cut points at even intervals
        cut_points = [
            start_offset + base_duration * (i + 1)
            for i in range(num_shots - 1)
        ]

        # Snap each cut point to nearest beat
        snapped_cuts = [self._find_nearest(t, self._beats) for t in cut_points]

        # Ensure monotonically increasing and within bounds
        snapped_cuts = self._enforce_monotonic(
            snapped_cuts, start_offset, start_offset + total_duration
        )

        # Convert cut points to durations
        durations = []
        prev = start_offset
        for cut in snapped_cuts:
            durations.append(round(cut - prev, 4))
            prev = cut
        durations.append(round(start_offset + total_duration - prev, 4))

        return durations

    def _suggest_musical(
        self,
        total_duration: float,
        num_shots: int,
        start_offset: float,
    ) -> list[float]:
        """Follow musical energy for shot duration suggestions.

        Low-energy sections (verses, intros) get longer shots.
        High-energy sections (choruses, drops) get shorter, faster cuts.

        Args:
            total_duration: Total duration in seconds.
            num_shots: Number of shots.
            start_offset: Time offset from audio start.

        Returns:
            List of durations.
        """
        _require_librosa()

        analysis_fps = 10
        energy = self.analyzer.get_energy_curve(fps=analysis_fps)

        if not energy:
            return self._suggest_even(total_duration, num_shots, start_offset)

        base_duration = total_duration / num_shots

        # Compute energy at each candidate cut point
        cut_points = [
            start_offset + base_duration * (i + 1)
            for i in range(num_shots - 1)
        ]

        # Weight durations inversely by energy: high energy = shorter shots
        segment_energies = []
        for i in range(num_shots):
            seg_start = start_offset if i == 0 else cut_points[i - 1]
            seg_end = (
                start_offset + total_duration
                if i == num_shots - 1
                else cut_points[i]
            )
            start_frame = int(seg_start * analysis_fps)
            end_frame = int(seg_end * analysis_fps)
            start_frame = max(0, min(start_frame, len(energy) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(energy)))
            avg_e = float(np.mean(energy[start_frame:end_frame]))
            segment_energies.append(avg_e)

        # Inverse energy weighting: low energy -> longer duration
        # energy=0 -> weight=1.5, energy=1 -> weight=0.5
        weights = [1.5 - e for e in segment_energies]
        total_weight = sum(weights)

        # Distribute total_duration proportional to weights
        raw_durations = [
            total_duration * (w / total_weight) for w in weights
        ]

        # Build cut points from these durations
        cut_points = []
        cumulative = start_offset
        for dur in raw_durations[:-1]:
            cumulative += dur
            cut_points.append(cumulative)

        # Snap to beats
        snapped_cuts = [self._find_nearest(t, self._beats) for t in cut_points]
        snapped_cuts = self._enforce_monotonic(
            snapped_cuts, start_offset, start_offset + total_duration
        )

        # Convert back to durations
        durations = []
        prev = start_offset
        for cut in snapped_cuts:
            durations.append(round(cut - prev, 4))
            prev = cut
        durations.append(round(start_offset + total_duration - prev, 4))

        return durations

    def _suggest_accelerating(
        self,
        total_duration: float,
        num_shots: int,
        start_offset: float,
    ) -> list[float]:
        """Start slow, cut faster toward the end — builds tension.

        Uses a harmonic series (1/1, 1/2, 1/3, ...) reversed so the
        longest shots are first and the shortest are last.

        Args:
            total_duration: Total duration in seconds.
            num_shots: Number of shots.
            start_offset: Time offset from audio start.

        Returns:
            List of durations.
        """
        # Harmonic weights: [1/1, 1/2, 1/3, ...] — reversed for deceleration
        # then reversed again so longest is first
        weights = [1.0 / (i + 1) for i in range(num_shots)]
        weights.reverse()  # longest first, shortest last
        total_weight = sum(weights)

        raw_durations = [
            total_duration * (w / total_weight) for w in weights
        ]

        # Build cut points
        cut_points = []
        cumulative = start_offset
        for dur in raw_durations[:-1]:
            cumulative += dur
            cut_points.append(cumulative)

        # Snap to beats
        snapped_cuts = [self._find_nearest(t, self._beats) for t in cut_points]
        snapped_cuts = self._enforce_monotonic(
            snapped_cuts, start_offset, start_offset + total_duration
        )

        # Convert back to durations
        durations = []
        prev = start_offset
        for cut in snapped_cuts:
            durations.append(round(cut - prev, 4))
            prev = cut
        durations.append(round(start_offset + total_duration - prev, 4))

        return durations

    def _enforce_monotonic(
        self,
        cuts: list[float],
        min_time: float,
        max_time: float,
    ) -> list[float]:
        """Ensure cut points are strictly monotonically increasing.

        After snapping to beats, two adjacent cuts might land on the same
        beat. This method resolves collisions by nudging forward to the
        next available beat.

        Args:
            cuts: List of snapped cut point times.
            min_time: Minimum allowed time (start of sequence).
            max_time: Maximum allowed time (end of sequence).

        Returns:
            Adjusted list of strictly increasing cut points.
        """
        if not cuts:
            return cuts

        result = []
        prev = min_time + self.MIN_SHOT_DURATION

        for cut in cuts:
            adjusted = max(cut, prev)
            # Also ensure we leave room for the last shot
            adjusted = min(adjusted, max_time - self.MIN_SHOT_DURATION)
            result.append(adjusted)
            prev = adjusted + self.MIN_SHOT_DURATION

        return result


# ---------------------------------------------------------------------------
# CLI — quick test / debug entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Analyze beats in an audio file and preview shot alignment."
    )
    parser.add_argument(
        "audio", help="Path to audio file (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=6,
        help="Number of shots to simulate (default: 6)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Video frame rate (default: 24)",
    )
    parser.add_argument(
        "--mode",
        choices=["snap_to_beat", "snap_to_downbeat", "energy_match"],
        default="snap_to_beat",
        help="Alignment mode (default: snap_to_beat)",
    )
    args = parser.parse_args()

    if not LIBROSA_AVAILABLE:
        print(
            "ERROR: librosa is not installed.\n"
            "Install it with: pip install librosa",
            file=sys.stderr,
        )
        sys.exit(1)

    # Analyze
    analyzer = BeatAnalyzer(args.audio)
    info = analyzer.get_info()

    print(f"\n{'=' * 60}")
    print(f"Audio: {args.audio}")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Tempo: {info.tempo:.1f} BPM")
    print(f"Beats: {len(info.beat_times)}")
    print(f"Downbeats: {len(info.downbeat_times)}")
    print(f"{'=' * 60}")

    # Show first 20 beats
    print(f"\nFirst {min(20, len(info.beat_times))} beats (seconds):")
    for i, t in enumerate(info.beat_times[:20]):
        marker = " [DOWNBEAT]" if t in info.downbeat_times else ""
        print(f"  Beat {i + 1:3d}: {t:7.3f}s{marker}")
    if len(info.beat_times) > 20:
        print(f"  ... ({len(info.beat_times) - 20} more)")

    # Simulate shot alignment
    print(f"\n{'=' * 60}")
    print(f"Shot alignment simulation ({args.num_shots} shots, mode={args.mode})")
    print(f"{'=' * 60}")

    # Create evenly-spaced shots as input
    shot_duration = info.duration / args.num_shots
    shots = [
        {
            "duration_seconds": shot_duration,
            "num_frames": round(shot_duration * args.fps),
            "fps": args.fps,
        }
        for _ in range(args.num_shots)
    ]

    aligner = ShotBeatAligner(analyzer)
    aligned = aligner.align_shots(shots, mode=args.mode)

    print(f"\n{'Shot':<6} {'Original':>10} {'Aligned':>10} {'Delta':>10} {'Frames':>8}")
    print(f"{'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")

    for i, (orig, adj) in enumerate(zip(shots, aligned)):
        delta = adj["duration_seconds"] - orig["duration_seconds"]
        print(
            f"{i + 1:<6d} {orig['duration_seconds']:>9.2f}s "
            f"{adj['duration_seconds']:>9.2f}s "
            f"{delta:>+9.2f}s {adj['num_frames']:>7d}"
        )

    total_orig = sum(s["duration_seconds"] for s in shots)
    total_adj = sum(s["duration_seconds"] for s in aligned)
    print(f"\nTotal: {total_orig:.2f}s -> {total_adj:.2f}s")

    # Show suggested durations
    print(f"\n{'=' * 60}")
    print("Suggested shot durations (all modes)")
    print(f"{'=' * 60}")

    for suggest_mode in ("even", "musical", "accelerating"):
        durations = aligner.suggest_shot_durations(
            total_duration=info.duration,
            num_shots=args.num_shots,
            mode=suggest_mode,
        )
        dur_str = " | ".join(f"{d:.2f}s" for d in durations)
        print(f"  {suggest_mode:<14}: {dur_str}")

    print()
