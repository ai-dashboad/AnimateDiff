"""
Video Compositor — assemble shots, transitions, audio into final video.

Handles:
- Shot concatenation with transition effects (cut, fade, dissolve)
- Audio track merging (narration + BGM + SFX)
- Final video encoding via ffmpeg or moviepy
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VideoCompositor:
    """Assemble multiple shots into a final video with audio."""

    def __init__(self, fps: int = 16, output_format: str = "mp4"):
        self.fps = fps
        self.output_format = output_format

    def compose(
        self,
        shot_frame_lists: List[List[Image.Image]],
        output_path: str,
        transitions: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        bgm_path: Optional[str] = None,
        bgm_volume: float = 0.3,
    ) -> str:
        """Compose multiple shot frame lists into a single video.

        Args:
            shot_frame_lists: List of frame lists (one per shot).
            output_path: Final output video path.
            transitions: Transition type between shots ("cut", "fade", "dissolve").
            audio_paths: Per-shot audio file paths (narration/dialogue).
            bgm_path: Background music file path.
            bgm_volume: BGM volume relative to narration (0.0-1.0).

        Returns:
            Path to the final video file.
        """
        if not shot_frame_lists:
            raise ValueError("No shots to compose")

        transitions = transitions or ["cut"] * (len(shot_frame_lists) - 1)

        # Apply transitions between shots
        all_frames = self._apply_transitions(shot_frame_lists, transitions)

        logger.info(f"Composing {len(all_frames)} total frames from {len(shot_frame_lists)} shots")

        # Save frames to temporary video
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if audio_paths or bgm_path:
            # Need ffmpeg for audio mixing
            return self._compose_with_audio(
                all_frames, output_path, audio_paths, bgm_path, bgm_volume
            )
        else:
            # Simple frame-only output
            return self._save_frames_as_video(all_frames, output_path)

    def _apply_transitions(
        self,
        shot_frame_lists: List[List[Image.Image]],
        transitions: List[str],
    ) -> List[Image.Image]:
        """Apply transition effects between shots."""
        if len(shot_frame_lists) == 1:
            return shot_frame_lists[0]

        all_frames = []
        for i, frames in enumerate(shot_frame_lists):
            if i == 0:
                all_frames.extend(frames)
                continue

            transition = transitions[i - 1] if i - 1 < len(transitions) else "cut"

            if transition == "fade":
                all_frames.extend(self._fade_transition(all_frames[-1], frames[0], duration_frames=8))
                all_frames.extend(frames[1:])
            elif transition == "dissolve":
                overlap = min(8, len(frames) // 2)
                all_frames.extend(self._dissolve_transition(
                    all_frames[-overlap:], frames[:overlap]
                ))
                all_frames.extend(frames[overlap:])
            else:  # cut
                all_frames.extend(frames)

        return all_frames

    def _fade_transition(
        self, last_frame: Image.Image, first_frame: Image.Image, duration_frames: int = 8
    ) -> List[Image.Image]:
        """Create a fade-through-black transition."""
        result = []
        arr_last = np.array(last_frame, dtype=np.float32)
        arr_first = np.array(first_frame, dtype=np.float32)

        half = duration_frames // 2

        # Fade to black
        for i in range(half):
            alpha = 1.0 - (i + 1) / half
            faded = (arr_last * alpha).astype(np.uint8)
            result.append(Image.fromarray(faded))

        # Fade from black
        for i in range(half):
            alpha = (i + 1) / half
            faded = (arr_first * alpha).astype(np.uint8)
            result.append(Image.fromarray(faded))

        return result

    def _dissolve_transition(
        self, end_frames: List[Image.Image], start_frames: List[Image.Image]
    ) -> List[Image.Image]:
        """Create a cross-dissolve transition."""
        n = min(len(end_frames), len(start_frames))
        result = []

        for i in range(n):
            alpha = (i + 1) / (n + 1)
            arr_a = np.array(end_frames[i], dtype=np.float32)
            arr_b = np.array(start_frames[i], dtype=np.float32)
            blended = (arr_a * (1 - alpha) + arr_b * alpha).astype(np.uint8)
            result.append(Image.fromarray(blended))

        return result

    def _save_frames_as_video(self, frames: List[Image.Image], output_path: str) -> str:
        """Save frames as video using diffusers utility or ffmpeg."""
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, output_path, fps=self.fps)
            logger.info(f"Saved video: {output_path}")
            return output_path
        except ImportError:
            return self._save_with_ffmpeg(frames, output_path)

    def _compose_with_audio(
        self,
        frames: List[Image.Image],
        output_path: str,
        audio_paths: Optional[List[str]] = None,
        bgm_path: Optional[str] = None,
        bgm_volume: float = 0.3,
    ) -> str:
        """Compose video with audio tracks using ffmpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save silent video first
            silent_video = os.path.join(tmpdir, "silent.mp4")
            self._save_frames_as_video(frames, silent_video)

            # Build ffmpeg command for audio mixing
            cmd = ["ffmpeg", "-y", "-i", silent_video]
            filter_parts = []
            input_idx = 1

            # Add narration audio tracks
            valid_audios = []
            if audio_paths:
                for path in audio_paths:
                    if path and os.path.exists(path):
                        cmd.extend(["-i", path])
                        valid_audios.append(input_idx)
                        input_idx += 1

            # Add BGM
            bgm_idx = None
            if bgm_path and os.path.exists(bgm_path):
                cmd.extend(["-i", bgm_path])
                bgm_idx = input_idx
                input_idx += 1

            if not valid_audios and bgm_idx is None:
                # No audio to mix, just copy
                os.rename(silent_video, output_path)
                return output_path

            # Build audio mixing filter
            audio_inputs = []

            if valid_audios:
                # Concatenate narration tracks
                concat_parts = "".join(f"[{i}:a]" for i in valid_audios)
                filter_parts.append(
                    f"{concat_parts}concat=n={len(valid_audios)}:v=0:a=1[narration]"
                )
                audio_inputs.append("[narration]")

            if bgm_idx is not None:
                filter_parts.append(
                    f"[{bgm_idx}:a]volume={bgm_volume}[bgm]"
                )
                audio_inputs.append("[bgm]")

            if len(audio_inputs) > 1:
                mix_inputs = "".join(audio_inputs)
                filter_parts.append(
                    f"{mix_inputs}amix=inputs={len(audio_inputs)}:duration=first[aout]"
                )
                audio_label = "[aout]"
            elif len(audio_inputs) == 1:
                audio_label = audio_inputs[0]
            else:
                audio_label = None

            if audio_label and filter_parts:
                cmd.extend(["-filter_complex", ";".join(filter_parts)])
                cmd.extend(["-map", "0:v", "-map", audio_label])
            elif audio_label:
                cmd.extend(["-map", "0:v", "-map", f"{valid_audios[0] if valid_audios else bgm_idx}:a"])

            cmd.extend([
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path,
            ])

            logger.info(f"Running ffmpeg for audio mixing...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                # Fallback: return silent video
                os.rename(silent_video, output_path)
            else:
                logger.info(f"Composed final video: {output_path}")

        return output_path

    def compose_with_timed_audio(
        self,
        shot_frame_lists: List[List[Image.Image]],
        output_path: str,
        transitions: Optional[List[str]] = None,
        narration_paths: Optional[List[str]] = None,
        bgm_path: Optional[str] = None,
        bgm_volume: float = 0.3,
        subtitle_texts: Optional[List[str]] = None,
    ) -> str:
        """Compose video with per-shot narration precisely aligned to shot start times.

        Unlike compose(), this method calculates the exact start time of each shot
        in the assembled timeline and uses ffmpeg adelay to align each narration
        clip to its corresponding shot, ensuring audio-visual sync.

        Args:
            shot_frame_lists: List of frame lists (one per shot).
            output_path: Final output video path.
            transitions: Transition type between shots.
            narration_paths: Per-shot narration WAV paths (empty string = no narration).
            bgm_path: Background music file.
            bgm_volume: BGM volume (0.0-1.0).
            subtitle_texts: Optional per-shot subtitle/scene text overlay.

        Returns:
            Path to the final video file.
        """
        if not shot_frame_lists:
            raise ValueError("No shots to compose")

        transitions = transitions or ["cut"] * (len(shot_frame_lists) - 1)

        # Calculate shot start times in the timeline (in seconds)
        shot_start_times = []
        current_time = 0.0
        for i, frames in enumerate(shot_frame_lists):
            shot_start_times.append(current_time)
            shot_duration = len(frames) / self.fps
            if i < len(transitions):
                transition = transitions[i] if i < len(transitions) else "cut"
                if transition == "fade":
                    shot_duration -= 8 / self.fps  # fade overlap
                elif transition == "dissolve":
                    overlap = min(8, len(frames) // 2)
                    shot_duration -= overlap / self.fps
            current_time += shot_duration

        # Apply transitions between shots
        all_frames = self._apply_transitions(shot_frame_lists, transitions)

        logger.info(
            f"Composing {len(all_frames)} frames from {len(shot_frame_lists)} shots "
            f"with timed audio alignment"
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save silent video
            silent_video = os.path.join(tmpdir, "silent.mp4")
            self._save_frames_as_video(all_frames, silent_video)

            # Build ffmpeg command with timed audio
            cmd = ["ffmpeg", "-y", "-i", silent_video]
            filter_parts = []
            input_idx = 1

            # Add each narration track as a separate input
            narration_inputs = []  # (input_idx, start_time_ms)
            if narration_paths:
                for i, path in enumerate(narration_paths):
                    if path and os.path.exists(path) and i < len(shot_start_times):
                        cmd.extend(["-i", path])
                        delay_ms = int(shot_start_times[i] * 1000)
                        narration_inputs.append((input_idx, delay_ms))
                        input_idx += 1

            # Add BGM
            bgm_idx = None
            if bgm_path and os.path.exists(bgm_path):
                cmd.extend(["-i", bgm_path])
                bgm_idx = input_idx
                input_idx += 1

            if not narration_inputs and bgm_idx is None:
                os.rename(silent_video, output_path)
                return output_path

            # Build filter: delay each narration to its shot start time
            delayed_labels = []
            for idx, (inp_idx, delay_ms) in enumerate(narration_inputs):
                label = f"narr{idx}"
                filter_parts.append(
                    f"[{inp_idx}:a]adelay={delay_ms}|{delay_ms}[{label}]"
                )
                delayed_labels.append(f"[{label}]")

            # Mix all narration tracks together
            if len(delayed_labels) > 1:
                mix_inputs = "".join(delayed_labels)
                filter_parts.append(
                    f"{mix_inputs}amix=inputs={len(delayed_labels)}:"
                    f"duration=longest:normalize=0[narration]"
                )
                narration_label = "[narration]"
            elif len(delayed_labels) == 1:
                narration_label = delayed_labels[0]
            else:
                narration_label = None

            # Add BGM with volume control
            audio_inputs = []
            if narration_label:
                audio_inputs.append(narration_label)
            if bgm_idx is not None:
                filter_parts.append(f"[{bgm_idx}:a]volume={bgm_volume}[bgm]")
                audio_inputs.append("[bgm]")

            # Final mix
            if len(audio_inputs) > 1:
                mix_all = "".join(audio_inputs)
                filter_parts.append(
                    f"{mix_all}amix=inputs={len(audio_inputs)}:"
                    f"duration=first:normalize=0[aout]"
                )
                audio_label = "[aout]"
            elif len(audio_inputs) == 1:
                audio_label = audio_inputs[0]
            else:
                audio_label = None

            if audio_label and filter_parts:
                cmd.extend(["-filter_complex", ";".join(filter_parts)])
                cmd.extend(["-map", "0:v", "-map", audio_label])

            cmd.extend([
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path,
            ])

            logger.info("Running ffmpeg for timed audio composition...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                # Fallback to simple composition
                logger.info("Falling back to simple audio composition...")
                return self.compose(
                    shot_frame_lists, output_path, transitions,
                    narration_paths, bgm_path, bgm_volume
                )
            else:
                logger.info(f"Composed final video with timed audio: {output_path}")

        return output_path

    def _save_with_ffmpeg(self, frames: List[Image.Image], output_path: str) -> str:
        """Save frames using ffmpeg pipe (fallback when diffusers not available)."""
        if not frames:
            return output_path

        h, w = np.array(frames[0]).shape[:2]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(np.array(frame).tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            logger.error(f"ffmpeg failed: {proc.stderr.read().decode()}")
        else:
            logger.info(f"Saved video via ffmpeg: {output_path}")

        return output_path
