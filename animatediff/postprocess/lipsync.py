"""
Lip Sync Post-Processor — apply lip sync to video shots with dialogue.

Supports:
- MuseTalk: High-quality audio-driven lip sync (requires CUDA GPU)
- JoyVASA: Anime-friendly lip sync alternative
- Fallback: Skip lip sync gracefully if neither tool works

Usage:
    processor = LipSyncProcessor(backend="auto")
    synced_frames = processor.apply(frames, audio_path, fps=24)
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class LipSyncProcessor:
    """Apply lip sync to video frames using audio-driven face animation."""

    SUPPORTED_BACKENDS = ("musetalk", "joyvasa", "auto", "none")

    def __init__(self, backend: str = "auto", device: str = "cuda"):
        """
        Args:
            backend: "musetalk", "joyvasa", "auto", or "none".
            device: torch device (lip sync requires CUDA for most backends).
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown lip sync backend: {backend}. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )
        self.device = device
        self.backend = self._resolve_backend(backend)
        self._model = None

    def _resolve_backend(self, backend: str) -> str:
        """Auto-detect available lip sync backend."""
        if backend != "auto":
            return backend

        if self.device == "mps":
            logger.warning("Lip sync requires CUDA GPU. MPS not supported.")
            return "none"

        # Try MuseTalk first
        if self._check_musetalk():
            logger.info("Auto-detected lip sync backend: musetalk")
            return "musetalk"

        # Try JoyVASA
        if self._check_joyvasa():
            logger.info("Auto-detected lip sync backend: joyvasa")
            return "joyvasa"

        logger.warning("No lip sync backend available. Install MuseTalk or JoyVASA.")
        return "none"

    @staticmethod
    def _check_musetalk() -> bool:
        """Check if MuseTalk is available."""
        try:
            import musetalk  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_joyvasa() -> bool:
        """Check if JoyVASA is available."""
        try:
            import joyvasa  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        """Whether a working lip sync backend is available."""
        return self.backend not in ("none",)

    def apply(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int = 24,
    ) -> List[Image.Image]:
        """Apply lip sync to a sequence of video frames.

        Args:
            frames: Input video frames (PIL Images).
            audio_path: Path to the dialogue audio WAV file.
            fps: Frame rate of the video.

        Returns:
            Lip-synced frames (same length as input). If lip sync fails,
            returns the original frames unchanged.
        """
        if self.backend == "none":
            logger.info("Lip sync disabled, returning original frames")
            return frames

        if not frames or not audio_path or not os.path.exists(audio_path):
            logger.warning("Missing frames or audio, skipping lip sync")
            return frames

        try:
            if self.backend == "musetalk":
                return self._apply_musetalk(frames, audio_path, fps)
            elif self.backend == "joyvasa":
                return self._apply_joyvasa(frames, audio_path, fps)
            else:
                return frames
        except Exception as e:
            logger.error(f"Lip sync failed ({self.backend}): {e}")
            logger.info("Returning original frames without lip sync")
            return frames

    def _apply_musetalk(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int,
    ) -> List[Image.Image]:
        """Apply lip sync using MuseTalk.

        MuseTalk pipeline:
        1. Save frames as temporary video
        2. Run MuseTalk inference (audio + video -> lip-synced video)
        3. Extract frames from output video
        """
        from diffusers.utils import export_to_video
        import musetalk
        from musetalk.utils.utils import get_file_type, get_video_fps, datagen
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
        from musetalk.utils.blending import get_image

        logger.info(f"Applying MuseTalk lip sync ({len(frames)} frames)...")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input frames as video
            input_video = os.path.join(tmpdir, "input.mp4")
            export_to_video(frames, input_video, fps=fps)

            # Run MuseTalk inference
            output_video = os.path.join(tmpdir, "output.mp4")

            # Use MuseTalk real-time inference API
            from musetalk.real_time_inference import MuseTalkRealtimeInference

            if self._model is None:
                self._model = MuseTalkRealtimeInference()
                self._model.init_model()

            self._model.process(
                video_path=input_video,
                audio_path=audio_path,
                output_path=output_video,
            )

            # Read output frames
            return self._read_video_frames(output_video, len(frames))

        return frames  # fallback

    def _apply_joyvasa(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int,
    ) -> List[Image.Image]:
        """Apply lip sync using JoyVASA (anime-friendly).

        JoyVASA supports anime/cartoon face animation which makes it
        more suitable for our xianxia anime style.
        """
        from diffusers.utils import export_to_video

        logger.info(f"Applying JoyVASA lip sync ({len(frames)} frames)...")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save first frame as reference image
            ref_image_path = os.path.join(tmpdir, "ref.png")
            frames[0].save(ref_image_path)

            # Save input frames as video
            input_video = os.path.join(tmpdir, "input.mp4")
            export_to_video(frames, input_video, fps=fps)

            output_video = os.path.join(tmpdir, "output.mp4")

            # JoyVASA CLI inference
            cmd = [
                "python", "-m", "joyvasa.inference",
                "--source_image", ref_image_path,
                "--driving_audio", audio_path,
                "--output", output_video,
                "--fps", str(fps),
                "--anime_mode",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"JoyVASA failed: {result.stderr}")
                return frames

            return self._read_video_frames(output_video, len(frames))

        return frames  # fallback

    @staticmethod
    def _read_video_frames(video_path: str, expected_count: int) -> List[Image.Image]:
        """Read frames from a video file using ffmpeg."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract frames with ffmpeg
            frame_pattern = os.path.join(tmpdir, "frame_%06d.png")
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "fps=fps",
                frame_pattern,
            ]
            subprocess.run(cmd, capture_output=True, text=True)

            # Read extracted frames
            frames = []
            for i in range(1, expected_count + 1):
                path = os.path.join(tmpdir, f"frame_{i:06d}.png")
                if os.path.exists(path):
                    frames.append(Image.open(path).convert("RGB"))
                else:
                    break

        # Pad or truncate to expected count
        if len(frames) < expected_count and frames:
            # Repeat last frame to match expected count
            while len(frames) < expected_count:
                frames.append(frames[-1].copy())
        elif len(frames) > expected_count:
            frames = frames[:expected_count]

        return frames


def apply_lipsync_to_shots(
    shot_frame_lists: List[List[Image.Image]],
    narration_paths: List[str],
    lip_sync_flags: List[bool],
    fps: int = 24,
    backend: str = "auto",
    device: str = "cuda",
) -> List[List[Image.Image]]:
    """Convenience function: apply lip sync to selected shots.

    Args:
        shot_frame_lists: List of frame lists (one per shot).
        narration_paths: Per-shot narration audio paths.
        lip_sync_flags: Per-shot boolean flags for lip sync.
        fps: Video frame rate.
        backend: Lip sync backend.
        device: Torch device.

    Returns:
        Updated frame lists with lip sync applied to flagged shots.
    """
    # Quick check: any shots need lip sync?
    if not any(lip_sync_flags):
        logger.info("No shots flagged for lip sync, skipping")
        return shot_frame_lists

    processor = LipSyncProcessor(backend=backend, device=device)
    if not processor.available:
        logger.warning("No lip sync backend available, skipping all lip sync")
        return shot_frame_lists

    result = []
    for i, frames in enumerate(shot_frame_lists):
        should_sync = (
            i < len(lip_sync_flags) and lip_sync_flags[i]
            and i < len(narration_paths) and narration_paths[i]
        )
        if should_sync:
            logger.info(f"[Shot {i}] Applying lip sync...")
            synced = processor.apply(frames, narration_paths[i], fps=fps)
            result.append(synced)
        else:
            result.append(frames)

    return result
