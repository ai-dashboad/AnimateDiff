"""
Shot Scheduler — orchestrates multi-shot video generation with temporal consistency.

Handles:
- Sequential shot generation with inter-shot consistency
- LoRA swapping between shots (different characters per shot)
- Frame count calculation from duration + fps
- Progress tracking and resumption
"""

import logging
import math
import os
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict

import torch

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.story_engine import StoryBoard, ShotSpec
from animatediff.core.character_manager import CharacterManager

logger = logging.getLogger(__name__)


@dataclass
class ShotResult:
    """Result of generating a single shot."""
    shot_id: int
    output: Optional[VideoOutput] = None
    output_path: str = ""
    duration_seconds: float = 0.0
    generation_time: float = 0.0
    error: Optional[str] = None


@dataclass
class SchedulerConfig:
    """Configuration for the shot scheduler."""
    fps: int = 16
    default_width: int = 1280
    default_height: int = 720
    max_frames_per_shot: int = 121
    output_dir: str = "output/shots"
    output_format: str = "mp4"
    quality: str = "standard"  # draft/standard/high/max
    save_intermediate: bool = True  # save each shot separately


# Map quality to generation parameters
QUALITY_MAP = {
    "draft": dict(num_inference_steps=10, guidance_scale=3.0),
    "standard": dict(num_inference_steps=20, guidance_scale=5.0),
    "high": dict(num_inference_steps=30, guidance_scale=6.0),
    "max": dict(num_inference_steps=50, guidance_scale=7.5),
}


class ShotScheduler:
    """Schedule and execute multi-shot video generation."""

    def __init__(
        self,
        pipeline: BasePipeline,
        character_manager: Optional[CharacterManager] = None,
        config: Optional[SchedulerConfig] = None,
    ):
        self.pipeline = pipeline
        self.characters = character_manager or CharacterManager()
        self.config = config or SchedulerConfig()
        self.results: List[ShotResult] = []

    def generate_all(
        self,
        board: StoryBoard,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[ShotResult]:
        """Generate all shots in a storyboard sequentially.

        Args:
            board: The storyboard to generate.
            progress_callback: Optional (current, total, message) callback.

        Returns:
            List of ShotResult for each shot.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.results = []
        total = board.num_shots

        quality_params = QUALITY_MAP.get(self.config.quality, QUALITY_MAP["standard"])

        for i, shot in enumerate(board.shots):
            if progress_callback:
                progress_callback(i, total, f"Generating shot {i+1}/{total}: {shot.scene[:50]}...")

            logger.info(f"[Shot {i+1}/{total}] {shot.prompt[:80]}...")
            result = self._generate_shot(shot, quality_params)
            self.results.append(result)

            if result.error:
                logger.error(f"[Shot {i+1}] Failed: {result.error}")
            else:
                logger.info(f"[Shot {i+1}] Done in {result.generation_time:.1f}s -> {result.output_path}")

        if progress_callback:
            progress_callback(total, total, "All shots complete.")

        return self.results

    def _generate_shot(self, shot: ShotSpec, quality_params: dict) -> ShotResult:
        """Generate a single shot."""
        result = ShotResult(shot_id=shot.shot_id)

        try:
            # Calculate frame count from duration
            fps = self.config.fps
            num_frames = shot.num_frames or self._duration_to_frames(shot.duration_seconds, fps)
            num_frames = min(num_frames, self.config.max_frames_per_shot)

            width = shot.width or self.config.default_width
            height = shot.height or self.config.default_height

            # Build enriched prompt with character descriptions
            prompt = shot.prompt
            if shot.characters:
                char_desc = self.characters.build_character_prompt(shot.characters)
                if char_desc and char_desc not in prompt:
                    prompt = f"{char_desc}, {prompt}"
            if shot.camera and shot.camera != "static":
                prompt = f"{prompt}, {shot.camera} camera movement"

            # Get character reference image (for I2V mode)
            ref_image = None
            if shot.characters:
                ref_image = self.characters.get_reference_image(shot.characters[0])

            t0 = time.time()

            output = self.pipeline.generate(
                prompt=prompt,
                negative_prompt=shot.negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=quality_params.get("num_inference_steps", 20),
                guidance_scale=quality_params.get("guidance_scale", 5.0),
                seed=shot.seed,
                image=ref_image,
            )

            result.generation_time = time.time() - t0
            result.output = output
            result.duration_seconds = len(output.frames) / fps

            # Save intermediate
            if self.config.save_intermediate:
                path = os.path.join(
                    self.config.output_dir,
                    f"shot_{shot.shot_id:04d}.{self.config.output_format}"
                )
                self.pipeline.save(output, path, fps=fps)
                result.output_path = path

        except Exception as e:
            result.error = str(e)
            logger.exception(f"Shot {shot.shot_id} failed")

        return result

    def _duration_to_frames(self, duration_seconds: float, fps: int) -> int:
        """Convert duration to frame count, aligned to common multiples.

        Wan models prefer frame counts of 4N+1 (e.g., 17, 33, 49, 65, 81, 97, 113, 121).
        """
        raw_frames = int(duration_seconds * fps)
        # Align to 4N+1
        aligned = ((raw_frames - 1) // 4) * 4 + 1
        return max(17, aligned)  # minimum 17 frames

    def compute_audio_first_durations(
        self,
        board: StoryBoard,
        narration_dir: str,
        transition_pad: float = 0.3,
    ) -> None:
        """Compute per-shot frame counts from narration audio durations.

        For each shot with narration text: reads the corresponding WAV file,
        computes duration, adds transition padding, converts to frames aligned
        to 4N+1, and updates shot.num_frames in-place.

        Shots with duration_seconds > 0 and no narration keep their explicit duration.
        Shots with duration_seconds == 0 and no narration get the default (5.0s).

        Args:
            board: StoryBoard whose shots will be updated in-place.
            narration_dir: Directory containing narration_XXXX.wav files.
            transition_pad: Extra seconds to add after narration ends (default 0.3s).
        """
        fps = self.config.fps
        max_frames = self.config.max_frames_per_shot

        for i, shot in enumerate(board.shots):
            wav_path = os.path.join(narration_dir, f"narration_{i:04d}.wav")

            if shot.narration and os.path.exists(wav_path):
                audio_duration = self._get_wav_duration(wav_path)
                total_duration = audio_duration + transition_pad
                num_frames = self._duration_to_frames(total_duration, fps)
                num_frames = min(num_frames, max_frames)

                shot.num_frames = num_frames
                shot.duration_seconds = num_frames / fps

                logger.info(
                    f"[Shot {i}] Audio={audio_duration:.2f}s + pad={transition_pad}s "
                    f"-> {num_frames} frames ({shot.duration_seconds:.2f}s)"
                )

                # Flag if narration exceeds max frame duration
                if audio_duration > max_frames / fps:
                    logger.warning(
                        f"[Shot {i}] Narration ({audio_duration:.2f}s) exceeds max "
                        f"({max_frames / fps:.2f}s). Last frames will hold."
                    )
            elif shot.duration_seconds == 0:
                # No narration and no explicit duration: use default
                shot.duration_seconds = 5.0
                shot.num_frames = self._duration_to_frames(5.0, fps)
                logger.info(f"[Shot {i}] No narration, default 5.0s -> {shot.num_frames} frames")

    @staticmethod
    def _get_wav_duration(wav_path: str) -> float:
        """Get duration of a WAV file in seconds."""
        with wave.open(wav_path, 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate

    def get_all_frames(self) -> List:
        """Collect all frames from all successful shots in order."""
        all_frames = []
        for result in self.results:
            if result.output and result.output.frames:
                all_frames.extend(result.output.frames)
        return all_frames
