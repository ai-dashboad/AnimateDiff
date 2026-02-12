"""
Shot Scheduler — orchestrates multi-shot video generation with temporal consistency.

Handles:
- Sequential shot generation with inter-shot consistency
- LoRA swapping between shots (different characters per shot)
- Frame count calculation from duration + fps
- Progress tracking and resumption
- Camera movement orchestration and shot rhythm planning (ShotOrchestrator)
"""

import logging
import math
import os
import random
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

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


# ===========================================================================
# Scene-type keyword tables for prompt classification
# ===========================================================================

# Each key is a scene type; values are lowercase keyword tokens that, when
# found inside a shot prompt, vote for that scene type.

_SCENE_KEYWORDS: Dict[str, List[str]] = {
    "action": [
        "fight", "sword", "battle", "chase", "combat", "attack", "clash",
        "strike", "martial", "duel", "kick", "punch", "slash", "war",
        "explosion", "charge", "dodge", "parry", "weapon", "arrow",
        "剑", "战", "斗", "攻", "武", "刺", "杀",
    ],
    "emotional": [
        "dialogue", "tears", "whisper", "embrace", "farewell", "meditation",
        "love", "cry", "hug", "kiss", "pray", "mourn", "smile", "tender",
        "gentle", "sorrow", "longing", "reunion", "goodbye",
        "泪", "拥", "别", "柔", "悲", "思", "念",
    ],
    "establishing": [
        "landscape", "mountain", "village", "sky", "sunrise", "valley",
        "city", "ocean", "forest", "desert", "wide shot", "panorama",
        "sunset", "horizon", "establishing", "aerial view", "overview",
        "全景", "远景", "风景", "山", "水", "天",
    ],
    "xianxia": [
        "flying", "breakthrough", "qi", "cultivation", "spiritual",
        "formation", "array", "ascend", "transcend", "immortal", "sect",
        "elixir", "pill", "tribulation", "heavenly", "dao", "enlighten",
        "realm", "aura", "energy pillar", "levitat",
        "仙", "修", "道", "灵", "突破", "飞", "阵", "丹", "劫", "悟",
    ],
    "transition": [
        "title", "end", "card", "credits", "opening", "closing",
        "logo", "text card", "chapter", "epilogue", "prologue",
        "片头", "片尾", "字幕",
    ],
}

# Preferred camera preset categories for each scene type.  The orchestrator
# picks from this mapping when auto-assigning cameras.
_SCENE_TYPE_TO_CATEGORIES: Dict[str, List[str]] = {
    "action":       ["dynamic", "dramatic"],
    "emotional":    ["intimate", "dramatic"],
    "establishing": ["establishing", "aerial"],
    "xianxia":      ["xianxia_specific", "aerial", "dramatic"],
    "transition":   [],  # always "static" for title/end cards
}

# When auto-selecting a camera from a category, prefer presets whose
# motion_intensity matches the scene energy.
_SCENE_TYPE_DEFAULT_INTENSITY: Dict[str, str] = {
    "action":       "high",
    "emotional":    "low",
    "establishing": "low",
    "xianxia":      "medium",
    "transition":   "none",
}


# ===========================================================================
# ShotOrchestrator
# ===========================================================================

class ShotOrchestrator:
    """Orchestrates camera movements and shot rhythm across a storyboard.

    Sits between the StoryEngine (which produces a list of ShotSpecs) and the
    ShotScheduler (which generates video). Its job is to:

    1. **Auto-assign** camera presets to shots that don't already have one,
       using keyword-based scene classification.
    2. **Shape rhythm** -- adjust durations so the edit has cinematic pacing
       rather than uniform 5-second clips.
    3. **Generate trajectories** -- produce a ``CameraTrajectory`` for every
       shot so the scheduler can inject Plucker / optical-flow / prompt data.
    4. **Validate** the plan and warn about common mistakes (e.g. the same
       camera used back-to-back, no variety in intensity).

    Usage::

        from animatediff.core.camera_ctrl import CameraController
        from animatediff.core.shot_scheduler import ShotOrchestrator

        ctrl = CameraController(method="prompt_only")
        orchestrator = ShotOrchestrator(ctrl)

        # board.shots are ShotSpec dicts (or objects) from StoryEngine
        shots = orchestrator.plan_cameras(board.shots)
        shots = orchestrator.plan_rhythm(shots)
        warnings = orchestrator.validate_plan(shots)
        trajectories = orchestrator.generate_trajectories(shots)
    """

    def __init__(
        self,
        camera_controller: Any = None,
        *,
        default_fps: int = 24,
        default_width: int = 832,
        default_height: int = 480,
        seed: Optional[int] = None,
    ):
        """Initialize the orchestrator.

        Args:
            camera_controller: A ``CameraController`` instance used for
                applying trajectories to pipeline kwargs. If ``None``, a
                default prompt-only controller is created lazily.
            default_fps: Frames per second used when computing frame counts
                from durations.
            default_width: Default output width in pixels (for trajectory
                generation).
            default_height: Default output height in pixels (for trajectory
                generation).
            seed: Optional random seed for deterministic camera selection
                when multiple candidates score equally.
        """
        self._camera_controller = camera_controller
        self.default_fps = default_fps
        self.default_width = default_width
        self.default_height = default_height
        self._rng = random.Random(seed)

        # Lazy-loaded preset catalog from camera_presets.json
        self._preset_catalog: Optional[Dict[str, dict]] = None

    # ------------------------------------------------------------------
    # Internal: lazy loaders
    # ------------------------------------------------------------------

    @property
    def camera_controller(self):
        """Lazily create a default CameraController if none was provided."""
        if self._camera_controller is None:
            from animatediff.core.camera_ctrl import CameraController
            self._camera_controller = CameraController(method="prompt_only")
        return self._camera_controller

    def _load_preset_catalog(self) -> Dict[str, dict]:
        """Load and cache the preset catalog from camera_presets.json."""
        if self._preset_catalog is None:
            try:
                from animatediff.data.camera_utils import load_presets
                self._preset_catalog = load_presets()
            except Exception:
                logger.warning("Could not load camera_presets.json; "
                               "falling back to built-in trajectory presets only.")
                self._preset_catalog = {}
        return self._preset_catalog

    # ------------------------------------------------------------------
    # Scene classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_scene(prompt: str) -> Tuple[str, float]:
        """Classify a shot prompt into a scene type using keyword matching.

        Args:
            prompt: The shot's text prompt.

        Returns:
            A tuple of ``(scene_type, confidence)`` where *scene_type* is one
            of ``"action"``, ``"emotional"``, ``"establishing"``,
            ``"xianxia"``, ``"transition"``, or ``"general"`` (fallback),
            and *confidence* is a float in ``[0, 1]``.
        """
        prompt_lower = prompt.lower()

        scores: Dict[str, int] = {}
        for scene_type, keywords in _SCENE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > 0:
                scores[scene_type] = score

        if not scores:
            return ("general", 0.0)

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_type]
        # Normalise confidence: 1 keyword = 0.3, 2 = 0.5, 3+ = 0.7+, 5+ = 1.0
        confidence = min(1.0, 0.15 + best_score * 0.17)
        return (best_type, round(confidence, 2))

    # ------------------------------------------------------------------
    # plan_cameras
    # ------------------------------------------------------------------

    def plan_cameras(self, shots: List[ShotSpec]) -> List[ShotSpec]:
        """Auto-assign ``camera`` presets to shots that lack one.

        Shots that already have a non-empty ``camera`` field (other than a
        bare ``"static"`` when the scene type suggests motion) are left
        untouched.

        Rules applied:
        - Title / end cards are always set to ``"static"``.
        - Action scenes prefer presets from the ``"dynamic"`` category.
        - Emotional scenes prefer ``"intimate"`` presets.
        - Establishing shots prefer ``"establishing"`` or ``"aerial"``.
        - Xianxia scenes prefer ``"xianxia_specific"`` presets.
        - When multiple presets are equally good, the orchestrator uses
          ``recommend_for_shot()`` from ``camera_utils`` as a tiebreaker
          and randomises among top candidates.

        After assignment, a second pass ensures no two consecutive shots
        use the same preset (the *variety pass*).

        Args:
            shots: List of ``ShotSpec`` objects. Modified **in place** and
                also returned for chaining convenience.

        Returns:
            The same list of shots, with ``camera`` fields populated.
        """
        catalog = self._load_preset_catalog()
        presets_by_category: Dict[str, List[dict]] = {}
        for preset in catalog.values():
            cat = preset.get("category", "")
            presets_by_category.setdefault(cat, []).append(preset)

        for shot in shots:
            # Skip shots that already have an explicit camera preset that is
            # not the generic placeholder "static".
            if shot.camera and shot.camera != "static":
                continue

            scene_type, confidence = self.classify_scene(shot.prompt)

            # Transition / title cards are always static.
            if scene_type == "transition":
                shot.camera = "static"
                shot.metadata["scene_type"] = scene_type
                shot.metadata["camera_confidence"] = confidence
                continue

            # Pick candidate presets from the preferred categories.
            preferred_cats = _SCENE_TYPE_TO_CATEGORIES.get(scene_type, [])
            target_intensity = _SCENE_TYPE_DEFAULT_INTENSITY.get(scene_type, "medium")
            candidates = self._gather_candidates(
                preferred_cats, target_intensity, presets_by_category,
            )

            # Also ask recommend_for_shot for extra candidates scored by
            # the prompt-based matching system.
            recommended = self._recommend_from_camera_utils(
                camera_hint=shot.camera or None,
                emotion=shot.emotion or None,
            )
            # Merge -- prefer recommended if it overlaps with our candidates.
            recommended_ids = {r["id"] for r in recommended}
            candidate_ids = {c["id"] for c in candidates}
            # Boost: recommended candidates that are also in our category list
            # go to the front.
            boosted = [c for c in candidates if c["id"] in recommended_ids]
            rest = [c for c in candidates if c["id"] not in recommended_ids]
            # Then append any recommended that we didn't already have.
            extra = [r for r in recommended if r["id"] not in candidate_ids]
            final_candidates = boosted + rest + extra

            if final_candidates:
                chosen = self._rng.choice(
                    final_candidates[:5]  # pick from top 5
                )
                shot.camera = chosen["id"]
            elif scene_type == "general":
                # No strong signal -- pick a conservative default.
                shot.camera = "slow_push_in"
            else:
                # Shouldn't happen, but be defensive.
                shot.camera = "static"

            shot.metadata["scene_type"] = scene_type
            shot.metadata["camera_confidence"] = confidence

        # --- Variety pass: avoid consecutive repeats -----------------------
        self._deduplicate_consecutive(shots, catalog)

        return shots

    def _gather_candidates(
        self,
        preferred_categories: List[str],
        target_intensity: str,
        presets_by_category: Dict[str, List[dict]],
    ) -> List[dict]:
        """Collect camera presets from preferred categories, prioritised by
        motion_intensity match.

        Returns a list sorted so that intensity-matching presets come first.
        """
        exact = []
        fallback = []
        for cat in preferred_categories:
            for preset in presets_by_category.get(cat, []):
                if preset.get("motion_intensity") == target_intensity:
                    exact.append(preset)
                else:
                    fallback.append(preset)
        return exact + fallback

    def _recommend_from_camera_utils(
        self,
        camera_hint: Optional[str],
        emotion: Optional[str],
    ) -> List[dict]:
        """Delegate to ``camera_utils.recommend_for_shot`` if available."""
        try:
            from animatediff.data.camera_utils import recommend_for_shot
            return recommend_for_shot(camera_hint=camera_hint, emotion=emotion)
        except Exception:
            return []

    def _deduplicate_consecutive(
        self,
        shots: List[ShotSpec],
        catalog: Dict[str, dict],
    ) -> None:
        """Ensure no two adjacent shots share the same camera preset.

        When a collision is found the second shot is reassigned to a preset
        from the same category with a different ID. If no alternative is
        available the collision is left in place (a warning will be surfaced
        by ``validate_plan``).
        """
        for i in range(1, len(shots)):
            if shots[i].camera == shots[i - 1].camera and shots[i].camera != "static":
                current_id = shots[i].camera
                current_preset = catalog.get(current_id, {})
                current_cat = current_preset.get("category", "")

                # Find alternatives in the same category.
                alternatives = [
                    pid for pid, p in catalog.items()
                    if p.get("category") == current_cat and pid != current_id
                ]
                if alternatives:
                    shots[i].camera = self._rng.choice(alternatives)
                    logger.debug(
                        f"[Variety] Shot {i}: '{current_id}' collided with "
                        f"shot {i-1}; swapped to '{shots[i].camera}'."
                    )

    # ------------------------------------------------------------------
    # plan_rhythm
    # ------------------------------------------------------------------

    def plan_rhythm(
        self,
        shots: List[ShotSpec],
        energy_curve: Optional[List[float]] = None,
    ) -> List[ShotSpec]:
        """Adjust shot durations for cinematic rhythm.

        Two modes are supported:

        **Audio-driven** (when *energy_curve* is provided):
            The curve is expected to come from a ``BeatAnalyzer`` and contains
            one float per shot representing normalised energy in ``[0, 1]``.
            High-energy shots get shorter durations (fast cuts); low-energy
            shots get longer, more contemplative durations.

        **Heuristic** (no energy curve):
            Applies a simple dramatic arc: start medium, alternate long/short,
            build intensity toward a climax at roughly 70-80 % through the
            sequence, then wind down.

        In both modes the orchestrator also ensures that motion_intensity
        and duration are coherent (high-intensity cameras get shorter
        durations to avoid dizziness; low-intensity cameras can breathe).

        After duration adjustment a variety check ensures the same camera
        is not used twice in a row (already handled by ``plan_cameras`` but
        re-checked here since durations can shift which shots feel
        "identical").

        Args:
            shots: List of ``ShotSpec`` objects, modified in place.
            energy_curve: Optional list of floats (one per shot) in
                ``[0, 1]``, where 1.0 is maximum energy.

        Returns:
            The same list of shots with adjusted ``duration_seconds``.
        """
        n = len(shots)
        if n == 0:
            return shots

        catalog = self._load_preset_catalog()

        if energy_curve is not None and len(energy_curve) == n:
            self._apply_energy_curve(shots, energy_curve, catalog)
        else:
            self._apply_heuristic_rhythm(shots, catalog)

        return shots

    def _apply_energy_curve(
        self,
        shots: List[ShotSpec],
        energy: List[float],
        catalog: Dict[str, dict],
    ) -> None:
        """Map an energy curve to shot durations.

        Mapping: energy 1.0 -> min_duration (fast cuts), energy 0.0 ->
        max_duration (contemplative).
        """
        MIN_DURATION = 2.0
        MAX_DURATION = 7.0

        for i, shot in enumerate(shots):
            e = max(0.0, min(1.0, energy[i]))
            # Invert: high energy = short duration
            target = MAX_DURATION - e * (MAX_DURATION - MIN_DURATION)

            # Respect recommended_duration from the preset if available.
            preset = catalog.get(shot.camera, {})
            rec = preset.get("recommended_duration", 0)
            if rec > 0:
                # Blend: 60 % energy-driven, 40 % preset recommendation.
                target = 0.6 * target + 0.4 * rec

            shot.duration_seconds = round(target, 1)

        logger.info("Applied energy-curve rhythm to %d shots.", len(shots))

    def _apply_heuristic_rhythm(
        self,
        shots: List[ShotSpec],
        catalog: Dict[str, dict],
    ) -> None:
        """Apply a dramatic-arc heuristic when no energy curve is available.

        The arc shape:
          - **Intro** (first ~15 %): medium-length establishing shots.
          - **Rising action** (~15-65 %): alternating short/long,
            trending shorter.
          - **Climax** (~65-80 %): shortest durations, fastest cuts.
          - **Denouement** (~80-100 %): durations lengthen again.
        """
        n = len(shots)

        for i, shot in enumerate(shots):
            frac = i / max(n - 1, 1)  # position in [0, 1]
            scene_type = shot.metadata.get("scene_type", "general")

            # Base duration from a dramatic arc curve.
            # cos-based wave: peaks at start/end, dip at 0.75.
            arc = 0.5 + 0.5 * math.cos(2.0 * math.pi * (frac - 0.75))
            # Scale to [2.5, 6.5]
            base = 2.5 + arc * 4.0

            # Scene-type modifiers.
            if scene_type == "transition":
                base = 3.0  # title cards are short and fixed
            elif scene_type == "action":
                base *= 0.75  # tighter cuts
            elif scene_type == "establishing":
                base *= 1.15  # let the audience absorb the vista
            elif scene_type == "emotional":
                base *= 1.10  # breathing room

            # Intensity modifier: high-intensity cameras feel dizzying if
            # they run too long.
            preset = catalog.get(shot.camera, {})
            intensity = preset.get("motion_intensity", "medium")
            if intensity == "high":
                base = min(base, 4.0)
            elif intensity == "none":
                # Static shots can be any length.
                base = max(base, 3.5)

            # Respect preset recommended duration as a soft anchor.
            rec = preset.get("recommended_duration", 0)
            if rec > 0:
                base = 0.7 * base + 0.3 * rec

            # Alternation nudge: adjacent shots should not be the same
            # length -- nudge odd-indexed shots slightly shorter.
            if i % 2 == 1:
                base *= 0.9

            shot.duration_seconds = round(max(2.0, min(8.0, base)), 1)

        logger.info("Applied heuristic rhythm to %d shots.", n)

    # ------------------------------------------------------------------
    # generate_trajectories
    # ------------------------------------------------------------------

    def generate_trajectories(
        self,
        shots: List[ShotSpec],
    ) -> Dict[int, Any]:
        """Generate ``CameraTrajectory`` objects for each shot.

        Each shot's ``camera`` field is interpreted as a preset ID (from
        ``camera_presets.json`` or the built-in trajectory system).  The
        method tries ``get_trajectory_for_preset`` first, falling back to
        ``CameraTrajectory.from_preset`` for bare trajectory names (e.g.
        ``"dolly_in"``).

        Args:
            shots: List of ``ShotSpec`` objects. Their ``camera`` and
                ``duration_seconds`` fields must be populated (call
                ``plan_cameras`` and ``plan_rhythm`` first).

        Returns:
            A dict mapping shot index to a ``CameraTrajectory`` instance.
        """
        from animatediff.core.camera_ctrl import (
            CameraTrajectory,
            get_trajectory_for_preset,
        )

        trajectories: Dict[int, Any] = {}

        for i, shot in enumerate(shots):
            preset_id = self._normalize_camera_id(shot.camera)
            fps = self.default_fps
            num_frames = shot.num_frames or self._duration_to_frames(
                shot.duration_seconds, fps
            )
            width = shot.width or self.default_width
            height = shot.height or self.default_height

            try:
                traj = get_trajectory_for_preset(
                    preset_id,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                )
            except KeyError:
                # Fallback: try as a direct trajectory preset name.
                try:
                    traj = CameraTrajectory.from_preset(
                        preset_id,
                        num_frames=num_frames,
                        width=width,
                        height=height,
                    )
                except KeyError:
                    logger.warning(
                        f"[Shot {i}] Unknown camera preset '{preset_id}'; "
                        f"falling back to 'static'."
                    )
                    traj = CameraTrajectory.from_preset(
                        "static",
                        num_frames=num_frames,
                        width=width,
                        height=height,
                    )

            trajectories[i] = traj
            logger.debug(
                f"[Shot {i}] Trajectory: {traj.preset_name or preset_id}, "
                f"{num_frames} frames, {width}x{height}"
            )

        logger.info(
            "Generated %d camera trajectories for %d shots.",
            len(trajectories), len(shots),
        )
        return trajectories

    # ------------------------------------------------------------------
    # validate_plan
    # ------------------------------------------------------------------

    def validate_plan(self, shots: List[ShotSpec]) -> List[str]:
        """Check a shot plan for common cinematic issues.

        Returns a list of human-readable warning strings. An empty list
        means the plan looks good.

        Checks performed:
        1. **Consecutive duplicate cameras** -- the same preset used back
           to back feels monotonous.
        2. **Uniform duration** -- if all shots are the same length the
           edit has no rhythm.
        3. **No intensity variety** -- a sequence where every shot has the
           same motion_intensity feels flat.
        4. **Overlong high-intensity shots** -- high-motion presets running
           longer than 5 s can cause motion sickness.
        5. **Missing camera assignment** -- shots with empty camera fields.

        Args:
            shots: The shot list to validate.

        Returns:
            List of warning strings (may be empty).
        """
        warnings: List[str] = []
        catalog = self._load_preset_catalog()

        if not shots:
            return warnings

        # 1. Consecutive duplicate cameras
        for i in range(1, len(shots)):
            if (shots[i].camera == shots[i - 1].camera
                    and shots[i].camera
                    and shots[i].camera != "static"):
                warnings.append(
                    f"Shots {i-1} and {i} both use camera '{shots[i].camera}'. "
                    f"Consider varying camera movement for visual interest."
                )

        # 2. Uniform duration
        durations = [s.duration_seconds for s in shots]
        if len(set(round(d, 1) for d in durations)) == 1 and len(shots) > 2:
            warnings.append(
                f"All {len(shots)} shots have the same duration "
                f"({durations[0]:.1f}s). Varying duration creates better "
                f"cinematic rhythm."
            )

        # 3. No intensity variety
        intensities = []
        for shot in shots:
            preset = catalog.get(shot.camera, {})
            intensities.append(preset.get("motion_intensity", "unknown"))
        unique_intensities = set(intensities)
        if len(unique_intensities) == 1 and len(shots) > 3:
            warnings.append(
                f"All shots have '{intensities[0]}' motion intensity. "
                f"Mixing intensity levels (low/medium/high) creates a "
                f"more dynamic viewing experience."
            )

        # 4. Overlong high-intensity shots
        for i, shot in enumerate(shots):
            preset = catalog.get(shot.camera, {})
            if (preset.get("motion_intensity") == "high"
                    and shot.duration_seconds > 5.0):
                warnings.append(
                    f"Shot {i} uses high-intensity camera '{shot.camera}' "
                    f"for {shot.duration_seconds:.1f}s. Consider keeping "
                    f"high-motion shots under 5s to avoid viewer fatigue."
                )

        # 5. Missing camera assignment
        for i, shot in enumerate(shots):
            if not shot.camera:
                warnings.append(
                    f"Shot {i} has no camera preset assigned."
                )

        return warnings

    # ------------------------------------------------------------------
    # Convenience: full orchestration pipeline
    # ------------------------------------------------------------------

    def orchestrate(
        self,
        shots: List[ShotSpec],
        energy_curve: Optional[List[float]] = None,
    ) -> Tuple[List[ShotSpec], Dict[int, Any], List[str]]:
        """Run the complete orchestration pipeline in one call.

        Equivalent to calling ``plan_cameras``, ``plan_rhythm``,
        ``generate_trajectories``, and ``validate_plan`` in sequence.

        Args:
            shots: List of ``ShotSpec`` objects (modified in place).
            energy_curve: Optional energy curve for rhythm planning.

        Returns:
            A tuple of ``(shots, trajectories, warnings)`` where
            *trajectories* maps shot index to ``CameraTrajectory`` and
            *warnings* is a list of validation messages.
        """
        self.plan_cameras(shots)
        self.plan_rhythm(shots, energy_curve=energy_curve)
        trajectories = self.generate_trajectories(shots)
        warnings = self.validate_plan(shots)

        if warnings:
            for w in warnings:
                logger.warning(f"[Orchestrator] {w}")
        else:
            logger.info("[Orchestrator] Plan validated -- no issues found.")

        return shots, trajectories, warnings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Map informal camera names (as typed by storyboard authors) to
    # canonical preset IDs that the trajectory system understands.
    _CAMERA_ALIASES: Dict[str, str] = {
        # Informal name            -> canonical preset ID
        "zoom in":                  "zoom_in",
        "zoom out":                 "zoom_out",
        "pan left":                 "pan_left",
        "pan right":                "pan_right",
        "tilt up":                  "tilt_up",
        "tilt down":                "tilt_down",
        "dolly in":                 "dolly_in",
        "dolly out":                "dolly_out",
        "dolly forward":            "dolly_in",
        "dolly backward":           "dolly_out",
        "dolly back":               "dolly_out",
        "push in":                  "slow_push_in",
        "push forward":             "dolly_in",
        "pull back":                "pull_back_reveal",
        "orbit":                    "orbit_cw",
        "orbit left":               "orbit_ccw",
        "orbit right":              "orbit_cw",
        "crane up":                 "crane_up",
        "crane down":               "crane_down",
        "truck left":               "truck_left",
        "truck right":              "truck_right",
        "fly forward":              "fly_forward",
        "fly through":              "fly_through_clouds",
        "spiral":                   "spiral_ascend",
        "dutch":                    "dutch_roll",
        "dutch angle":              "dutch_angle_tension",
        "handheld":                 "sword_fight_handheld",
        "whip pan":                 "whip_pan",
        "time lapse":               "time_lapse_pan",
    }

    @classmethod
    def _normalize_camera_id(cls, camera: str) -> str:
        """Convert informal camera names to canonical preset IDs.

        Tries exact match, then lowered match, then underscore-normalized
        match. Returns the original string if no alias is found.
        """
        if not camera:
            return "static"

        # Try direct alias lookup (case-insensitive)
        lowered = camera.lower().strip()
        if lowered in cls._CAMERA_ALIASES:
            return cls._CAMERA_ALIASES[lowered]

        # Try replacing spaces with underscores
        underscored = lowered.replace(" ", "_").replace("-", "_")
        if underscored != lowered:
            # Check if this is already a valid preset name
            return underscored

        return camera

    @staticmethod
    def _duration_to_frames(duration_seconds: float, fps: int) -> int:
        """Convert duration to frame count aligned to 4N+1 (Wan model preference)."""
        raw = int(duration_seconds * fps)
        aligned = ((raw - 1) // 4) * 4 + 1
        return max(17, aligned)
