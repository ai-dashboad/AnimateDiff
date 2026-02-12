"""
Director Engine — unified multi-reference orchestrator for SeedAnce-style generation.

Implements the "@" reference system: binds images, videos, and audio to semantic
roles (character, camera, style, soundtrack, voice) and routes them to the optimal
backend + postprocess pipeline for each shot.

This is the central coordinator that replaces ad-hoc per-shot wiring with a
declarative reference-based approach:

    1. plan()    — Parse storyboard + references → per-shot ExecutionPlan
    2. execute() — Run plan shot-by-shot with quality gates + chaining
    3. compose() — Assemble shots into final video with audio + transitions

Usage:
    from animatediff.core.director import DirectorEngine

    director = DirectorEngine()
    plan = director.plan(storyboard, references)
    shots = director.execute(plan)
    director.compose(shots, output_path="output/trailer.mp4")
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.reference_parser import Reference, ReferenceSet, ReferenceParser
from animatediff.core.story_engine import ShotSpec, StoryBoard

logger = logging.getLogger(__name__)

__all__ = [
    "ShotPlan",
    "ExecutionPlan",
    "ShotResult",
    "DirectorEngine",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShotPlan:
    """Execution plan for a single shot.

    Contains all information needed to generate one shot: which backend to use,
    what references to inject, generation kwargs, and quality gates.
    """
    shot_id: int = 0
    shot_spec: Optional[ShotSpec] = None

    # Backend routing
    backend_name: str = "wan22"
    mode: str = "t2v"  # t2v, i2v, continuation, reference_to_video, audio_driven

    # Generation kwargs (passed to backend.generate())
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Bound references for this shot
    character_refs: List[Reference] = field(default_factory=list)
    camera_ref: Optional[Reference] = None
    style_ref: Optional[Reference] = None
    audio_refs: List[Reference] = field(default_factory=list)

    # Quality gate
    quality_threshold: float = 0.0  # 0 = no gate, >0 = retry if below
    max_attempts: int = 1

    # Chaining
    chain_from_previous: bool = False
    chain_method: str = "last_frame_i2v"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_image_ref(self) -> bool:
        return bool(self.character_refs) or self.style_ref is not None

    @property
    def has_audio_ref(self) -> bool:
        return bool(self.audio_refs)


@dataclass
class ExecutionPlan:
    """Complete execution plan for a storyboard."""
    title: str = ""
    shot_plans: List[ShotPlan] = field(default_factory=list)
    global_refs: Optional[ReferenceSet] = None
    acceleration_preset: str = "balanced"  # fast, balanced, quality
    output_dir: str = "output/"

    # Global generation params
    default_width: int = 832
    default_height: int = 480
    default_fps: int = 24
    default_steps: int = 50
    default_guidance: float = 5.0

    @property
    def num_shots(self) -> int:
        return len(self.shot_plans)

    @property
    def total_duration(self) -> float:
        return sum(
            (sp.shot_spec.duration_seconds if sp.shot_spec else 0.0)
            for sp in self.shot_plans
        )

    def summary(self) -> str:
        backends = {}
        for sp in self.shot_plans:
            backends[sp.backend_name] = backends.get(sp.backend_name, 0) + 1
        backend_str = ", ".join(f"{k}:{v}" for k, v in backends.items())
        return (
            f"ExecutionPlan '{self.title}': {self.num_shots} shots, "
            f"~{self.total_duration:.1f}s, backends=[{backend_str}], "
            f"accel={self.acceleration_preset}"
        )


@dataclass
class ShotResult:
    """Result of generating a single shot."""
    shot_id: int = 0
    output: Optional[VideoOutput] = None
    output_path: str = ""
    score: float = 0.0
    attempts: int = 1
    generation_time: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.output is not None and self.error is None


# ---------------------------------------------------------------------------
# Backend routing logic
# ---------------------------------------------------------------------------

# Mode → preferred backend mapping
_MODE_BACKEND_MAP = {
    "t2v": ["wan22", "hunyuan", "cogvideo", "ltx", "wan"],
    "i2v": ["wan22", "wan22_vace", "skyreels_v3"],
    "audio_driven": ["wan22_s2v", "ltx2"],
    "continuation": ["wan22_vace"],
    "reference_to_video": ["wan22_vace", "skyreels_v3"],
    "animate": ["wan22_animate"],
    "joint_av": ["ltx2"],
}


def _select_backend(
    mode: str,
    preferred: str = "",
    available: Optional[List[str]] = None,
) -> str:
    """Select the best backend for a given generation mode.

    Args:
        mode: Generation mode (t2v, i2v, audio_driven, etc.).
        preferred: User-preferred backend (used if compatible).
        available: List of available/loaded backends.

    Returns:
        Backend name string.
    """
    candidates = _MODE_BACKEND_MAP.get(mode, ["wan22"])

    # If user specified a preferred backend and it's compatible, use it
    if preferred and preferred in candidates:
        return preferred
    if preferred and available and preferred in available:
        return preferred

    # If available list is given, filter candidates
    if available:
        for c in candidates:
            if c in available:
                return c

    # Return first preference
    return candidates[0]


def _determine_mode(shot_spec: ShotSpec, refs: List[Reference]) -> str:
    """Determine generation mode from shot spec and references.

    Logic:
      - If audio references exist and shot has lip_sync → audio_driven
      - If character image references exist → i2v
      - If shot has explicit mode in metadata → use it
      - Otherwise → t2v
    """
    explicit_mode = shot_spec.metadata.get("mode", "")
    if explicit_mode:
        return explicit_mode

    has_char_image = any(
        r.role in ("character", "face") and r.type == "image"
        for r in refs
    )
    has_audio = any(r.type == "audio" for r in refs)

    if has_audio and shot_spec.lip_sync:
        return "audio_driven"
    if has_char_image:
        return "i2v"
    return "t2v"


# ---------------------------------------------------------------------------
# Director Engine
# ---------------------------------------------------------------------------

class DirectorEngine:
    """SeedAnce-style unified generation orchestrator.

    The Director manages the full lifecycle of multi-shot video generation:
      1. Planning: Route references to backends per shot
      2. Execution: Generate with quality gates + shot chaining
      3. Composition: Assemble final video with audio + transitions

    Designed to work with any combination of backends and postprocessors.
    """

    def __init__(
        self,
        default_backend: str = "wan22",
        quality_threshold: float = 0.0,
        max_attempts: int = 1,
        chain_method: str = "last_frame_i2v",
    ):
        """
        Args:
            default_backend: Fallback backend when no better match exists.
            quality_threshold: Global quality gate (0 = disabled).
            max_attempts: Max retries per shot when quality gate is active.
            chain_method: Default shot chaining method.
        """
        self.default_backend = default_backend
        self.quality_threshold = quality_threshold
        self.max_attempts = max_attempts
        self.chain_method = chain_method
        self._loaded_backends: Dict[str, BasePipeline] = {}

        logger.info(
            f"DirectorEngine initialized: default_backend={default_backend}, "
            f"quality_threshold={quality_threshold}, chain_method={chain_method}"
        )

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        storyboard: StoryBoard,
        references: Optional[ReferenceSet] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Parse storyboard + references into a per-shot execution plan.

        Routes each shot to the optimal backend based on its references:
          - Character images → I2V backend (Wan TI2V-5B, VACE, SkyReels)
          - Camera videos → camera motion extraction
          - Audio files → beat sync timing + lip sync
          - Style images → style harmonization reference

        Args:
            storyboard: Parsed storyboard with shots and characters.
            references: Global reference set (shared across all shots).
            config: Optional config overrides (resolution, fps, steps, etc.).

        Returns:
            ExecutionPlan with per-shot ShotPlans.
        """
        config = config or {}
        references = references or ReferenceSet()

        # Parse character definitions into references if not already present
        if storyboard.characters:
            parser = ReferenceParser()
            char_refs = parser.parse_characters(
                {name: {"image_path": c.image_path, "description": c.description,
                         "lora_path": c.lora_path, "lora_scale": c.lora_scale}
                 for name, c in storyboard.characters.items()}
            )
            for ref in char_refs.refs:
                references.add(ref)

        exec_plan = ExecutionPlan(
            title=storyboard.title,
            global_refs=references,
            output_dir=config.get("output_dir", "output/"),
            acceleration_preset=config.get("acceleration_preset", "balanced"),
            default_width=config.get("width", 832),
            default_height=config.get("height", 480),
            default_fps=config.get("fps", 24),
            default_steps=config.get("num_inference_steps", 50),
            default_guidance=config.get("guidance_scale", 5.0),
        )

        available_backends = config.get("available_backends", None)

        for shot in storyboard.shots:
            shot_plan = self._plan_shot(
                shot, references, exec_plan, available_backends,
            )
            exec_plan.shot_plans.append(shot_plan)

        logger.info(f"Planned: {exec_plan.summary()}")
        return exec_plan

    def _plan_shot(
        self,
        shot: ShotSpec,
        global_refs: ReferenceSet,
        plan: ExecutionPlan,
        available_backends: Optional[List[str]],
    ) -> ShotPlan:
        """Create a ShotPlan for a single shot."""
        # Gather references for this shot
        shot_refs = self._gather_shot_references(shot, global_refs)

        # Determine mode
        mode = _determine_mode(shot, shot_refs)

        # Select backend
        preferred = shot.metadata.get("backend", self.default_backend)
        backend = _select_backend(mode, preferred, available_backends)

        # Separate refs by role
        char_refs = [r for r in shot_refs if r.role in ("character", "face")]
        camera_refs = [r for r in shot_refs if r.role == "camera"]
        style_refs = [r for r in shot_refs if r.role == "style"]
        audio_refs = [r for r in shot_refs if r.type == "audio"]

        # Build generation kwargs
        gen_kwargs = self._build_gen_kwargs(shot, plan, char_refs, camera_refs, mode)

        # Quality threshold: per-shot override or global
        threshold = shot.metadata.get("quality_threshold", self.quality_threshold)
        max_att = shot.metadata.get("max_attempts", self.max_attempts)

        # Chaining
        chain = shot.shot_id > 0 and mode not in ("title_card", "end_card")

        return ShotPlan(
            shot_id=shot.shot_id,
            shot_spec=shot,
            backend_name=backend,
            mode=mode,
            gen_kwargs=gen_kwargs,
            character_refs=char_refs,
            camera_ref=camera_refs[0] if camera_refs else None,
            style_ref=style_refs[0] if style_refs else None,
            audio_refs=audio_refs,
            quality_threshold=threshold,
            max_attempts=max_att,
            chain_from_previous=chain,
            chain_method=self.chain_method,
        )

    def _gather_shot_references(
        self,
        shot: ShotSpec,
        global_refs: ReferenceSet,
    ) -> List[Reference]:
        """Gather references relevant to a specific shot.

        Matches global references to the shot's character list, plus any
        shot-level references from ShotSpec.references.
        """
        refs: List[Reference] = []

        # Shot-level references (from storyboard JSON "references" field)
        for ref in getattr(shot, "references", []):
            if isinstance(ref, Reference):
                refs.append(ref)
            elif isinstance(ref, dict):
                parser = ReferenceParser()
                parsed = parser.parse_dict_list([ref])
                refs.extend(parsed.refs)

        # Match global character references to shot's character list
        for char_name in shot.characters:
            for gref in global_refs.refs:
                if gref.role in ("character", "face") and gref.tag == char_name:
                    if gref not in refs:
                        refs.append(gref)

        # Global style references apply to all shots
        for gref in global_refs.style_refs():
            if gref not in refs:
                refs.append(gref)

        # Global audio references (soundtrack) apply to all shots
        for gref in global_refs.audio_refs():
            if gref.role in ("soundtrack", "sfx") and gref not in refs:
                refs.append(gref)

        return refs

    def _build_gen_kwargs(
        self,
        shot: ShotSpec,
        plan: ExecutionPlan,
        char_refs: List[Reference],
        camera_refs: List[Reference],
        mode: str,
    ) -> Dict[str, Any]:
        """Build generation kwargs for a shot."""
        kwargs: Dict[str, Any] = {
            "prompt": shot.prompt,
            "negative_prompt": shot.negative_prompt,
            "width": shot.width or plan.default_width,
            "height": shot.height or plan.default_height,
            "num_frames": shot.num_frames or int(shot.duration_seconds * plan.default_fps),
            "num_inference_steps": plan.default_steps,
            "guidance_scale": plan.default_guidance,
            "seed": shot.seed,
        }

        # I2V: use first character image as reference
        if mode == "i2v" and char_refs:
            ref_path = char_refs[0].path
            if Path(ref_path).exists():
                kwargs["image"] = Image.open(ref_path).convert("RGB")
            else:
                logger.warning(f"Character reference not found: {ref_path}, falling back to t2v")

        # Camera reference: extract motion params
        if camera_refs:
            kwargs["_camera_ref"] = camera_refs[0].path

        return kwargs

    # ------------------------------------------------------------------
    # Phase 2: Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: ExecutionPlan,
        backend_loader: Optional[Callable[[str], BasePipeline]] = None,
        scorer: Optional[Any] = None,
        chainer: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        dry_run: bool = False,
    ) -> List[ShotResult]:
        """Execute plan shot-by-shot with quality gates + chaining.

        Args:
            plan: The execution plan from plan().
            backend_loader: Callable that loads a backend by name.
                If None, uses animatediff.backends.get_backend().
            scorer: Optional VideoQualityScorer instance for quality gates.
            chainer: Optional ShotChainer instance for cross-shot consistency.
            progress_callback: Optional (shot_idx, total, status_msg) callback.
            dry_run: If True, log plan but don't generate.

        Returns:
            List of ShotResult, one per shot.
        """
        results: List[ShotResult] = []
        output_dir = Path(plan.output_dir) / "shots"
        output_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            logger.info(f"DRY RUN: {plan.summary()}")
            for sp in plan.shot_plans:
                logger.info(
                    f"  Shot {sp.shot_id}: backend={sp.backend_name}, mode={sp.mode}, "
                    f"refs={len(sp.character_refs)}char/{1 if sp.camera_ref else 0}cam/"
                    f"{len(sp.audio_refs)}aud, quality_gate={sp.quality_threshold}"
                )
                results.append(ShotResult(shot_id=sp.shot_id))
            return results

        current_backend_name = ""
        current_backend = None

        for i, shot_plan in enumerate(plan.shot_plans):
            if progress_callback:
                progress_callback(i, plan.num_shots, f"Generating shot {shot_plan.shot_id}")

            # Load backend if needed
            if shot_plan.backend_name != current_backend_name:
                current_backend = self._load_backend(
                    shot_plan.backend_name, backend_loader,
                )
                current_backend_name = shot_plan.backend_name

            if current_backend is None:
                results.append(ShotResult(
                    shot_id=shot_plan.shot_id,
                    error=f"Failed to load backend: {shot_plan.backend_name}",
                ))
                continue

            # Apply chaining from previous shot
            gen_kwargs = dict(shot_plan.gen_kwargs)
            if shot_plan.chain_from_previous and chainer and results:
                prev_result = results[-1]
                if prev_result.success and prev_result.output_path:
                    chain_overrides = chainer.chain(
                        shot_data={"mode": shot_plan.mode},
                        prev_output_path=prev_result.output_path,
                        backend_name=shot_plan.backend_name,
                    )
                    gen_kwargs.update(chain_overrides)

            # Remove internal keys not understood by backends
            gen_kwargs.pop("_camera_ref", None)

            # Generate with quality gate
            result = self._generate_with_gate(
                shot_plan, current_backend, gen_kwargs,
                scorer, output_dir,
            )
            results.append(result)

            if result.success:
                logger.info(
                    f"Shot {shot_plan.shot_id} complete: "
                    f"score={result.score:.3f}, "
                    f"time={result.generation_time:.1f}s, "
                    f"attempts={result.attempts}"
                )
            else:
                logger.error(f"Shot {shot_plan.shot_id} failed: {result.error}")

        return results

    def _generate_with_gate(
        self,
        shot_plan: ShotPlan,
        backend: BasePipeline,
        gen_kwargs: Dict[str, Any],
        scorer: Optional[Any],
        output_dir: Path,
    ) -> ShotResult:
        """Generate a shot with optional quality-gated retries."""
        best_result: Optional[ShotResult] = None

        for attempt in range(1, shot_plan.max_attempts + 1):
            t0 = time.time()
            try:
                output = backend.generate(**gen_kwargs)
            except Exception as e:
                logger.warning(f"Shot {shot_plan.shot_id} attempt {attempt} failed: {e}")
                if attempt == shot_plan.max_attempts:
                    return ShotResult(
                        shot_id=shot_plan.shot_id,
                        attempts=attempt,
                        error=str(e),
                    )
                continue

            gen_time = time.time() - t0

            # Save output
            out_path = str(output_dir / f"shot_{shot_plan.shot_id:04d}.mp4")
            try:
                backend.save(output, out_path, fps=output.fps)
            except Exception as e:
                logger.warning(f"Failed to save shot {shot_plan.shot_id}: {e}")

            # Score if scorer available
            score = 0.0
            if scorer and output.frames:
                try:
                    prompt = gen_kwargs.get("prompt", "")
                    vs = scorer.score_video(output.frames, prompt)
                    score = vs.overall
                except Exception as e:
                    logger.warning(f"Scoring failed for shot {shot_plan.shot_id}: {e}")

            result = ShotResult(
                shot_id=shot_plan.shot_id,
                output=output,
                output_path=out_path,
                score=score,
                attempts=attempt,
                generation_time=gen_time,
            )

            # Quality gate check
            if not shot_plan.quality_threshold or score >= shot_plan.quality_threshold:
                return result

            # Keep best so far
            if best_result is None or score > best_result.score:
                best_result = result

            logger.info(
                f"Shot {shot_plan.shot_id} attempt {attempt}: "
                f"score={score:.3f} < threshold={shot_plan.quality_threshold:.3f}, retrying"
            )

            # Adjust seed for retry
            if gen_kwargs.get("seed", -1) >= 0:
                gen_kwargs["seed"] += 1

        return best_result or ShotResult(
            shot_id=shot_plan.shot_id, error="All attempts below quality threshold"
        )

    def _load_backend(
        self,
        name: str,
        loader: Optional[Callable[[str], BasePipeline]],
    ) -> Optional[BasePipeline]:
        """Load a backend, caching loaded instances."""
        if name in self._loaded_backends:
            return self._loaded_backends[name]

        try:
            if loader:
                backend = loader(name)
            else:
                from animatediff.backends import get_backend
                backend_cls = get_backend(name)
                backend = backend_cls.load()
            self._loaded_backends[name] = backend
            logger.info(f"Loaded backend: {name}")
            return backend
        except Exception as e:
            logger.error(f"Failed to load backend '{name}': {e}")
            return None

    # ------------------------------------------------------------------
    # Phase 3: Composition
    # ------------------------------------------------------------------

    def compose(
        self,
        results: List[ShotResult],
        storyboard: Optional[StoryBoard] = None,
        audio_tracks: Optional[Dict[str, str]] = None,
        output_path: str = "output/final.mp4",
        fps: int = 24,
        color_lut: str = "",
    ) -> str:
        """Assemble generated shots into final video with transitions and audio.

        Args:
            results: List of ShotResult from execute().
            storyboard: Optional storyboard for transition info.
            audio_tracks: Dict of audio track name → path
                (e.g., {"bgm": "bgm.mp3", "narration_0": "narr_0.wav"}).
            output_path: Final output file path.
            fps: Output framerate.
            color_lut: Optional color LUT name for grading.

        Returns:
            Path to the composed video file.
        """
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor()

        # Collect successful shot outputs
        shot_paths = []
        shot_specs = []
        for r in results:
            if r.success and r.output_path:
                shot_paths.append(r.output_path)
                # Find matching ShotSpec for transition info
                if storyboard:
                    matching = [s for s in storyboard.shots if s.shot_id == r.shot_id]
                    shot_specs.append(matching[0] if matching else None)
                else:
                    shot_specs.append(None)

        if not shot_paths:
            logger.error("No successful shots to compose")
            return ""

        # Build timed audio list for compositor
        timed_audio = []
        if audio_tracks:
            # Narration tracks: keyed as "narration_<shot_id>"
            cumulative_time = 0.0
            for i, r in enumerate(results):
                if not r.success:
                    continue
                narr_key = f"narration_{r.shot_id}"
                if narr_key in audio_tracks:
                    timed_audio.append({
                        "path": audio_tracks[narr_key],
                        "start_time": cumulative_time,
                        "volume": 1.0,
                    })
                if r.output and r.output.frames:
                    cumulative_time += len(r.output.frames) / fps

        # Compose
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            bgm_path = audio_tracks.get("bgm", "") if audio_tracks else ""
            compositor.compose_with_timed_audio(
                shot_paths=shot_paths,
                timed_narration=timed_audio,
                bgm_path=bgm_path,
                bgm_volume=0.15,
                output_path=output_path,
                fps=fps,
            )

            # Apply color LUT if specified
            if color_lut:
                compositor.apply_color_lut(output_path, output_path, lut_name=color_lut)

            logger.info(f"Final video composed: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Composition failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run(
        self,
        storyboard: StoryBoard,
        references: Optional[ReferenceSet] = None,
        config: Optional[Dict[str, Any]] = None,
        output_path: str = "output/final.mp4",
        dry_run: bool = False,
    ) -> str:
        """End-to-end: plan → execute → compose.

        Convenience method that chains all three phases.

        Returns:
            Path to the final composed video.
        """
        plan = self.plan(storyboard, references, config)

        if dry_run:
            self.execute(plan, dry_run=True)
            return ""

        results = self.execute(plan)
        return self.compose(results, storyboard, output_path=output_path)
