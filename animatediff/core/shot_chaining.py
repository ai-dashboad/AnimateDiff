"""
Shot Chaining — cross-shot character consistency via frame-based continuity.

Ensures visual consistency between consecutive shots in a multi-shot storyboard
by feeding the last frame(s) of shot N into shot N+1 as conditioning input.

Three chaining methods:
  1. last_frame_i2v:     Extract last frame of prev shot, use as I2V reference for next.
                         Works with any backend that supports image= param (wan22, etc).
  2. vace_continuation:  Use VACE continuation mode with overlap frames from prev shot.
                         Requires the wan22_vace backend.
  3. style_transfer:     Extract a style embedding from prev shot and blend into the
                         next shot's prompt (lightweight, prompt-only approach).

The chainer is called DURING generation (Phase 2), not after. The ShotScheduler
generates shots sequentially, and the chainer modifies each shot's generation
params before the backend is invoked.

Usage:
    from animatediff.core.shot_chaining import ShotChainer

    chainer = ShotChainer(method="last_frame_i2v")
    # After generating shot N, before generating shot N+1:
    gen_kwargs = chainer.chain(
        shot=next_shot_spec,
        prev_result=prev_shot_result,
        backend_name="wan22",
    )
    # gen_kwargs now contains image=<last_frame> and any mode adjustments
"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any

from PIL import Image

logger = logging.getLogger(__name__)

# Shot modes that should NEVER be chained (static graphics, not generated video)
NON_CHAINABLE_MODES = frozenset({
    "title_card",
    "end_card",
    "credits",
    "interstitial",
    "chapter_card",
})


class ShotChainer:
    """Chains shots together for cross-shot visual consistency.

    Designed to be called between sequential shot generations. The chainer
    inspects the previous shot's output and modifies the next shot's generation
    parameters to maintain character/scene continuity.
    """

    def __init__(
        self,
        method: str = "last_frame_i2v",
        overlap_frames: int = 4,
        style_weight: float = 0.5,
    ):
        """
        Args:
            method: Chaining strategy.
                - "last_frame_i2v":     Use last frame of prev shot as I2V reference.
                - "vace_continuation":  Use VACE continuation mode with overlap frames.
                - "style_transfer":     Prompt-based style consistency (no image ref).
            overlap_frames: Number of overlap frames for VACE continuation mode.
                            Typically 1-8; higher = stronger continuity but fewer
                            new frames generated per shot.
            style_weight: Blending weight for style_transfer method (0.0-1.0).
        """
        valid_methods = ("last_frame_i2v", "vace_continuation", "style_transfer")
        if method not in valid_methods:
            raise ValueError(f"Unknown chaining method: {method}. Supported: {valid_methods}")

        self.method = method
        self.overlap_frames = overlap_frames
        self.style_weight = style_weight

        logger.info(
            f"ShotChainer initialized: method={method}, "
            f"overlap_frames={overlap_frames}, style_weight={style_weight}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_chain(self, shot_data: dict, prev_output_path: Optional[str] = None) -> bool:
        """Determine whether this shot should be chained from the previous one.

        A shot is chained when ALL of the following are true:
          1. The shot's "chain_from_previous" field is True (or absent, defaults True
             for non-first shots in the storyboard).
          2. The shot's mode is not in NON_CHAINABLE_MODES.
          3. A previous shot output exists to chain from.

        Args:
            shot_data: Raw shot dict from the storyboard (not ShotSpec — this allows
                       checking fields like "mode" and "chain_from_previous" that may
                       not be on ShotSpec).
            prev_output_path: Path to the previous shot's output video/frames.
                              If None or empty, chaining is impossible.

        Returns:
            True if this shot should be chained.
        """
        # No previous output to chain from
        if not prev_output_path:
            return False

        # Explicitly disabled
        if shot_data.get("chain_from_previous") is False:
            return False

        # Non-chainable mode (title cards, end cards, etc.)
        mode = shot_data.get("mode", "t2v")
        if mode in NON_CHAINABLE_MODES:
            logger.debug(f"Shot mode '{mode}' is non-chainable, skipping chain.")
            return False

        return True

    def chain(
        self,
        shot_data: dict,
        prev_output_path: str,
        prev_frames: Optional[List[Image.Image]] = None,
        backend_name: str = "wan22",
    ) -> Dict[str, Any]:
        """Compute chaining overrides for the next shot's generation kwargs.

        This method does NOT modify shot_data in-place. Instead it returns a
        dict of keyword arguments to merge into the backend.generate() call.

        Args:
            shot_data: The raw shot dict from the storyboard for the NEXT shot.
            prev_output_path: File path to the previous shot's rendered video.
            prev_frames: Optional pre-loaded frames from the previous shot. If
                         None, frames are loaded from prev_output_path on demand.
            backend_name: Name of the active backend ("wan22", "wan22_vace", etc.).

        Returns:
            Dict of kwargs to merge into generate(). Possible keys:
              - image: PIL.Image (last frame reference for I2V)
              - mode: str (e.g., "continuation" for VACE)
              - source_frames: List[PIL.Image] (for VACE continuation)
              - overlap_frames: int (for VACE continuation)
              - prompt_suffix: str (for style_transfer — caller should append)

        Raises:
            ValueError: If the chaining method is incompatible with the backend.
        """
        overrides: Dict[str, Any] = {}

        if self.method == "last_frame_i2v":
            overrides = self._chain_last_frame_i2v(
                prev_output_path, prev_frames, backend_name
            )

        elif self.method == "vace_continuation":
            overrides = self._chain_vace_continuation(
                prev_output_path, prev_frames, backend_name
            )

        elif self.method == "style_transfer":
            overrides = self._chain_style_transfer(
                prev_output_path, prev_frames
            )

        if overrides:
            logger.info(
                f"Chaining shot via '{self.method}': "
                f"overrides={list(overrides.keys())}"
            )

        return overrides

    def prepare_chain(
        self,
        shots: List[dict],
        shot_outputs: Dict[int, str],
        backend_name: str = "wan22",
    ) -> List[dict]:
        """Batch-prepare chaining overrides for a full shot list.

        Given a shot list and a mapping of already-generated shot_id -> output_path,
        modify remaining shots (in-place copies) to include chaining parameters.

        This is useful when the caller wants to pre-compute all chaining params
        before starting generation (e.g., for dry-run / planning).

        For each shot where chain_from_previous=True:
          - Extract last frame from previous shot's output
          - Set it as the reference_image for the current shot
          - Adjust mode to "i2v" if using last_frame_i2v method
          - Or set up VACE continuation params if using vace_continuation

        Args:
            shots: List of raw shot dicts from the storyboard.
            shot_outputs: Mapping of shot_id -> output file path for
                          already-generated shots.
            backend_name: Active backend name.

        Returns:
            Modified (deep-copied) shot list with chaining metadata injected
            into each shot's "chain_overrides" field.
        """
        result = []
        for i, shot in enumerate(shots):
            shot_copy = deepcopy(shot)

            if i == 0:
                # First shot has nothing to chain from
                result.append(shot_copy)
                continue

            prev_shot_id = shots[i - 1].get("shot_id", i - 1)
            prev_path = shot_outputs.get(prev_shot_id)

            if self.should_chain(shot_copy, prev_path):
                overrides = self.chain(
                    shot_data=shot_copy,
                    prev_output_path=prev_path,
                    backend_name=backend_name,
                )
                shot_copy["_chain_overrides"] = overrides
            else:
                shot_copy["_chain_overrides"] = {}

            result.append(shot_copy)

        return result

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def extract_reference_frame(
        self,
        video_path: str,
        position: str = "last",
    ) -> Image.Image:
        """Extract a single frame from a video file.

        Args:
            video_path: Path to an MP4/AVI/MOV/WEBM/GIF file or a directory
                        of frame images (PNG/JPG).
            position: Which frame to extract.
                - "first": First frame (index 0).
                - "last":  Last frame (index -1).
                - "middle": Middle frame (index len//2).

        Returns:
            PIL.Image.Image in RGB mode.

        Raises:
            ValueError: If no frames found or unsupported format.
        """
        frames = self._load_frames(video_path)
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        if position == "first":
            frame = frames[0]
        elif position == "last":
            frame = frames[-1]
        elif position == "middle":
            frame = frames[len(frames) // 2]
        else:
            raise ValueError(f"Unknown position '{position}'. Use 'first', 'last', or 'middle'.")

        return frame.convert("RGB")

    def create_transition_overlap(
        self,
        shot_a_path: str,
        shot_b_path: str,
        overlap_frames: int = 4,
    ) -> List[Image.Image]:
        """Create smooth overlap frames between two shots for VACE continuation.

        Extracts the last N frames from shot A and the first N frames from shot B,
        then blends them with a linear crossfade to create smooth transition frames.

        If shot B has not been generated yet (typical case during sequential
        generation), only the last N frames from shot A are returned — the
        VACE backend will generate the continuation from these.

        Args:
            shot_a_path: Path to the previous shot's video.
            shot_b_path: Path to the next shot's video (may not exist yet).
            overlap_frames: Number of overlap frames to produce.

        Returns:
            List of PIL.Image.Image — the overlap/conditioning frames.
        """
        frames_a = self._load_frames(shot_a_path)
        if not frames_a:
            raise ValueError(f"No frames found in shot A: {shot_a_path}")

        # Take the last N frames from shot A
        tail_a = frames_a[-overlap_frames:]

        # If shot B doesn't exist yet (normal during sequential generation),
        # just return shot A's tail frames as conditioning
        try:
            frames_b = self._load_frames(shot_b_path)
        except (ValueError, FileNotFoundError, OSError):
            logger.debug(
                f"Shot B not yet generated ({shot_b_path}), "
                f"returning {len(tail_a)} tail frames from shot A."
            )
            return tail_a

        if not frames_b:
            return tail_a

        # Both shots exist — create a crossfade blend
        head_b = frames_b[:overlap_frames]
        blended = self._crossfade_frames(tail_a, head_b)
        return blended

    # ------------------------------------------------------------------
    # Private: chaining strategies
    # ------------------------------------------------------------------

    def _chain_last_frame_i2v(
        self,
        prev_output_path: str,
        prev_frames: Optional[List[Image.Image]],
        backend_name: str,
    ) -> Dict[str, Any]:
        """Last-frame I2V: extract last frame, pass as image= to the backend.

        Works with:
          - wan22 (WanPipeline / WanImageToVideoPipeline accept image= kwarg)
          - wan22_vace (reference_to_video mode)
          - Any backend whose generate() accepts image=
        """
        if prev_frames:
            last_frame = prev_frames[-1].convert("RGB")
        else:
            last_frame = self.extract_reference_frame(prev_output_path, position="last")

        overrides: Dict[str, Any] = {"image": last_frame}

        # For VACE backend, explicitly set mode to reference_to_video
        if backend_name == "wan22_vace":
            overrides["mode"] = "reference_to_video"

        return overrides

    def _chain_vace_continuation(
        self,
        prev_output_path: str,
        prev_frames: Optional[List[Image.Image]],
        backend_name: str,
    ) -> Dict[str, Any]:
        """VACE continuation: use last N frames as overlap conditioning.

        Requires the wan22_vace backend. Falls back to last_frame_i2v if the
        active backend does not support continuation mode.
        """
        if backend_name != "wan22_vace":
            logger.warning(
                f"VACE continuation requested but backend is '{backend_name}' "
                f"(not wan22_vace). Falling back to last_frame_i2v."
            )
            return self._chain_last_frame_i2v(prev_output_path, prev_frames, backend_name)

        # Load source frames for continuation conditioning
        if prev_frames:
            source = prev_frames
        else:
            source = self._load_frames(prev_output_path)

        if not source:
            logger.warning(f"No frames from prev shot, cannot chain via VACE continuation.")
            return {}

        return {
            "mode": "continuation",
            "source_frames": source,
            "overlap_frames": self.overlap_frames,
        }

    def _chain_style_transfer(
        self,
        prev_output_path: str,
        prev_frames: Optional[List[Image.Image]],
    ) -> Dict[str, Any]:
        """Style transfer: extract dominant colors/mood from prev shot, append to prompt.

        This is a lightweight, backend-agnostic approach. It analyzes the last
        frame of the previous shot and generates a style description string
        that can be appended to the next shot's prompt.
        """
        if prev_frames:
            last_frame = prev_frames[-1].convert("RGB")
        else:
            last_frame = self.extract_reference_frame(prev_output_path, position="last")

        style_desc = self._analyze_frame_style(last_frame)

        if style_desc:
            return {"prompt_suffix": f", consistent with previous scene: {style_desc}"}
        return {}

    # ------------------------------------------------------------------
    # Private: frame loading and utilities
    # ------------------------------------------------------------------

    def _load_frames(self, path: str) -> List[Image.Image]:
        """Load video frames from a file or directory.

        Reuses the helper from Wan22VACEBackend for consistency.
        """
        from pathlib import Path as PathLib

        p = PathLib(path)

        if not p.exists():
            raise FileNotFoundError(f"Video path does not exist: {path}")

        # Delegate to the VACE backend's static helper if available
        try:
            from animatediff.backends.wan22_vace import Wan22VACEBackend
            return Wan22VACEBackend.load_video_frames(path)
        except ImportError:
            pass

        # Fallback: use diffusers load_video for video files
        if p.suffix in (".mp4", ".avi", ".mov", ".webm", ".gif"):
            try:
                from diffusers.utils import load_video
                return load_video(str(p))
            except ImportError:
                pass

        # Fallback: directory of images
        if p.is_dir():
            frame_files = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
            if not frame_files:
                raise ValueError(f"No PNG/JPG frames found in directory: {path}")
            return [Image.open(f).convert("RGB") for f in frame_files]

        raise ValueError(f"Cannot load frames from: {path}")

    @staticmethod
    def _crossfade_frames(
        frames_a: List[Image.Image],
        frames_b: List[Image.Image],
    ) -> List[Image.Image]:
        """Create a linear crossfade between two lists of frames.

        Both lists should have the same length. If they differ, the shorter
        one is padded by repeating its last frame.

        Returns:
            List of blended PIL.Image.Image (same length as the longer input).
        """
        import numpy as np

        max_len = max(len(frames_a), len(frames_b))

        # Pad shorter list
        while len(frames_a) < max_len:
            frames_a.append(frames_a[-1])
        while len(frames_b) < max_len:
            frames_b.append(frames_b[-1])

        blended = []
        for i in range(max_len):
            alpha = i / max(max_len - 1, 1)  # 0.0 -> 1.0
            arr_a = np.array(frames_a[i], dtype=np.float32)
            arr_b = np.array(frames_b[i], dtype=np.float32)

            # Ensure same dimensions
            if arr_a.shape != arr_b.shape:
                from PIL import Image as PILImage
                frames_b[i] = frames_b[i].resize(frames_a[i].size)
                arr_b = np.array(frames_b[i], dtype=np.float32)

            mixed = (1.0 - alpha) * arr_a + alpha * arr_b
            blended.append(Image.fromarray(mixed.clip(0, 255).astype(np.uint8)))

        return blended

    @staticmethod
    def _analyze_frame_style(frame: Image.Image) -> str:
        """Analyze a frame to extract a rough style description.

        Uses simple color analysis (dominant hue, brightness, contrast) to
        generate a text description. This is intentionally lightweight --
        no neural network required.

        Returns:
            A short style description string, e.g., "warm golden tones, high contrast,
            dark shadows". Returns empty string if analysis fails.
        """
        try:
            import numpy as np

            arr = np.array(frame.resize((128, 128)), dtype=np.float32)
            mean_rgb = arr.mean(axis=(0, 1))
            std_rgb = arr.std(axis=(0, 1))

            # Overall brightness
            brightness = mean_rgb.mean()
            if brightness > 180:
                bright_desc = "bright, high-key lighting"
            elif brightness > 120:
                bright_desc = "balanced mid-tone lighting"
            elif brightness > 60:
                bright_desc = "moody low-key lighting"
            else:
                bright_desc = "dark, dramatic shadows"

            # Contrast
            contrast = std_rgb.mean()
            if contrast > 70:
                contrast_desc = "high contrast"
            elif contrast > 40:
                contrast_desc = "moderate contrast"
            else:
                contrast_desc = "soft low contrast"

            # Dominant color channel
            r, g, b = mean_rgb
            if r > g and r > b:
                if r > g + 30:
                    color_desc = "warm red-orange tones"
                else:
                    color_desc = "warm golden tones"
            elif g > r and g > b:
                color_desc = "cool green tones"
            elif b > r and b > g:
                if b > r + 30:
                    color_desc = "cool blue tones"
                else:
                    color_desc = "cool blue-purple tones"
            else:
                color_desc = "neutral balanced tones"

            return f"{color_desc}, {bright_desc}, {contrast_desc}"

        except Exception as e:
            logger.debug(f"Frame style analysis failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# Factory / convenience
# ---------------------------------------------------------------------------

def create_chainer_from_config(config: dict) -> Optional[ShotChainer]:
    """Create a ShotChainer from a storyboard's shot_chaining config block.

    Expected config format (from storyboard JSON):
        {
            "enabled": true,
            "method": "last_frame_i2v",
            "overlap_frames": 4
        }

    Returns:
        ShotChainer instance if enabled, None otherwise.
    """
    if not config or not config.get("enabled", False):
        return None

    return ShotChainer(
        method=config.get("method", "last_frame_i2v"),
        overlap_frames=config.get("overlap_frames", 4),
        style_weight=config.get("style_weight", 0.5),
    )
