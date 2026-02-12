"""
Multi-Dimensional Reward Model — physics, aesthetic, and alignment scoring.

Approximates SeedAnce 2.0's three reward models using:
  - MotionRewardModel: Optical flow consistency, gravity, motion magnitude
  - AestheticRewardModel: Wraps existing quality_scorer + optional ImageReward
  - AlignmentRewardModel: CLIP frame-text similarity + object detection
  - RewardEnsemble: Weighted combination → single score + retry decision

The ensemble enables generate→score→retry loops for RLHF-style refinement
at inference time (no RL training required).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "RewardScore",
    "MotionRewardModel",
    "AestheticRewardModel",
    "AlignmentRewardModel",
    "RewardEnsemble",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RewardScore:
    """Multi-dimensional reward score for a generated video."""
    overall: float = 0.0
    motion: float = 0.0
    aesthetic: float = 0.0
    alignment: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"overall={self.overall:.3f} "
            f"motion={self.motion:.3f} "
            f"aesthetic={self.aesthetic:.3f} "
            f"alignment={self.alignment:.3f}"
        )


# ---------------------------------------------------------------------------
# Motion Reward Model
# ---------------------------------------------------------------------------

def _compute_optical_flow_magnitude(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
) -> np.ndarray:
    """Compute gradient-based optical flow magnitude between two grayscale frames.

    Returns per-pixel flow magnitude as a 2D array.
    """
    # Temporal gradient
    dt = gray_b.astype(np.float32) - gray_a.astype(np.float32)

    # Spatial gradients (central difference)
    dx = np.zeros_like(gray_a, dtype=np.float32)
    dy = np.zeros_like(gray_a, dtype=np.float32)
    dx[:, 1:-1] = (gray_a[:, 2:].astype(np.float32) - gray_a[:, :-2].astype(np.float32)) / 2.0
    dy[1:-1, :] = (gray_a[2:, :].astype(np.float32) - gray_a[:-2, :].astype(np.float32)) / 2.0

    grad_mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    flow_mag = np.abs(dt) / grad_mag
    return np.clip(flow_mag, 0.0, 20.0)


class MotionRewardModel:
    """Physics plausibility scoring via optical flow analysis.

    Checks:
      - Flow consistency: no teleportation (sudden large displacements)
      - Gravity direction: downward flow bias when applicable
      - Motion magnitude vs prompt expectations
      - Trajectory smoothness: smooth acceleration, no jitter
    """

    def __init__(self, max_sample_frames: int = 30):
        self.max_sample_frames = max_sample_frames

    def score(
        self,
        frames: List[Image.Image],
        prompt: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        """Score motion quality of a video.

        Returns:
            (score, details) where score is in [0, 1].
        """
        if len(frames) < 3:
            return 0.5, {"note": "too_few_frames"}

        details: Dict[str, float] = {}

        # Convert to grayscale, downsample for speed
        grays = []
        step = max(1, len(frames) // self.max_sample_frames)
        for i in range(0, len(frames), step):
            arr = np.array(frames[i].convert("L").resize((128, 128)))
            grays.append(arr)

        # 1. Flow consistency (no teleportation)
        flow_mags = []
        for i in range(len(grays) - 1):
            flow = _compute_optical_flow_magnitude(grays[i], grays[i + 1])
            flow_mags.append(float(np.mean(flow)))

        flow_arr = np.array(flow_mags)
        flow_mean = float(np.mean(flow_arr))
        flow_std = float(np.std(flow_arr))

        # Teleportation detection: any frame with flow > 3 * mean is suspicious
        if flow_mean > 0.01:
            teleport_ratio = float(np.sum(flow_arr > 3 * flow_mean) / len(flow_arr))
        else:
            teleport_ratio = 0.0
        consistency_score = max(0.0, 1.0 - teleport_ratio * 3.0)
        details["flow_consistency"] = consistency_score
        details["flow_mean"] = flow_mean
        details["teleport_ratio"] = teleport_ratio

        # 2. Motion magnitude appropriateness
        # Check if prompt suggests motion level
        prompt_lower = prompt.lower()
        expected_motion = "medium"
        for word in ["slow", "still", "calm", "frozen", "static"]:
            if word in prompt_lower:
                expected_motion = "low"
                break
        for word in ["fast", "explosion", "run", "fight", "rapid", "flying"]:
            if word in prompt_lower:
                expected_motion = "high"
                break

        if expected_motion == "low":
            mag_score = max(0.0, 1.0 - flow_mean / 2.0) if flow_mean > 0.5 else 1.0
        elif expected_motion == "high":
            mag_score = min(1.0, flow_mean / 1.0) if flow_mean < 1.0 else 1.0
        else:
            # Medium: penalize extremes
            if flow_mean < 0.05:
                mag_score = flow_mean / 0.05
            elif flow_mean > 4.0:
                mag_score = max(0.0, 1.0 - (flow_mean - 4.0) / 4.0)
            else:
                mag_score = 1.0
        details["motion_magnitude"] = mag_score
        details["expected_motion"] = {"low": 0.0, "medium": 0.5, "high": 1.0}[expected_motion]

        # 3. Trajectory smoothness (low acceleration jitter)
        if len(flow_mags) > 2:
            accelerations = np.diff(flow_arr)
            acc_std = float(np.std(accelerations))
            # Lower acceleration variance = smoother
            smoothness = max(0.0, 1.0 - acc_std / (flow_mean + 0.01))
        else:
            smoothness = 0.5
        details["trajectory_smoothness"] = smoothness

        # 4. Gravity check (downward flow bias in vertical component)
        gravity_score = self._check_gravity(grays, prompt)
        details["gravity_plausibility"] = gravity_score

        # Combine
        score = (
            0.30 * consistency_score
            + 0.25 * mag_score
            + 0.25 * smoothness
            + 0.20 * gravity_score
        )
        return float(np.clip(score, 0.0, 1.0)), details

    def _check_gravity(
        self, grays: List[np.ndarray], prompt: str,
    ) -> float:
        """Check if vertical motion follows gravity expectations.

        For prompts mentioning falling/dropping, expects downward flow.
        Returns 1.0 if consistent, lower if contradictory.
        """
        # Only check if prompt implies gravity-relevant motion
        gravity_words = {"fall", "drop", "rain", "snow", "pour", "descend"}
        prompt_lower = prompt.lower()
        if not any(w in prompt_lower for w in gravity_words):
            return 0.8  # Neutral — no gravity expectation

        # Compute vertical flow direction bias
        down_flow = 0.0
        up_flow = 0.0
        for i in range(len(grays) - 1):
            a = grays[i].astype(np.float32)
            b = grays[i + 1].astype(np.float32)
            dy = b - a  # Positive = content moved down
            h = dy.shape[0]
            # Check lower half vs upper half movement
            lower_change = float(np.mean(np.abs(dy[h // 2:, :])))
            upper_change = float(np.mean(np.abs(dy[:h // 2, :])))
            if lower_change > upper_change:
                down_flow += 1
            else:
                up_flow += 1

        total = down_flow + up_flow
        if total == 0:
            return 0.8

        # Gravity-consistent if downward flow dominates
        return float(down_flow / total)


# ---------------------------------------------------------------------------
# Aesthetic Reward Model
# ---------------------------------------------------------------------------

class AestheticRewardModel:
    """Visual quality scoring wrapping the existing VideoQualityScorer.

    Adds optional ImageReward model integration for learned aesthetic
    preferences.
    """

    def __init__(self, device: str = "auto"):
        self.device = device
        self._scorer = None
        self._scorer_checked = False

    def _get_scorer(self):
        if not self._scorer_checked:
            self._scorer_checked = True
            try:
                from animatediff.core.quality_scorer import VideoQualityScorer
                self._scorer = VideoQualityScorer(
                    backends=["aesthetic", "technical"],
                    device=self.device,
                )
            except Exception as e:
                logger.warning(f"Could not load VideoQualityScorer: {e}")
        return self._scorer

    def score(
        self,
        frames: List[Image.Image],
        prompt: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        """Score aesthetic quality.

        Returns:
            (score, details) where score is in [0, 1].
        """
        scorer = self._get_scorer()
        if scorer is None:
            return 0.5, {"note": "scorer_unavailable"}

        vs = scorer.score_video(frames, prompt)
        details = {
            "aesthetic_raw": vs.aesthetic,
            "technical_raw": vs.technical,
        }
        details.update(vs.details)

        # Combine aesthetic + technical
        score = 0.6 * vs.aesthetic + 0.4 * vs.technical
        return float(np.clip(score, 0.0, 1.0)), details


# ---------------------------------------------------------------------------
# Alignment Reward Model
# ---------------------------------------------------------------------------

class AlignmentRewardModel:
    """Prompt-video alignment scoring using CLIP similarity.

    Measures how well the generated video matches the text prompt by
    computing CLIP cosine similarity between keyframes and the prompt text.
    """

    def __init__(self, device: str = "auto", keyframe_count: int = 6):
        self.device = device
        self.keyframe_count = keyframe_count
        self._scorer = None
        self._scorer_checked = False

    def _get_scorer(self):
        if not self._scorer_checked:
            self._scorer_checked = True
            try:
                from animatediff.core.quality_scorer import VideoQualityScorer
                self._scorer = VideoQualityScorer(
                    backends=["prompt_alignment"],
                    device=self.device,
                    keyframe_count=self.keyframe_count,
                )
            except Exception as e:
                logger.warning(f"Could not load alignment scorer: {e}")
        return self._scorer

    def score(
        self,
        frames: List[Image.Image],
        prompt: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        """Score prompt-video alignment.

        Returns:
            (score, details) where score is in [0, 1].
        """
        if not prompt:
            return 0.5, {"note": "no_prompt"}

        scorer = self._get_scorer()
        if scorer is None:
            return 0.5, {"note": "clip_unavailable"}

        vs = scorer.score_video(frames, prompt)
        details = {"clip_alignment": vs.prompt_alignment}
        details.update(vs.details)

        return float(np.clip(vs.prompt_alignment, 0.0, 1.0)), details


# ---------------------------------------------------------------------------
# Reward Ensemble
# ---------------------------------------------------------------------------

class RewardEnsemble:
    """Weighted ensemble of reward models → single score + retry decision.

    Default weights approximate SeedAnce 2.0's multi-reward approach:
      - motion:    0.30 (physics plausibility)
      - aesthetic:  0.30 (visual quality)
      - alignment: 0.40 (prompt fidelity)
    """

    DEFAULT_WEIGHTS = {
        "motion": 0.30,
        "aesthetic": 0.30,
        "alignment": 0.40,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        device: str = "auto",
    ):
        """
        Args:
            weights: Custom weights for {motion, aesthetic, alignment}.
            device: Compute device for model-based scorers.
        """
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        self.motion_model = MotionRewardModel()
        self.aesthetic_model = AestheticRewardModel(device=device)
        self.alignment_model = AlignmentRewardModel(device=device)

        logger.info(f"RewardEnsemble weights: {self.weights}")

    def score(
        self,
        frames: List[Image.Image],
        prompt: str = "",
        audio_path: Optional[str] = None,
    ) -> RewardScore:
        """Score a video across all reward dimensions.

        Args:
            frames: Video frames (PIL Images).
            prompt: Text prompt used for generation.
            audio_path: Optional audio file (for future A/V sync scoring).

        Returns:
            RewardScore with per-dimension and overall scores.
        """
        all_details: Dict[str, float] = {}

        # Motion reward
        motion_score, motion_details = self.motion_model.score(frames, prompt)
        all_details.update({f"motion_{k}": v for k, v in motion_details.items()
                           if isinstance(v, (int, float))})

        # Aesthetic reward
        aesthetic_score, aesthetic_details = self.aesthetic_model.score(frames, prompt)
        all_details.update({f"aesthetic_{k}": v for k, v in aesthetic_details.items()
                           if isinstance(v, (int, float))})

        # Alignment reward
        alignment_score, alignment_details = self.alignment_model.score(frames, prompt)
        all_details.update({f"alignment_{k}": v for k, v in alignment_details.items()
                           if isinstance(v, (int, float))})

        # Weighted overall
        overall = (
            self.weights.get("motion", 0.3) * motion_score
            + self.weights.get("aesthetic", 0.3) * aesthetic_score
            + self.weights.get("alignment", 0.4) * alignment_score
        )

        result = RewardScore(
            overall=float(np.clip(overall, 0.0, 1.0)),
            motion=motion_score,
            aesthetic=aesthetic_score,
            alignment=alignment_score,
            details=all_details,
        )

        logger.info(f"Reward: {result.summary()}")
        return result

    def should_retry(
        self,
        score: RewardScore,
        threshold: float = 0.6,
    ) -> bool:
        """Decide whether to retry generation based on score.

        Args:
            score: RewardScore from score().
            threshold: Minimum acceptable overall score.

        Returns:
            True if the video should be regenerated.
        """
        return score.overall < threshold

    def pick_best(
        self,
        candidates: List[List[Image.Image]],
        prompt: str = "",
    ) -> Tuple[int, RewardScore]:
        """Score multiple candidates and return the best one.

        Args:
            candidates: List of frame-lists, one per candidate.
            prompt: Text prompt.

        Returns:
            (best_index, best_score).
        """
        if not candidates:
            raise ValueError("No candidates provided")
        if len(candidates) == 1:
            return 0, self.score(candidates[0], prompt)

        scores = []
        for i, frames in enumerate(candidates):
            logger.info(f"Scoring candidate {i + 1}/{len(candidates)}")
            s = self.score(frames, prompt)
            scores.append((i, s))

        best_idx, best_score = max(scores, key=lambda x: x[1].overall)
        logger.info(f"Best candidate: #{best_idx + 1} ({best_score.summary()})")
        return best_idx, best_score
