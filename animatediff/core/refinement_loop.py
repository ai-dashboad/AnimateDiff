"""
Refinement Loop — RLHF-style generate→score→retry at inference time.

Approximates SeedAnce 2.0's RLHF training by performing multiple generation
attempts, scoring each with the RewardEnsemble, and selecting the best.

Optionally enhances prompts between retries by adding quality tokens or
adjusting generation parameters (seed, guidance scale).

Usage::

    from animatediff.core.refinement_loop import RefinementLoop
    from animatediff.core.reward_model import RewardEnsemble

    loop = RefinementLoop(reward=RewardEnsemble())
    result = loop.generate_with_refinement(
        backend=my_backend,
        gen_kwargs={"prompt": "A cat walking", "seed": 42, ...},
        max_attempts=3,
        min_score=0.7,
    )
    print(f"Best score: {result.score.summary()}")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput

logger = logging.getLogger(__name__)

__all__ = ["RefinementResult", "RefinementLoop"]

# Quality tokens appended on retries to nudge generation
_QUALITY_BOOST_TOKENS = [
    ", masterpiece, best quality, highly detailed",
    ", professional cinematic lighting, sharp focus, 8k",
    ", award-winning photography, perfect composition, vivid colors",
]


@dataclass
class RefinementResult:
    """Result of a refinement loop."""
    output: Optional[VideoOutput] = None
    score: Optional[Any] = None  # RewardScore
    attempts: int = 0
    total_time: float = 0.0
    all_scores: List[float] = field(default_factory=list)
    accepted: bool = False

    @property
    def best_score_value(self) -> float:
        if self.score and hasattr(self.score, "overall"):
            return self.score.overall
        return 0.0


class RefinementLoop:
    """Generate multiple candidates, score, pick best, optionally retry.

    The loop implements inference-time RLHF:
      1. Generate N candidates per attempt
      2. Score each with RewardEnsemble
      3. If best > threshold → accept
      4. If all below → enhance prompt + adjust params + retry
      5. After max_attempts → return best overall
    """

    def __init__(
        self,
        reward: Optional[Any] = None,
        candidates_per_attempt: int = 1,
        prompt_enhancement: bool = True,
    ):
        """
        Args:
            reward: RewardEnsemble instance. If None, creates a default one.
            candidates_per_attempt: How many candidates to generate per attempt.
                Higher = better quality but slower. 1-3 recommended.
            prompt_enhancement: Whether to add quality tokens on retries.
        """
        self.reward = reward
        self.candidates_per_attempt = max(1, candidates_per_attempt)
        self.prompt_enhancement = prompt_enhancement

        if self.reward is None:
            try:
                from animatediff.core.reward_model import RewardEnsemble
                self.reward = RewardEnsemble()
            except Exception as e:
                logger.warning(f"Could not create RewardEnsemble: {e}")

    def generate_with_refinement(
        self,
        backend: BasePipeline,
        gen_kwargs: Dict[str, Any],
        max_attempts: int = 3,
        min_score: float = 0.7,
        save_path: str = "",
    ) -> RefinementResult:
        """Generate with quality-gated retries.

        Args:
            backend: Video generation backend.
            gen_kwargs: Generation keyword arguments (prompt, seed, etc.).
            max_attempts: Maximum retry attempts.
            min_score: Minimum acceptable RewardScore.overall.
            save_path: Optional path to save the best output.

        Returns:
            RefinementResult with the best output and scoring details.
        """
        t0 = time.time()
        best_output: Optional[VideoOutput] = None
        best_score = None
        best_score_val = -1.0
        all_scores: List[float] = []

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"Refinement attempt {attempt}/{max_attempts} "
                f"(candidates={self.candidates_per_attempt})"
            )

            # Generate candidates
            candidates = []
            for c in range(self.candidates_per_attempt):
                kwargs = self._prepare_kwargs(gen_kwargs, attempt, c)
                try:
                    output = backend.generate(**kwargs)
                    candidates.append(output)
                except Exception as e:
                    logger.warning(
                        f"Candidate {c + 1} failed in attempt {attempt}: {e}"
                    )

            if not candidates:
                logger.warning(f"No candidates produced in attempt {attempt}")
                continue

            # Score candidates
            for c_idx, output in enumerate(candidates):
                if not output.frames:
                    all_scores.append(0.0)
                    continue

                if self.reward:
                    prompt = gen_kwargs.get("prompt", "")
                    score = self.reward.score(output.frames, prompt)
                    score_val = score.overall
                else:
                    score = None
                    score_val = 0.5  # Default if no reward model

                all_scores.append(score_val)
                logger.info(
                    f"  Candidate {c_idx + 1}: score={score_val:.3f}"
                    + (f" ({score.summary()})" if score else "")
                )

                if score_val > best_score_val:
                    best_output = output
                    best_score = score
                    best_score_val = score_val

            # Check if we meet the threshold
            if best_score_val >= min_score:
                logger.info(
                    f"Accepted at attempt {attempt}: "
                    f"score={best_score_val:.3f} >= {min_score}"
                )
                result = RefinementResult(
                    output=best_output,
                    score=best_score,
                    attempts=attempt,
                    total_time=time.time() - t0,
                    all_scores=all_scores,
                    accepted=True,
                )
                if save_path and best_output:
                    backend.save(best_output, save_path, fps=best_output.fps)
                return result

            logger.info(
                f"Attempt {attempt}: best={best_score_val:.3f} < "
                f"threshold={min_score}, {'retrying' if attempt < max_attempts else 'giving up'}"
            )

        # Return best we got
        total_time = time.time() - t0
        result = RefinementResult(
            output=best_output,
            score=best_score,
            attempts=max_attempts,
            total_time=total_time,
            all_scores=all_scores,
            accepted=best_score_val >= min_score,
        )

        if save_path and best_output:
            backend.save(best_output, save_path, fps=best_output.fps)

        logger.info(
            f"Refinement complete: {max_attempts} attempts, "
            f"best={best_score_val:.3f}, "
            f"accepted={result.accepted}, "
            f"time={total_time:.1f}s"
        )
        return result

    def _prepare_kwargs(
        self,
        base_kwargs: Dict[str, Any],
        attempt: int,
        candidate_idx: int,
    ) -> Dict[str, Any]:
        """Prepare generation kwargs for a specific attempt + candidate.

        Adjusts seed and optionally enhances prompt for retries.
        """
        kwargs = dict(base_kwargs)

        # Vary seed across candidates and attempts
        base_seed = kwargs.get("seed", -1)
        if base_seed >= 0:
            kwargs["seed"] = base_seed + (attempt - 1) * 100 + candidate_idx
        elif candidate_idx > 0:
            # Random seed for additional candidates
            import random
            kwargs["seed"] = random.randint(0, 2**31)

        # Enhance prompt on retries (attempt > 1)
        if self.prompt_enhancement and attempt > 1:
            token_idx = min(attempt - 2, len(_QUALITY_BOOST_TOKENS) - 1)
            boost = _QUALITY_BOOST_TOKENS[token_idx]
            prompt = kwargs.get("prompt", "")
            if boost not in prompt:
                kwargs["prompt"] = prompt + boost
                logger.debug(f"Enhanced prompt with: {boost}")

        # Slightly increase guidance on later attempts
        if attempt > 1:
            guidance = kwargs.get("guidance_scale", 5.0)
            kwargs["guidance_scale"] = min(guidance + 0.5 * (attempt - 1), 12.0)

        return kwargs
