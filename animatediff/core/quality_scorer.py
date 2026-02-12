"""
Video Quality Scorer -- automatic multi-metric video quality assessment.

Scores generated videos across five dimensions to enable multi-candidate
selection (generate N candidates, pick the best one).  The numpy-only
fallback always works; optional backends (CLIP, ImageReward) are used
when available.

Scoring dimensions:
  - aesthetic:        Frame-level visual beauty (brightness, saturation, edges)
  - technical:        Sharpness, noise, exposure correctness
  - temporal:         Frame-to-frame consistency (penalises flicker)
  - motion:           Motion smoothness and diversity (penalises jitter)
  - prompt_alignment: Text-video semantic match (requires CLIP)

References & inspiration:
  - ImageReward  (NeurIPS 2023)  -- RLHF-trained image preference scoring
  - HPS v2                       -- Human Preference Score for T2I
  - DOVER       (ICCV 2023)      -- disentangled aesthetic + technical VQA
  - Q-Align     (ICML 2024)      -- VLM-based visual scoring
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VideoScore:
    """Detailed quality score for a single video."""

    overall: float = 0.0           # 0-1 composite score
    aesthetic: float = 0.0         # visual beauty
    technical: float = 0.0         # sharpness, exposure, noise
    temporal: float = 0.0          # frame consistency
    motion: float = 0.0            # motion smoothness
    prompt_alignment: float = 0.0  # text-video semantic match
    details: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"overall={self.overall:.3f}  "
            f"aesthetic={self.aesthetic:.3f}  "
            f"technical={self.technical:.3f}  "
            f"temporal={self.temporal:.3f}  "
            f"motion={self.motion:.3f}  "
            f"prompt={self.prompt_alignment:.3f}"
        )


# ---------------------------------------------------------------------------
# Helpers -- numpy-only image analysis
# ---------------------------------------------------------------------------

def _pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 numpy array (H, W, 3) in [0, 1]."""
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) float RGB to (H, W) float grayscale."""
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    """Compute the variance of a 3x3 Laplacian filter -- measures sharpness.

    Higher variance = sharper image.
    """
    # 3x3 Laplacian kernel applied via numpy (no cv2 needed)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)

    # Pad to avoid boundary effects
    padded = np.pad(gray, 1, mode="reflect")
    h, w = gray.shape
    lap = np.zeros_like(gray)
    for di in range(3):
        for dj in range(3):
            lap += kernel[di, dj] * padded[di:di + h, dj:dj + w]

    return float(np.var(lap))


def _sobel_edge_density(gray: np.ndarray) -> float:
    """Compute normalised edge density using Sobel gradients."""
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = kx.T

    padded = np.pad(gray, 1, mode="reflect")
    h, w = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for di in range(3):
        for dj in range(3):
            gx += kx[di, dj] * padded[di:di + h, dj:dj + w]
            gy += ky[di, dj] * padded[di:di + h, dj:dj + w]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # Normalise to [0, 1] -- theoretical max for Sobel on [0,1] input is ~4
    return float(np.mean(magnitude) / 4.0)


def _brightness_stats(arr: np.ndarray) -> Tuple[float, float]:
    """Return (mean_brightness, brightness_std) for an RGB image in [0, 1]."""
    gray = _to_gray(arr)
    return float(np.mean(gray)), float(np.std(gray))


def _color_saturation(arr: np.ndarray) -> float:
    """Mean colour saturation -- distance of each pixel from the gray axis."""
    gray = _to_gray(arr)[:, :, np.newaxis]
    diff = arr - gray
    sat = np.sqrt(np.mean(diff ** 2, axis=2))
    return float(np.mean(sat))


def _histogram_spread(gray: np.ndarray) -> float:
    """Evaluate how well the gray histogram uses the full [0, 1] range.

    Returns 1.0 for a perfectly uniform distribution; lower for
    clipped / narrow histograms.
    """
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0.0, 1.0))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-12)

    # Shannon entropy normalised to [0, 1]
    nonzero = hist[hist > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    max_entropy = np.log2(64)
    return float(entropy / max_entropy)


def _high_freq_energy_ratio(gray: np.ndarray) -> float:
    """Ratio of high-frequency energy -- proxy for noise level.

    Uses the 2-D DCT-like approach via FFT.  High ratio = noisy.
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    # Low-frequency radius = 1/8 of image diagonal
    r = int(math.sqrt(h ** 2 + w ** 2) / 8)

    # Create circular mask for low frequencies
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = dist <= r

    total = mag.sum() + 1e-12
    low = mag[low_mask].sum()

    # High-frequency ratio (higher = noisier)
    return float(1.0 - low / total)


def _ssim_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified SSIM between two grayscale images (same shape).

    Returns a value in [-1, 1]; higher = more similar.
    We use the basic SSIM formula with a small window average.
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)

    return float(num / (den + 1e-12))


def _select_keyframes(
    n_frames: int,
    n_samples: int = 8,
) -> List[int]:
    """Select keyframe indices: first, last, middle, + evenly spaced."""
    if n_frames <= n_samples:
        return list(range(n_frames))

    indices = {0, n_frames - 1, n_frames // 2}
    # Fill remaining with evenly spaced samples
    step = n_frames / (n_samples - len(indices) + 1)
    pos = step
    while len(indices) < n_samples and pos < n_frames:
        indices.add(int(pos))
        pos += step
    return sorted(indices)


# ---------------------------------------------------------------------------
# Optional backend: ImageReward
# ---------------------------------------------------------------------------

def _try_load_image_reward(device: str):
    """Attempt to load ImageReward model.  Returns (model, preprocess) or None."""
    try:
        import ImageReward as ir
        model = ir.load("ImageReward-v1.0", device=device)
        logger.info("ImageReward v1.0 loaded for aesthetic scoring")
        return model
    except Exception as e:
        logger.debug(f"ImageReward not available: {e}")
        return None


# ---------------------------------------------------------------------------
# Optional backend: CLIP
# ---------------------------------------------------------------------------

def _try_load_clip(device: str):
    """Attempt to load CLIP for prompt alignment scoring.

    Returns (model, processor, tokenizer) or None.
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)

        import torch
        model = model.to(device).eval()
        logger.info(f"CLIP ({model_name}) loaded for prompt alignment scoring")
        return model, processor
    except Exception as e:
        logger.debug(f"CLIP not available: {e}")
        return None


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class VideoQualityScorer:
    """Score video quality using multiple metrics.

    The numpy-only fallback (aesthetic, technical, temporal, motion) always
    works with zero extra dependencies.  Optional enhanced backends
    (ImageReward for aesthetic, CLIP for prompt alignment) are used when
    their packages are installed.

    Usage::

        scorer = VideoQualityScorer(backends=["aesthetic", "technical", "temporal"])
        score = scorer.score_video(frames, prompt="a cinematic landscape")
        print(score.summary())

        # Multi-candidate selection
        idx, best_frames = scorer.pick_best([candidate_a, candidate_b], prompt)
    """

    # Weights for combining dimension scores into ``overall``
    DEFAULT_WEIGHTS = {
        "aesthetic": 0.25,
        "technical": 0.20,
        "temporal": 0.20,
        "motion": 0.15,
        "prompt_alignment": 0.20,
    }

    VALID_BACKENDS = {"aesthetic", "technical", "temporal", "motion", "prompt_alignment"}

    def __init__(
        self,
        backends: Optional[List[str]] = None,
        device: str = "auto",
        weights: Optional[Dict[str, float]] = None,
        keyframe_count: int = 8,
    ):
        """
        Args:
            backends: Which scoring dimensions to compute.  Defaults to
                ``["aesthetic", "technical", "temporal", "motion"]``.
                Add ``"prompt_alignment"`` to include CLIP-based text-video
                alignment (requires transformers + CLIP).
            device: ``"auto"`` picks CUDA > MPS > CPU.
            weights: Optional custom weights for combining dimension scores.
            keyframe_count: Number of keyframes to sample for per-frame metrics.
        """
        if backends is None:
            backends = ["aesthetic", "technical", "temporal", "motion"]

        unknown = set(backends) - self.VALID_BACKENDS
        if unknown:
            raise ValueError(f"Unknown scoring backends: {unknown}")

        self.backends = list(backends)
        self.device = self._resolve_device(device)
        self.keyframe_count = max(2, keyframe_count)

        # Merge custom weights
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        # Normalise weights to only active backends
        active_w = {k: self.weights.get(k, 0.0) for k in self.backends}
        w_sum = sum(active_w.values()) or 1.0
        self._norm_weights = {k: v / w_sum for k, v in active_w.items()}

        # Lazy-loaded optional models
        self._image_reward = None
        self._image_reward_checked = False
        self._clip_model = None
        self._clip_processor = None
        self._clip_checked = False

        logger.info(
            f"VideoQualityScorer initialized  "
            f"backends={self.backends}  device={self.device}  "
            f"keyframes={self.keyframe_count}"
        )

    # ----- device resolution -------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # =========================================================================
    # Public API
    # =========================================================================

    def score_video(
        self,
        frames: List[Image.Image],
        prompt: str = "",
    ) -> VideoScore:
        """Score a single video.

        Args:
            frames: List of PIL Images (RGB) representing the video.
            prompt: The text prompt used to generate the video (for alignment).

        Returns:
            A ``VideoScore`` with per-dimension and composite scores.
        """
        if not frames:
            logger.warning("score_video called with empty frame list")
            return VideoScore()

        details: Dict[str, float] = {}
        dim_scores: Dict[str, float] = {}

        # Convert frames to numpy once
        np_frames = [_pil_to_np(f) for f in frames]
        gray_frames = [_to_gray(a) for a in np_frames]
        key_idx = _select_keyframes(len(frames), self.keyframe_count)

        # --- Aesthetic ---
        if "aesthetic" in self.backends:
            dim_scores["aesthetic"] = self._score_aesthetic(
                frames, np_frames, gray_frames, key_idx, prompt, details,
            )

        # --- Technical ---
        if "technical" in self.backends:
            dim_scores["technical"] = self._score_technical(
                np_frames, gray_frames, key_idx, details,
            )

        # --- Temporal ---
        if "temporal" in self.backends:
            dim_scores["temporal"] = self._score_temporal(
                gray_frames, details,
            )

        # --- Motion ---
        if "motion" in self.backends:
            dim_scores["motion"] = self._score_motion(
                gray_frames, details,
            )

        # --- Prompt alignment ---
        if "prompt_alignment" in self.backends:
            dim_scores["prompt_alignment"] = self._score_prompt_alignment(
                frames, key_idx, prompt, details,
            )

        # Composite
        overall = sum(
            self._norm_weights.get(k, 0.0) * v
            for k, v in dim_scores.items()
        )

        score = VideoScore(
            overall=overall,
            aesthetic=dim_scores.get("aesthetic", 0.0),
            technical=dim_scores.get("technical", 0.0),
            temporal=dim_scores.get("temporal", 0.0),
            motion=dim_scores.get("motion", 0.0),
            prompt_alignment=dim_scores.get("prompt_alignment", 0.0),
            details=details,
        )

        logger.info(f"Video score: {score.summary()}")
        return score

    def score_candidates(
        self,
        candidates: List[List[Image.Image]],
        prompt: str = "",
    ) -> List[VideoScore]:
        """Score multiple video candidates and return ranked results.

        Args:
            candidates: List of frame-lists, one per candidate.
            prompt: Text prompt used for generation.

        Returns:
            List of ``VideoScore`` objects, one per candidate, sorted by
            descending ``overall`` score.
        """
        scores = []
        for i, frames in enumerate(candidates):
            logger.info(f"Scoring candidate {i + 1}/{len(candidates)} ({len(frames)} frames)")
            s = self.score_video(frames, prompt)
            s.details["candidate_index"] = float(i)
            scores.append(s)

        # Sort by overall descending
        scores.sort(key=lambda s: s.overall, reverse=True)
        return scores

    def pick_best(
        self,
        candidates: List[List[Image.Image]],
        prompt: str = "",
    ) -> Tuple[int, List[Image.Image]]:
        """Pick the best candidate video.

        Args:
            candidates: List of frame-lists.
            prompt: Text prompt used for generation.

        Returns:
            Tuple of (best_index, best_frames).
        """
        if not candidates:
            raise ValueError("No candidates provided")
        if len(candidates) == 1:
            logger.info("Single candidate -- skipping quality comparison")
            return 0, candidates[0]

        scores = []
        for i, frames in enumerate(candidates):
            logger.info(f"Scoring candidate {i + 1}/{len(candidates)} ({len(frames)} frames)")
            s = self.score_video(frames, prompt)
            scores.append((i, s))

        # Find best
        best_idx, best_score = max(scores, key=lambda x: x[1].overall)
        logger.info(
            f"Best candidate: #{best_idx + 1} "
            f"(overall={best_score.overall:.3f})  "
            f"scores: {best_score.summary()}"
        )

        return best_idx, candidates[best_idx]

    # =========================================================================
    # Dimension scorers
    # =========================================================================

    def _score_aesthetic(
        self,
        pil_frames: List[Image.Image],
        np_frames: List[np.ndarray],
        gray_frames: List[np.ndarray],
        key_idx: List[int],
        prompt: str,
        details: Dict[str, float],
    ) -> float:
        """Aesthetic score: visual beauty of keyframes.

        Combines brightness balance, colour saturation, and edge density.
        Optionally enhanced with ImageReward if available.
        """
        # --- Try ImageReward first ---
        ir_score = self._score_image_reward(pil_frames, key_idx, prompt, details)
        if ir_score is not None:
            return ir_score

        # --- Numpy fallback ---
        brightness_scores = []
        saturation_scores = []
        edge_scores = []

        for idx in key_idx:
            arr = np_frames[idx]
            gray = gray_frames[idx]

            # Brightness: penalise too dark (<0.2) or too bright (>0.8)
            mean_b, std_b = _brightness_stats(arr)
            # Ideal brightness around 0.45-0.55, with some contrast (std ~0.15-0.25)
            b_score = 1.0 - 2.0 * abs(mean_b - 0.5)
            b_score = max(0.0, b_score)
            # Bonus for good contrast
            contrast_score = min(1.0, std_b / 0.20)
            brightness_scores.append(0.6 * b_score + 0.4 * contrast_score)

            # Colour saturation: moderate saturation is usually pleasing
            sat = _color_saturation(arr)
            # Sweet spot around 0.1-0.25
            sat_score = min(1.0, sat / 0.15) if sat < 0.15 else max(0.0, 1.0 - (sat - 0.25) / 0.3)
            saturation_scores.append(max(0.0, min(1.0, sat_score)))

            # Edge density: some detail is good, too much is noise-like
            edge = _sobel_edge_density(gray)
            # Sweet spot around 0.03-0.10
            edge_score = min(1.0, edge / 0.05) if edge < 0.05 else max(0.0, 1.0 - (edge - 0.10) / 0.15)
            edge_scores.append(max(0.0, min(1.0, edge_score)))

        aes = (
            0.35 * np.mean(brightness_scores)
            + 0.35 * np.mean(saturation_scores)
            + 0.30 * np.mean(edge_scores)
        )

        details["aesthetic_brightness"] = float(np.mean(brightness_scores))
        details["aesthetic_saturation"] = float(np.mean(saturation_scores))
        details["aesthetic_edges"] = float(np.mean(edge_scores))

        return float(np.clip(aes, 0.0, 1.0))

    def _score_image_reward(
        self,
        pil_frames: List[Image.Image],
        key_idx: List[int],
        prompt: str,
        details: Dict[str, float],
    ) -> Optional[float]:
        """Try scoring with ImageReward.  Returns None if unavailable."""
        if not self._image_reward_checked:
            self._image_reward_checked = True
            self._image_reward = _try_load_image_reward(self.device)

        if self._image_reward is None:
            return None

        if not prompt:
            logger.debug("ImageReward requires a prompt; falling back to numpy aesthetic")
            return None

        try:
            scores = []
            for idx in key_idx:
                s = self._image_reward.score(prompt, pil_frames[idx])
                scores.append(float(s))

            # ImageReward scores are roughly in [-2, 2]; normalise to [0, 1]
            raw_mean = np.mean(scores)
            normalised = float(np.clip((raw_mean + 2.0) / 4.0, 0.0, 1.0))

            details["image_reward_raw"] = float(raw_mean)
            details["image_reward_per_frame"] = scores
            logger.debug(f"ImageReward raw={raw_mean:.3f} -> normalised={normalised:.3f}")
            return normalised

        except Exception as e:
            logger.warning(f"ImageReward scoring failed: {e}")
            return None

    def _score_technical(
        self,
        np_frames: List[np.ndarray],
        gray_frames: List[np.ndarray],
        key_idx: List[int],
        details: Dict[str, float],
    ) -> float:
        """Technical quality: sharpness, noise level, exposure.

        All computed with pure numpy -- no external dependencies.

        Key insight: noise inflates both Laplacian variance and HF energy.
        We detect noise first via HF ratio, then discount sharpness when
        noise is high.  Real sharpness shows structured edges (high kurtosis
        in the Laplacian), while noise shows uniform high-frequency energy
        (low kurtosis).
        """
        sharpness_scores = []
        noise_scores = []
        exposure_scores = []

        for idx in key_idx:
            gray = gray_frames[idx]
            arr = np_frames[idx]

            # --- Noise (high-frequency energy ratio) --- compute FIRST
            hf_ratio = _high_freq_energy_ratio(gray)
            # Lower is better (less noise).  Typical for clean images: 0.70-0.88
            # Noisy images push toward 0.95+
            if hf_ratio > 0.88:
                noise_s = max(0.0, 1.0 - (hf_ratio - 0.88) / 0.12)
            elif hf_ratio < 0.55:
                noise_s = max(0.0, hf_ratio / 0.55)
            else:
                noise_s = 1.0
            noise_scores.append(noise_s)

            # --- Sharpness (Laplacian variance, noise-discounted) ---
            lap_var = _laplacian_variance(gray)
            # Base sharpness from Laplacian variance
            raw_sharp = float(1.0 - math.exp(-lap_var / 0.005))
            # Discount sharpness when noise is high: noise inflates Laplacian
            # variance uniformly, but real sharpness comes from structured edges.
            # Use noise_s as a gate: if noise_s is low (noisy), discount sharpness.
            sharp = raw_sharp * (0.3 + 0.7 * noise_s)
            sharpness_scores.append(sharp)

            # --- Exposure (histogram spread) ---
            spread = _histogram_spread(gray)
            # Good exposure has moderate entropy (0.75-0.90).
            # Very high entropy (>0.95) can indicate noise spreading pixel values.
            if spread > 0.95:
                exp_score = max(0.5, 1.0 - (spread - 0.95) / 0.10)
            else:
                exp_score = min(1.0, spread / 0.85)
            # Discount if noise is adding artificial histogram spread
            exp_score = exp_score * (0.5 + 0.5 * noise_s)
            exposure_scores.append(exp_score)

        tech = (
            0.40 * np.mean(sharpness_scores)
            + 0.30 * np.mean(noise_scores)
            + 0.30 * np.mean(exposure_scores)
        )

        details["technical_sharpness"] = float(np.mean(sharpness_scores))
        details["technical_noise"] = float(np.mean(noise_scores))
        details["technical_exposure"] = float(np.mean(exposure_scores))

        return float(np.clip(tech, 0.0, 1.0))

    def _score_temporal(
        self,
        gray_frames: List[np.ndarray],
        details: Dict[str, float],
    ) -> float:
        """Temporal consistency: penalise sudden brightness/content jumps.

        Measures SSIM and pixel-difference between consecutive frames.
        High consistency = smooth; sudden jumps = flicker.
        """
        if len(gray_frames) < 2:
            details["temporal_ssim_mean"] = 1.0
            details["temporal_flicker_penalty"] = 0.0
            return 1.0

        ssims = []
        diffs = []

        # Sample pairs (skip expensive all-pairs for long videos)
        n = len(gray_frames)
        step = max(1, n // 30)  # Sample up to ~30 consecutive pairs
        pair_indices = list(range(0, n - 1, step))

        for i in pair_indices:
            a = gray_frames[i]
            b = gray_frames[i + 1]

            # Downsample for speed if frames are large
            if a.shape[0] > 256:
                factor = a.shape[0] // 256
                a = a[::factor, ::factor]
                b = b[::factor, ::factor]

            ssim = _ssim_pair(a, b)
            ssims.append(ssim)

            diff = np.mean(np.abs(a - b))
            diffs.append(diff)

        ssim_mean = np.mean(ssims)
        diff_mean = np.mean(diffs)

        # Flicker detection: high variance in frame-to-frame differences
        diff_std = np.std(diffs) if len(diffs) > 1 else 0.0
        # Penalise if some transitions are much larger than average
        flicker_penalty = min(1.0, diff_std / (diff_mean + 1e-6))

        # SSIM score (higher = better consistency)
        ssim_score = max(0.0, (ssim_mean - 0.5) / 0.5)  # Map [0.5, 1.0] -> [0, 1]

        # Diff score (lower diff = better)
        diff_score = max(0.0, 1.0 - diff_mean / 0.15)

        # Combine
        temporal = 0.40 * ssim_score + 0.35 * diff_score + 0.25 * (1.0 - flicker_penalty)

        details["temporal_ssim_mean"] = float(ssim_mean)
        details["temporal_diff_mean"] = float(diff_mean)
        details["temporal_flicker_penalty"] = float(flicker_penalty)

        return float(np.clip(temporal, 0.0, 1.0))

    def _score_motion(
        self,
        gray_frames: List[np.ndarray],
        details: Dict[str, float],
    ) -> float:
        """Motion quality: smooth, diverse motion is good; jitter/static is bad.

        Uses gradient-based optical flow magnitude estimation.
        """
        if len(gray_frames) < 3:
            details["motion_magnitude_mean"] = 0.0
            details["motion_smoothness"] = 1.0
            details["motion_diversity"] = 0.0
            return 0.5

        flow_magnitudes = []

        # Sample triplets for efficiency
        n = len(gray_frames)
        step = max(1, n // 20)
        sample_indices = list(range(0, n - 1, step))

        for i in sample_indices:
            a = gray_frames[i]
            b = gray_frames[min(i + 1, n - 1)]

            # Downsample for speed
            if a.shape[0] > 128:
                factor = max(1, a.shape[0] // 128)
                a = a[::factor, ::factor]
                b = b[::factor, ::factor]

            # Simple gradient-based flow magnitude estimate
            # Temporal gradient
            dt = b - a

            # Spatial gradients of frame a
            # Central differences
            dx = np.zeros_like(a)
            dy = np.zeros_like(a)
            dx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / 2.0
            dy[1:-1, :] = (a[2:, :] - a[:-2, :]) / 2.0

            # Flow magnitude ~= |dt| / (|dx| + |dy| + eps)
            grad_mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
            flow_mag = np.abs(dt) / grad_mag
            flow_mag = np.clip(flow_mag, 0.0, 10.0)

            flow_magnitudes.append(float(np.mean(flow_mag)))

        mag_array = np.array(flow_magnitudes)
        mag_mean = float(np.mean(mag_array))
        mag_std = float(np.std(mag_array))

        # --- Motion diversity ---
        # Some motion is good (not static).  Ideal mag_mean around 0.3-2.0
        if mag_mean < 0.05:
            # Nearly static -- boring
            diversity_score = mag_mean / 0.05
        elif mag_mean > 3.0:
            # Extremely high motion -- probably jittery
            diversity_score = max(0.0, 1.0 - (mag_mean - 3.0) / 5.0)
        else:
            diversity_score = 1.0

        # --- Motion smoothness ---
        # Low variance in flow magnitudes = smooth motion
        # High variance = jittery, inconsistent motion
        if mag_mean > 0.01:
            cv = mag_std / (mag_mean + 1e-6)  # Coefficient of variation
            smoothness = max(0.0, 1.0 - cv / 2.0)
        else:
            smoothness = 1.0

        motion = 0.5 * diversity_score + 0.5 * smoothness

        details["motion_magnitude_mean"] = mag_mean
        details["motion_magnitude_std"] = mag_std
        details["motion_smoothness"] = smoothness
        details["motion_diversity"] = diversity_score

        return float(np.clip(motion, 0.0, 1.0))

    def _score_prompt_alignment(
        self,
        pil_frames: List[Image.Image],
        key_idx: List[int],
        prompt: str,
        details: Dict[str, float],
    ) -> float:
        """Text-video alignment using CLIP cosine similarity.

        Returns 0.0 if CLIP is unavailable or no prompt is provided.
        """
        if not prompt:
            details["prompt_alignment_note"] = "no prompt provided"
            return 0.0

        # Lazy-load CLIP
        if not self._clip_checked:
            self._clip_checked = True
            result = _try_load_clip(self.device)
            if result is not None:
                self._clip_model, self._clip_processor = result

        if self._clip_model is None:
            details["prompt_alignment_note"] = "CLIP not available"
            return 0.0

        try:
            import torch

            clip_scores = []
            for idx in key_idx:
                inputs = self._clip_processor(
                    text=[prompt],
                    images=[pil_frames[idx]],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                # Move to device
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._clip_model(**inputs)
                    # Cosine similarity between text and image embeddings
                    logits = outputs.logits_per_image  # (1, 1)
                    # CLIP logits are already cosine_sim * 100
                    sim = float(logits[0, 0].cpu()) / 100.0
                    clip_scores.append(sim)

            raw_mean = float(np.mean(clip_scores))
            # CLIP cosine similarities for matching pairs typically range 0.20-0.35
            # Map to [0, 1]: 0.15 -> 0.0, 0.35 -> 1.0
            normalised = float(np.clip((raw_mean - 0.15) / 0.20, 0.0, 1.0))

            details["clip_raw_mean"] = raw_mean
            details["clip_per_frame"] = clip_scores

            logger.debug(f"CLIP alignment: raw={raw_mean:.3f} -> normalised={normalised:.3f}")
            return normalised

        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            details["prompt_alignment_note"] = f"CLIP error: {e}"
            return 0.0
