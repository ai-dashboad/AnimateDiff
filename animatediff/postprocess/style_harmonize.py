"""
Style Harmonize --- cross-shot visual consistency for multi-shot video generation.

Ensures that all shots in a storyboard share a consistent visual style:
colour palette, brightness, contrast, and optional texture/grain characteristics.

Backends:
- histogram_match:  Reinhard-style colour transfer in LAB space (PIL + numpy only)
- color_palette:    Extract dominant colours via k-means and transfer palettes (numpy only)
- neural_style:     Neural style transfer using VGG features (requires torch)
- auto:             Picks the best available backend

All methods operate on lists of PIL Images, matching the post-processing conventions
used by deflicker.py, interpolation.py, and compositor.py.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LAB conversion helpers (shared with deflicker.py, duplicated here
# to keep the module self-contained and dependency-free)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> linear RGB [0,1]."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear RGB [0,1] -> sRGB [0,1]."""
    return np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.power(np.maximum(c, 1e-10), 1.0 / 2.4) - 0.055,
    )


def _rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] image (H,W,3) to CIE-LAB (H,W,3).

    L in [0,100], a/b roughly in [-128,127].
    """
    linear = _srgb_to_linear(img_rgb)

    # RGB -> XYZ (D65 illuminant, sRGB primaries)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ M.T

    # Normalise to D65 white point
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 1] /= 1.00000
    xyz[:, :, 2] /= 1.08883

    epsilon = 0.008856
    kappa = 903.3
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])

    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    """Convert CIE-LAB (H,W,3) back to float32 RGB [0,1]."""
    L, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    epsilon = 0.008856
    kappa = 903.3

    xr = np.where(fx ** 3 > epsilon, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * epsilon, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > epsilon, fz ** 3, (116.0 * fz - 16.0) / kappa)

    x = xr * 0.95047
    y = yr * 1.00000
    z = zr * 1.08883

    xyz = np.stack([x, y, z], axis=-1)

    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    linear = xyz @ M_inv.T

    rgb = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# StyleDescriptor
# ---------------------------------------------------------------------------

@dataclass
class StyleDescriptor:
    """Captures the visual style of a shot for cross-shot harmonization.

    All statistics are computed in CIE-LAB colour space for perceptual
    uniformity. Colour palette entries are stored as LAB triplets.
    """

    # Per-channel (L, a, b) statistics
    lab_mean: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    lab_std: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))

    # Brightness / contrast (derived from L channel)
    brightness: float = 50.0       # mean L
    contrast: float = 20.0         # std L

    # Dominant colour palette: (K, 3) array of LAB colours + (K,) weights
    palette_lab: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    palette_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Optional texture / grain statistics (std of high-frequency residual)
    grain_intensity: float = 0.0   # mean absolute Laplacian magnitude

    # Number of frames and pixels this descriptor was computed from
    num_frames: int = 0
    num_pixels: int = 0

    def __repr__(self) -> str:
        palette_str = f"{len(self.palette_lab)} colours" if len(self.palette_lab) else "none"
        return (
            f"StyleDescriptor("
            f"L={self.brightness:.1f}+/-{self.contrast:.1f}, "
            f"a={self.lab_mean[1]:.1f}+/-{self.lab_std[1]:.1f}, "
            f"b={self.lab_mean[2]:.1f}+/-{self.lab_std[2]:.1f}, "
            f"palette={palette_str}, grain={self.grain_intensity:.3f}, "
            f"frames={self.num_frames})"
        )


# ---------------------------------------------------------------------------
# K-means for palette extraction (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

def _kmeans_numpy(
    data: np.ndarray,
    k: int = 6,
    max_iter: int = 30,
    seed: int = 42,
) -> tuple:
    """Run k-means clustering on (N, D) float data.

    Returns:
        (centroids (k, D), labels (N,), weights (k,))
    """
    rng = np.random.RandomState(seed)
    n = data.shape[0]
    if n == 0 or k <= 0:
        return np.zeros((0, data.shape[1])), np.zeros(0, dtype=int), np.zeros(0)

    k = min(k, n)

    # k-means++ initialisation
    idx = [rng.randint(n)]
    for _ in range(1, k):
        dists = np.min(
            np.sum((data[:, None, :] - data[idx][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = dists.sum()
        if total < 1e-12:
            # All remaining points are identical to existing centroids; pick random
            idx.append(rng.randint(n))
        else:
            prob = dists / total
            # Ensure valid probability distribution (handle floating-point drift)
            prob = np.clip(prob, 0, None)
            prob = prob / prob.sum()
            idx.append(rng.choice(n, p=prob))

    centroids = data[idx].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = data[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    # Compute weights (fraction of pixels in each cluster)
    weights = np.zeros(k, dtype=np.float32)
    for j in range(k):
        weights[j] = (labels == j).sum()
    weights = weights / (weights.sum() + 1e-10)

    return centroids, labels, weights


def _compute_grain(img_rgb: np.ndarray) -> float:
    """Estimate texture/grain intensity via Laplacian magnitude.

    Uses a simple 3x3 Laplacian kernel on the luminance channel.
    Returns the mean absolute value.
    """
    # Convert to grayscale
    gray = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]

    # Laplacian kernel via convolution (numpy-only)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape

    # Pad and convolve
    padded = np.pad(gray, 1, mode="reflect")
    lap = np.zeros_like(gray)
    for di in range(3):
        for dj in range(3):
            lap += padded[di:di + h, dj:dj + w] * kernel[di, dj]

    return float(np.abs(lap).mean())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StyleHarmonizer:
    """Harmonize visual style across multiple video shots."""

    BACKENDS = ("histogram_match", "color_palette", "neural_style", "auto")

    def __init__(
        self,
        backend: Literal["histogram_match", "color_palette", "neural_style", "auto"] = "auto",
        device: str = "auto",
    ):
        """
        Args:
            backend:
                - histogram_match: Match histograms to a reference style (always available).
                - color_palette:   Extract and transfer dominant colour palettes (always available).
                - neural_style:    Use VGG-based neural style transfer (requires torch).
                - auto:            Best available backend.
            device:
                - auto: pick CUDA > MPS > CPU
                - cuda / mps / cpu: force a device
        """
        self._requested_backend = backend
        self.device = self._resolve_device(device)
        self.backend = self._resolve_backend(backend)
        self._vgg_model = None
        logger.info(f"StyleHarmonizer ready  backend={self.backend}  device={self.device}")

    # ----- device / backend resolution ----------------------------------------

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

    def _resolve_backend(self, backend: str) -> str:
        if backend == "histogram_match":
            return "histogram_match"
        if backend == "color_palette":
            return "color_palette"
        if backend == "neural_style":
            self._assert_torch("neural_style backend")
            return "neural_style"

        # auto: try neural_style -> histogram_match
        try:
            self._assert_torch("auto backend probe")
            return "neural_style"
        except ImportError:
            logger.info("torch unavailable; style harmonizer will use histogram_match backend")
            return "histogram_match"

    @staticmethod
    def _assert_torch(context: str):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                f"PyTorch is required for {context}. "
                "Install it or use backend='histogram_match'."
            )

    # ======================================================================
    # Public API
    # ======================================================================

    def extract_style(
        self,
        frames: List[Image.Image],
        palette_k: int = 6,
        sample_stride: int = 1,
    ) -> StyleDescriptor:
        """Extract a style descriptor from a shot's frames.

        Args:
            frames:        List of PIL Images (RGB) for one shot.
            palette_k:     Number of dominant colours to extract.
            sample_stride: Process every Nth frame (for speed on long shots).

        Returns:
            A StyleDescriptor capturing the visual statistics.
        """
        if not frames:
            return StyleDescriptor()

        sampled = frames[::max(1, sample_stride)]

        all_lab_means: List[np.ndarray] = []
        all_lab_stds: List[np.ndarray] = []
        all_grains: List[float] = []
        pixel_samples: List[np.ndarray] = []
        total_pixels = 0

        for img in sampled:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            lab = _rgb_to_lab(arr)

            flat = lab.reshape(-1, 3)
            mean = flat.mean(axis=0)
            std = flat.std(axis=0) + 1e-6

            all_lab_means.append(mean)
            all_lab_stds.append(std)
            all_grains.append(_compute_grain(arr))
            total_pixels += flat.shape[0]

            # Sub-sample pixels for palette clustering (max ~10k per frame)
            stride = max(1, flat.shape[0] // 10000)
            pixel_samples.append(flat[::stride])

        # Aggregate statistics
        agg_mean = np.mean(all_lab_means, axis=0).astype(np.float32)
        agg_std = np.mean(all_lab_stds, axis=0).astype(np.float32)

        # Palette extraction
        all_pixels = np.concatenate(pixel_samples, axis=0)
        # Limit total pixels for k-means to ~50k
        if all_pixels.shape[0] > 50000:
            idx = np.random.RandomState(42).choice(all_pixels.shape[0], 50000, replace=False)
            all_pixels = all_pixels[idx]

        palette_lab, _, palette_weights = _kmeans_numpy(all_pixels, k=palette_k)

        # Sort palette by weight (most dominant first)
        if len(palette_weights) > 0:
            order = np.argsort(-palette_weights)
            palette_lab = palette_lab[order]
            palette_weights = palette_weights[order]

        return StyleDescriptor(
            lab_mean=agg_mean,
            lab_std=agg_std,
            brightness=float(agg_mean[0]),
            contrast=float(agg_std[0]),
            palette_lab=palette_lab.astype(np.float32),
            palette_weights=palette_weights.astype(np.float32),
            grain_intensity=float(np.mean(all_grains)),
            num_frames=len(sampled),
            num_pixels=total_pixels,
        )

    def compute_reference_style(
        self,
        all_shots: List[List[Image.Image]],
        palette_k: int = 6,
        sample_stride: int = 2,
    ) -> StyleDescriptor:
        """Compute a unified reference style from all shots.

        The reference is the weighted median of all per-shot styles, which
        minimises the total adjustment needed across the storyboard.

        Args:
            all_shots:     List of shot frame-lists.
            palette_k:     Number of palette colours.
            sample_stride: Frame sampling stride per shot.

        Returns:
            A unified StyleDescriptor that all shots should target.
        """
        if not all_shots:
            return StyleDescriptor()

        descriptors = [
            self.extract_style(shot, palette_k=palette_k, sample_stride=sample_stride)
            for shot in all_shots
        ]

        # Weight by number of frames (longer shots contribute more)
        total_frames = sum(d.num_frames for d in descriptors)
        if total_frames == 0:
            total_frames = 1
        weights = np.array([d.num_frames / total_frames for d in descriptors], dtype=np.float32)

        # Weighted average of statistics
        ref_mean = np.zeros(3, dtype=np.float32)
        ref_std = np.zeros(3, dtype=np.float32)
        ref_grain = 0.0

        for d, w in zip(descriptors, weights):
            ref_mean += d.lab_mean * w
            ref_std += d.lab_std * w
            ref_grain += d.grain_intensity * w

        # Merge palettes: collect all palette entries weighted, then re-cluster
        all_palette_pixels = []
        for d, w in zip(descriptors, weights):
            if len(d.palette_lab) > 0:
                # Repeat each centre proportionally to its weight in the shot
                repeat_count = np.maximum((d.palette_weights * w * 1000).astype(int), 1)
                for lab, rc in zip(d.palette_lab, repeat_count):
                    all_palette_pixels.extend([lab] * rc)

        if all_palette_pixels:
            palette_data = np.array(all_palette_pixels, dtype=np.float32)
            ref_palette, _, ref_palette_w = _kmeans_numpy(palette_data, k=palette_k)
            order = np.argsort(-ref_palette_w)
            ref_palette = ref_palette[order]
            ref_palette_w = ref_palette_w[order]
        else:
            ref_palette = np.zeros((0, 3), dtype=np.float32)
            ref_palette_w = np.zeros(0, dtype=np.float32)

        ref = StyleDescriptor(
            lab_mean=ref_mean,
            lab_std=ref_std,
            brightness=float(ref_mean[0]),
            contrast=float(ref_std[0]),
            palette_lab=ref_palette.astype(np.float32),
            palette_weights=ref_palette_w.astype(np.float32),
            grain_intensity=ref_grain,
            num_frames=sum(d.num_frames for d in descriptors),
            num_pixels=sum(d.num_pixels for d in descriptors),
        )

        logger.info(f"Computed reference style from {len(all_shots)} shots: {ref}")
        return ref

    def harmonize(
        self,
        frames: List[Image.Image],
        target_style: StyleDescriptor,
        strength: float = 0.5,
    ) -> List[Image.Image]:
        """Apply target style to frames with given strength.

        Args:
            frames:       List of PIL Images (RGB) for one shot.
            target_style: The desired visual style (from extract_style or compute_reference_style).
            strength:     0.0 (no change) to 1.0 (full style transfer).

        Returns:
            New list of harmonized PIL Images.
        """
        if not frames:
            return []

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == 0.0:
            return list(frames)

        logger.info(
            f"Harmonizing {len(frames)} frames  "
            f"backend={self.backend}  strength={strength:.2f}"
        )

        if self.backend == "histogram_match":
            return self._harmonize_histogram(frames, target_style, strength)
        elif self.backend == "color_palette":
            return self._harmonize_palette(frames, target_style, strength)
        elif self.backend == "neural_style":
            return self._harmonize_neural(frames, target_style, strength)
        else:
            return self._harmonize_histogram(frames, target_style, strength)

    def harmonize_all_shots(
        self,
        shot_frames: List[List[Image.Image]],
        strength: float = 0.5,
        reference_style: Optional[StyleDescriptor] = None,
        palette_k: int = 6,
    ) -> List[List[Image.Image]]:
        """Harmonize all shots toward a unified style.

        If no reference_style is provided, one is computed automatically
        from the median of all shots.

        Args:
            shot_frames:     List of shot frame-lists.
            strength:        Harmonization strength (0.0 to 1.0).
            reference_style: Optional pre-computed reference; computed if None.
            palette_k:       Number of palette colours for style extraction.

        Returns:
            New list of shot frame-lists with harmonized visuals.
        """
        if len(shot_frames) <= 1:
            return [list(s) for s in shot_frames]

        if reference_style is None:
            reference_style = self.compute_reference_style(
                shot_frames, palette_k=palette_k,
            )

        logger.info(
            f"Harmonizing {len(shot_frames)} shots "
            f"(strength={strength:.2f}, backend={self.backend})"
        )

        result: List[List[Image.Image]] = []
        for i, shot in enumerate(shot_frames):
            logger.debug(f"  Harmonizing shot {i + 1}/{len(shot_frames)} ({len(shot)} frames)")
            harmonized = self.harmonize(shot, reference_style, strength)
            result.append(harmonized)

        return result

    # ======================================================================
    # Backend 1 -- Histogram matching (Reinhard LAB transfer)
    # ======================================================================

    def _harmonize_histogram(
        self,
        frames: List[Image.Image],
        target: StyleDescriptor,
        strength: float,
    ) -> List[Image.Image]:
        """Reinhard-style LAB colour transfer toward target statistics.

        Shifts each frame's per-channel mean and standard deviation to match
        the target, blended by strength.
        """
        result: List[Image.Image] = []

        for idx, img in enumerate(frames):
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            lab = _rgb_to_lab(arr)

            flat = lab.reshape(-1, 3)
            cur_mean = flat.mean(axis=0)
            cur_std = flat.std(axis=0) + 1e-6

            # Reinhard transfer
            adjusted = (lab - cur_mean) * (target.lab_std / cur_std) + target.lab_mean

            # Blend original and adjusted
            blended = lab * (1.0 - strength) + adjusted * strength

            rgb = _lab_to_rgb(blended)
            result.append(
                Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))
            )

            if (idx + 1) % 50 == 0:
                logger.debug(f"  Histogram harmonize: {idx + 1}/{len(frames)}")

        return result

    # ======================================================================
    # Backend 2 -- Colour palette transfer
    # ======================================================================

    def _harmonize_palette(
        self,
        frames: List[Image.Image],
        target: StyleDescriptor,
        strength: float,
    ) -> List[Image.Image]:
        """Transfer colour palette from target to each frame.

        For each pixel, find its nearest palette colour in the source palette,
        then shift it toward the corresponding target palette colour. Falls
        back to histogram matching if the target has no palette.
        """
        if len(target.palette_lab) == 0:
            logger.warning("Target style has no palette; falling back to histogram_match")
            return self._harmonize_histogram(frames, target, strength)

        target_palette = target.palette_lab  # (K, 3)

        result: List[Image.Image] = []

        for idx, img in enumerate(frames):
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            lab = _rgb_to_lab(arr)
            h, w, _ = lab.shape
            flat = lab.reshape(-1, 3)

            # Find nearest target palette colour for each pixel
            # (N, K) distance matrix
            dists = np.sum(
                (flat[:, None, :] - target_palette[None, :, :]) ** 2, axis=2
            )
            nearest = np.argmin(dists, axis=1)  # (N,)

            # Compute per-cluster shift
            # For each cluster, compute the difference between target centre
            # and the mean of pixels assigned to that cluster
            adjusted = flat.copy()
            for k in range(len(target_palette)):
                mask = nearest == k
                if mask.sum() == 0:
                    continue
                cluster_pixels = flat[mask]
                cluster_mean = cluster_pixels.mean(axis=0)
                shift = target_palette[k] - cluster_mean
                adjusted[mask] = cluster_pixels + shift * strength

            blended_lab = flat * (1.0 - strength) + adjusted * strength
            blended_lab = blended_lab.reshape(h, w, 3)

            rgb = _lab_to_rgb(blended_lab)
            result.append(
                Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))
            )

            if (idx + 1) % 50 == 0:
                logger.debug(f"  Palette harmonize: {idx + 1}/{len(frames)}")

        return result

    # ======================================================================
    # Backend 3 -- Neural style transfer (VGG Gram matrices)
    # ======================================================================

    def _harmonize_neural(
        self,
        frames: List[Image.Image],
        target: StyleDescriptor,
        strength: float,
    ) -> List[Image.Image]:
        """Neural style harmonization using VGG feature statistics.

        Matches the Gram matrices (second-order feature statistics) of each
        frame to those of a reference style image synthesized from the target
        descriptor. This captures texture and colour correlations that
        histogram matching cannot.

        For efficiency, each frame undergoes a fixed number of optimisation
        steps (not full artistic style transfer).
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # First pass: use histogram matching as the base correction
        # Neural refinement adjusts texture/correlation on top
        base_harmonized = self._harmonize_histogram(frames, target, strength)

        # Load VGG for feature extraction
        if self._vgg_model is None:
            self._vgg_model = self._load_vgg()

        if self._vgg_model is None:
            logger.warning("VGG model unavailable; returning histogram-harmonized frames")
            return base_harmonized

        # Extract target style Gram matrices from a synthetic reference frame
        # We create a reference by applying full histogram transfer to the median frame
        mid_idx = len(frames) // 2
        ref_frame = base_harmonized[mid_idx]
        ref_grams = self._extract_gram_matrices(ref_frame)

        result: List[Image.Image] = []

        # Neural refinement strength: scale down to keep adjustments subtle
        neural_weight = strength * 0.3
        num_steps = 10  # few steps for speed

        for idx, img in enumerate(base_harmonized):
            refined = self._neural_refine_frame(
                img, ref_grams, weight=neural_weight, steps=num_steps,
            )
            result.append(refined)

            if (idx + 1) % 20 == 0:
                logger.debug(f"  Neural harmonize: {idx + 1}/{len(frames)}")

        return result

    def _load_vgg(self):
        """Load VGG16 feature extractor for style transfer."""
        try:
            import torch
            from torchvision import models

            vgg = models.vgg16(weights="IMAGENET1K_V1").features[:16].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            vgg = vgg.to(self.device)
            logger.info("Loaded VGG16 for neural style harmonization")
            return vgg
        except Exception as e:
            logger.warning(f"Could not load VGG16: {e}")
            return None

    def _extract_gram_matrices(self, img: Image.Image) -> list:
        """Extract Gram matrices from VGG feature maps for a single image."""
        import torch

        tensor = self._pil_to_tensor(img)
        features = []
        x = tensor
        for i, layer in enumerate(self._vgg_model):
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                features.append(x)

        grams = []
        for feat in features:
            b, c, h, w = feat.shape
            f = feat.view(c, h * w)
            gram = torch.mm(f, f.t()) / (c * h * w)
            grams.append(gram.detach())

        return grams

    def _neural_refine_frame(
        self,
        img: Image.Image,
        target_grams: list,
        weight: float = 0.1,
        steps: int = 10,
    ) -> Image.Image:
        """Refine a single frame by minimising Gram matrix difference."""
        import torch
        import torch.nn.functional as F

        canvas = self._pil_to_tensor(img).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([canvas], lr=0.01)

        original = self._pil_to_tensor(img).detach()

        for _ in range(steps):
            optimizer.zero_grad()

            # Content loss (keep close to input)
            content_loss = F.mse_loss(canvas, original)

            # Style loss (match Gram matrices)
            features = []
            x = canvas
            for layer in self._vgg_model:
                x = layer(x)
                if isinstance(layer, torch.nn.ReLU):
                    features.append(x)

            style_loss = torch.tensor(0.0, device=self.device)
            for feat, target_gram in zip(features, target_grams):
                b, c, h, w = feat.shape
                f = feat.view(c, h * w)
                gram = torch.mm(f, f.t()) / (c * h * w)
                style_loss = style_loss + F.mse_loss(gram, target_gram)

            loss = content_loss + weight * style_loss
            loss.backward()
            optimizer.step()

        result = canvas.detach().clamp(0, 1)
        return self._tensor_to_pil(result)

    def _pil_to_tensor(self, img: Image.Image):
        """Convert PIL Image to tensor (1, 3, H, W) float32 [0, 1]."""
        import torch
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _tensor_to_pil(tensor) -> Image.Image:
        """Convert tensor (1, 3, H, W) to PIL Image."""
        arr = (
            tensor.squeeze(0)
            .clamp(0, 1)
            .cpu()
            .numpy()
            .transpose(1, 2, 0) * 255
        ).astype(np.uint8)
        return Image.fromarray(arr)
