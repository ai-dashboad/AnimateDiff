"""
Video Deflicker — reduce temporal flickering within and across shots.

Strategies:
- histogram:     Fast LAB histogram matching between consecutive frames (PIL + numpy only)
- optical_flow:  Optical-flow-guided temporal blending (requires torch)
- neural:        Placeholder for future INR-Smooth / deep temporal consistency models
- auto:          Tries neural -> optical_flow -> histogram

Cross-shot harmonization matches color distributions across shot boundaries
so that cuts between different lighting or color palettes feel smooth.
"""

import logging
from typing import List, Literal, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — LAB conversion (pure numpy, no cv2 dependency)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> linear RGB [0,1]."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear RGB [0,1] -> sRGB [0,1]."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 1e-10), 1.0 / 2.4) - 0.055)


def _rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] image (H,W,3) to CIE-LAB (H,W,3).

    L in [0,100], a/b roughly in [-128,127].
    """
    linear = _srgb_to_linear(img_rgb)

    # RGB -> XYZ (D65 illuminant, sRGB primaries)
    # fmt: off
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    # fmt: on
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

    # XYZ -> linear RGB
    # fmt: off
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    # fmt: on
    linear = xyz @ M_inv.T

    rgb = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    return np.clip(rgb, 0.0, 1.0)


def _compute_lab_stats(img_lab: np.ndarray):
    """Return per-channel (mean, std) for a LAB image."""
    means = img_lab.reshape(-1, 3).mean(axis=0)
    stds = img_lab.reshape(-1, 3).std(axis=0) + 1e-6
    return means, stds


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VideoDeflicker:
    """Remove temporal flickering from video frame sequences."""

    def __init__(
        self,
        backend: Literal["histogram", "optical_flow", "neural", "auto"] = "auto",
        device: str = "auto",
    ):
        """
        Args:
            backend:
                - histogram:     Fast, lightweight histogram matching in LAB space.
                                 No dependencies beyond PIL + numpy.
                - optical_flow:  Optical-flow-guided temporal consistency (needs torch).
                - neural:        Placeholder for future INR-Smooth integration.
                - auto:          Tries neural -> optical_flow -> histogram.
            device:
                - auto: pick CUDA > MPS > CPU
                - cuda / mps / cpu: force a device
        """
        self._requested_backend = backend
        self.device = self._resolve_device(device)
        self.backend = self._resolve_backend(backend)
        logger.info(f"VideoDeflicker ready  backend={self.backend}  device={self.device}")

    # ----- device / backend resolution ------------------------------------

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
        if backend == "histogram":
            return "histogram"
        if backend == "optical_flow":
            self._assert_torch("optical_flow backend")
            return "optical_flow"
        if backend == "neural":
            logger.warning(
                "Neural deflicker backend is a placeholder — "
                "falling back to optical_flow -> histogram"
            )
            return self._resolve_backend("auto")

        # auto: try neural -> optical_flow -> histogram
        try:
            self._assert_torch("auto backend probe")
            return "optical_flow"
        except ImportError:
            logger.info("torch unavailable; deflicker will use histogram backend")
            return "histogram"

    @staticmethod
    def _assert_torch(context: str):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                f"PyTorch is required for {context}. "
                "Install it or use backend='histogram'."
            )

    # ======================================================================
    # Public API
    # ======================================================================

    def deflicker_shot(
        self,
        frames: List[Image.Image],
        strength: float = 0.5,
    ) -> List[Image.Image]:
        """Deflicker a single shot's frames.

        Args:
            frames:   List of PIL Images (RGB) representing one shot.
            strength: 0.0 (no change) to 1.0 (maximum temporal smoothing).

        Returns:
            New list of deflickered PIL Images.
        """
        if len(frames) < 2:
            return list(frames)

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == 0.0:
            return list(frames)

        logger.info(
            f"Deflickering shot: {len(frames)} frames, "
            f"strength={strength:.2f}, backend={self.backend}"
        )

        if self.backend == "optical_flow":
            return self._deflicker_optical_flow(frames, strength)
        else:
            return self._deflicker_histogram(frames, strength)

    def harmonize_shots(
        self,
        shot_frames: List[List[Image.Image]],
        overlap: int = 5,
    ) -> List[List[Image.Image]]:
        """Harmonize color / brightness across multiple shots.

        For each pair of adjacent shots the method analyses the last *overlap*
        frames of shot A and the first *overlap* frames of shot B.  It then
        computes a LAB color transfer and gradually applies it to both sides
        so the transition feels natural.

        Args:
            shot_frames: List of shot frame-lists.
            overlap:     Number of boundary frames to consider on each side.

        Returns:
            A new list of shot frame-lists with harmonized colours.
        """
        if len(shot_frames) <= 1:
            return [list(s) for s in shot_frames]

        overlap = max(1, overlap)
        logger.info(
            f"Harmonizing {len(shot_frames)} shots with overlap={overlap}"
        )

        result: List[List[Image.Image]] = [list(shot_frames[0])]

        for i in range(1, len(shot_frames)):
            prev_shot = result[i - 1]
            next_shot = list(shot_frames[i])

            # Collect boundary frames
            n_prev = min(overlap, len(prev_shot))
            n_next = min(overlap, len(next_shot))
            tail_frames = prev_shot[-n_prev:]
            head_frames = next_shot[:n_next]

            # Compute aggregate LAB statistics for each side
            tail_lab_stats = self._aggregate_lab_stats(tail_frames)
            head_lab_stats = self._aggregate_lab_stats(head_frames)

            # Midpoint statistics — the target both sides will blend toward
            mid_mean = (tail_lab_stats[0] + head_lab_stats[0]) / 2.0
            mid_std = (tail_lab_stats[1] + head_lab_stats[1]) / 2.0

            # Gradually adjust the tail of the previous shot toward midpoint
            for j in range(n_prev):
                # Ramp from 0 (no change, far from boundary) to 1 (full change)
                t = (j + 1) / (n_prev + 1)
                prev_shot[-(n_prev - j)] = self._adjust_frame_lab(
                    prev_shot[-(n_prev - j)], mid_mean, mid_std, blend=t * 0.5,
                )

            # Gradually adjust the head of the next shot toward midpoint
            for j in range(n_next):
                t = 1.0 - j / (n_next + 1)
                next_shot[j] = self._adjust_frame_lab(
                    next_shot[j], mid_mean, mid_std, blend=t * 0.5,
                )

            result[i - 1] = prev_shot
            result.append(next_shot)

        return result

    def match_color_distribution(
        self,
        source: Image.Image,
        target: Image.Image,
    ) -> Image.Image:
        """Match the colour distribution of *source* to *target*.

        Uses Reinhard-style colour transfer in LAB space:
        the mean and standard deviation of each LAB channel of *source*
        is shifted to match those of *target*.

        Args:
            source: The image whose colours will be changed.
            target: The reference image whose colour distribution to match.

        Returns:
            A new PIL Image with source content but target colour palette.
        """
        src_arr = np.array(source, dtype=np.float32) / 255.0
        tgt_arr = np.array(target, dtype=np.float32) / 255.0

        src_lab = _rgb_to_lab(src_arr)
        tgt_lab = _rgb_to_lab(tgt_arr)

        src_mean, src_std = _compute_lab_stats(src_lab)
        tgt_mean, tgt_std = _compute_lab_stats(tgt_lab)

        result_lab = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
        result_rgb = _lab_to_rgb(result_lab)
        return Image.fromarray((result_rgb * 255).astype(np.uint8))

    # ======================================================================
    # Strategy 1 — Histogram matching (LAB space, pure numpy)
    # ======================================================================

    def _deflicker_histogram(
        self,
        frames: List[Image.Image],
        strength: float,
    ) -> List[Image.Image]:
        """Temporal histogram smoothing in LAB colour space.

        Maintains a running exponential average of LAB channel statistics
        (mean, std) and nudges each frame toward that running average.
        This removes frame-to-frame brightness / colour flicker while
        preserving intentional scene-level changes.
        """
        alpha = 0.15 + 0.65 * strength  # EMA weight: higher = more smoothing

        # Convert first frame to LAB and seed the running statistics
        first_arr = np.array(frames[0], dtype=np.float32) / 255.0
        first_lab = _rgb_to_lab(first_arr)
        run_mean, run_std = _compute_lab_stats(first_lab)

        result: List[Image.Image] = [frames[0]]

        for idx in range(1, len(frames)):
            arr = np.array(frames[idx], dtype=np.float32) / 255.0
            lab = _rgb_to_lab(arr)
            cur_mean, cur_std = _compute_lab_stats(lab)

            # Update running stats (EMA)
            run_mean = run_mean * alpha + cur_mean * (1 - alpha)
            run_std = run_std * alpha + cur_std * (1 - alpha)

            # Shift this frame's distribution toward the running average
            adjusted = (lab - cur_mean) * (run_std / cur_std) + run_mean

            # Blend between original and adjusted based on strength
            blended_lab = lab * (1 - strength) + adjusted * strength

            rgb = _lab_to_rgb(blended_lab)
            result.append(
                Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))
            )

            if (idx + 1) % 50 == 0:
                logger.debug(f"  Histogram deflicker: {idx + 1}/{len(frames)}")

        return result

    # ======================================================================
    # Strategy 2 — Optical flow based (torch)
    # ======================================================================

    def _deflicker_optical_flow(
        self,
        frames: List[Image.Image],
        strength: float,
    ) -> List[Image.Image]:
        """Optical-flow-guided temporal blending.

        For each frame we:
        1. Estimate dense optical flow from the previous frame.
        2. Warp the previous (already-deflickered) frame toward the current.
        3. Blend the warped previous frame with the current frame.

        This preserves motion while smoothing brightness / colour flicker.
        """
        import torch
        import torch.nn.functional as F

        blend_weight = 0.2 + 0.5 * strength  # how much of warped-prev to mix in

        def pil_to_tensor(img: Image.Image) -> torch.Tensor:
            """(H,W,3) uint8 PIL -> (1,3,H,W) float32 [0,1]."""
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        def tensor_to_pil(t: torch.Tensor) -> Image.Image:
            arr = (
                t.squeeze(0)
                .clamp(0, 1)
                .cpu()
                .numpy()
                .transpose(1, 2, 0) * 255
            ).astype(np.uint8)
            return Image.fromarray(arr)

        def estimate_flow(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
            """Estimate backward optical flow (cur -> prev) using phase correlation.

            Returns a (1,2,H,W) flow field in pixel units.
            This is a lightweight gradient-based approach — not as accurate as
            RAFT but requires no extra model weights.
            """
            # Convert to grayscale
            weights = torch.tensor([0.299, 0.587, 0.114], device=prev.device).view(1, 3, 1, 1)
            gray_prev = (prev * weights).sum(dim=1, keepdim=True)
            gray_cur = (cur * weights).sum(dim=1, keepdim=True)

            # Compute spatial gradients of current frame
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32, device=prev.device,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32, device=prev.device,
            ).view(1, 1, 3, 3)

            Ix = F.conv2d(gray_cur, sobel_x, padding=1)
            Iy = F.conv2d(gray_cur, sobel_y, padding=1)
            It = gray_cur - gray_prev

            # Lucas-Kanade style: solve in local 5x5 windows via pooling
            k = 5
            pool = lambda t: F.avg_pool2d(t, k, stride=1, padding=k // 2)

            Ix2 = pool(Ix * Ix) + 1e-6
            Iy2 = pool(Iy * Iy) + 1e-6
            Ixy = pool(Ix * Iy)
            Ixt = pool(Ix * It)
            Iyt = pool(Iy * It)

            det = Ix2 * Iy2 - Ixy * Ixy + 1e-8
            u = -(Iy2 * Ixt - Ixy * Iyt) / det
            v = -(Ix2 * Iyt - Ixy * Ixt) / det

            # Clamp extreme flows
            max_flow = 15.0
            u = u.clamp(-max_flow, max_flow)
            v = v.clamp(-max_flow, max_flow)

            return torch.cat([u, v], dim=1)  # (1,2,H,W)

        def warp_with_flow(
            img: torch.Tensor, flow: torch.Tensor,
        ) -> torch.Tensor:
            """Warp *img* (1,C,H,W) using *flow* (1,2,H,W) in pixel units."""
            _, _, H, W = img.shape
            # Build base grid in [-1,1]
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=img.device),
                torch.linspace(-1, 1, W, device=img.device),
                indexing="ij",
            )
            grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1,H,W,2)

            # Convert pixel flow to normalised flow
            flow_norm = torch.zeros_like(flow)
            flow_norm[:, 0] = flow[:, 0] / (W / 2)
            flow_norm[:, 1] = flow[:, 1] / (H / 2)

            # flow is (1,2,H,W) -> (1,H,W,2) for grid_sample
            flow_perm = flow_norm.permute(0, 2, 3, 1)
            sample_grid = grid + flow_perm

            # MPS doesn't support "border" padding — fall back to "zeros"
            pad_mode = "zeros" if img.device.type == "mps" else "border"
            return F.grid_sample(
                img, sample_grid, mode="bilinear", padding_mode=pad_mode,
                align_corners=True,
            )

        # First frame is kept as-is
        prev_tensor = pil_to_tensor(frames[0])
        result: List[Image.Image] = [frames[0]]

        for idx in range(1, len(frames)):
            cur_tensor = pil_to_tensor(frames[idx])

            with torch.no_grad():
                flow = estimate_flow(prev_tensor, cur_tensor)
                warped_prev = warp_with_flow(prev_tensor, flow)

                # Blend warped-previous with current
                blended = cur_tensor * (1 - blend_weight) + warped_prev * blend_weight
                blended = blended.clamp(0, 1)

            result.append(tensor_to_pil(blended))
            prev_tensor = blended  # chain: use the smoothed frame going forward

            if (idx + 1) % 50 == 0:
                logger.debug(f"  Optical-flow deflicker: {idx + 1}/{len(frames)}")

        return result

    # ======================================================================
    # Cross-shot harmonization helpers
    # ======================================================================

    def _aggregate_lab_stats(
        self, frames: List[Image.Image],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute average LAB (mean, std) across a list of frames."""
        all_means: List[np.ndarray] = []
        all_stds: List[np.ndarray] = []
        for f in frames:
            arr = np.array(f, dtype=np.float32) / 255.0
            lab = _rgb_to_lab(arr)
            m, s = _compute_lab_stats(lab)
            all_means.append(m)
            all_stds.append(s)
        agg_mean = np.mean(all_means, axis=0)
        agg_std = np.mean(all_stds, axis=0)
        return agg_mean, agg_std

    @staticmethod
    def _adjust_frame_lab(
        frame: Image.Image,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        blend: float = 1.0,
    ) -> Image.Image:
        """Shift a frame's LAB distribution toward a target, by *blend* fraction."""
        arr = np.array(frame, dtype=np.float32) / 255.0
        lab = _rgb_to_lab(arr)
        cur_mean, cur_std = _compute_lab_stats(lab)

        adjusted = (lab - cur_mean) * (target_std / cur_std) + target_mean
        blended = lab * (1.0 - blend) + adjusted * blend

        rgb = _lab_to_rgb(blended)
        return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))
