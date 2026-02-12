"""
Video Extend --- extend short video clips to longer duration.

Strategies:
- loop_blend:         Seamless loop with crossfade blending (always available, PIL + numpy only)
- reverse_bounce:     Play forward then reverse for smooth extension (PIL + numpy only)
- interpolation:      Use RIFE frame interpolation to slow down / temporally upsample
- vace_continuation:  Use Wan VACE backend for generative continuation (requires torch + diffusers)
- auto:               Picks the best available method

All methods operate on lists of PIL Images, matching the post-processing conventions
used by deflicker.py, interpolation.py, and compositor.py.
"""

import logging
import math
import os
import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_array(img: Image.Image) -> np.ndarray:
    """PIL Image -> float32 array [0, 1], shape (H, W, 3)."""
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """Float32 array [0, 1] -> PIL Image."""
    return Image.fromarray((arr.clip(0.0, 1.0) * 255).astype(np.uint8))


def _crossfade(
    frames_a: List[Image.Image],
    frames_b: List[Image.Image],
    blend_frames: int,
) -> List[Image.Image]:
    """Crossfade the last *blend_frames* of A with the first *blend_frames* of B.

    Returns a single merged list:
        frames_a[:-blend] + blended_region + frames_b[blend:]
    """
    blend_frames = min(blend_frames, len(frames_a), len(frames_b))
    if blend_frames <= 0:
        return list(frames_a) + list(frames_b)

    result: List[Image.Image] = list(frames_a[:-blend_frames])

    for i in range(blend_frames):
        alpha = (i + 1) / (blend_frames + 1)
        a = _pil_to_array(frames_a[len(frames_a) - blend_frames + i])
        b = _pil_to_array(frames_b[i])
        blended = a * (1.0 - alpha) + b * alpha
        result.append(_array_to_pil(blended))

    result.extend(frames_b[blend_frames:])
    return result


def _load_frames_from_video(path: str) -> tuple:
    """Load frames and FPS from a video file.

    Returns:
        (frames, fps) where frames is a list of PIL Images.
    """
    try:
        from diffusers.utils import load_video
        frames = load_video(path)
        # Try to get fps via ffprobe
        fps = _get_video_fps(path)
        return frames, fps
    except ImportError:
        pass

    # Fallback: ffmpeg frame extraction
    fps = _get_video_fps(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-i", path,
            "-vsync", "0",
            os.path.join(tmpdir, "frame_%06d.png"),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        frame_files = sorted(
            f for f in os.listdir(tmpdir) if f.endswith(".png")
        )
        frames = [Image.open(os.path.join(tmpdir, f)).convert("RGB") for f in frame_files]
    return frames, fps


def _get_video_fps(path: str) -> int:
    """Extract FPS from a video file using ffprobe. Defaults to 24."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and "/" in result.stdout.strip():
            num, den = result.stdout.strip().split("/")
            return round(int(num) / int(den))
    except Exception:
        pass
    return 24


def _save_frames_as_video(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 24,
) -> str:
    """Save PIL frames to an MP4 file using ffmpeg or diffusers."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Prefer diffusers export
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        return output_path
    except ImportError:
        pass

    # Fallback: ffmpeg pipe
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "Neither diffusers nor ffmpeg are available for video export. "
            "Install diffusers or ffmpeg."
        )

    w, h = frames[0].size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.array(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()[-300:]
        raise RuntimeError(f"ffmpeg encode failed: {err}")
    return output_path


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VideoExtender:
    """Extend short video clips to longer duration."""

    METHODS = ("loop_blend", "reverse_bounce", "vace_continuation", "interpolation", "auto")

    def __init__(
        self,
        method: Literal["loop_blend", "reverse_bounce", "vace_continuation", "interpolation", "auto"] = "auto",
        device: str = "auto",
    ):
        """
        Args:
            method:
                - loop_blend:         Seamless loop with crossfade blending (always available).
                - reverse_bounce:     Play forward then reverse for smooth extension.
                - vace_continuation:  Use VACE backend for generative continuation.
                - interpolation:      Use RIFE to slow down (temporal upsampling).
                - auto:               Pick the best available method.
            device:
                - auto: pick CUDA > MPS > CPU
                - cuda / mps / cpu: force a device
        """
        self._requested_method = method
        self.device = self._resolve_device(device)
        self.method = self._resolve_method(method)
        self._interpolator = None
        self._vace_backend = None
        logger.info(f"VideoExtender ready  method={self.method}  device={self.device}")

    # ----- device / method resolution ----------------------------------------

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

    def _resolve_method(self, method: str) -> str:
        if method != "auto":
            return method

        # Auto: prefer vace_continuation if torch is available, else interpolation, else loop_blend
        try:
            import torch  # noqa: F401
            try:
                from diffusers import WanVACEPipeline  # noqa: F401
                logger.info("Auto method: vace_continuation available (torch + diffusers)")
                return "vace_continuation"
            except ImportError:
                logger.info("Auto method: falling back to interpolation (torch available)")
                return "interpolation"
        except ImportError:
            logger.info("Auto method: falling back to loop_blend (no torch)")
            return "loop_blend"

    # ======================================================================
    # Public API
    # ======================================================================

    def extend(
        self,
        frames: List[Image.Image],
        target_frames: int,
        fps: int = 24,
        prompt: str = "",
        blend_frames: int = 8,
    ) -> List[Image.Image]:
        """Extend frame list to target frame count.

        Args:
            frames:        Input PIL Images (RGB).
            target_frames: Desired total number of frames in the output.
            fps:           Frames per second (used for generative methods).
            prompt:        Text prompt (used only by vace_continuation).
            blend_frames:  Number of crossfade frames at loop/transition boundaries.

        Returns:
            Extended list of PIL Images.
        """
        if len(frames) == 0:
            raise ValueError("Cannot extend an empty frame list")
        if target_frames <= len(frames):
            logger.info(
                f"Already have {len(frames)} frames >= target {target_frames}; "
                "returning original frames"
            )
            return list(frames[:target_frames])

        logger.info(
            f"Extending {len(frames)} frames -> {target_frames} frames "
            f"(method={self.method}, fps={fps})"
        )

        if self.method == "loop_blend":
            return self._extend_loop_blend(frames, target_frames, blend_frames)
        elif self.method == "reverse_bounce":
            return self._extend_reverse_bounce(frames, target_frames, blend_frames)
        elif self.method == "interpolation":
            return self._extend_interpolation(frames, target_frames)
        elif self.method == "vace_continuation":
            return self._extend_vace(frames, target_frames, fps, prompt)
        else:
            raise ValueError(f"Unknown extension method: {self.method}")

    def extend_video_file(
        self,
        input_path: str,
        target_duration: float,
        output_path: str,
        **kwargs,
    ) -> str:
        """Extend a video file to target duration in seconds.

        Args:
            input_path:      Path to input video (MP4/AVI/MOV/WEBM/GIF).
            target_duration: Desired total duration in seconds.
            output_path:     Path for the output video file.
            **kwargs:        Additional keyword arguments passed to extend().

        Returns:
            Path to the extended video file.
        """
        frames, fps = _load_frames_from_video(input_path)
        logger.info(f"Loaded {len(frames)} frames at {fps} fps from {input_path}")

        target_frames = max(1, round(target_duration * fps))
        extended = self.extend(frames, target_frames, fps=fps, **kwargs)

        _save_frames_as_video(extended, output_path, fps=fps)
        logger.info(
            f"Extended video saved: {output_path} "
            f"({len(extended)} frames, {len(extended)/fps:.1f}s @ {fps}fps)"
        )
        return output_path

    def create_seamless_loop(
        self,
        frames: List[Image.Image],
        blend_frames: int = 8,
    ) -> List[Image.Image]:
        """Create a seamlessly looping version of the frames.

        The end of the clip is crossfaded back into the beginning, so that
        playing the result on repeat produces no visible seam.

        Args:
            frames:       Input PIL Images.
            blend_frames: Number of frames to crossfade at the loop boundary.

        Returns:
            Loopable frame list (slightly shorter than input by blend_frames).
        """
        if len(frames) < blend_frames * 2:
            logger.warning(
                f"Not enough frames ({len(frames)}) for {blend_frames}-frame "
                "crossfade; reducing blend region"
            )
            blend_frames = max(1, len(frames) // 4)

        # We take frames[0:N-blend] as the body, and crossfade
        # frames[N-blend:N] with frames[0:blend] to create the transition.
        body = list(frames[:-blend_frames])
        tail = list(frames[-blend_frames:])
        head = list(frames[:blend_frames])

        transition: List[Image.Image] = []
        for i in range(blend_frames):
            alpha = (i + 1) / (blend_frames + 1)
            a = _pil_to_array(tail[i])
            b = _pil_to_array(head[i])
            blended = a * (1.0 - alpha) + b * alpha
            transition.append(_array_to_pil(blended))

        result = body + transition
        logger.info(
            f"Created seamless loop: {len(frames)} -> {len(result)} frames "
            f"(blend_frames={blend_frames})"
        )
        return result

    # ======================================================================
    # Strategy 1 -- Loop blend
    # ======================================================================

    def _extend_loop_blend(
        self,
        frames: List[Image.Image],
        target_frames: int,
        blend_frames: int,
    ) -> List[Image.Image]:
        """Extend by repeating the clip with crossfade blending at loop points.

        Each repetition crossfades the end of one copy with the start of the
        next for a smooth transition.
        """
        blend_frames = min(blend_frames, len(frames) // 2)
        if blend_frames < 1:
            blend_frames = 1

        result = list(frames)

        while len(result) < target_frames:
            # How many frames do we still need?
            remaining = target_frames - len(result)

            # Next chunk is a full copy of the original frames (or partial if close)
            next_chunk = list(frames[:min(len(frames), remaining + blend_frames)])

            result = _crossfade(result, next_chunk, blend_frames)

        result = result[:target_frames]
        logger.info(f"Loop-blend extension: {len(frames)} -> {len(result)} frames")
        return result

    # ======================================================================
    # Strategy 2 -- Reverse bounce
    # ======================================================================

    def _extend_reverse_bounce(
        self,
        frames: List[Image.Image],
        target_frames: int,
        blend_frames: int,
    ) -> List[Image.Image]:
        """Extend by alternating forward and reverse playback.

        Creates a bounce effect: forward -> reverse -> forward -> ...
        with crossfade blending at each direction change.
        """
        blend_frames = min(blend_frames, len(frames) // 2)
        if blend_frames < 1:
            blend_frames = 1

        forward = list(frames)
        # Exclude first and last to avoid doubled frames at direction changes
        reversed_frames = list(frames[-2:0:-1]) if len(frames) > 2 else list(reversed(frames))

        result = list(forward)
        use_reversed = True

        while len(result) < target_frames:
            remaining = target_frames - len(result)
            if use_reversed:
                next_chunk = reversed_frames[:min(len(reversed_frames), remaining + blend_frames)]
            else:
                next_chunk = forward[:min(len(forward), remaining + blend_frames)]

            result = _crossfade(result, next_chunk, blend_frames)
            use_reversed = not use_reversed

        result = result[:target_frames]
        logger.info(f"Reverse-bounce extension: {len(frames)} -> {len(result)} frames")
        return result

    # ======================================================================
    # Strategy 3 -- Interpolation (temporal upsampling = slow motion)
    # ======================================================================

    def _extend_interpolation(
        self,
        frames: List[Image.Image],
        target_frames: int,
    ) -> List[Image.Image]:
        """Extend by slowing down the video through frame interpolation.

        Uses RIFE (or fallback blend) to generate intermediate frames,
        effectively increasing the frame count by temporal upsampling.
        """
        from animatediff.postprocess.interpolation import FrameInterpolator

        if self._interpolator is None:
            self._interpolator = FrameInterpolator(backend="auto", device=self.device)
            # Warm up: FrameInterpolator may resolve to "rife" but fail to load
            # the model on the first call, falling back to "blend" internally.
            # Trigger a tiny dummy interpolation to finalise backend resolution.
            dummy = [frames[0], frames[0]]
            try:
                self._interpolator.interpolate(dummy, multiplier=2)
            except Exception:
                logger.info("RIFE unavailable, interpolator falling back to blend")
                self._interpolator = FrameInterpolator(backend="blend", device=self.device)

        current = list(frames)

        # Determine required multiplier: we need at least target_frames
        # Each 2x pass roughly doubles frame count: (N-1)*2 + 1
        while len(current) < target_frames:
            # Calculate how many 2x passes we can afford
            expected_after = (len(current) - 1) * 2 + 1
            if expected_after > target_frames * 2:
                # One more pass would overshoot by a lot; do a partial pass
                break

            logger.debug(
                f"Interpolation pass: {len(current)} -> ~{expected_after} frames"
            )
            current = self._interpolator.interpolate(current, multiplier=2)

        # If we overshot, trim uniformly to target length
        if len(current) > target_frames:
            current = self._uniform_subsample(current, target_frames)
        elif len(current) < target_frames:
            # Still short: do one more 2x pass then subsample
            current = self._interpolator.interpolate(current, multiplier=2)
            current = self._uniform_subsample(current, target_frames)

        logger.info(f"Interpolation extension: {len(frames)} -> {len(current)} frames")
        return current

    @staticmethod
    def _uniform_subsample(
        frames: List[Image.Image], target: int,
    ) -> List[Image.Image]:
        """Uniformly subsample frames to exactly *target* count."""
        if target >= len(frames):
            return list(frames)
        indices = np.linspace(0, len(frames) - 1, target, dtype=int)
        return [frames[i] for i in indices]

    # ======================================================================
    # Strategy 4 -- VACE continuation (generative)
    # ======================================================================

    def _extend_vace(
        self,
        frames: List[Image.Image],
        target_frames: int,
        fps: int,
        prompt: str,
    ) -> List[Image.Image]:
        """Extend by generating new frames using Wan VACE continuation mode.

        This produces genuinely new content that continues the scene, rather
        than looping or slowing down existing frames.
        """
        try:
            from animatediff.backends.wan22_vace import Wan22VACEBackend
        except ImportError:
            logger.warning(
                "VACE backend unavailable (missing torch / diffusers). "
                "Falling back to loop_blend."
            )
            return self._extend_loop_blend(frames, target_frames, blend_frames=8)

        result = list(frames)

        # VACE generates in chunks of ~81 frames with overlap_frames conditioning
        overlap = min(4, len(frames))
        chunk_size = 81  # default VACE generation length

        while len(result) < target_frames:
            remaining = target_frames - len(result)
            gen_frames = min(chunk_size, remaining + overlap)

            logger.info(
                f"VACE continuation: generating {gen_frames} frames "
                f"(overlap={overlap}, total so far={len(result)})"
            )

            if self._vace_backend is None:
                # Lazy-load the VACE backend
                device = self.device
                dtype_str = "float32" if device == "mps" else "bfloat16"
                import torch
                torch_dtype = torch.float32 if device == "mps" else torch.bfloat16

                self._vace_backend = Wan22VACEBackend.load(
                    torch_dtype=torch_dtype,
                    device=device,
                    offload_strategy="model_cpu" if device != "mps" else "none",
                )

            try:
                output = self._vace_backend.generate(
                    prompt=prompt or "continue the video naturally",
                    mode="continuation",
                    source_frames=result,
                    overlap_frames=overlap,
                    num_frames=gen_frames,
                )
                # The output includes the overlap frames; skip those
                new_frames = output.frames[overlap:]
                result.extend(new_frames)
            except Exception as e:
                logger.error(f"VACE continuation failed: {e}. Falling back to loop_blend.")
                return self._extend_loop_blend(frames, target_frames, blend_frames=8)

        result = result[:target_frames]
        logger.info(f"VACE continuation extension: {len(frames)} -> {len(result)} frames")
        return result
