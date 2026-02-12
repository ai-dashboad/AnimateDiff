"""
Lip Synchronization — synchronize lip movements to audio in video frames.

Backends:
- musetalk:  MuseTalk (TMElyralab) — real-time latent-space inpainting, best quality
             Requires CUDA GPU. Uses frozen VAE + Whisper audio encoder + UNet.
             256x256 face region, 30fps+ on V100.  MIT license.
- wav2lip:   Wav2Lip (Rudrabha) — classic audio-driven lip sync (ACM MM 2020)
             Lightweight, widely available, good sync accuracy.
             Requires face detection + GAN lip generation.
- sadtalker: SadTalker (OpenTalker) — head pose + lip motion from 3DMM (CVPR 2023)
             Generates natural head movement alongside lip sync.
             Uses ExpNet + PoseVAE for realistic motion coefficients.
- hallo2:    Hallo2 (Fudan/Baidu) — long-duration portrait animation (ICLR 2025)
             4K resolution, hour-long generation, Wav2Vec2 audio encoder.
             Best for extended dialogue sequences.
- basic:     Fallback — audio-energy-based mouth warping using face detection.
             No neural model required; uses mediapipe/dlib/OpenCV cascade.
             Not real lip sync, but provides visual speaking feedback.
- auto:      Try musetalk -> hallo2 -> wav2lip -> sadtalker -> basic

Face detection cascade: mediapipe -> dlib -> opencv haarcascade -> manual bbox

Usage:
    syncer = LipSyncer(backend="auto")
    synced_frames = syncer.sync(frames, "dialogue.wav")
    syncer.sync_video_file("input.mp4", "dialogue.wav", "output.mp4")
"""

import logging
import math
import os
import subprocess
import tempfile
from typing import List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Backend priority for auto-detection
_BACKEND_PRIORITY = ("musetalk", "hallo2", "wav2lip", "sadtalker", "basic")

_SUPPORTED_BACKENDS = (
    "musetalk", "wav2lip", "sadtalker", "hallo2", "basic", "auto", "none",
)


# ---------------------------------------------------------------------------
# Helpers — device resolution, audio analysis, video I/O
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
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


def _audio_energy_envelope(
    audio_path: str,
    num_frames: int,
    fps: float = 24.0,
) -> np.ndarray:
    """Compute per-frame audio energy envelope from a WAV file.

    Returns a (num_frames,) float32 array in [0, 1] representing the
    normalized RMS energy of each frame-aligned audio window.
    """
    try:
        import wave
        import struct

        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_audio_frames = wf.getnframes()
            raw = wf.readframes(n_audio_frames)

        # Convert to float samples
        if sample_width == 2:
            fmt = f"<{n_audio_frames * n_channels}h"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
        elif sample_width == 4:
            fmt = f"<{n_audio_frames * n_channels}i"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 2147483648.0
        else:
            # Fallback: uniform energy
            return np.ones(num_frames, dtype=np.float32) * 0.5

        # Mix to mono
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Compute per-frame RMS energy
        frame_duration = 1.0 / fps
        energy = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            t_start = i * frame_duration
            t_end = (i + 1) * frame_duration
            s_start = int(t_start * sample_rate)
            s_end = int(t_end * sample_rate)
            s_start = min(s_start, len(samples))
            s_end = min(s_end, len(samples))
            if s_end > s_start:
                window = samples[s_start:s_end]
                energy[i] = np.sqrt(np.mean(window ** 2))

        # Normalize to [0, 1]
        max_energy = energy.max()
        if max_energy > 1e-6:
            energy = energy / max_energy

        return energy

    except Exception as e:
        logger.warning(f"Could not read audio energy from {audio_path}: {e}")
        return np.ones(num_frames, dtype=np.float32) * 0.5


def _frames_to_video(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 24,
) -> str:
    """Save PIL Image frames to an MP4 video using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write frames as PNGs
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmpdir, f"frame_{i:06d}.png"))

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg encode failed: {result.stderr}")

    return output_path


def _video_to_frames(
    video_path: str,
    expected_count: Optional[int] = None,
) -> List[Image.Image]:
    """Read frames from a video file using ffmpeg."""
    frames: List[Image.Image] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            frame_pattern,
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        i = 1
        while True:
            path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            if os.path.exists(path):
                frames.append(Image.open(path).convert("RGB").copy())
                i += 1
            else:
                break

    # Pad or truncate to expected count
    if expected_count is not None and frames:
        while len(frames) < expected_count:
            frames.append(frames[-1].copy())
        if len(frames) > expected_count:
            frames = frames[:expected_count]

    return frames


# ---------------------------------------------------------------------------
# Face detection cascade
# ---------------------------------------------------------------------------

class FaceDetector:
    """Detect face bounding boxes with a cascade of backends.

    Tries: mediapipe -> dlib -> opencv haarcascade.
    Caches results across consecutive frames for stability.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend: Optional[str] = None
        self._detector = None
        self._cached_bbox: Optional[Tuple[int, int, int, int]] = None
        self._cache_frame_idx: int = -1
        self._init_detector()

    def _init_detector(self):
        """Initialize the best available face detection backend."""
        # Try mediapipe
        try:
            import mediapipe as mp
            self._mp_face_detection = mp.solutions.face_detection
            self._detector = self._mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5,
            )
            self._backend = "mediapipe"
            logger.info("Face detection backend: mediapipe")
            return
        except ImportError:
            pass

        # Try dlib
        try:
            import dlib
            self._detector = dlib.get_frontal_face_detector()
            self._backend = "dlib"
            logger.info("Face detection backend: dlib")
            return
        except ImportError:
            pass

        # Try OpenCV Haar cascade
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self._detector = cv2.CascadeClassifier(cascade_path)
                self._backend = "opencv"
                logger.info("Face detection backend: opencv haarcascade")
                return
        except (ImportError, AttributeError):
            pass

        self._backend = None
        logger.warning(
            "No face detection backend available. "
            "Install mediapipe, dlib, or opencv-python for auto face detection."
        )

    def detect(
        self,
        frame: Image.Image,
        frame_idx: int = -1,
        cache_interval: int = 5,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box as (x1, y1, x2, y2).

        Args:
            frame: PIL Image (RGB).
            frame_idx: Current frame index (for caching).
            cache_interval: Re-detect every N frames; use cached bbox otherwise.

        Returns:
            (x1, y1, x2, y2) or None if no face found.
        """
        # Use cached bbox for intermediate frames
        if (
            self._cached_bbox is not None
            and frame_idx >= 0
            and (frame_idx - self._cache_frame_idx) < cache_interval
        ):
            return self._cached_bbox

        bbox = self._detect_impl(frame)

        if bbox is not None:
            self._cached_bbox = bbox
            self._cache_frame_idx = frame_idx

        return bbox if bbox is not None else self._cached_bbox

    def _detect_impl(
        self, frame: Image.Image,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Run actual face detection on a single frame."""
        if self._backend is None:
            return None

        arr = np.array(frame)
        h, w = arr.shape[:2]

        if self._backend == "mediapipe":
            return self._detect_mediapipe(arr, w, h)
        elif self._backend == "dlib":
            return self._detect_dlib(arr, w, h)
        elif self._backend == "opencv":
            return self._detect_opencv(arr, w, h)

        return None

    def _detect_mediapipe(
        self, arr: np.ndarray, w: int, h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using mediapipe."""
        results = self._detector.process(arr)
        if not results.detections:
            return None

        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w, int((bb.xmin + bb.width) * w))
        y2 = min(h, int((bb.ymin + bb.height) * h))
        return (x1, y1, x2, y2)

    def _detect_dlib(
        self, arr: np.ndarray, w: int, h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using dlib."""
        # dlib expects RGB uint8 or grayscale
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = self._detector(gray, 1)
        if not faces:
            return None

        rect = faces[0]
        x1 = max(0, rect.left())
        y1 = max(0, rect.top())
        x2 = min(w, rect.right())
        y2 = min(h, rect.bottom())
        return (x1, y1, x2, y2)

    def _detect_opencv(
        self, arr: np.ndarray, w: int, h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using OpenCV Haar cascade."""
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
        )
        if len(faces) == 0:
            return None

        # Pick largest face
        areas = [fw * fh for (_, _, fw, fh) in faces]
        idx = np.argmax(areas)
        fx, fy, fw, fh = faces[idx]
        return (fx, fy, fx + fw, fy + fh)

    def detect_mouth_region(
        self,
        frame: Image.Image,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_idx: int = -1,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Estimate mouth bounding box within the face region.

        The mouth region is heuristically estimated as the lower third
        of the face bounding box, centered horizontally.

        Returns:
            (x1, y1, x2, y2) for the mouth region, or None.
        """
        if face_bbox is None:
            face_bbox = self.detect(frame, frame_idx)
        if face_bbox is None:
            return None

        x1, y1, x2, y2 = face_bbox
        face_w = x2 - x1
        face_h = y2 - y1

        # Mouth is roughly in the lower 35% of the face, centered 60% width
        mouth_y1 = y1 + int(face_h * 0.65)
        mouth_y2 = y2
        mouth_cx = (x1 + x2) // 2
        mouth_hw = int(face_w * 0.30)
        mouth_x1 = max(0, mouth_cx - mouth_hw)
        mouth_x2 = min(frame.width, mouth_cx + mouth_hw)

        return (mouth_x1, mouth_y1, mouth_x2, mouth_y2)

    def close(self):
        """Release detector resources."""
        if self._backend == "mediapipe" and self._detector is not None:
            self._detector.close()
            self._detector = None


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

class _MuseTalkBackend:
    """Wrapper for TMElyralab/MuseTalk real-time lip sync.

    MuseTalk operates by inpainting in the latent space with a single step
    using a frozen VAE + Whisper audio encoder + UNet architecture.
    256x256 face region, 30fps+ on V100.

    Requires: pip install musetalk (or git clone TMElyralab/MuseTalk)
    Requires: CUDA GPU
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    @staticmethod
    def is_available() -> bool:
        try:
            import musetalk  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading MuseTalk model (first use)...")
        try:
            from musetalk.real_time_inference import Avatar
            self._model = Avatar(avatar_id="lip_sync_session", gpu_id=0)
            logger.info("MuseTalk model loaded")
        except ImportError:
            # Try alternative import path
            from musetalk.utils.utils import load_all_model
            self._model = load_all_model()
            logger.info("MuseTalk models loaded (utils path)")

    def process(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int = 24,
    ) -> List[Image.Image]:
        """Process frames through MuseTalk pipeline."""
        self._load_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write input video
            input_video = os.path.join(tmpdir, "input.mp4")
            _frames_to_video(frames, input_video, fps=fps)

            output_video = os.path.join(tmpdir, "output.mp4")

            # MuseTalk expects video + audio paths
            try:
                from musetalk.real_time_inference import Avatar

                avatar = Avatar(
                    avatar_id="lip_sync_tmp",
                    video_path=input_video,
                    bbox_shift=0,
                    batch_size=min(20, len(frames)),
                    preparation=True,
                )
                avatar.init()
                avatar.inference(audio_path, output_video, fps=fps)

            except (ImportError, Exception) as e:
                # Fallback: try CLI invocation
                logger.info(f"MuseTalk API call failed ({e}), trying CLI...")
                cmd = [
                    "python", "-m", "musetalk.realtime_inference",
                    "--video_path", input_video,
                    "--audio_path", audio_path,
                    "--output_path", output_video,
                    "--fps", str(fps),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"MuseTalk CLI failed: {result.stderr}")

            return _video_to_frames(output_video, expected_count=len(frames))


class _Wav2LipBackend:
    """Wrapper for Rudrabha/Wav2Lip classic lip sync.

    ACM Multimedia 2020.  Uses a lip sync discriminator to generate
    accurate mouth movements from audio.  Lightweight and widely available.

    Requires: pip install wav2lipy  (or git clone Rudrabha/Wav2Lip)
    Model checkpoint: wav2lip_gan.pth or wav2lip.pth
    """

    # Standard model checkpoint locations
    _CHECKPOINT_SEARCH = [
        "checkpoints/wav2lip_gan.pth",
        "checkpoints/wav2lip.pth",
        os.path.expanduser("~/.cache/wav2lip/wav2lip_gan.pth"),
        os.path.expanduser("~/.cache/wav2lip/wav2lip.pth"),
    ]

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._checkpoint: Optional[str] = None
        self._model = None

    @staticmethod
    def is_available() -> bool:
        """Check if Wav2Lip is available."""
        # Check for the wav2lipy PyPI package
        try:
            import Wav2Lip  # noqa: F401
            return True
        except ImportError:
            pass

        # Check for wav2lip module (git clone)
        try:
            import wav2lip  # noqa: F401
            return True
        except ImportError:
            pass

        # Check if inference script is accessible
        for ckpt in _Wav2LipBackend._CHECKPOINT_SEARCH:
            if os.path.exists(ckpt):
                return True

        return False

    def _find_checkpoint(self) -> Optional[str]:
        """Locate a Wav2Lip model checkpoint."""
        if self._checkpoint is not None:
            return self._checkpoint

        for path in self._CHECKPOINT_SEARCH:
            if os.path.exists(path):
                self._checkpoint = path
                return path

        # Try HuggingFace cache
        try:
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(
                repo_id="wav2lip/wav2lip_gan",
                filename="wav2lip_gan.pth",
                cache_dir=os.path.expanduser("~/.cache/wav2lip"),
            )
            self._checkpoint = ckpt
            return ckpt
        except Exception:
            pass

        return None

    def process(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int = 24,
    ) -> List[Image.Image]:
        """Process frames through Wav2Lip pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = os.path.join(tmpdir, "input.mp4")
            _frames_to_video(frames, input_video, fps=fps)

            output_video = os.path.join(tmpdir, "output.mp4")
            checkpoint = self._find_checkpoint()

            if checkpoint is None:
                raise RuntimeError(
                    "Wav2Lip checkpoint not found. Download wav2lip_gan.pth "
                    "and place it in checkpoints/ or ~/.cache/wav2lip/"
                )

            # Wav2Lip CLI inference
            cmd = [
                "python", "-m", "Wav2Lip.inference",
                "--checkpoint_path", checkpoint,
                "--face", input_video,
                "--audio", audio_path,
                "--outfile", output_video,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                # Try alternative module path
                cmd[2] = "wav2lip.inference"
                result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Wav2Lip failed: {result.stderr}")

            return _video_to_frames(output_video, expected_count=len(frames))


class _SadTalkerBackend:
    """Wrapper for OpenTalker/SadTalker head + lip animation.

    CVPR 2023.  Generates 3D motion coefficients (head pose + expression)
    from audio via ExpNet + PoseVAE, then renders with 3DMM face renderer.
    Produces natural head movement alongside lip sync.

    Requires: git clone OpenTalker/SadTalker + download pretrained models
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @staticmethod
    def is_available() -> bool:
        try:
            import sadtalker  # noqa: F401
            return True
        except ImportError:
            pass

        # Check if SadTalker directory exists in common locations
        for path in [
            "SadTalker",
            os.path.expanduser("~/SadTalker"),
            os.path.expanduser("~/.cache/SadTalker"),
        ]:
            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "inference.py")
            ):
                return True

        return False

    def process(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int = 24,
    ) -> List[Image.Image]:
        """Process frames through SadTalker pipeline.

        SadTalker works from a single source image, so we use the first
        frame as the reference portrait and apply audio-driven animation.
        The resulting frames are then blended with the originals.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # SadTalker takes a single reference image
            ref_image_path = os.path.join(tmpdir, "reference.png")
            frames[0].save(ref_image_path)

            result_dir = os.path.join(tmpdir, "results")
            os.makedirs(result_dir, exist_ok=True)

            cmd = [
                "python", "-m", "sadtalker.inference",
                "--driven_audio", audio_path,
                "--source_image", ref_image_path,
                "--result_dir", result_dir,
                "--still",
                "--preprocess", "full",
                "--enhancer", "gfpgan",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                # Try direct script path
                for sadtalker_dir in [
                    "SadTalker",
                    os.path.expanduser("~/SadTalker"),
                ]:
                    script = os.path.join(sadtalker_dir, "inference.py")
                    if os.path.exists(script):
                        cmd = [
                            "python", script,
                            "--driven_audio", audio_path,
                            "--source_image", ref_image_path,
                            "--result_dir", result_dir,
                            "--still",
                            "--preprocess", "full",
                        ]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True,
                        )
                        break

            if result.returncode != 0:
                raise RuntimeError(f"SadTalker failed: {result.stderr}")

            # Find output video in result_dir
            output_video = None
            for fname in os.listdir(result_dir):
                if fname.endswith(".mp4"):
                    output_video = os.path.join(result_dir, fname)
                    break

            # Check subdirectories too
            if output_video is None:
                for root, dirs, files in os.walk(result_dir):
                    for fname in files:
                        if fname.endswith(".mp4"):
                            output_video = os.path.join(root, fname)
                            break
                    if output_video:
                        break

            if output_video is None:
                raise RuntimeError("SadTalker produced no output video")

            return _video_to_frames(output_video, expected_count=len(frames))


class _Hallo2Backend:
    """Wrapper for fudan-generative-vision/Hallo2 long-duration lip sync.

    ICLR 2025.  Uses Wav2Vec2 audio encoder + diffusion-based portrait
    animation.  Supports 4K resolution and hour-long generation.
    Best suited for extended dialogue sequences.

    Requires: git clone fudan-generative-vision/hallo2 + pretrained models
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @staticmethod
    def is_available() -> bool:
        try:
            import hallo2  # noqa: F401
            return True
        except ImportError:
            pass

        # Check for Hallo2 directory
        for path in [
            "hallo2",
            os.path.expanduser("~/hallo2"),
            os.path.expanduser("~/.cache/hallo2"),
        ]:
            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "scripts", "inference_long.py")
            ):
                return True

        return False

    def process(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int = 24,
    ) -> List[Image.Image]:
        """Process frames through Hallo2 pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Hallo2 takes a source image + driving audio
            ref_image_path = os.path.join(tmpdir, "source.png")
            frames[0].save(ref_image_path)

            output_video = os.path.join(tmpdir, "output.mp4")

            # Try Python module invocation
            cmd = [
                "python", "-m", "hallo2.scripts.inference_long",
                "--source_image", ref_image_path,
                "--driving_audio", audio_path,
                "--output", output_video,
                "--pose_weight", "1.0",
                "--face_weight", "1.0",
                "--lip_weight", "1.0",
                "--face_expand_ratio", "1.2",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                # Try direct script path
                for hallo2_dir in [
                    "hallo2",
                    os.path.expanduser("~/hallo2"),
                ]:
                    script = os.path.join(
                        hallo2_dir, "scripts", "inference_long.py",
                    )
                    if os.path.exists(script):
                        cmd = [
                            "python", script,
                            "--source_image", ref_image_path,
                            "--driving_audio", audio_path,
                            "--output", output_video,
                        ]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True,
                        )
                        break

            if result.returncode != 0:
                raise RuntimeError(f"Hallo2 failed: {result.stderr}")

            return _video_to_frames(output_video, expected_count=len(frames))


class _BasicBackend:
    """Fallback lip sync using audio-energy-driven mouth warping.

    No neural model required.  Detects the mouth region using available
    face detection (mediapipe / dlib / OpenCV cascade) and applies subtle
    spatial warping to the mouth area based on audio RMS energy.

    This is NOT real lip sync — it just makes the mouth area move slightly
    in time with audio energy, providing a visual cue that the character
    is speaking.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._face_detector: Optional[FaceDetector] = None

    @staticmethod
    def is_available() -> bool:
        # Basic backend is always available (degrades gracefully)
        return True

    def _get_face_detector(self) -> FaceDetector:
        if self._face_detector is None:
            self._face_detector = FaceDetector(device=self.device)
        return self._face_detector

    def process(
        self,
        frames: List[Image.Image],
        audio_path: str,
        fps: int = 24,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        strength: float = 1.0,
    ) -> List[Image.Image]:
        """Apply audio-energy-based mouth warping to frames.

        For each frame:
        1. Detect or use provided face/mouth region
        2. Compute audio energy for the corresponding time window
        3. Apply a vertical stretch to the mouth area proportional to energy
        4. Blend the warped region back into the original frame
        """
        if not frames:
            return frames

        energy = _audio_energy_envelope(audio_path, len(frames), fps)
        detector = self._get_face_detector()

        result: List[Image.Image] = []
        for i, frame in enumerate(frames):
            e = energy[i] * strength

            # Skip frames with very low energy (silence)
            if e < 0.05:
                result.append(frame)
                continue

            # Detect mouth region
            if face_bbox is not None:
                mouth_bbox = detector.detect_mouth_region(
                    frame, face_bbox=face_bbox, frame_idx=i,
                )
            else:
                mouth_bbox = detector.detect_mouth_region(
                    frame, frame_idx=i,
                )

            if mouth_bbox is None:
                result.append(frame)
                continue

            # Apply mouth warp
            warped = self._warp_mouth(frame, mouth_bbox, e)
            result.append(warped)

            if (i + 1) % 50 == 0:
                logger.debug(
                    f"  Basic lip sync: {i + 1}/{len(frames)} "
                    f"(energy={e:.2f})"
                )

        return result

    @staticmethod
    def _warp_mouth(
        frame: Image.Image,
        mouth_bbox: Tuple[int, int, int, int],
        energy: float,
    ) -> Image.Image:
        """Apply vertical stretch warp to mouth region based on energy.

        The warp simulates mouth opening by stretching the lower portion
        of the mouth region downward proportional to audio energy.
        """
        x1, y1, x2, y2 = mouth_bbox
        w = x2 - x1
        h = y2 - y1

        if w < 4 or h < 4:
            return frame

        # Clamp warp amount: up to 15% of mouth height at full energy
        max_warp_px = max(2, int(h * 0.15))
        warp_amount = int(max_warp_px * energy)

        if warp_amount < 1:
            return frame

        arr = np.array(frame, dtype=np.uint8).copy()
        img_h, img_w = arr.shape[:2]

        # Extract mouth region with padding
        pad = max(4, warp_amount + 2)
        ry1 = max(0, y1 - pad)
        ry2 = min(img_h, y2 + pad + warp_amount)
        rx1 = max(0, x1 - 2)
        rx2 = min(img_w, x2 + 2)

        mouth_region = arr[ry1:ry2, rx1:rx2].copy()
        rh, rw = mouth_region.shape[:2]

        if rh < 4 or rw < 4:
            return frame

        # Create vertical displacement map
        # The center of the mouth moves down; top stays fixed
        new_region = mouth_region.copy()
        center_y_local = (y1 - ry1) + h // 2

        for local_y in range(rh):
            # Displacement is zero at top, maximum at mouth center, fades below
            if local_y < center_y_local:
                # Above center: slight upward displacement (mouth opens)
                t = local_y / max(1, center_y_local)
                dy = -int(warp_amount * 0.3 * t * math.sin(t * math.pi))
            else:
                # Below center: downward displacement (jaw drops)
                t = (local_y - center_y_local) / max(1, rh - center_y_local)
                dy = int(warp_amount * (1.0 - t) * math.sin((1.0 - t) * math.pi * 0.5))

            src_y = local_y - dy
            src_y = max(0, min(rh - 1, src_y))
            new_region[local_y] = mouth_region[src_y]

        # Blend warped region back with feathered edges
        blend_margin = max(2, pad // 2)
        for local_y in range(rh):
            # Vertical feathering
            fy = 1.0
            if local_y < blend_margin:
                fy = local_y / blend_margin
            elif local_y > rh - blend_margin:
                fy = (rh - local_y) / blend_margin

            for local_x in range(rw):
                # Horizontal feathering
                fx = 1.0
                if local_x < blend_margin:
                    fx = local_x / blend_margin
                elif local_x > rw - blend_margin:
                    fx = (rw - local_x) / blend_margin

                alpha = fy * fx
                arr[ry1 + local_y, rx1 + local_x] = (
                    new_region[local_y, local_x] * alpha
                    + arr[ry1 + local_y, rx1 + local_x] * (1.0 - alpha)
                ).astype(np.uint8)

        return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Main class: LipSyncer
# ---------------------------------------------------------------------------

class LipSyncer:
    """Synchronize lip movements to audio in video frames.

    Supports multiple backends with automatic fallback and lazy model loading.
    Face detection is cached across frames for stability and performance.

    Example::

        syncer = LipSyncer(backend="auto")

        # Sync PIL Image frames with audio
        synced = syncer.sync(frames, "dialogue.wav")

        # Or sync a video file directly
        syncer.sync_video_file("input.mp4", "dialogue.wav", "output.mp4")

        # Detect face in a frame
        bbox = syncer.detect_face(frame)
    """

    def __init__(
        self,
        backend: Literal[
            "musetalk", "wav2lip", "sadtalker", "hallo2",
            "basic", "auto", "none",
        ] = "auto",
        device: str = "auto",
    ):
        """
        Args:
            backend:
                - "musetalk":  MuseTalk (best quality, real-time, CUDA only)
                - "wav2lip":   Wav2Lip (classic, widely available)
                - "sadtalker": SadTalker (head + lip motion)
                - "hallo2":    Hallo2 (best for long video, 4K)
                - "basic":     Audio-energy mouth warping (no model needed)
                - "auto":      Try musetalk -> hallo2 -> wav2lip -> sadtalker -> basic
                - "none":      Disabled (pass-through)
            device:
                - "auto": pick CUDA > MPS > CPU
                - "cuda" / "mps" / "cpu": force a specific device
        """
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown lip sync backend: {backend!r}. "
                f"Supported: {_SUPPORTED_BACKENDS}"
            )
        self.device = _resolve_device(device)
        self.backend = self._resolve_backend(backend)
        self._backend_impl = None  # Lazy-loaded
        self._face_detector: Optional[FaceDetector] = None
        logger.info(
            f"LipSyncer ready  backend={self.backend}  device={self.device}"
        )

    # ----- Backend resolution ------------------------------------------------

    def _resolve_backend(self, backend: str) -> str:
        """Resolve the requested backend, falling back if unavailable."""
        if backend == "none":
            return "none"

        if backend == "auto":
            return self._auto_detect_backend()

        # Explicit backend requested — verify availability
        if backend == "musetalk" and not _MuseTalkBackend.is_available():
            logger.warning("MuseTalk not available, falling back to auto")
            return self._auto_detect_backend()
        if backend == "wav2lip" and not _Wav2LipBackend.is_available():
            logger.warning("Wav2Lip not available, falling back to auto")
            return self._auto_detect_backend()
        if backend == "sadtalker" and not _SadTalkerBackend.is_available():
            logger.warning("SadTalker not available, falling back to auto")
            return self._auto_detect_backend()
        if backend == "hallo2" and not _Hallo2Backend.is_available():
            logger.warning("Hallo2 not available, falling back to auto")
            return self._auto_detect_backend()

        return backend

    def _auto_detect_backend(self) -> str:
        """Auto-detect the best available lip sync backend."""
        checks = {
            "musetalk": _MuseTalkBackend.is_available,
            "hallo2": _Hallo2Backend.is_available,
            "wav2lip": _Wav2LipBackend.is_available,
            "sadtalker": _SadTalkerBackend.is_available,
            "basic": _BasicBackend.is_available,
        }

        # Neural backends require CUDA (except basic)
        cuda_only = {"musetalk", "hallo2", "wav2lip", "sadtalker"}
        has_cuda = self.device == "cuda"

        for name in _BACKEND_PRIORITY:
            if name in cuda_only and not has_cuda:
                continue
            if checks[name]():
                logger.info(f"Auto-detected lip sync backend: {name}")
                return name

        # Basic is always available and works on any device
        logger.info("No GPU lip sync backend available, using basic fallback")
        return "basic"

    def _get_backend(self):
        """Lazy-load the backend implementation."""
        if self._backend_impl is not None:
            return self._backend_impl

        if self.backend == "musetalk":
            self._backend_impl = _MuseTalkBackend(self.device)
        elif self.backend == "wav2lip":
            self._backend_impl = _Wav2LipBackend(self.device)
        elif self.backend == "sadtalker":
            self._backend_impl = _SadTalkerBackend(self.device)
        elif self.backend == "hallo2":
            self._backend_impl = _Hallo2Backend(self.device)
        elif self.backend == "basic":
            self._backend_impl = _BasicBackend(self.device)
        else:
            self._backend_impl = None

        return self._backend_impl

    def _get_face_detector(self) -> FaceDetector:
        """Lazy-load the face detector."""
        if self._face_detector is None:
            self._face_detector = FaceDetector(device=self.device)
        return self._face_detector

    # ======================================================================
    # Public API
    # ======================================================================

    @property
    def available(self) -> bool:
        """Whether a working lip sync backend is configured."""
        return self.backend != "none"

    def sync(
        self,
        frames: List[Image.Image],
        audio_path: str,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        strength: float = 1.0,
        fps: int = 24,
    ) -> List[Image.Image]:
        """Apply lip sync to frames using the audio.

        Args:
            frames:     List of PIL Images (RGB) representing video frames.
            audio_path: Path to the dialogue audio file (WAV preferred).
            face_bbox:  (x1, y1, x2, y2) face region, or None for auto-detect.
            strength:   Blending strength 0.0 (no change) to 1.0 (full sync).
            fps:        Frame rate of the video.

        Returns:
            New list of lip-synced PIL Images (same length as input).
            If lip sync fails, returns the original frames unchanged.
        """
        if not self.available:
            logger.info("Lip sync disabled, returning original frames")
            return list(frames)

        if not frames:
            return []

        if not audio_path or not os.path.exists(audio_path):
            logger.warning(
                f"Audio file not found: {audio_path}. Skipping lip sync."
            )
            return list(frames)

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == 0.0:
            return list(frames)

        logger.info(
            f"Lip sync: {len(frames)} frames, backend={self.backend}, "
            f"strength={strength:.2f}, fps={fps}"
        )

        try:
            backend = self._get_backend()
            if backend is None:
                return list(frames)

            # Basic backend supports face_bbox and strength natively
            if self.backend == "basic":
                return backend.process(
                    frames, audio_path, fps=fps,
                    face_bbox=face_bbox, strength=strength,
                )

            # Neural backends process the full pipeline
            synced = backend.process(frames, audio_path, fps=fps)

            # Apply strength blending if < 1.0
            if strength < 1.0:
                synced = self._blend_frames(frames, synced, strength)

            return synced

        except Exception as e:
            logger.error(f"Lip sync failed ({self.backend}): {e}")
            logger.info("Returning original frames without lip sync")
            return list(frames)

    def detect_face(
        self, frame: Image.Image,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box in a frame.

        Args:
            frame: PIL Image (RGB).

        Returns:
            (x1, y1, x2, y2) bounding box, or None if no face found.
        """
        detector = self._get_face_detector()
        return detector.detect(frame, frame_idx=-1)

    def detect_mouth(
        self,
        frame: Image.Image,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect mouth region in a frame.

        Args:
            frame:     PIL Image (RGB).
            face_bbox: Known face bbox, or None to auto-detect.

        Returns:
            (x1, y1, x2, y2) mouth bounding box, or None.
        """
        detector = self._get_face_detector()
        return detector.detect_mouth_region(frame, face_bbox=face_bbox)

    def sync_video_file(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        strength: float = 1.0,
        fps: int = 24,
    ) -> str:
        """Convenience: apply lip sync to a video file.

        Args:
            video_path:  Path to the input video file.
            audio_path:  Path to the dialogue audio file.
            output_path: Path for the output lip-synced video.
            face_bbox:   (x1, y1, x2, y2) face region, or None for auto-detect.
            strength:    Blending strength (0.0 to 1.0).
            fps:         Frame rate (used for frame extraction).

        Returns:
            Path to the output video file.
        """
        logger.info(
            f"Lip sync video: {video_path} + {audio_path} -> {output_path}"
        )

        # Read input frames
        frames = _video_to_frames(video_path)
        if not frames:
            raise RuntimeError(f"Could not read frames from {video_path}")

        logger.info(f"Read {len(frames)} frames from {video_path}")

        # Apply lip sync
        synced = self.sync(
            frames, audio_path,
            face_bbox=face_bbox, strength=strength, fps=fps,
        )

        # Write output video
        _frames_to_video(synced, output_path, fps=fps)

        # Mux audio into the output
        muxed_path = output_path.replace(".mp4", "_muxed.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            muxed_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.replace(muxed_path, output_path)
        else:
            logger.warning(
                f"Audio muxing failed, output has no audio: {result.stderr}"
            )
            if os.path.exists(muxed_path):
                os.remove(muxed_path)

        logger.info(f"Lip sync video saved: {output_path}")
        return output_path

    # ----- Helpers -----------------------------------------------------------

    @staticmethod
    def _blend_frames(
        original: List[Image.Image],
        synced: List[Image.Image],
        strength: float,
    ) -> List[Image.Image]:
        """Blend original and synced frames by strength factor.

        strength=0.0 returns originals, 1.0 returns fully synced.
        """
        result: List[Image.Image] = []
        for orig, sync in zip(original, synced):
            orig_arr = np.array(orig, dtype=np.float32)
            sync_arr = np.array(sync, dtype=np.float32)
            blended = orig_arr * (1.0 - strength) + sync_arr * strength
            result.append(
                Image.fromarray(blended.clip(0, 255).astype(np.uint8))
            )

        # If synced is shorter, append remaining originals
        if len(synced) < len(original):
            result.extend(original[len(synced):])

        return result

    @staticmethod
    def list_backends() -> dict:
        """List all backends and their availability status.

        Returns:
            Dict mapping backend name to availability boolean.
        """
        return {
            "musetalk": _MuseTalkBackend.is_available(),
            "hallo2": _Hallo2Backend.is_available(),
            "wav2lip": _Wav2LipBackend.is_available(),
            "sadtalker": _SadTalkerBackend.is_available(),
            "basic": _BasicBackend.is_available(),
        }

    def close(self):
        """Release resources held by the syncer."""
        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None
        self._backend_impl = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"LipSyncer(backend={self.backend!r}, device={self.device!r}, "
            f"available={self.available})"
        )
