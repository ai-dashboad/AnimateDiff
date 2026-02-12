#!/usr/bin/env python3
"""
凡人修仙传 Trailer V4 — Beat Sync + Camera Presets + VACE + Shot Chaining + Deflicker
                         + V4.2/V4.3: SkyReels, S2V, CameraCtrl, LTX-2, ShotOrchestrator,
                           MTV Sync, VideoExtender, StyleHarmonizer, VoiceProfiles

Upgrades from V3:
- Beat sync: BGM beat detection via librosa, shot durations aligned to musical beats
- Camera presets: Per-shot cinematic camera movement prompts from camera_presets.json
- Shot chaining: Last-frame-I2V continuation for visual coherence between shots
- VACE backend: Wan VACE for continuation shots (fallback to standard wan22)
- Deflicker: Cross-shot color harmonization + per-shot temporal deflicker
- Beat-aware transitions: Transition durations adjusted by beat timing

V4.2 additions:
- VoiceProfile: Per-character voice cloning with speed/language support
- SkyReels-V3: Multi-reference character-consistent video generation
- Wan 2.2 S2V: Audio-driven lip-sync video generation
- CameraController: Trajectory-based camera control (prompt_only / plucker / optical_flow)

V4.3 additions:
- LTX-2: Joint audio-video generation backend
- ShotOrchestrator: Auto-camera assignment + rhythm planning
- StyleHarmonizer: Cross-shot visual consistency (before deflicker)
- VideoExtender: Extend short clips to minimum duration
- VideoAudioSync: 3-stream audio sync for frame-level alignment
- Audio post-processing: normalize_audio, add_reverb for cinematic narration

Backwards compatible with V3 storyboards (missing V4 fields default to disabled).

Three-phase production:
  Phase 1 (local Mac MPS): Generate narration audio, compute frame counts, beat sync,
                            orchestrate cameras + rhythm
  Phase 2 (cloud GPU):     Generate portraits + T2V/I2V shots (with camera presets +
                            chaining + multi-backend selection + camera trajectories)
  Phase 3 (local Mac MPS): StyleHarmonize + deflicker + extend short shots +
                            RIFE + color grading + MTV sync + audio post + compose

Usage:
  # Run Phase 1 only (local, ~2h)
  python scripts/produce_trailer_v4.py --phase 1

  # Run Phase 2 only (needs CUDA GPU)
  python scripts/produce_trailer_v4.py --phase 2

  # Run Phase 3 only (local, ~30min)
  python scripts/produce_trailer_v4.py --phase 3

  # Run all phases
  python scripts/produce_trailer_v4.py --phase all
"""

import argparse
import gc
import json
import logging
import os
import time
import wave
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# V4 storyboard path, with fallback to V3 for backwards compatibility
STORYBOARD_PATH_V4 = PROJECT_ROOT / "examples" / "fanren_trailer_v4.json"
STORYBOARD_PATH_V3 = PROJECT_ROOT / "examples" / "fanren_trailer_v3.json"

def _get_storyboard_path() -> Path:
    """Return the best available storyboard path."""
    if STORYBOARD_PATH_V4.exists():
        return STORYBOARD_PATH_V4
    elif STORYBOARD_PATH_V3.exists():
        logger.info("V4 storyboard not found, falling back to V3 storyboard")
        return STORYBOARD_PATH_V3
    else:
        raise FileNotFoundError(
            f"No storyboard found. Expected:\n"
            f"  {STORYBOARD_PATH_V4}\n"
            f"  {STORYBOARD_PATH_V3}"
        )

OUTPUT_DIR = PROJECT_ROOT / "output" / "fanren-trailer-v4"

NARRATION_DIR = OUTPUT_DIR / "narration"
PORTRAIT_DIR = OUTPUT_DIR / "portraits"
SHOTS_DIR = OUTPUT_DIR / "shots"
FINAL_DIR = OUTPUT_DIR / "final"

TRANSITION_PAD = 0.3  # seconds of padding after narration

# BGM (Creative Commons)
BGM_URL = "https://peritune.com/music/PerituneMaterial_Wuxia3.mp3"
BGM_FILENAME = "PerituneMaterial_Wuxia3.mp3"

# Beat sync: lazy import guard
LIBROSA_AVAILABLE = False
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    pass

# V4.2 imports: guarded so script works even if modules aren't installed yet
SKYREELS_V3_AVAILABLE = False
try:
    from animatediff.backends.skyreels_v3 import SkyReelsV3Backend
    SKYREELS_V3_AVAILABLE = True
except ImportError:
    pass

WAN22_S2V_AVAILABLE = False
try:
    from animatediff.backends.wan22_s2v import Wan22S2VBackend
    WAN22_S2V_AVAILABLE = True
except ImportError:
    pass

CAMERA_CTRL_AVAILABLE = False
try:
    from animatediff.core.camera_ctrl import CameraController, CameraTrajectory, get_trajectory_for_preset
    CAMERA_CTRL_AVAILABLE = True
except ImportError:
    pass

VOICE_PROFILE_AVAILABLE = False
try:
    from animatediff.postprocess.audio import VoiceProfile, AudioGenerator, normalize_audio, add_reverb
    VOICE_PROFILE_AVAILABLE = True
except ImportError:
    pass

# V4.3 imports: guarded
LTX2_AVAILABLE = False
try:
    from animatediff.backends.ltx2 import LTX2Backend
    LTX2_AVAILABLE = True
except ImportError:
    pass

SHOT_ORCHESTRATOR_AVAILABLE = False
try:
    from animatediff.core.shot_scheduler import ShotOrchestrator
    from animatediff.core.story_engine import ShotSpec
    SHOT_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    pass

STYLE_HARMONIZER_AVAILABLE = False
try:
    from animatediff.postprocess.style_harmonize import StyleHarmonizer
    STYLE_HARMONIZER_AVAILABLE = True
except ImportError:
    pass

VIDEO_EXTENDER_AVAILABLE = False
try:
    from animatediff.postprocess.video_extend import VideoExtender
    VIDEO_EXTENDER_AVAILABLE = True
except ImportError:
    pass

MTV_SYNC_AVAILABLE = False
try:
    from animatediff.postprocess.mtv_sync import AudioStreamAnalyzer, VideoAudioSync
    MTV_SYNC_AVAILABLE = True
except ImportError:
    pass

# V4.4 SeedAnce-style systems: guarded imports
IDENTITY_KEEPER_AVAILABLE = False
try:
    from animatediff.core.identity_keeper import IdentityKeeper, IdentityEmbedding
    IDENTITY_KEEPER_AVAILABLE = True
except ImportError:
    pass

ACCELERATION_AVAILABLE = False
try:
    from animatediff.core.acceleration import AccelerationManager
    ACCELERATION_AVAILABLE = True
except ImportError:
    pass

REFINEMENT_AVAILABLE = False
try:
    from animatediff.core.refinement_loop import RefinementLoop
    REFINEMENT_AVAILABLE = True
except ImportError:
    pass

REWARD_MODEL_AVAILABLE = False
try:
    from animatediff.core.reward_model import RewardEnsemble
    REWARD_MODEL_AVAILABLE = True
except ImportError:
    pass


def _load_storyboard() -> dict:
    """Load and return the storyboard JSON."""
    sb_path = _get_storyboard_path()
    with open(sb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_computed_storyboard() -> dict:
    """Load computed storyboard (with durations from Phase 1), falling back to original."""
    computed_sb = OUTPUT_DIR / "storyboard_computed.json"
    if computed_sb.exists():
        with open(computed_sb, "r", encoding="utf-8") as f:
            storyboard = json.load(f)
        print(f"  Using computed storyboard: {computed_sb}")
        return storyboard
    else:
        print("  No computed storyboard found, using original with default durations")
        return _load_storyboard()


def _get_model_params(storyboard: dict) -> dict:
    """Extract model_params from storyboard with defaults."""
    mp = storyboard.get("model_params", {})
    return {
        "width": mp.get("width", 832),
        "height": mp.get("height", 480),
        "fps": mp.get("fps", 24),
        "max_frames": mp.get("max_frames", 81),
        "guidance_scale": mp.get("guidance_scale", 5.0),
        "guidance_scale_2": mp.get("guidance_scale_2", None),
        "num_inference_steps": mp.get("num_inference_steps", 50),
    }


def _is_v4_storyboard(storyboard: dict) -> bool:
    """Check if the storyboard uses V4 features."""
    version = storyboard.get("version", "3.0")
    return version.startswith("4")


# ============================================================================
# V4 Feature Helpers
# ============================================================================

def _get_beat_sync_config(storyboard: dict) -> dict:
    """Extract beat_sync config, defaulting to disabled for V3 storyboards."""
    cfg = storyboard.get("beat_sync", {})
    return {
        "enabled": cfg.get("enabled", False),
        "mode": cfg.get("mode", "snap_to_beat"),
        "bgm_path": cfg.get("bgm_path", ""),
    }


def _get_shot_chaining_config(storyboard: dict) -> dict:
    """Extract shot_chaining config, defaulting to disabled for V3 storyboards."""
    cfg = storyboard.get("shot_chaining", {})
    return {
        "enabled": cfg.get("enabled", False),
        "method": cfg.get("method", "last_frame_i2v"),
        "overlap_frames": cfg.get("overlap_frames", 4),
    }


def _get_deflicker_config(storyboard: dict) -> dict:
    """Extract deflicker config, defaulting to disabled for V3 storyboards."""
    cfg = storyboard.get("deflicker", {})
    return {
        "enabled": cfg.get("enabled", False),
        "backend": cfg.get("backend", "auto"),
        "strength": cfg.get("strength", 0.5),
        "cross_shot_harmonize": cfg.get("cross_shot_harmonize", True),
    }


def _get_style_harmonize_config(storyboard: dict) -> dict:
    """Extract style_harmonize config, defaulting to disabled for V3 storyboards."""
    cfg = storyboard.get("style_harmonize", {})
    return {
        "enabled": cfg.get("enabled", False),
        "backend": cfg.get("backend", "auto"),
        "strength": cfg.get("strength", 0.5),
    }


def _get_video_extend_config(storyboard: dict) -> dict:
    """Extract video_extend config, defaulting to disabled for V3 storyboards."""
    cfg = storyboard.get("video_extend", {})
    return {
        "enabled": cfg.get("enabled", False),
        "method": cfg.get("method", "auto"),
        "min_duration_seconds": cfg.get("min_duration_seconds", 3.0),
        "blend_frames": cfg.get("blend_frames", 8),
    }


def _get_mtv_sync_config(storyboard: dict) -> dict:
    """Extract mtv_sync config, defaulting to disabled for V3 storyboards."""
    cfg = storyboard.get("mtv_sync", {})
    return {
        "enabled": cfg.get("enabled", False),
        "backend": cfg.get("backend", "auto"),
        "align_transitions": cfg.get("align_transitions", True),
    }


def _get_audio_post_config(storyboard: dict) -> dict:
    """Extract audio_post config for narration normalization and reverb."""
    cfg = storyboard.get("audio_post", {})
    return {
        "enabled": cfg.get("enabled", False),
        "normalize_db": cfg.get("normalize_db", -20.0),
        "reverb": cfg.get("reverb", False),
        "reverb_room_size": cfg.get("reverb_room_size", 0.3),
        "reverb_damping": cfg.get("reverb_damping", 0.5),
        "reverb_wet": cfg.get("reverb_wet", 0.15),
    }


def _get_multi_backend_config(storyboard: dict) -> dict:
    """Extract multi-backend selection config."""
    cfg = storyboard.get("multi_backend", {})
    return {
        "enabled": cfg.get("enabled", False),
        "default_backend": cfg.get("default_backend", "wan22"),
    }


def _get_camera_ctrl_config(storyboard: dict) -> dict:
    """Extract camera_ctrl trajectory config, defaulting to disabled."""
    cfg = storyboard.get("camera_ctrl", {})
    return {
        "enabled": cfg.get("enabled", False),
        "method": cfg.get("method", "prompt_only"),
    }


def _get_shot_orchestrator_config(storyboard: dict) -> dict:
    """Extract shot_orchestrator config, defaulting to disabled."""
    cfg = storyboard.get("shot_orchestrator", {})
    return {
        "enabled": cfg.get("enabled", False),
        "auto_cameras": cfg.get("auto_cameras", True),
        "auto_rhythm": cfg.get("auto_rhythm", True),
    }


def _get_seedance_config(storyboard: dict) -> dict:
    """Extract V4.4 SeedAnce-style config fields."""
    return {
        "acceleration_preset": storyboard.get("acceleration_preset", "balanced"),
        "quality_threshold": storyboard.get("quality_threshold", 0.65),
        "identity_enforcement": storyboard.get("identity_enforcement", "reference"),
    }


def _shots_to_shotspecs(shots: list, storyboard: dict) -> list:
    """Convert raw storyboard shot dicts to ShotSpec objects for ShotOrchestrator.

    Returns list of ShotSpec or empty list if ShotSpec is unavailable.
    """
    if not SHOT_ORCHESTRATOR_AVAILABLE:
        return []
    specs = []
    for i, shot in enumerate(shots):
        trans_raw = shot.get("transition", "cut")
        if isinstance(trans_raw, dict):
            trans = trans_raw.get("type", "cut")
        else:
            trans = trans_raw
        spec = ShotSpec(
            shot_id=i,
            prompt=shot.get("prompt", ""),
            negative_prompt=storyboard.get("negative_prompt", ""),
            duration_seconds=shot.get("duration_seconds", 5.0),
            camera=shot.get("camera_preset", ""),
            characters=shot.get("characters", []),
            emotion=shot.get("emotion", ""),
            scene=shot.get("scene", ""),
            transition=trans,
            width=shot.get("width", 0),
            height=shot.get("height", 0),
            num_frames=shot.get("num_frames", 0),
            seed=shot.get("seed", -1),
            narration=shot.get("narration", ""),
            voice_id=shot.get("voice_id", ""),
            metadata=shot.get("metadata", {}),
        )
        specs.append(spec)
    return specs


def _apply_shotspecs_to_shots(specs: list, shots: list):
    """Write ShotSpec fields back into raw storyboard dicts (in-place)."""
    for spec, shot in zip(specs, shots):
        if spec.camera:
            shot["camera_preset"] = spec.camera
        if spec.duration_seconds:
            shot["duration_seconds"] = spec.duration_seconds
        if spec.num_frames:
            shot["num_frames"] = spec.num_frames


def _apply_camera_preset(prompt: str, preset_id: str) -> str:
    """Apply a camera preset to a prompt. Returns original prompt if preset unavailable."""
    if not preset_id:
        return prompt
    try:
        from animatediff.data.camera_utils import apply_preset
        augmented = apply_preset(prompt, preset_id)
        logger.info(f"  Applied camera preset: {preset_id}")
        return augmented
    except (ImportError, KeyError) as e:
        logger.warning(f"  Camera preset '{preset_id}' unavailable: {e}")
        return prompt


def _extract_last_frame(video_path: str) -> "Image.Image | None":
    """Extract the last frame from a video file. Returns None on failure."""
    try:
        frames = _read_video_frames(video_path)
        if frames:
            return frames[-1]
    except Exception as e:
        logger.warning(f"  Failed to extract last frame from {video_path}: {e}")
    return None


def _score_portrait(img):
    """Score portrait quality (higher = better). Detects NaN/VAE corruption.

    Uses multiple heuristics since VAE corruption can be statistically subtle:
    1. Row-to-row pixel jumps (banding artifacts)
    2. Inter-channel edge correlation (color fringing detection)
    3. Channel correlation (RGB decorrelation from NaN)
    """
    import numpy as np
    arr = np.array(img, dtype=np.float32)

    # 1. Row smoothness (lower row_jumps = smoother = better)
    row_jumps = np.abs(arr[1:] - arr[:-1]).mean()

    # 2. Inter-channel Laplacian correlation (edges should be aligned across RGB)
    laps = []
    for c in range(3):
        ch = arr[:, :, c]
        lap = ch[2:, 1:-1] + ch[:-2, 1:-1] + ch[1:-1, 2:] + ch[1:-1, :-2] - 4 * ch[1:-1, 1:-1]
        laps.append(lap.flatten())
    rg_corr = np.corrcoef(laps[0], laps[1])[0, 1]
    rb_corr = np.corrcoef(laps[0], laps[2])[0, 1]
    edge_corr = (rg_corr + rb_corr) / 2  # higher = better

    # 3. Channel correlation (RGB should be correlated in natural images)
    r, g, b = arr[:, :, 0].flatten(), arr[:, :, 1].flatten(), arr[:, :, 2].flatten()
    ch_corr = (np.corrcoef(r, g)[0, 1] + np.corrcoef(r, b)[0, 1]) / 2

    # Combined score: weighted sum (all terms: higher = better)
    score = -row_jumps + 10 * edge_corr + 5 * ch_corr
    return score


# ============================================================================
# Utility Functions
# ============================================================================

def _get_wav_duration(path: str) -> float:
    """Get WAV file duration in seconds."""
    with wave.open(path, 'r') as wf:
        return wf.getnframes() / wf.getframerate()


def _duration_to_frames(duration_seconds: float, fps: int) -> int:
    """Convert duration to 4N+1 aligned frame count."""
    raw = int(duration_seconds * fps)
    aligned = ((raw - 1) // 4) * 4 + 1
    return max(17, aligned)


def _read_video_frames(video_path: str) -> list:
    """Read frames from an MP4 file using ffmpeg."""
    import subprocess
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = os.path.join(tmpdir, "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg read failed: {result.stderr[:200]}")
            return []

        frames = []
        i = 1
        while True:
            path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            if not os.path.exists(path):
                break
            frames.append(Image.open(path).convert("RGB"))
            i += 1

    return frames


def _load_frames_from_dir(frames_dir: str) -> list:
    """Load PNG frames from a directory."""
    from PIL import Image

    frames = []
    i = 0
    while True:
        path = os.path.join(frames_dir, f"{i:04d}.png")
        if not os.path.exists(path):
            break
        frames.append(Image.open(path).convert("RGB"))
        i += 1
    return frames


def _create_title_card(
    title: str,
    subtitle: str,
    size: tuple = (1664, 960),
    num_frames: int = 72,
) -> list:
    """Create a simple title card with text on black background."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    w, h = size
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to load a CJK font, fall back to default
    font_title = None
    font_sub = None
    try:
        # macOS system fonts
        for font_path in [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]:
            if os.path.exists(font_path):
                font_title = ImageFont.truetype(font_path, size=min(w, h) // 8)
                font_sub = ImageFont.truetype(font_path, size=min(w, h) // 16)
                break
    except Exception:
        pass

    # Draw title
    kwargs = {}
    if font_title:
        kwargs["font"] = font_title
    bbox = draw.textbbox((0, 0), title, **kwargs)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, h // 2 - th - 20), title, fill=(255, 215, 0), **kwargs)

    # Draw subtitle
    kwargs_sub = {}
    if font_sub:
        kwargs_sub["font"] = font_sub
    bbox = draw.textbbox((0, 0), subtitle, **kwargs_sub)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, h // 2 + 20), subtitle, fill=(200, 200, 200), **kwargs_sub)

    # Create frame list with fade-in effect
    frames = []
    arr = np.array(img, dtype=np.float32)
    fade_frames = min(num_frames // 3, 24)

    for i in range(num_frames):
        if i < fade_frames:
            alpha = i / fade_frames
        elif i > num_frames - fade_frames:
            alpha = (num_frames - i) / fade_frames
        else:
            alpha = 1.0
        frame = Image.fromarray((arr * alpha).astype(np.uint8))
        frames.append(frame)

    return frames


def _save_frames_as_mp4(frames: list, output_path: str, fps: int):
    """Save a list of PIL Image frames as an MP4 using ffmpeg."""
    import subprocess
    import numpy as np

    if not frames:
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    h, w = np.array(frames[0]).shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.array(frame).tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        logger.error(f"ffmpeg save failed: {proc.stderr.read().decode()[:200]}")


# ============================================================================
# Phase 1: Local Preparation (Mac MPS)
# ============================================================================

def phase1_generate_narration():
    """Generate narration audio for all shots using F5-TTS-MLX."""
    print("\n" + "=" * 60)
    print("PHASE 1: Generating Narration Audio (F5-TTS-MLX)")
    print("=" * 60 + "\n")

    NARRATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load storyboard
    storyboard = _load_storyboard()
    model_params = _get_model_params(storyboard)

    shots = storyboard["shots"]
    voice_profiles = storyboard.get("voice_profiles", {})

    print(f"Storyboard: {storyboard['title']}")
    print(f"Version: {storyboard.get('version', '3.0')}")
    print(f"Shots: {len(shots)}")
    print(f"Voice profiles: {list(voice_profiles.keys())}")
    print(f"Model params: fps={model_params['fps']}, max_frames={model_params['max_frames']}")
    print()

    # V4.2: Load VoiceProfile objects for per-character voice cloning
    voice_profile_objs = {}
    if VOICE_PROFILE_AVAILABLE and voice_profiles:
        try:
            voice_profile_objs = VoiceProfile.load_profiles_from_storyboard(storyboard)
            if voice_profile_objs:
                print(f"  VoiceProfile objects loaded: {list(voice_profile_objs.keys())}")
        except Exception as e:
            logger.warning(f"  VoiceProfile loading failed: {e}")

    # V4.2: Try AudioGenerator for enhanced voice cloning, fallback to F5-TTS-MLX
    audio_gen = None
    if VOICE_PROFILE_AVAILABLE and voice_profile_objs:
        try:
            audio_gen = AudioGenerator(tts_engine="auto", device="cpu")
            print("  AudioGenerator loaded for enhanced voice cloning")
        except Exception as e:
            logger.warning(f"  AudioGenerator init failed ({e}), falling back to F5-TTS-MLX")
            audio_gen = None

    # Import F5-TTS-MLX as fallback
    f5_generate = None
    try:
        from f5_tts_mlx.generate import generate as _f5_gen
        f5_generate = _f5_gen
    except ImportError:
        if audio_gen is None:
            logger.error("  Neither AudioGenerator nor F5-TTS-MLX available for TTS")

    generated = []
    for i, shot in enumerate(shots):
        narration = shot.get("narration") or ""
        voice_id = shot.get("voice_id", "narrator")

        if not narration.strip():
            print(f"  Shot {i:2d}: (no narration, skipping)")
            generated.append("")
            continue

        out_path = str(NARRATION_DIR / f"narration_{i:04d}.wav")

        # Check if already generated
        if os.path.exists(out_path):
            duration = _get_wav_duration(out_path)
            print(f"  Shot {i:2d}: [cached] {duration:.2f}s -- {voice_id}: {narration[:40]}...")
            generated.append(out_path)
            continue

        print(f"  Shot {i:2d}: Generating ({voice_id})... {narration[:40]}...")
        t0 = time.time()

        try:
            # V4.2: Prefer AudioGenerator with VoiceProfile for enhanced cloning
            vp = voice_profile_objs.get(voice_id)
            if audio_gen is not None and vp is not None:
                result = audio_gen.generate_narration_with_voice(
                    text=narration,
                    voice_profile=vp,
                    output_path=out_path,
                    seed=42 + i,
                )
                if result and os.path.exists(result):
                    duration = _get_wav_duration(result)
                    elapsed = time.time() - t0
                    print(f"           Done (VoiceProfile): {duration:.2f}s audio ({elapsed:.1f}s generation)")
                    generated.append(result)
                    continue
                else:
                    logger.warning(f"  Shot {i}: AudioGenerator produced no output, trying F5-TTS-MLX fallback")

            # Fallback: F5-TTS-MLX direct generation
            if f5_generate is not None:
                kwargs = dict(
                    generation_text=narration,
                    output_path=out_path,
                    seed=42 + i,  # reproducible per shot
                )

                # If we have voice profile with reference audio, use it
                profile = voice_profiles.get(voice_id, {})
                ref_audio = profile.get("ref_audio")
                ref_text = profile.get("ref_text")
                if ref_audio and os.path.exists(ref_audio):
                    kwargs["ref_audio_path"] = ref_audio
                if ref_text:
                    kwargs["ref_audio_text"] = ref_text

                f5_generate(**kwargs)

                if os.path.exists(out_path):
                    duration = _get_wav_duration(out_path)
                    elapsed = time.time() - t0
                    print(f"           Done: {duration:.2f}s audio ({elapsed:.1f}s generation)")
                    generated.append(out_path)
                else:
                    logger.error(f"  Shot {i}: TTS produced no output file")
                    generated.append("")
            else:
                logger.error(f"  Shot {i}: No TTS engine available")
                generated.append("")
        except Exception as e:
            logger.error(f"  Shot {i}: TTS failed: {e}")
            import traceback
            traceback.print_exc()
            generated.append("")

    print(f"\nGenerated {sum(1 for g in generated if g)} / {len(shots)} narration clips")
    return generated


def phase1_compute_durations():
    """Compute frame counts from narration audio, optionally align to beats."""
    print("\n" + "-" * 60)
    print("PHASE 1.2: Computing Audio-First Durations")
    print("-" * 60 + "\n")

    # Load storyboard and model params
    storyboard = _load_storyboard()
    model_params = _get_model_params(storyboard)

    gen_fps = model_params["fps"]
    max_frames = model_params["max_frames"]

    print(f"  Using model_params: fps={gen_fps}, max_frames={max_frames}")

    shots = storyboard["shots"]
    timing = []

    for i, shot in enumerate(shots):
        wav_path = NARRATION_DIR / f"narration_{i:04d}.wav"
        narration = shot.get("narration") or ""
        mode = shot.get("mode", "t2v")

        if narration.strip() and wav_path.exists():
            audio_dur = _get_wav_duration(str(wav_path))
            total_dur = audio_dur + TRANSITION_PAD
            num_frames = _duration_to_frames(total_dur, gen_fps)
            num_frames = min(num_frames, max_frames)
            actual_dur = num_frames / gen_fps

            shot["num_frames"] = num_frames
            shot["duration_seconds"] = round(actual_dur, 3)

            overflow = ""
            if audio_dur > max_frames / gen_fps:
                overflow = " [!] narration exceeds max, last frames will hold"

            print(f"  Shot {i:2d} ({mode:10s}): audio={audio_dur:.2f}s -> {num_frames}f ({actual_dur:.2f}s){overflow}")
        else:
            # No narration: default 5s
            num_frames = _duration_to_frames(5.0, gen_fps)
            num_frames = min(num_frames, max_frames)
            shot["num_frames"] = num_frames
            shot["duration_seconds"] = round(num_frames / gen_fps, 3)
            print(f"  Shot {i:2d} ({mode:10s}): (no narration) -> default {num_frames}f ({num_frames / gen_fps:.2f}s)")

        timing.append({
            "shot_id": i,
            "scene": shot.get("scene", ""),
            "mode": mode,
            "num_frames": shot["num_frames"],
            "duration_seconds": shot["duration_seconds"],
            "fps": gen_fps,
        })

    # --- V4: Beat Sync ---
    beat_cfg = _get_beat_sync_config(storyboard)
    beat_info = None

    if beat_cfg["enabled"]:
        print("\n" + "-" * 60)
        print("PHASE 1.3: Beat Sync Alignment")
        print("-" * 60 + "\n")

        if not LIBROSA_AVAILABLE:
            print("  [SKIP] librosa not installed -- beat sync disabled")
            print("  Install with: pip install librosa")
        else:
            # Resolve BGM path
            bgm_rel = beat_cfg.get("bgm_path", "")
            bgm_path = None
            if bgm_rel:
                bgm_path = Path(bgm_rel) if Path(bgm_rel).is_absolute() else PROJECT_ROOT / bgm_rel
            if bgm_path is None or not bgm_path.exists():
                bgm_path = OUTPUT_DIR / BGM_FILENAME

            if bgm_path.exists():
                try:
                    from animatediff.postprocess.beat_sync import BeatAnalyzer, ShotBeatAligner

                    print(f"  Analyzing beats in: {bgm_path}")
                    analyzer = BeatAnalyzer(str(bgm_path))
                    beat_info = analyzer.get_info()

                    print(f"  Tempo: {beat_info.tempo:.1f} BPM")
                    print(f"  Beats: {len(beat_info.beat_times)}")
                    print(f"  Downbeats: {len(beat_info.downbeat_times)}")
                    print(f"  Mode: {beat_cfg['mode']}")

                    # Align shot durations to beats
                    aligner = ShotBeatAligner(analyzer)
                    aligned = aligner.align_shots(timing, mode=beat_cfg["mode"])

                    # Apply aligned durations back to shots
                    for idx, (shot, aligned_info) in enumerate(zip(shots, aligned)):
                        old_dur = shot.get("duration_seconds", 0)
                        new_dur = aligned_info["duration_seconds"]
                        new_frames = aligned_info["num_frames"]

                        shot["num_frames"] = new_frames
                        shot["duration_seconds"] = new_dur

                        timing[idx]["num_frames"] = new_frames
                        timing[idx]["duration_seconds"] = new_dur

                        delta = new_dur - old_dur
                        if abs(delta) > 0.01:
                            print(f"    Shot {idx:2d}: {old_dur:.2f}s -> {new_dur:.2f}s (delta={delta:+.2f}s)")

                    print(f"\n  Beat sync applied ({beat_cfg['mode']})")

                except Exception as e:
                    logger.warning(f"  Beat sync failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  [SKIP] BGM not found at {bgm_path} -- download in Phase 3 first, then re-run Phase 1")

    # --- V4.3: ShotOrchestrator — auto-camera + rhythm planning ---
    orchestrator_cfg = _get_shot_orchestrator_config(storyboard)
    if orchestrator_cfg["enabled"] and SHOT_ORCHESTRATOR_AVAILABLE:
        print("\n" + "-" * 60)
        print("PHASE 1.4: ShotOrchestrator (Auto-Camera + Rhythm)")
        print("-" * 60 + "\n")

        try:
            orchestrator = ShotOrchestrator()
            shot_specs = _shots_to_shotspecs(shots, storyboard)

            if shot_specs:
                # Auto-assign camera presets
                if orchestrator_cfg["auto_cameras"]:
                    print("  Auto-assigning camera presets...")
                    orchestrator.plan_cameras(shot_specs)
                    for spec in shot_specs:
                        if spec.camera:
                            print(f"    Shot {spec.shot_id:2d}: camera={spec.camera}")

                # Rhythm planning with optional beat energy curve
                if orchestrator_cfg["auto_rhythm"]:
                    energy_curve = None
                    if beat_info is not None:
                        # Build per-shot energy from beat analysis
                        try:
                            from animatediff.postprocess.beat_sync import BeatAnalyzer
                            bgm_rel = beat_cfg.get("bgm_path", "")
                            bgm_path = None
                            if bgm_rel:
                                bgm_path = Path(bgm_rel) if Path(bgm_rel).is_absolute() else PROJECT_ROOT / bgm_rel
                            if bgm_path is None or not bgm_path.exists():
                                bgm_path = OUTPUT_DIR / BGM_FILENAME
                            if bgm_path.exists() and LIBROSA_AVAILABLE:
                                y, sr = librosa.load(str(bgm_path), sr=None)
                                # Compute per-shot energy as RMS of audio in each shot's time window
                                energy_curve = []
                                current_time = 0.0
                                for t in timing:
                                    dur = t["duration_seconds"]
                                    start_sample = int(current_time * sr)
                                    end_sample = int((current_time + dur) * sr)
                                    segment = y[start_sample:end_sample]
                                    rms = float((segment ** 2).mean() ** 0.5) if len(segment) > 0 else 0.0
                                    energy_curve.append(rms)
                                    current_time += dur
                                # Normalize to [0, 1]
                                max_e = max(energy_curve) if energy_curve else 1.0
                                if max_e > 0:
                                    energy_curve = [e / max_e for e in energy_curve]
                                print(f"  Energy curve: {[f'{e:.2f}' for e in energy_curve[:8]]}{'...' if len(energy_curve) > 8 else ''}")
                        except Exception as e:
                            logger.warning(f"  Energy curve computation failed: {e}")
                            energy_curve = None

                    print("  Planning rhythm...")
                    orchestrator.plan_rhythm(shot_specs, energy_curve=energy_curve)

                # Write results back to raw shot dicts
                _apply_shotspecs_to_shots(shot_specs, shots)

                # Update timing entries
                for idx, (spec, t) in enumerate(zip(shot_specs, timing)):
                    if spec.duration_seconds > 0:
                        t["duration_seconds"] = spec.duration_seconds
                    if spec.num_frames > 0:
                        t["num_frames"] = spec.num_frames

                print("  ShotOrchestrator planning complete.")

        except Exception as e:
            logger.warning(f"  ShotOrchestrator failed: {e}")
            import traceback
            traceback.print_exc()
    elif orchestrator_cfg["enabled"] and not SHOT_ORCHESTRATOR_AVAILABLE:
        print("\n  [SKIP] ShotOrchestrator not available (module not installed)")

    # Save updated storyboard (with computed durations)
    computed_path = OUTPUT_DIR / "storyboard_computed.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(computed_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)
    print(f"\nSaved computed storyboard: {computed_path}")

    # Save timing summary
    timing_path = OUTPUT_DIR / "shot_timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    # Save beat info if available
    if beat_info is not None:
        beat_path = OUTPUT_DIR / "beat_info.json"
        with open(beat_path, "w", encoding="utf-8") as f:
            json.dump({
                "tempo": beat_info.tempo,
                "beat_times": beat_info.beat_times,
                "downbeat_times": beat_info.downbeat_times,
                "duration": beat_info.duration,
            }, f, indent=2)
        print(f"Beat info saved: {beat_path}")

    total_dur = sum(t["duration_seconds"] for t in timing)
    total_frames = sum(t["num_frames"] for t in timing)
    t2v_count = sum(1 for t in timing if t["mode"] == "t2v")
    i2v_count = sum(1 for t in timing if t["mode"] == "i2v")
    other_count = sum(1 for t in timing if t["mode"] not in ("t2v", "i2v"))
    print(f"\nTotal: {total_frames} frames, {total_dur:.1f}s ({total_dur/60:.1f}min)")
    print(f"Shots: {t2v_count} T2V + {i2v_count} I2V + {other_count} title/end cards")
    print(f"Timing saved: {timing_path}")

    return timing


# ============================================================================
# Phase 2: Cloud Generation (Vast.ai / CUDA GPU)
# ============================================================================

def phase2_generate_on_gpu():
    """Generate character portraits + T2V/I2V video shots on GPU.

    V4 additions:
    - Camera presets applied to prompts
    - Shot chaining (last frame -> I2V reference for next shot)
    - VACE backend for continuation shots (fallback to standard wan22)

    V4.2 additions:
    - Multi-backend selection: skyreels_v3 (reference_images), wan22_s2v (audio_path), ltx2 (joint A/V)
    - CameraController trajectory injection for compatible backends

    V4.3 additions:
    - ShotOrchestrator trajectory generation (per-shot camera paths)
    """
    print("\n" + "=" * 60)
    print("PHASE 2: GPU Generation (V4 — Portraits + T2V/I2V + Camera + Chaining + Multi-Backend)")
    print("=" * 60 + "\n")

    import torch
    from animatediff.core.gpu_config import GPUConfig
    from animatediff.backends import get_backend

    # Auto-detect GPU and configure generation parameters
    gpu = GPUConfig.auto_detect()
    print(gpu.summary())

    device = gpu.device
    torch_dtype = gpu.torch_dtype
    model_variant = gpu.model_variant
    gen_width = gpu.width
    gen_height = gpu.height
    gen_fps = gpu.fps
    max_frames = gpu.max_frames
    guidance_scale = gpu.guidance_scale
    guidance_scale_2 = gpu.guidance_scale_2
    num_inference_steps = gpu.num_inference_steps
    offload = gpu.offload
    quantization_override = gpu.quantization
    portrait_width = gpu.portrait_width
    portrait_height = gpu.portrait_height

    # Load storyboard (computed if available, else original)
    storyboard = _load_computed_storyboard()
    model_params = _get_model_params(storyboard)

    # Allow storyboard to override max_frames if specified
    storyboard_max_frames = model_params.get("max_frames")
    if storyboard_max_frames and storyboard_max_frames != 81:
        max_frames = storyboard_max_frames

    # V4 configs
    chaining_cfg = _get_shot_chaining_config(storyboard)
    is_v4 = _is_v4_storyboard(storyboard)

    # V4.2/V4.3 configs
    multi_backend_cfg = _get_multi_backend_config(storyboard)
    camera_ctrl_cfg = _get_camera_ctrl_config(storyboard)
    orchestrator_cfg = _get_shot_orchestrator_config(storyboard)

    # V4.4: SeedAnce-style systems
    seedance_cfg = _get_seedance_config(storyboard)

    # Build character identity sheet
    identity_keeper = None
    if IDENTITY_KEEPER_AVAILABLE and seedance_cfg["identity_enforcement"] != "none":
        try:
            identity_keeper = IdentityKeeper()
            characters = storyboard.get("characters", {})
            for name, char_info in characters.items():
                img_path = char_info.get("image_path", "")
                # Resolve relative paths
                if img_path:
                    img_abs = Path(img_path) if Path(img_path).is_absolute() else PROJECT_ROOT / img_path
                    if img_abs.exists():
                        from PIL import Image
                        img = Image.open(str(img_abs)).convert("RGB")
                        identity = identity_keeper.extract_identity(img, name=name)
                        identity.reference_path = str(img_abs)
                        lora_path = char_info.get("lora_path")
                        if lora_path:
                            identity.metadata["lora_path"] = lora_path
                            identity.metadata["lora_scale"] = char_info.get("lora_scale", 1.0)
                        identity_keeper._character_sheet[name] = identity
            logger.info(f"  IdentityKeeper: {len(identity_keeper._character_sheet)} characters")
        except Exception as e:
            logger.warning(f"  IdentityKeeper init failed: {e}")
            identity_keeper = None

    # Initialize reward ensemble for quality scoring
    reward_ensemble = None
    if REWARD_MODEL_AVAILABLE:
        try:
            reward_ensemble = RewardEnsemble()
        except Exception as e:
            logger.warning(f"  RewardEnsemble init failed: {e}")

    if is_v4:
        print(f"\nV4 features:")
        print(f"  Shot chaining: {'enabled' if chaining_cfg['enabled'] else 'disabled'}")
        print(f"  Camera presets: enabled (per-shot)")
        print(f"  Multi-backend: {'enabled' if multi_backend_cfg['enabled'] else 'disabled'}"
              f" (default={multi_backend_cfg['default_backend']})")
        print(f"  CameraCtrl: {'enabled' if camera_ctrl_cfg['enabled'] else 'disabled'}"
              f" (method={camera_ctrl_cfg['method']})")
        print(f"  ShotOrchestrator: {'enabled' if orchestrator_cfg['enabled'] else 'disabled'}")

        # V4.4 SeedAnce features
        print(f"  Identity: {'enabled' if identity_keeper else 'disabled'}"
              f" (method={seedance_cfg['identity_enforcement']})")
        print(f"  Quality gate: threshold={seedance_cfg['quality_threshold']}")
        print(f"  Acceleration: {seedance_cfg['acceleration_preset']}"
              f" ({'available' if ACCELERATION_AVAILABLE else 'not installed'})")
        print(f"  Reward model: {'available' if reward_ensemble else 'not installed'}")

        # Report availability of optional backends
        if multi_backend_cfg["enabled"]:
            print(f"    SkyReels-V3: {'available' if SKYREELS_V3_AVAILABLE else 'not installed'}")
            print(f"    Wan2.2-S2V:  {'available' if WAN22_S2V_AVAILABLE else 'not installed'}")
            print(f"    LTX-2:       {'available' if LTX2_AVAILABLE else 'not installed'}")
        if camera_ctrl_cfg["enabled"]:
            print(f"    CameraCtrl:  {'available' if CAMERA_CTRL_AVAILABLE else 'not installed'}")

    # V4.3: Generate camera trajectories if orchestrator is enabled
    shot_trajectories = {}  # shot_index -> CameraTrajectory
    if orchestrator_cfg["enabled"] and SHOT_ORCHESTRATOR_AVAILABLE and CAMERA_CTRL_AVAILABLE:
        try:
            shots_list = storyboard["shots"]
            shot_specs = _shots_to_shotspecs(shots_list, storyboard)
            if shot_specs:
                orch = ShotOrchestrator()
                shot_trajectories = orch.generate_trajectories(shot_specs)
                if shot_trajectories:
                    print(f"\n  Generated {len(shot_trajectories)} camera trajectories")
                    for idx, traj in shot_trajectories.items():
                        print(f"    Shot {idx}: {traj}")
        except Exception as e:
            logger.warning(f"  Trajectory generation failed: {e}")

    # Write actual generation params to computed storyboard so Phase 3 picks up correct fps
    storyboard["model_params"]["fps"] = gen_fps
    storyboard["model_params"]["width"] = gen_width
    storyboard["model_params"]["height"] = gen_height
    computed_path = OUTPUT_DIR / "storyboard_computed.json"
    with open(computed_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)

    BackendClass = get_backend("wan22")

    # --- Try to load VACE backend for continuation shots ---
    VACEBackendClass = None
    if chaining_cfg["enabled"] and is_v4:
        try:
            VACEBackendClass = get_backend("wan22_vace")
            print("  VACE backend available for continuation shots")
        except Exception as e:
            logger.warning(f"  VACE backend not available ({e}), will use standard wan22 for all shots")

    # --- Try to load ShotChainer ---
    ShotChainer = None
    if chaining_cfg["enabled"] and is_v4:
        try:
            from animatediff.core.shot_chaining import ShotChainer as _ShotChainer
            ShotChainer = _ShotChainer
            print(f"  ShotChainer loaded (method={chaining_cfg['method']}, overlap={chaining_cfg['overlap_frames']})")
        except ImportError:
            logger.warning("  ShotChainer module not found -- using built-in last-frame extraction")
            ShotChainer = None

    # ---------------------------------------------------------------
    # Step 2.1: Generate character portraits (T2V mode)
    # ---------------------------------------------------------------
    print("\n--- Step 2.1: Generating Character Portraits (T2V) ---")
    PORTRAIT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading Wan 2.2 {model_variant} pipeline (T2V mode for portraits)...")
    pipeline_t2v = BackendClass.load(
        model_variant=model_variant,
        mode="t2v",
        torch_dtype=torch_dtype,
        device=device,
        quantization=quantization_override,
        offload_strategy=offload,
        enable_vae_slicing=True,
        enable_vae_tiling=True,
    )
    print("  T2V pipeline loaded.")

    portrait_results = {}
    for name, info in storyboard.get("characters", {}).items():
        # Check storyboard image_path first (user-provided reference)
        storyboard_img = info.get("image_path")
        if storyboard_img:
            img_abs = Path(storyboard_img) if Path(storyboard_img).is_absolute() else PROJECT_ROOT / storyboard_img
            if img_abs.exists():
                print(f"  {name}: [storyboard ref] {img_abs}")
                portrait_results[name] = str(img_abs)
                continue

        # Check cached .png or .jpg
        portrait_path = PORTRAIT_DIR / f"{name}.png"
        if not portrait_path.exists():
            portrait_path = PORTRAIT_DIR / f"{name}.jpg"
        if portrait_path.exists():
            print(f"  {name}: [cached] {portrait_path}")
            portrait_results[name] = str(portrait_path)
            continue

        prompt = info.get("portrait_prompt", "")
        if not prompt:
            desc = info.get("description", name)
            prompt = f"portrait of {desc}, anime style, high detail, upper body"

        print(f"  {name}: Generating portrait...")
        print(f"    Prompt: {prompt[:70]}...")

        candidates = []
        for seed_offset in range(3):  # 3 candidates
            seed = 42 + seed_offset
            output = pipeline_t2v.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed, ugly, watermark, multiple people",
                width=portrait_width,
                height=portrait_height,
                num_frames=17,
                num_inference_steps=30,
                guidance_scale=6.0,
                seed=seed,
            )
            if output.frames:
                mid = len(output.frames) // 2
                candidate = output.frames[mid]
                candidate_path = PORTRAIT_DIR / f"{name}_candidate_{seed_offset}.png"
                candidate.save(str(candidate_path))
                score = _score_portrait(candidate)
                candidates.append((candidate, output.seed, score))
                print(f"    Candidate {seed_offset} (seed={output.seed}): score={score:.2f}")

        if candidates:
            # Pick the candidate with highest quality score
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_frame, best_seed, best_score = candidates[0]
            portrait_path = PORTRAIT_DIR / f"{name}.png"
            best_frame.save(str(portrait_path))
            portrait_results[name] = str(portrait_path)
            print(f"    -> Best: {portrait_path} (seed={best_seed}, score={best_score:.2f})")

    # ---------------------------------------------------------------
    # Step 2.2: Generate video shots (T2V + I2V, sequential for chaining)
    # ---------------------------------------------------------------
    print("\n--- Step 2.2: Generating Video Shots (T2V + I2V with camera presets + chaining) ---")
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    shots = storyboard["shots"]

    # Separate shots by type for pipeline management
    gen_shots = [(i, s) for i, s in enumerate(shots) if s.get("mode") in ("t2v", "i2v")]
    card_shots = [(i, s) for i, s in enumerate(shots) if s.get("mode") in ("title_card", "end_card")]

    # When chaining is enabled, we must generate shots sequentially so each shot
    # can use the previous shot's last frame. When disabled, we group by mode.
    chaining_enabled = chaining_cfg["enabled"] and is_v4

    if chaining_enabled:
        print(f"  Shot chaining enabled -- generating {len(gen_shots)} shots sequentially")
        _generate_shots_chained(
            gen_shots=gen_shots,
            storyboard=storyboard,
            BackendClass=BackendClass,
            VACEBackendClass=VACEBackendClass,
            ShotChainer=ShotChainer,
            chaining_cfg=chaining_cfg,
            portrait_results=portrait_results,
            pipeline_t2v=pipeline_t2v,
            model_variant=model_variant,
            torch_dtype=torch_dtype,
            device=device,
            quantization_override=quantization_override,
            offload=offload,
            gen_width=gen_width,
            gen_height=gen_height,
            gen_fps=gen_fps,
            max_frames=max_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            multi_backend_cfg=multi_backend_cfg,
            camera_ctrl_cfg=camera_ctrl_cfg,
            shot_trajectories=shot_trajectories,
            identity_keeper=identity_keeper,
            reward_ensemble=reward_ensemble,
            seedance_cfg=seedance_cfg,
        )
    else:
        print(f"  Shot chaining disabled -- generating by mode (T2V first, then I2V)")
        _generate_shots_by_mode(
            gen_shots=gen_shots,
            storyboard=storyboard,
            BackendClass=BackendClass,
            portrait_results=portrait_results,
            pipeline_t2v=pipeline_t2v,
            model_variant=model_variant,
            torch_dtype=torch_dtype,
            device=device,
            quantization_override=quantization_override,
            offload=offload,
            gen_width=gen_width,
            gen_height=gen_height,
            gen_fps=gen_fps,
            max_frames=max_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            multi_backend_cfg=multi_backend_cfg,
            camera_ctrl_cfg=camera_ctrl_cfg,
            shot_trajectories=shot_trajectories,
        )

    # ---------------------------------------------------------------
    # Step 2.3: Generate title_card / end_card shots in code
    # ---------------------------------------------------------------
    if card_shots:
        print(f"\n--- Step 2.3: Generating Title/End Cards ({len(card_shots)}) ---")
        for i, shot in card_shots:
            shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
            if shot_path.exists():
                print(f"  Shot {i:2d}: [cached] {shot_path}")
                continue

            num_frames = shot.get("num_frames", 0)
            if num_frames == 0:
                num_frames = _duration_to_frames(shot.get("duration_seconds", 3.0), gen_fps)

            card_title = shot.get("title_text", shot.get("scene", ""))
            card_subtitle = shot.get("subtitle_text", "")

            frames = _create_title_card(
                card_title,
                card_subtitle,
                size=(gen_width, gen_height),
                num_frames=num_frames,
            )

            _save_frames_as_mp4(frames, str(shot_path), gen_fps)
            print(f"  Shot {i:2d}: {shot.get('mode')} -> {shot_path} ({num_frames}f)")

    print("\nPhase 2 complete!")


def _generate_shots_chained(
    gen_shots, storyboard, BackendClass, VACEBackendClass, ShotChainer,
    chaining_cfg, portrait_results, pipeline_t2v,
    model_variant, torch_dtype, device, quantization_override, offload,
    gen_width, gen_height, gen_fps, max_frames,
    num_inference_steps, guidance_scale, guidance_scale_2,
    multi_backend_cfg=None, camera_ctrl_cfg=None, shot_trajectories=None,
    identity_keeper=None, reward_ensemble=None, seedance_cfg=None,
):
    """Generate shots sequentially with chaining: each shot can use the
    previous shot's last frame as I2V reference for visual continuity.

    For shots with chain_from_previous=True:
    1. Try VACE continuation backend if available
    2. Fall back to I2V with last frame as reference
    3. Fall back to T2V if all else fails

    V4.2: Multi-backend selection based on shot properties.
    V4.3: Camera trajectory injection via CameraController.
    """
    import torch
    from PIL import Image

    # Initialize ShotChainer if available
    chainer = None
    if ShotChainer is not None:
        try:
            chainer = ShotChainer(
                method=chaining_cfg.get("method", "last_frame_i2v"),
                overlap_frames=chaining_cfg.get("overlap_frames", 4),
            )
        except Exception as e:
            logger.warning(f"  ShotChainer init failed: {e}")

    # We need both T2V and I2V pipelines. Start with T2V (already loaded).
    # We will lazily load I2V and VACE pipelines when first needed.
    current_pipeline = pipeline_t2v
    current_mode = "t2v"
    pipeline_i2v = None
    pipeline_vace = None

    prev_shot_path = None  # Path to the previously generated shot (for chaining)

    for i, shot in gen_shots:
        shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
        mode = shot.get("mode", "t2v")
        should_chain = shot.get("chain_from_previous", False) and prev_shot_path is not None

        if shot_path.exists():
            print(f"  Shot {i:2d}: [cached] {shot_path}")
            prev_shot_path = str(shot_path)
            continue

        num_frames = shot.get("num_frames", 0)
        if num_frames == 0:
            num_frames = _duration_to_frames(shot.get("duration_seconds", 5.0), gen_fps)
            num_frames = min(num_frames, max_frames)

        # V4: Apply camera preset to prompt
        prompt = shot["prompt"]
        camera_preset = shot.get("camera_preset")
        if camera_preset:
            prompt = _apply_camera_preset(prompt, camera_preset)

        scene = shot.get("scene", "")
        print(f"  Shot {i:2d}: {mode} — {scene} ({num_frames}f)", end="")
        if should_chain:
            print(" [chained]", end="")
        if camera_preset:
            print(f" [cam:{camera_preset}]", end="")
        print()

        # ---- Determine generation strategy ----
        ref_image = None
        use_vace_continuation = False

        if should_chain:
            # Extract reference frame from previous shot
            if chainer is not None:
                try:
                    ref_image = chainer.extract_reference_frame(prev_shot_path, "last")
                except Exception as e:
                    logger.warning(f"    Chainer extract failed: {e}, trying manual extraction")
                    ref_image = _extract_last_frame(prev_shot_path)
            else:
                ref_image = _extract_last_frame(prev_shot_path)

            if ref_image is not None:
                # Try VACE continuation first, then I2V fallback
                if VACEBackendClass is not None and mode == "t2v":
                    use_vace_continuation = True
                else:
                    # Override mode to i2v for chaining
                    mode = "i2v"
            else:
                logger.warning(f"    No reference frame available, generating as standalone {mode}")

        # If i2v mode, resolve character portrait as reference (when not chaining)
        if mode == "i2v" and ref_image is None:
            characters = shot.get("characters", [])
            if characters:
                char_name = characters[0]
                portrait_path_str = portrait_results.get(char_name)
                if portrait_path_str and os.path.exists(portrait_path_str):
                    ref_image = Image.open(portrait_path_str).convert("RGB")
                    print(f"           Using portrait ref: {char_name}")
                else:
                    print(f"           No portrait for {char_name}, using black frame")
                    ref_image = Image.new("RGB", (gen_width, gen_height), (0, 0, 0))
            else:
                print(f"           No characters specified, using black frame")
                ref_image = Image.new("RGB", (gen_width, gen_height), (0, 0, 0))

        # ---- V4.2: Multi-backend selection ----
        # Check if this shot should use a specialized backend instead of default wan22
        use_alt_backend = None  # will be set to a backend name if applicable
        if multi_backend_cfg and multi_backend_cfg.get("enabled", False) and not should_chain:
            # SkyReels-V3: for shots with reference_images (multi-ref character consistency)
            reference_images = shot.get("reference_images", [])
            if reference_images and SKYREELS_V3_AVAILABLE and device == "cuda":
                use_alt_backend = "skyreels_v3"

            # Wan 2.2 S2V: for shots with audio_path (audio-driven lip sync)
            audio_path = shot.get("audio_path", "")
            if audio_path and WAN22_S2V_AVAILABLE and device == "cuda":
                use_alt_backend = "wan22_s2v"

            # LTX-2: for shots requesting joint audio-video generation
            if shot.get("joint_av", False) and LTX2_AVAILABLE and device == "cuda":
                use_alt_backend = "ltx2"

        # ---- V4.3: Camera trajectory injection ----
        trajectory = None
        if shot_trajectories and i in shot_trajectories:
            trajectory = shot_trajectories[i]

        # ---- Generate ----
        t0 = time.time()

        # V4.2: Dispatch to specialized backend if selected
        if use_alt_backend is not None:
            try:
                if use_alt_backend == "skyreels_v3":
                    print(f"           Using SkyReels-V3 backend (multi-reference)")
                    reference_images_pil = []
                    for ref_path in shot.get("reference_images", []):
                        ref_abs = Path(ref_path) if Path(ref_path).is_absolute() else PROJECT_ROOT / ref_path
                        if ref_abs.exists():
                            reference_images_pil.append(Image.open(str(ref_abs)).convert("RGB"))
                    if reference_images_pil:
                        alt_pipeline = SkyReelsV3Backend.load(
                            model_variant="R2V-14B",
                            torch_dtype=torch_dtype,
                            device=device,
                            quantization=quantization_override,
                            offload_strategy=offload,
                        )
                        output = alt_pipeline.generate(
                            prompt=prompt,
                            negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                            reference_images=reference_images_pil,
                            width=gen_width, height=gen_height,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            seed=shot.get("seed", -1),
                        )
                        elapsed = time.time() - t0
                        alt_pipeline.save(output, str(shot_path), fps=gen_fps)
                        print(f"           Done (SkyReels-V3): {elapsed:.1f}s -> {shot_path}")
                        del alt_pipeline; gc.collect()
                        if device == "cuda": torch.cuda.empty_cache()
                        prev_shot_path = str(shot_path)
                        continue

                elif use_alt_backend == "wan22_s2v":
                    print(f"           Using Wan2.2-S2V backend (audio-driven)")
                    audio_p = shot.get("audio_path", "")
                    audio_abs = Path(audio_p) if Path(audio_p).is_absolute() else PROJECT_ROOT / audio_p
                    s2v_image = ref_image
                    if s2v_image is None:
                        # Use portrait as reference image for lip sync
                        characters = shot.get("characters", [])
                        if characters:
                            pp = portrait_results.get(characters[0])
                            if pp and os.path.exists(pp):
                                s2v_image = Image.open(pp).convert("RGB")
                    if s2v_image is not None and audio_abs.exists():
                        alt_pipeline = Wan22S2VBackend.load(
                            torch_dtype=torch_dtype,
                            device=device,
                            quantization=quantization_override,
                            offload_strategy=offload,
                        )
                        output = alt_pipeline.generate(
                            prompt=prompt,
                            image=s2v_image,
                            audio_path=str(audio_abs),
                            width=gen_width, height=gen_height,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            seed=shot.get("seed", -1),
                        )
                        elapsed = time.time() - t0
                        alt_pipeline.save(output, str(shot_path), fps=gen_fps)
                        print(f"           Done (S2V): {elapsed:.1f}s -> {shot_path}")
                        del alt_pipeline; gc.collect()
                        if device == "cuda": torch.cuda.empty_cache()
                        prev_shot_path = str(shot_path)
                        continue

                elif use_alt_backend == "ltx2":
                    print(f"           Using LTX-2 backend (joint audio-video)")
                    alt_pipeline = LTX2Backend.load(
                        model_variant="19b",
                        mode="text_to_av",
                        torch_dtype=torch_dtype,
                        device=device,
                        quantization=quantization_override,
                        offload_strategy=offload,
                    )
                    output = alt_pipeline.generate(
                        prompt=prompt,
                        negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                        width=gen_width, height=gen_height,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=shot.get("seed", -1),
                        enable_audio=True,
                    )
                    elapsed = time.time() - t0
                    alt_pipeline.save(output, str(shot_path), fps=gen_fps)
                    # Save audio if generated
                    if hasattr(output, 'audio') and output.audio is not None:
                        audio_out_path = SHOTS_DIR / f"shot_{i:04d}_audio.wav"
                        try:
                            import torchaudio
                            torchaudio.save(str(audio_out_path), output.audio.cpu(),
                                          output.audio_sample_rate)
                            print(f"           Audio saved: {audio_out_path}")
                        except Exception as ae:
                            logger.warning(f"           LTX-2 audio save failed: {ae}")
                    print(f"           Done (LTX-2): {elapsed:.1f}s -> {shot_path}")
                    del alt_pipeline; gc.collect()
                    if device == "cuda": torch.cuda.empty_cache()
                    prev_shot_path = str(shot_path)
                    continue

            except Exception as e:
                logger.warning(f"    Alt backend '{use_alt_backend}' failed ({e}), falling back to default")

        if use_vace_continuation and ref_image is not None:
            # Try VACE continuation
            try:
                if pipeline_vace is None:
                    print("           Loading VACE pipeline for continuation...")
                    # Free current pipeline first
                    if current_pipeline is not pipeline_t2v:
                        del current_pipeline
                    if pipeline_i2v is not None:
                        del pipeline_i2v
                        pipeline_i2v = None
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    # Load VACE model params
                    vace_params = storyboard.get("model_params_vace", {})
                    vace_variant = "1.3B"  # VACE is available in 1.3B and 14B
                    if device == "cuda":
                        vace_variant = "14B"

                    pipeline_vace = VACEBackendClass.load(
                        model_variant=vace_variant,
                        torch_dtype=torch_dtype,
                        device=device,
                        quantization=quantization_override,
                        offload_strategy=offload,
                        enable_vae_slicing=True,
                        enable_vae_tiling=True,
                    )
                    print("           VACE pipeline loaded.")

                # Use reference_to_video mode with the last frame
                output = pipeline_vace.generate(
                    prompt=prompt,
                    negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                    width=gen_width,
                    height=gen_height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=shot.get("seed", -1),
                    image=ref_image,
                    mode="reference_to_video",
                )
                elapsed = time.time() - t0
                pipeline_vace.save(output, str(shot_path), fps=gen_fps)
                print(f"           Done (VACE): {elapsed:.1f}s -> {shot_path}")
                prev_shot_path = str(shot_path)
                continue

            except Exception as e:
                logger.warning(f"    VACE continuation failed ({e}), falling back to standard I2V")
                mode = "i2v"  # Fall back

        # Standard T2V or I2V generation
        if mode == "i2v" and ref_image is not None:
            # Ensure I2V pipeline is loaded
            if pipeline_i2v is None:
                print("           Loading I2V pipeline...")
                # Free T2V / VACE
                if pipeline_vace is not None:
                    del pipeline_vace
                    pipeline_vace = None
                del pipeline_t2v
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                pipeline_i2v = BackendClass.load(
                    model_variant=model_variant,
                    mode="i2v",
                    torch_dtype=torch_dtype,
                    device=device,
                    quantization=quantization_override,
                    offload_strategy=offload,
                    enable_vae_slicing=True,
                    enable_vae_tiling=True,
                )
                current_mode = "i2v"
                print("           I2V pipeline loaded.")

            gen_kwargs = dict(
                prompt=prompt,
                negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                width=gen_width,
                height=gen_height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=shot.get("seed", -1),
                image=ref_image,
            )
            if guidance_scale_2 is not None:
                gen_kwargs["guidance_scale_2"] = guidance_scale_2

            # V4.3: Apply camera trajectory if available
            if trajectory is not None and CAMERA_CTRL_AVAILABLE and camera_ctrl_cfg:
                try:
                    ctrl = CameraController(method=camera_ctrl_cfg.get("method", "prompt_only"))
                    gen_kwargs = ctrl.apply(gen_kwargs, trajectory)
                    print(f"           Applied camera trajectory (method={camera_ctrl_cfg['method']})")
                except Exception as e:
                    logger.warning(f"           Camera trajectory failed: {e}")

            output = pipeline_i2v.generate(**gen_kwargs)
            elapsed = time.time() - t0
            pipeline_i2v.save(output, str(shot_path), fps=gen_fps)
            print(f"           Done (I2V): {elapsed:.1f}s -> {shot_path}")

        else:
            # T2V generation (ensure T2V pipeline is available)
            # Note: pipeline_t2v may have been freed during I2V/VACE loading.
            # In chained mode this can happen. We need to reload.
            if pipeline_i2v is not None or pipeline_vace is not None:
                # Need to reload T2V
                if pipeline_i2v is not None:
                    del pipeline_i2v
                    pipeline_i2v = None
                if pipeline_vace is not None:
                    del pipeline_vace
                    pipeline_vace = None
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                print("           Reloading T2V pipeline...")
                pipeline_t2v_reload = BackendClass.load(
                    model_variant=model_variant,
                    mode="t2v",
                    torch_dtype=torch_dtype,
                    device=device,
                    quantization=quantization_override,
                    offload_strategy=offload,
                    enable_vae_slicing=True,
                    enable_vae_tiling=True,
                )
                current_pipeline = pipeline_t2v_reload
            else:
                current_pipeline = pipeline_t2v

            gen_kwargs = dict(
                prompt=prompt,
                negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                width=gen_width,
                height=gen_height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=shot.get("seed", -1),
            )
            if guidance_scale_2 is not None:
                gen_kwargs["guidance_scale_2"] = guidance_scale_2

            # V4.3: Apply camera trajectory if available
            if trajectory is not None and CAMERA_CTRL_AVAILABLE and camera_ctrl_cfg:
                try:
                    ctrl = CameraController(method=camera_ctrl_cfg.get("method", "prompt_only"))
                    gen_kwargs = ctrl.apply(gen_kwargs, trajectory)
                    print(f"           Applied camera trajectory (method={camera_ctrl_cfg['method']})")
                except Exception as e:
                    logger.warning(f"           Camera trajectory failed: {e}")

            output = current_pipeline.generate(**gen_kwargs)
            elapsed = time.time() - t0
            current_pipeline.save(output, str(shot_path), fps=gen_fps)
            print(f"           Done (T2V): {elapsed:.1f}s -> {shot_path}")

        # ---- V4.4: Post-generation quality scoring + identity verification ----
        if shot_path.exists() and (reward_ensemble or identity_keeper):
            try:
                gen_frames = _read_video_frames(str(shot_path))
                if gen_frames:
                    # Quality scoring
                    if reward_ensemble:
                        try:
                            score = reward_ensemble.score(gen_frames, prompt)
                            q_thresh = (seedance_cfg or {}).get("quality_threshold", 0.65)
                            status = "PASS" if score.overall >= q_thresh else "WARN"
                            print(f"           Quality: {score.overall:.3f} [{status}]"
                                  f" (motion={score.motion:.2f} aes={score.aesthetic:.2f}"
                                  f" align={score.alignment:.2f})")
                        except Exception as qe:
                            logger.debug(f"           Quality scoring failed: {qe}")

                    # Identity verification
                    if identity_keeper:
                        characters = shot.get("characters", [])
                        if characters:
                            try:
                                id_scores = identity_keeper.verify_shot_identity(
                                    gen_frames, characters, sample_count=3
                                )
                                for cname, iscore in id_scores.items():
                                    id_status = "PASS" if iscore.is_match else "WARN"
                                    print(f"           Identity '{cname}': "
                                          f"{iscore.overall:.3f} [{id_status}]")
                            except Exception as ie:
                                logger.debug(f"           Identity check failed: {ie}")
            except Exception as e:
                logger.debug(f"           Post-gen analysis failed: {e}")

        prev_shot_path = str(shot_path)

    # Cleanup all pipelines
    for p in [pipeline_i2v, pipeline_vace]:
        if p is not None:
            del p
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


def _generate_shots_by_mode(
    gen_shots, storyboard, BackendClass, portrait_results, pipeline_t2v,
    model_variant, torch_dtype, device, quantization_override, offload,
    gen_width, gen_height, gen_fps, max_frames,
    num_inference_steps, guidance_scale, guidance_scale_2,
    multi_backend_cfg=None, camera_ctrl_cfg=None, shot_trajectories=None,
):
    """Generate shots grouped by mode (V3-compatible path, no chaining).

    V4.2: Multi-backend selection for shots with reference_images / audio_path.
    V4.3: Camera trajectory injection.
    """
    import torch
    from PIL import Image

    shots = storyboard["shots"]

    # --- T2V shots ---
    t2v_shots = [(i, s) for i, s in gen_shots if s.get("mode") == "t2v"]
    print(f"\n  T2V shots to generate: {len(t2v_shots)}")

    for i, shot in t2v_shots:
        shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
        if shot_path.exists():
            print(f"  Shot {i:2d}: [cached] {shot_path}")
            continue

        num_frames = shot.get("num_frames", 0)
        if num_frames == 0:
            num_frames = _duration_to_frames(shot.get("duration_seconds", 5.0), gen_fps)
            num_frames = min(num_frames, max_frames)

        # V4: Apply camera preset
        prompt = shot["prompt"]
        camera_preset = shot.get("camera_preset")
        if camera_preset:
            prompt = _apply_camera_preset(prompt, camera_preset)

        scene = shot.get("scene", "")
        print(f"  Shot {i:2d}: T2V — {scene} ({num_frames}f)", end="")
        if camera_preset:
            print(f" [cam:{camera_preset}]", end="")
        print()

        # V4.2: Check for multi-backend dispatch (non-chained mode)
        use_alt_backend = None
        if multi_backend_cfg and multi_backend_cfg.get("enabled", False):
            reference_images = shot.get("reference_images", [])
            if reference_images and SKYREELS_V3_AVAILABLE and device == "cuda":
                use_alt_backend = "skyreels_v3"
            audio_path_shot = shot.get("audio_path", "")
            if audio_path_shot and WAN22_S2V_AVAILABLE and device == "cuda":
                use_alt_backend = "wan22_s2v"
            if shot.get("joint_av", False) and LTX2_AVAILABLE and device == "cuda":
                use_alt_backend = "ltx2"

        if use_alt_backend is not None:
            # Dispatch to specialized backend (same logic as chained path)
            try:
                from PIL import Image as _Image
                if use_alt_backend == "skyreels_v3":
                    print(f"           Using SkyReels-V3 backend")
                    ref_imgs = []
                    for rp in shot.get("reference_images", []):
                        rp_abs = Path(rp) if Path(rp).is_absolute() else PROJECT_ROOT / rp
                        if rp_abs.exists():
                            ref_imgs.append(_Image.open(str(rp_abs)).convert("RGB"))
                    if ref_imgs:
                        alt_p = SkyReelsV3Backend.load(
                            model_variant="R2V-14B", torch_dtype=torch_dtype,
                            device=device, quantization=quantization_override,
                            offload_strategy=offload)
                        out = alt_p.generate(prompt=prompt,
                            negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                            reference_images=ref_imgs, width=gen_width, height=gen_height,
                            num_frames=num_frames, num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale, seed=shot.get("seed", -1))
                        alt_p.save(out, str(shot_path), fps=gen_fps)
                        print(f"           Done (SkyReels-V3): -> {shot_path}")
                        del alt_p; gc.collect()
                        if device == "cuda": torch.cuda.empty_cache()
                        continue
                elif use_alt_backend == "wan22_s2v":
                    print(f"           Using Wan2.2-S2V backend")
                    ap = shot.get("audio_path", "")
                    ap_abs = Path(ap) if Path(ap).is_absolute() else PROJECT_ROOT / ap
                    s2v_img = _Image.new("RGB", (gen_width, gen_height), (0,0,0))
                    chars = shot.get("characters", [])
                    if chars:
                        pp = portrait_results.get(chars[0])
                        if pp and os.path.exists(pp):
                            s2v_img = _Image.open(pp).convert("RGB")
                    if ap_abs.exists():
                        alt_p = Wan22S2VBackend.load(torch_dtype=torch_dtype, device=device,
                            quantization=quantization_override, offload_strategy=offload)
                        out = alt_p.generate(prompt=prompt, image=s2v_img,
                            audio_path=str(ap_abs), width=gen_width, height=gen_height,
                            num_frames=num_frames, num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale, seed=shot.get("seed", -1))
                        alt_p.save(out, str(shot_path), fps=gen_fps)
                        print(f"           Done (S2V): -> {shot_path}")
                        del alt_p; gc.collect()
                        if device == "cuda": torch.cuda.empty_cache()
                        continue
                elif use_alt_backend == "ltx2":
                    print(f"           Using LTX-2 backend")
                    alt_p = LTX2Backend.load(model_variant="19b", mode="text_to_av",
                        torch_dtype=torch_dtype, device=device,
                        quantization=quantization_override, offload_strategy=offload)
                    out = alt_p.generate(prompt=prompt,
                        negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                        width=gen_width, height=gen_height, num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale, seed=shot.get("seed", -1),
                        enable_audio=True)
                    alt_p.save(out, str(shot_path), fps=gen_fps)
                    if hasattr(out, 'audio') and out.audio is not None:
                        audio_out = SHOTS_DIR / f"shot_{i:04d}_audio.wav"
                        try:
                            import torchaudio
                            torchaudio.save(str(audio_out), out.audio.cpu(), out.audio_sample_rate)
                        except Exception: pass
                    print(f"           Done (LTX-2): -> {shot_path}")
                    del alt_p; gc.collect()
                    if device == "cuda": torch.cuda.empty_cache()
                    continue
            except Exception as e:
                logger.warning(f"    Alt backend '{use_alt_backend}' failed ({e}), using default T2V")

        t0 = time.time()
        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
            width=gen_width,
            height=gen_height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=shot.get("seed", -1),
        )
        if guidance_scale_2 is not None:
            gen_kwargs["guidance_scale_2"] = guidance_scale_2

        # V4.3: Apply camera trajectory if available
        trajectory = shot_trajectories.get(i) if shot_trajectories else None
        if trajectory is not None and CAMERA_CTRL_AVAILABLE and camera_ctrl_cfg:
            try:
                ctrl = CameraController(method=camera_ctrl_cfg.get("method", "prompt_only"))
                gen_kwargs = ctrl.apply(gen_kwargs, trajectory)
                print(f"           Applied camera trajectory")
            except Exception as e:
                logger.warning(f"           Camera trajectory failed: {e}")

        output = pipeline_t2v.generate(**gen_kwargs)
        elapsed = time.time() - t0

        pipeline_t2v.save(output, str(shot_path), fps=gen_fps)
        print(f"           Done: {elapsed:.1f}s -> {shot_path}")

    # --- I2V shots ---
    i2v_shots = [(i, s) for i, s in gen_shots if s.get("mode") == "i2v"]
    print(f"\n  I2V shots to generate: {len(i2v_shots)}")

    if i2v_shots:
        # Free T2V pipeline memory
        del pipeline_t2v
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"\n  Loading Wan 2.2 {model_variant} pipeline (I2V mode)...")
        pipeline_i2v = BackendClass.load(
            model_variant=model_variant,
            mode="i2v",
            torch_dtype=torch_dtype,
            device=device,
            quantization=quantization_override,
            offload_strategy=offload,
            enable_vae_slicing=True,
            enable_vae_tiling=True,
        )
        print("  I2V pipeline loaded.")

        for i, shot in i2v_shots:
            shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
            if shot_path.exists():
                print(f"  Shot {i:2d}: [cached] {shot_path}")
                continue

            # Resolve character reference image
            ref_image = None
            characters = shot.get("characters", [])
            if characters:
                char_name = characters[0]
                portrait_path_str = portrait_results.get(char_name)
                if portrait_path_str and os.path.exists(portrait_path_str):
                    ref_image = Image.open(portrait_path_str).convert("RGB")
                    print(f"  Shot {i:2d}: I2V with ref={char_name}")
                else:
                    print(f"  Shot {i:2d}: No portrait for {char_name}, using black frame")
                    ref_image = Image.new("RGB", (gen_width, gen_height), (0, 0, 0))
            else:
                print(f"  Shot {i:2d}: No characters specified, using black frame")
                ref_image = Image.new("RGB", (gen_width, gen_height), (0, 0, 0))

            num_frames = shot.get("num_frames", 0)
            if num_frames == 0:
                num_frames = _duration_to_frames(shot.get("duration_seconds", 5.0), gen_fps)
                num_frames = min(num_frames, max_frames)

            # V4: Apply camera preset
            prompt = shot["prompt"]
            camera_preset = shot.get("camera_preset")
            if camera_preset:
                prompt = _apply_camera_preset(prompt, camera_preset)

            scene = shot.get("scene", "")
            print(f"           {scene} ({num_frames}f)", end="")
            if camera_preset:
                print(f" [cam:{camera_preset}]", end="")
            print()

            t0 = time.time()
            gen_kwargs = dict(
                prompt=prompt,
                negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
                width=gen_width,
                height=gen_height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=shot.get("seed", -1),
                image=ref_image,
            )
            if guidance_scale_2 is not None:
                gen_kwargs["guidance_scale_2"] = guidance_scale_2

            # V4.3: Apply camera trajectory if available
            trajectory = shot_trajectories.get(i) if shot_trajectories else None
            if trajectory is not None and CAMERA_CTRL_AVAILABLE and camera_ctrl_cfg:
                try:
                    ctrl = CameraController(method=camera_ctrl_cfg.get("method", "prompt_only"))
                    gen_kwargs = ctrl.apply(gen_kwargs, trajectory)
                    print(f"           Applied camera trajectory")
                except Exception as e:
                    logger.warning(f"           Camera trajectory failed: {e}")

            output = pipeline_i2v.generate(**gen_kwargs)
            elapsed = time.time() - t0

            pipeline_i2v.save(output, str(shot_path), fps=gen_fps)
            print(f"           Done: {elapsed:.1f}s -> {shot_path}")

        del pipeline_i2v
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        # No I2V shots -- free T2V pipeline
        del pipeline_t2v
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print("  No I2V shots to generate.")


# ============================================================================
# Phase 3: Local Post-Processing (Mac MPS)
# ============================================================================

def phase3_postprocess():
    """Upscale + style harmonize + deflicker + extend + RIFE + color grade + MTV sync + compose.

    V4 additions:
    - Deflicker: Cross-shot harmonization + per-shot deflicker after upscale
    - Beat-aware transitions: Transition durations driven by beat timing

    V4.2/V4.3 additions:
    - StyleHarmonizer: Cross-shot visual consistency (before deflicker)
    - VideoExtender: Extend short clips to minimum duration
    - VideoAudioSync: 3-stream sync for frame-level alignment + transition snapping
    - Audio post-processing: normalize_audio, add_reverb for cinematic narration
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Post-Processing (V4 — Style + Upscale + Deflicker + Extend + RIFE + MTV Sync + Compose)")
    print("=" * 60 + "\n")

    import torch
    import numpy as np
    from PIL import Image

    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load storyboard
    storyboard = _load_computed_storyboard()
    model_params = _get_model_params(storyboard)
    native_fps = model_params["fps"]
    is_v4 = _is_v4_storyboard(storyboard)

    # V4 configs
    deflicker_cfg = _get_deflicker_config(storyboard)
    beat_cfg = _get_beat_sync_config(storyboard)

    # V4.2/V4.3 configs
    style_cfg = _get_style_harmonize_config(storyboard)
    extend_cfg = _get_video_extend_config(storyboard)
    mtv_cfg = _get_mtv_sync_config(storyboard)
    audio_post_cfg = _get_audio_post_config(storyboard)

    # Postprocess config from storyboard
    postprocess_cfg = storyboard.get("postprocess", {})
    interp_cfg = postprocess_cfg.get("interpolation", {})
    rife_mult = interp_cfg.get("multiplier", 2)
    color_lut = postprocess_cfg.get("color_lut")

    # Output fps after RIFE interpolation
    output_fps = native_fps * rife_mult
    print(f"  Native FPS: {native_fps}")
    print(f"  RIFE multiplier: {rife_mult}x")
    print(f"  Output FPS: {output_fps}")
    print(f"  Color LUT: {color_lut or '(none)'}")
    if is_v4:
        print(f"  Deflicker: {'enabled' if deflicker_cfg['enabled'] else 'disabled'} "
              f"(strength={deflicker_cfg['strength']}, harmonize={deflicker_cfg['cross_shot_harmonize']})")
        print(f"  Beat sync: {'enabled' if beat_cfg['enabled'] else 'disabled'}")
        print(f"  Style harmonize: {'enabled' if style_cfg['enabled'] else 'disabled'}"
              f" (backend={style_cfg['backend']}, strength={style_cfg['strength']})")
        print(f"  Video extend: {'enabled' if extend_cfg['enabled'] else 'disabled'}"
              f" (min_dur={extend_cfg['min_duration_seconds']}s)")
        print(f"  MTV sync: {'enabled' if mtv_cfg['enabled'] else 'disabled'}")
        print(f"  Audio post: {'enabled' if audio_post_cfg['enabled'] else 'disabled'}")

    # --- Step 3.1: Download BGM if needed ---
    print("\n--- Step 3.1: Download BGM ---")
    bgm_path = OUTPUT_DIR / BGM_FILENAME
    if not bgm_path.exists():
        print(f"Downloading BGM: {BGM_URL}")
        import urllib.request
        try:
            urllib.request.urlretrieve(BGM_URL, str(bgm_path))
            print(f"  -> {bgm_path}")
        except Exception as e:
            logger.warning(f"  BGM download failed: {e}")
            bgm_path = None
    else:
        print(f"BGM: [cached] {bgm_path}")

    # --- Step 3.2: Real-ESRGAN 2x Upscale ---
    print("\n--- Step 3.2: Real-ESRGAN 2x Upscale ---")

    upscale_dir = OUTPUT_DIR / "shots_upscaled"
    upscale_dir.mkdir(parents=True, exist_ok=True)

    shots = storyboard["shots"]
    shot_frame_lists = []
    narration_paths = []
    transitions = []
    shot_modes = []

    for i, shot in enumerate(shots):
        shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
        mode = shot.get("mode", "t2v")
        shot_modes.append(mode)

        if not shot_path.exists():
            print(f"  Shot {i}: Not found: {shot_path}")
            continue

        # Read frames with upscale caching
        upscaled_marker = upscale_dir / f"shot_{i:04d}_done.marker"
        upscaled_frames_dir = upscale_dir / f"shot_{i:04d}"

        if upscaled_marker.exists():
            print(f"  Shot {i:2d}: [cached upscaled]")
            frames = _load_frames_from_dir(str(upscaled_frames_dir))
        else:
            print(f"  Shot {i:2d}: Reading + upscaling...")
            frames = _read_video_frames(str(shot_path))

            if frames:
                # Skip upscale for title/end cards (already at target resolution)
                if mode in ("title_card", "end_card"):
                    print(f"           {len(frames)} frames (card, skipping upscale)")
                else:
                    try:
                        from animatediff.postprocess.upscale import VideoUpscaler
                        upscaler = VideoUpscaler(
                            model_name="animevideov3",
                            scale=2,
                            device=device,
                        )
                        frames = upscaler.upscale_frames(frames)
                        print(f"           {len(frames)} frames upscaled to {frames[0].size}")
                    except Exception as e:
                        logger.warning(f"  Shot {i}: Upscale failed ({e}), using original")

                # Cache upscaled frames
                upscaled_frames_dir.mkdir(parents=True, exist_ok=True)
                for j, frame in enumerate(frames):
                    frame.save(str(upscaled_frames_dir / f"{j:04d}.png"))
                upscaled_marker.touch()

        shot_frame_lists.append(frames)

        # Narration path
        narr_path = NARRATION_DIR / f"narration_{i:04d}.wav"
        narration_paths.append(str(narr_path) if narr_path.exists() else "")

        # Transition -- V4 storyboard uses dict format, V3 uses string
        trans_raw = shot.get("transition", "cut")
        if isinstance(trans_raw, dict):
            transitions.append(trans_raw.get("type", "cut"))
        else:
            transitions.append(trans_raw)

    if not shot_frame_lists:
        print("  No shot frames available. Run Phase 2 first.")
        return

    # --- Step 3.2b: V4.3 StyleHarmonizer (before deflicker) ---
    if style_cfg["enabled"] and is_v4 and STYLE_HARMONIZER_AVAILABLE:
        print("\n--- Step 3.2b: Style Harmonization ---")
        try:
            harmonizer = StyleHarmonizer(
                backend=style_cfg["backend"],
                device=device,
            )
            # Only harmonize video shots (not cards)
            video_indices = [idx for idx, m in enumerate(shot_modes) if m not in ("title_card", "end_card")]
            if len(video_indices) > 1:
                video_frame_lists = [shot_frame_lists[idx] for idx in video_indices]
                print(f"  Harmonizing {len(video_frame_lists)} video shots (strength={style_cfg['strength']})...")
                harmonized = harmonizer.harmonize_all_shots(
                    video_frame_lists,
                    strength=style_cfg["strength"],
                )
                for list_idx, shot_idx in enumerate(video_indices):
                    shot_frame_lists[shot_idx] = harmonized[list_idx]
                print(f"  Style harmonization complete.")
            else:
                print(f"  Only {len(video_indices)} video shot(s), skipping style harmonization")
        except Exception as e:
            logger.warning(f"  Style harmonization failed: {e}")
            import traceback
            traceback.print_exc()
    elif style_cfg["enabled"] and not STYLE_HARMONIZER_AVAILABLE:
        print("\n--- Step 3.2b: Style Harmonization (skipped -- module not installed) ---")

    # --- Step 3.3: V4 Deflicker (after upscale, before color grading) ---
    if deflicker_cfg["enabled"] and is_v4:
        print("\n--- Step 3.3: Deflicker (Cross-shot harmonization + per-shot) ---")

        try:
            from animatediff.postprocess.deflicker import VideoDeflicker

            deflicker = VideoDeflicker(
                backend=deflicker_cfg["backend"],
                device=device,
            )
            strength = deflicker_cfg["strength"]

            # Cross-shot harmonization
            if deflicker_cfg["cross_shot_harmonize"]:
                print(f"  Cross-shot harmonization ({len(shot_frame_lists)} shots, overlap=5)...")
                # Only harmonize video shots (not cards)
                video_indices = [idx for idx, m in enumerate(shot_modes) if m not in ("title_card", "end_card")]
                if len(video_indices) > 1:
                    video_frame_lists = [shot_frame_lists[idx] for idx in video_indices]
                    harmonized = deflicker.harmonize_shots(video_frame_lists, overlap=5)
                    for list_idx, shot_idx in enumerate(video_indices):
                        shot_frame_lists[shot_idx] = harmonized[list_idx]
                    print(f"  Harmonized {len(video_indices)} video shots")
                else:
                    print(f"  Only {len(video_indices)} video shot(s), skipping harmonization")

            # Per-shot deflicker
            print(f"  Per-shot deflicker (strength={strength})...")
            for idx in range(len(shot_frame_lists)):
                if shot_modes[idx] in ("title_card", "end_card"):
                    continue
                if len(shot_frame_lists[idx]) < 2:
                    continue
                print(f"    Shot {idx:2d}: {len(shot_frame_lists[idx])} frames...")
                shot_frame_lists[idx] = deflicker.deflicker_shot(
                    shot_frame_lists[idx], strength=strength,
                )
            print("  Deflicker complete.")

        except Exception as e:
            logger.warning(f"  Deflicker failed: {e}. Skipping.")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Step 3.3: Deflicker (skipped -- disabled or V3 storyboard) ---")

    # --- Step 3.3b: V4.3 VideoExtender (extend short shots before RIFE) ---
    if extend_cfg["enabled"] and is_v4 and VIDEO_EXTENDER_AVAILABLE:
        print("\n--- Step 3.3b: VideoExtender (extend short shots) ---")
        min_dur = extend_cfg["min_duration_seconds"]
        min_frames = int(min_dur * native_fps)

        try:
            extender = VideoExtender(
                method=extend_cfg["method"],
                device=device,
            )
            for idx in range(len(shot_frame_lists)):
                if shot_modes[idx] in ("title_card", "end_card"):
                    continue
                if len(shot_frame_lists[idx]) >= min_frames:
                    continue
                if len(shot_frame_lists[idx]) < 2:
                    continue
                old_count = len(shot_frame_lists[idx])
                shot_prompt = shots[idx].get("prompt", "") if idx < len(shots) else ""
                print(f"  Shot {idx:2d}: {old_count} frames < {min_frames} minimum, extending...")
                shot_frame_lists[idx] = extender.extend(
                    frames=shot_frame_lists[idx],
                    target_frames=min_frames,
                    fps=native_fps,
                    prompt=shot_prompt,
                    blend_frames=extend_cfg["blend_frames"],
                )
                print(f"           -> {len(shot_frame_lists[idx])} frames")
            print("  VideoExtender complete.")
        except Exception as e:
            logger.warning(f"  VideoExtender failed: {e}")
            import traceback
            traceback.print_exc()
    elif extend_cfg["enabled"] and not VIDEO_EXTENDER_AVAILABLE:
        print("\n--- Step 3.3b: VideoExtender (skipped -- module not installed) ---")

    # --- Step 3.4: RIFE Frame Interpolation ---
    print("\n--- Step 3.4: RIFE Frame Interpolation ---")

    if rife_mult > 1:
        try:
            from animatediff.postprocess.interpolation import FrameInterpolator
            interpolator = FrameInterpolator(backend="auto", device=device)

            for i, frames in enumerate(shot_frame_lists):
                if rife_mult > 1 and len(frames) > 1:
                    original_count = len(frames)
                    expected_count = (original_count - 1) * rife_mult + 1
                    print(f"  Shot {i:2d}: RIFE {rife_mult}x interpolation ({original_count} -> ~{expected_count} frames)")
                    frames = interpolator.interpolate(frames, multiplier=rife_mult)
                    shot_frame_lists[i] = frames
                    print(f"           -> {len(frames)} frames")
        except Exception as e:
            print(f"  RIFE interpolation failed: {e}")
            print(f"  Skipping interpolation, using native {native_fps}fps")
            output_fps = native_fps
    else:
        print("  RIFE disabled (multiplier=1), skipping interpolation.")

    # --- Step 3.5: Color Grading ---
    print("\n--- Step 3.5: Color Grading ---")

    if color_lut:
        try:
            from animatediff.postprocess.compositor import VideoCompositor
            temp_compositor = VideoCompositor(fps=output_fps)
            for i, frames in enumerate(shot_frame_lists):
                shot_frame_lists[i] = temp_compositor.apply_color_lut(frames, lut_name=color_lut)
                print(f"  Shot {i:2d}: Applied {color_lut} color grading")
        except AttributeError:
            logger.warning(
                f"  VideoCompositor.apply_color_lut() not available yet. "
                f"Skipping color grading. (Will be added in compositor.py update)"
            )
        except Exception as e:
            logger.warning(f"  Color grading failed: {e}. Skipping.")
    else:
        print("  No color LUT specified, skipping color grading.")

    # --- Step 3.5b: V4.2 Audio Post-Processing (normalize + reverb) ---
    if audio_post_cfg["enabled"] and is_v4 and VOICE_PROFILE_AVAILABLE:
        print("\n--- Step 3.5b: Audio Post-Processing ---")
        for idx, narr_path in enumerate(narration_paths):
            if not narr_path or not os.path.exists(narr_path):
                continue
            try:
                # Normalize narration volume
                print(f"  Shot {idx:2d}: Normalizing narration to {audio_post_cfg['normalize_db']}dB...")
                normalize_audio(narr_path, target_db=audio_post_cfg["normalize_db"])

                # Add reverb if enabled
                if audio_post_cfg["reverb"]:
                    print(f"  Shot {idx:2d}: Adding reverb (room={audio_post_cfg['reverb_room_size']}, "
                          f"wet={audio_post_cfg['reverb_wet']})...")
                    add_reverb(
                        narr_path,
                        room_size=audio_post_cfg["reverb_room_size"],
                        damping=audio_post_cfg["reverb_damping"],
                        wet=audio_post_cfg["reverb_wet"],
                    )
            except Exception as e:
                logger.warning(f"  Shot {idx}: Audio post-processing failed: {e}")
        print("  Audio post-processing complete.")
    elif audio_post_cfg["enabled"] and not VOICE_PROFILE_AVAILABLE:
        print("\n--- Step 3.5b: Audio Post-Processing (skipped -- module not installed) ---")

    # --- Step 3.6: Create Title Card + End Card ---
    print("\n--- Step 3.6: Title & End Cards ---")

    # Determine target resolution from upscaled video shots (skip cards)
    target_size = None
    for idx, frames in enumerate(shot_frame_lists):
        if frames and shot_modes[idx] not in ("title_card", "end_card"):
            target_size = frames[0].size
            break
    if target_size is None:
        # Fallback: 2x of generation resolution
        mp = _get_model_params(storyboard)
        target_size = (mp["width"] * 2, mp["height"] * 2)

    print(f"  Target size: {target_size}")

    # Resize card frames in shot_frame_lists to match target_size
    if target_size:
        for idx, frames in enumerate(shot_frame_lists):
            if frames and shot_modes[idx] in ("title_card", "end_card"):
                if frames[0].size != target_size:
                    shot_frame_lists[idx] = [f.resize(target_size, Image.LANCZOS) for f in frames]
                    print(f"  Shot {idx:2d}: Resized card {frames[0].size} -> {target_size}")

    title_frames = _create_title_card(
        storyboard.get("title", "凡人修仙传"),
        storyboard.get("subtitle", "炼气篇 · 预告"),
        size=target_size,
        num_frames=int(output_fps * 3),  # 3 second title card
    )
    end_frames = _create_title_card(
        storyboard.get("title", "凡人修仙传"),
        storyboard.get("end_text", "即将到来"),
        size=target_size,
        num_frames=int(output_fps * 3),  # 3 second end card
    )

    # Assemble: title + shots + end
    all_shot_lists = [title_frames] + shot_frame_lists + [end_frames]
    all_narrations = [""] + narration_paths + [""]
    all_transitions = ["fade"] + transitions[1:] + ["fade"]

    # --- V4: Beat-aware transition durations ---
    beat_times = None
    if beat_cfg["enabled"] and is_v4:
        beat_info_path = OUTPUT_DIR / "beat_info.json"
        if beat_info_path.exists():
            try:
                with open(beat_info_path, "r") as f:
                    beat_data = json.load(f)
                beat_times = beat_data.get("beat_times", [])
                print(f"\n  Beat sync: loaded {len(beat_times)} beat timestamps for transition timing")
            except Exception as e:
                logger.warning(f"  Could not load beat info: {e}")

    # --- V4.3: MTV Sync — frame-level audio alignment + transition snapping ---
    if mtv_cfg["enabled"] and is_v4 and MTV_SYNC_AVAILABLE:
        print("\n--- Step 3.6b: MTV Sync (Audio-Video Alignment) ---")
        try:
            analyzer = AudioStreamAnalyzer(backend=mtv_cfg.get("backend", "auto"))

            # Create sync map from BGM if available
            sync_map = None
            if bgm_path and bgm_path.exists():
                total_frames_for_sync = sum(len(f) for f in all_shot_lists)
                print(f"  Creating sync map from BGM ({total_frames_for_sync} frames @ {output_fps}fps)...")
                va_sync = VideoAudioSync(analyzer)
                sync_map = va_sync.create_sync_map(
                    audio_path=str(bgm_path),
                    fps=output_fps,
                    total_frames=total_frames_for_sync,
                )
                print(f"  Sync map: {sync_map.total_frames} frames, "
                      f"beats={sum(sync_map.frame_is_beat)}, "
                      f"speech={sum(sync_map.frame_is_speech)}")

                # Align transitions to audio beats/pauses
                if mtv_cfg.get("align_transitions", True) and sync_map is not None:
                    # Compute shot boundaries (cumulative frame counts)
                    shot_boundaries = []
                    cumulative = 0
                    for shot_frames in all_shot_lists:
                        shot_boundaries.append(cumulative)
                        cumulative += len(shot_frames)

                    if len(shot_boundaries) > 2:
                        # Only align interior boundaries (skip title and end)
                        interior = shot_boundaries[1:-1]
                        print(f"  Aligning {len(interior)} transition points to audio...")
                        aligned_boundaries = va_sync.align_transitions_to_audio(
                            interior, sync_map
                        )
                        # Report alignment adjustments
                        for orig, aligned in zip(interior, aligned_boundaries):
                            if orig != aligned:
                                print(f"    Boundary {orig} -> {aligned} (delta={aligned - orig})")

            # Separate BGM into streams if mixed (speech/music/SFX)
            if bgm_path and bgm_path.exists():
                try:
                    streams = analyzer.separate(str(bgm_path))
                    if streams and hasattr(streams, 'music') and streams.music:
                        print(f"  BGM separated: music={streams.music}")
                        # Could use streams.music as cleaner BGM track
                except Exception as sep_e:
                    logger.info(f"  BGM separation skipped: {sep_e}")

        except Exception as e:
            logger.warning(f"  MTV sync failed: {e}")
            import traceback
            traceback.print_exc()
    elif mtv_cfg["enabled"] and not MTV_SYNC_AVAILABLE:
        print("\n--- Step 3.6b: MTV Sync (skipped -- module not installed) ---")

    # Handle flash_white transitions (in-place modification)
    _apply_flash_white_transitions(all_shot_lists, all_transitions, beat_times=beat_times, output_fps=output_fps)

    # --- Step 3.7: Compose Final Video ---
    print("\n--- Step 3.7: Composing Final Trailer ---")

    from animatediff.postprocess.compositor import VideoCompositor
    compositor = VideoCompositor(fps=output_fps)

    final_path = str(FINAL_DIR / "fanren_trailer_v4.mp4")

    total_frames = sum(len(f) for f in all_shot_lists)
    print(f"  Shots: {len(all_shot_lists)} (incl. title + end)")
    print(f"  Total frames: {total_frames}")
    print(f"  Narration clips: {sum(1 for n in all_narrations if n)}")
    print(f"  FPS: {output_fps}")
    print(f"  BGM: {bgm_path}")
    print(f"  Duration: ~{total_frames / output_fps:.1f}s")
    print(f"  Output: {final_path}")
    print()

    # Normalize transitions for compositor (flash_white already applied, pass as "cut")
    compositor_transitions = [
        t if t in ("cut", "fade", "dissolve") else "cut"
        for t in all_transitions
    ]

    compositor.compose_with_timed_audio(
        shot_frame_lists=all_shot_lists,
        output_path=final_path,
        transitions=compositor_transitions,
        narration_paths=all_narrations,
        bgm_path=str(bgm_path) if bgm_path and bgm_path.exists() else None,
        bgm_volume=0.15,
    )

    print(f"\n{'=' * 60}")
    print(f"DONE! Final trailer: {final_path}")
    print(f"Duration: {total_frames / output_fps:.1f}s ({total_frames} frames @ {output_fps}fps)")
    if rife_mult > 1:
        print(f"RIFE: {native_fps}fps -> {output_fps}fps ({rife_mult}x)")
    if color_lut:
        print(f"Color grading: {color_lut}")
    if deflicker_cfg["enabled"] and is_v4:
        print(f"Deflicker: strength={deflicker_cfg['strength']}, harmonized={deflicker_cfg['cross_shot_harmonize']}")
    if beat_cfg["enabled"] and is_v4:
        print(f"Beat sync: {beat_cfg['mode']}")
    if style_cfg["enabled"] and is_v4:
        print(f"Style harmonize: backend={style_cfg['backend']}, strength={style_cfg['strength']}")
    if extend_cfg["enabled"] and is_v4:
        print(f"Video extend: method={extend_cfg['method']}, min_dur={extend_cfg['min_duration_seconds']}s")
    if mtv_cfg["enabled"] and is_v4:
        print(f"MTV sync: align_transitions={mtv_cfg.get('align_transitions', True)}")
    if audio_post_cfg["enabled"] and is_v4:
        print(f"Audio post: normalize={audio_post_cfg['normalize_db']}dB, reverb={audio_post_cfg['reverb']}")
    print(f"{'=' * 60}")


def _apply_flash_white_transitions(
    shot_frame_lists: list,
    transitions: list,
    beat_times: list = None,
    output_fps: int = 48,
):
    """Apply flash_white transitions in-place by modifying frame lists.

    V4: If beat_times are available, adjust flash duration to align with nearest
    beat for a more rhythmically satisfying impact.

    A flash_white transition appends white fade-out frames to the end of the
    previous shot and prepends white fade-in frames to the start of the next shot.
    """
    import numpy as np
    from PIL import Image

    base_flash_duration = 6  # frames per side (default)

    # Compute cumulative shot start times for beat alignment
    shot_start_times = []
    current_time = 0.0
    for frames in shot_frame_lists:
        shot_start_times.append(current_time)
        current_time += len(frames) / output_fps

    for idx in range(1, len(transitions)):
        if transitions[idx] != "flash_white":
            continue
        if idx >= len(shot_frame_lists) or idx - 1 < 0:
            continue

        flash_duration = base_flash_duration

        # V4: Adjust flash duration based on beat proximity
        if beat_times and idx < len(shot_start_times):
            cut_time = shot_start_times[idx]
            # Find nearest beat
            nearest_beat = min(beat_times, key=lambda b: abs(b - cut_time), default=None)
            if nearest_beat is not None:
                beat_distance = abs(nearest_beat - cut_time)
                # If the cut is very close to a beat (< 0.1s), use a sharper, shorter flash
                if beat_distance < 0.1:
                    flash_duration = 4
                # If further from a beat, use a longer, more gradual flash
                elif beat_distance > 0.3:
                    flash_duration = 8

        # Fade last frames of previous shot to white
        prev_shot = shot_frame_lists[idx - 1]
        if len(prev_shot) > flash_duration:
            for j in range(flash_duration):
                alpha = (j + 1) / flash_duration  # 0->1 (increasingly white)
                arr = np.array(prev_shot[-(flash_duration - j)], dtype=np.float32)
                white = np.full_like(arr, 255.0)
                blended = (arr * (1 - alpha) + white * alpha).astype(np.uint8)
                prev_shot[-(flash_duration - j)] = Image.fromarray(blended)

        # Fade first frames of next shot from white
        next_shot = shot_frame_lists[idx]
        if len(next_shot) > flash_duration:
            for j in range(flash_duration):
                alpha = (j + 1) / flash_duration  # 0->1 (increasingly normal)
                arr = np.array(next_shot[j], dtype=np.float32)
                white = np.full_like(arr, 255.0)
                blended = (white * (1 - alpha) + arr * alpha).astype(np.uint8)
                next_shot[j] = Image.fromarray(blended)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="凡人修仙传 Trailer V4 — Beat Sync + Camera Presets + VACE + Shot Chaining + Deflicker",
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which phase to run (1=narration+beats, 2=GPU gen, 3=postprocess, all)")

    args = parser.parse_args()

    start = time.time()

    if args.phase in ("1", "all"):
        phase1_generate_narration()
        phase1_compute_durations()

    if args.phase in ("2", "all"):
        phase2_generate_on_gpu()

    if args.phase in ("3", "all"):
        phase3_postprocess()

    elapsed = time.time() - start
    print(f"\nTotal pipeline time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
