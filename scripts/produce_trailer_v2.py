#!/usr/bin/env python3
"""
凡人修仙传 Trailer V2 — Full Production Pipeline

Three-phase production:
  Phase 1 (local Mac MPS): Generate narration audio, compute frame counts
  Phase 2 (cloud GPU):     Generate portraits + I2V shots + lip sync
  Phase 3 (local Mac MPS): Upscale + compose final trailer

Usage:
  # Run Phase 1 only (local, ~2h)
  python scripts/produce_trailer_v2.py --phase 1

  # Run Phase 2 only (needs CUDA GPU)
  python scripts/produce_trailer_v2.py --phase 2

  # Run Phase 3 only (local, ~30min)
  python scripts/produce_trailer_v2.py --phase 3

  # Run all phases
  python scripts/produce_trailer_v2.py --phase all
"""

import argparse
import json
import logging
import os
import struct
import time
import wave
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORYBOARD_PATH = PROJECT_ROOT / "examples" / "fanren_trailer_v2.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "fanren-trailer-v2"

NARRATION_DIR = OUTPUT_DIR / "narration"
PORTRAIT_DIR = OUTPUT_DIR / "portraits"
SHOTS_DIR = OUTPUT_DIR / "shots"
FINAL_DIR = OUTPUT_DIR / "final"

FPS = 24  # TI2V-5B native fps
MAX_FRAMES = 121  # Wan 2.2 max
TRANSITION_PAD = 0.3  # seconds of padding after narration

# BGM (Creative Commons)
BGM_URL = "https://peritune.com/wp-content/uploads/2024/03/PeriTune-Wuxia3.mp3"
BGM_FILENAME = "PeriTune-Wuxia3.mp3"


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
# Phase 1: Local Preparation (Mac MPS)
# ============================================================================

def phase1_generate_narration():
    """Generate narration audio for all 12 shots using F5-TTS-MLX."""
    print("\n" + "=" * 60)
    print("PHASE 1: Generating Narration Audio (F5-TTS-MLX)")
    print("=" * 60 + "\n")

    NARRATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load storyboard
    with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    shots = storyboard["shots"]
    voice_profiles = storyboard.get("voice_profiles", {})

    print(f"Storyboard: {storyboard['title']}")
    print(f"Shots: {len(shots)}")
    print(f"Voice profiles: {list(voice_profiles.keys())}")
    print()

    # Import F5-TTS-MLX
    from f5_tts_mlx.generate import generate

    generated = []
    for i, shot in enumerate(shots):
        narration = shot.get("narration", "")
        voice_id = shot.get("voice_id", "narrator")

        if not narration.strip():
            print(f"  Shot {i:2d}: (no narration, skipping)")
            generated.append("")
            continue

        out_path = str(NARRATION_DIR / f"narration_{i:04d}.wav")

        # Check if already generated
        if os.path.exists(out_path):
            duration = _get_wav_duration(out_path)
            print(f"  Shot {i:2d}: [cached] {duration:.2f}s — {voice_id}: {narration[:40]}...")
            generated.append(out_path)
            continue

        print(f"  Shot {i:2d}: Generating ({voice_id})... {narration[:40]}...")
        t0 = time.time()

        try:
            # F5-TTS-MLX: generation_text is the parameter name
            # Without ref_audio, uses default voice model
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

            generate(**kwargs)

            if os.path.exists(out_path):
                duration = _get_wav_duration(out_path)
                elapsed = time.time() - t0
                print(f"           Done: {duration:.2f}s audio ({elapsed:.1f}s generation)")
                generated.append(out_path)
            else:
                logger.error(f"  Shot {i}: TTS produced no output file")
                generated.append("")
        except Exception as e:
            logger.error(f"  Shot {i}: TTS failed: {e}")
            import traceback
            traceback.print_exc()
            generated.append("")

    print(f"\nGenerated {sum(1 for g in generated if g)} / {len(shots)} narration clips")
    return generated


def phase1_compute_durations():
    """Compute frame counts from narration audio and update storyboard."""
    print("\n" + "-" * 60)
    print("PHASE 1.2: Computing Audio-First Durations")
    print("-" * 60 + "\n")

    # Load storyboard
    with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    shots = storyboard["shots"]
    timing = []

    for i, shot in enumerate(shots):
        wav_path = NARRATION_DIR / f"narration_{i:04d}.wav"
        narration = shot.get("narration", "")

        if narration.strip() and wav_path.exists():
            audio_dur = _get_wav_duration(str(wav_path))
            total_dur = audio_dur + TRANSITION_PAD
            num_frames = _duration_to_frames(total_dur, FPS)
            num_frames = min(num_frames, MAX_FRAMES)
            actual_dur = num_frames / FPS

            shot["num_frames"] = num_frames
            shot["duration_seconds"] = round(actual_dur, 3)

            overflow = ""
            if audio_dur > MAX_FRAMES / FPS:
                overflow = f" ⚠️ narration exceeds max, last frames will hold"

            print(f"  Shot {i:2d}: audio={audio_dur:.2f}s → {num_frames}f ({actual_dur:.2f}s){overflow}")
        else:
            # No narration: default 5s
            num_frames = _duration_to_frames(5.0, FPS)
            shot["num_frames"] = num_frames
            shot["duration_seconds"] = 5.0
            print(f"  Shot {i:2d}: (no narration) → default {num_frames}f (5.00s)")

        timing.append({
            "shot_id": i,
            "scene": shot.get("scene", ""),
            "num_frames": shot["num_frames"],
            "duration_seconds": shot["duration_seconds"],
        })

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

    total_dur = sum(t["duration_seconds"] for t in timing)
    total_frames = sum(t["num_frames"] for t in timing)
    print(f"\nTotal: {total_frames} frames, {total_dur:.1f}s ({total_dur/60:.1f}min)")
    print(f"Timing saved: {timing_path}")

    return timing


# ============================================================================
# Phase 2: Cloud Generation (Vast.ai / CUDA GPU)
# ============================================================================

def phase2_generate_on_gpu():
    """Generate character portraits + I2V video shots on GPU."""
    print("\n" + "=" * 60)
    print("PHASE 2: GPU Generation (Portraits + I2V Shots + Lip Sync)")
    print("=" * 60 + "\n")

    import torch
    from animatediff.core.vram_manager import get_vram_manager
    from animatediff.backends import get_backend
    from animatediff.core.story_engine import StoryEngine
    from animatediff.core.character_manager import CharacterManager
    from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")

    vm = get_vram_manager()
    print(vm.summary())

    rec = vm.recommend("wan22")
    torch_dtype = rec.torch_dtype
    if device == "mps":
        torch_dtype = torch.float32

    # --- Step 2.1: Load T2V pipeline (for portrait generation) ---
    print("\nLoading Wan 2.2 TI2V-5B pipeline (T2V mode for portraits)...")
    BackendClass = get_backend("wan22")
    offload = "model_cpu" if device == "cuda" else "none"
    pipeline = BackendClass.load(
        model_variant="5B",
        mode="t2v",
        torch_dtype=torch_dtype,
        device=device,
        quantization=rec.quantization,
        offload_strategy=offload,
        enable_vae_slicing=True,
        enable_vae_tiling=True,
    )

    # --- Step 2.2: Generate character portraits ---
    print("\n--- Step 2.2: Generating Character Portraits ---")
    PORTRAIT_DIR.mkdir(parents=True, exist_ok=True)

    with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    portrait_results = {}
    for name, info in storyboard.get("characters", {}).items():
        portrait_path = PORTRAIT_DIR / f"{name}.png"
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
            output = pipeline.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed, ugly, watermark, multiple people",
                width=832,
                height=480,
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
            best_frame.save(str(portrait_path))
            portrait_results[name] = str(portrait_path)
            print(f"    → Best: {portrait_path} (seed={best_seed}, score={best_score:.2f})")

    # --- Step 2.3: Reload as I2V pipeline for shot generation ---
    print("\n--- Step 2.3: Reloading pipeline in I2V mode ---")
    del pipeline
    import gc; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    pipeline = BackendClass.load(
        model_variant="5B",
        mode="i2v",
        torch_dtype=torch_dtype,
        device=device,
        quantization=rec.quantization,
        offload_strategy=offload,
        enable_vae_slicing=True,
        enable_vae_tiling=True,
    )
    print("  I2V pipeline loaded.")

    # --- Step 2.4: Generate I2V shots ---
    print("\n--- Step 2.4: Generating I2V Video Shots ---")
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load computed storyboard (with durations from Phase 1)
    computed_sb = OUTPUT_DIR / "storyboard_computed.json"
    if computed_sb.exists():
        with open(computed_sb, "r", encoding="utf-8") as f:
            storyboard = json.load(f)
        print(f"  Using computed storyboard: {computed_sb}")
    else:
        print(f"  ⚠️ No computed storyboard found, using original with default durations")

    from PIL import Image

    for i, shot in enumerate(storyboard["shots"]):
        shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
        if shot_path.exists():
            print(f"  Shot {i:2d}: [cached] {shot_path}")
            continue

        # Resolve character reference image (required for I2V pipeline)
        ref_image = None
        characters = shot.get("characters", [])
        if characters:
            char_name = characters[0]
            portrait_path_str = portrait_results.get(char_name)
            if portrait_path_str and os.path.exists(portrait_path_str):
                ref_image = Image.open(portrait_path_str).convert("RGB")
                print(f"  Shot {i:2d}: I2V with ref={char_name}")
            else:
                print(f"  Shot {i:2d}: ⚠️ No portrait for {char_name}, using black frame")
                ref_image = Image.new("RGB", (832, 480), (0, 0, 0))
        else:
            print(f"  Shot {i:2d}: No characters, using black frame")
            ref_image = Image.new("RGB", (832, 480), (0, 0, 0))

        num_frames = shot.get("num_frames", 0)
        if num_frames == 0:
            num_frames = _duration_to_frames(shot.get("duration_seconds", 5.0), FPS)
            num_frames = min(num_frames, MAX_FRAMES)

        prompt = shot["prompt"]
        scene = shot.get("scene", "")
        print(f"           {scene} ({num_frames}f)")

        t0 = time.time()
        output = pipeline.generate(
            prompt=prompt,
            negative_prompt=storyboard.get("negative_prompt", "blurry, low quality"),
            width=832,
            height=480,
            num_frames=num_frames,
            num_inference_steps=50,
            guidance_scale=5.0,
            seed=shot.get("seed", -1),
            image=ref_image,
        )
        elapsed = time.time() - t0

        pipeline.save(output, str(shot_path), fps=FPS)
        print(f"           Done: {elapsed:.1f}s → {shot_path}")

    # --- Step 2.5: Lip Sync ---
    print("\n--- Step 2.5: Lip Sync (Shot 2 & 9) ---")

    lip_sync_shots = [
        (i, shot) for i, shot in enumerate(storyboard["shots"])
        if shot.get("lip_sync", False)
    ]

    if not lip_sync_shots:
        print("  No lip sync shots found.")
    else:
        try:
            from animatediff.postprocess.lipsync import LipSyncProcessor
            processor = LipSyncProcessor(backend="auto", device=device)

            if processor.available:
                for i, shot in lip_sync_shots:
                    shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
                    audio_path = NARRATION_DIR / f"narration_{i:04d}.wav"
                    synced_path = SHOTS_DIR / f"shot_{i:04d}_lipsync.mp4"

                    if synced_path.exists():
                        print(f"  Shot {i}: [cached] {synced_path}")
                        continue

                    if not audio_path.exists():
                        print(f"  Shot {i}: ⚠️ No narration audio, skipping")
                        continue

                    print(f"  Shot {i}: Applying lip sync...")
                    # Read frames from video, apply lip sync, save back
                    from animatediff.postprocess.lipsync import LipSyncProcessor
                    frames = _read_video_frames(str(shot_path))
                    if frames:
                        synced = processor.apply(frames, str(audio_path), fps=FPS)
                        from diffusers.utils import export_to_video
                        export_to_video(synced, str(synced_path), fps=FPS)
                        print(f"  Shot {i}: → {synced_path}")
            else:
                print("  ⚠️ No lip sync backend available (MuseTalk/JoyVASA not installed)")
                print("  Skipping lip sync — shots will use original video")
        except Exception as e:
            logger.warning(f"  Lip sync setup failed: {e}")
            print("  Skipping lip sync — shots will use original video")

    print("\nPhase 2 complete!")


# ============================================================================
# Phase 3: Local Post-Processing (Mac MPS)
# ============================================================================

def phase3_postprocess():
    """Upscale shots + compose final trailer with timed audio."""
    print("\n" + "=" * 60)
    print("PHASE 3: Post-Processing + Final Composition")
    print("=" * 60 + "\n")

    import torch
    from PIL import Image

    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 3.1: Download BGM if needed ---
    bgm_path = OUTPUT_DIR / BGM_FILENAME
    if not bgm_path.exists():
        print(f"Downloading BGM: {BGM_URL}")
        import urllib.request
        try:
            urllib.request.urlretrieve(BGM_URL, str(bgm_path))
            print(f"  → {bgm_path}")
        except Exception as e:
            logger.warning(f"  BGM download failed: {e}")
            bgm_path = None
    else:
        print(f"BGM: [cached] {bgm_path}")

    # --- Step 3.2: Real-ESRGAN 2x Upscale ---
    print("\n--- Step 3.2: Real-ESRGAN 2x Upscale ---")

    upscale_dir = OUTPUT_DIR / "shots_upscaled"
    upscale_dir.mkdir(parents=True, exist_ok=True)

    # Load storyboard for shot count
    computed_sb = OUTPUT_DIR / "storyboard_computed.json"
    if computed_sb.exists():
        with open(computed_sb, "r", encoding="utf-8") as f:
            storyboard = json.load(f)
    else:
        with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
            storyboard = json.load(f)

    shots = storyboard["shots"]
    shot_frame_lists = []
    narration_paths = []
    lip_sync_flags = []
    transitions = []

    for i, shot in enumerate(shots):
        # Prefer lip-synced version if available
        lipsync_path = SHOTS_DIR / f"shot_{i:04d}_lipsync.mp4"
        shot_path = lipsync_path if lipsync_path.exists() else SHOTS_DIR / f"shot_{i:04d}.mp4"

        if not shot_path.exists():
            print(f"  Shot {i}: ⚠️ Not found: {shot_path}")
            continue

        # Read frames
        upscaled_marker = upscale_dir / f"shot_{i:04d}_done.marker"
        upscaled_frames_dir = upscale_dir / f"shot_{i:04d}"

        if upscaled_marker.exists():
            print(f"  Shot {i}: [cached upscaled]")
            frames = _load_frames_from_dir(str(upscaled_frames_dir))
        else:
            print(f"  Shot {i}: Reading + upscaling...")
            frames = _read_video_frames(str(shot_path))

            if frames:
                try:
                    from animatediff.postprocess.upscale import VideoUpscaler
                    upscaler = VideoUpscaler(
                        model_name="animevideov3",
                        scale=2,
                        device=device,
                    )
                    frames = upscaler.upscale_frames(frames)
                    # Cache upscaled frames
                    upscaled_frames_dir.mkdir(parents=True, exist_ok=True)
                    for j, frame in enumerate(frames):
                        frame.save(str(upscaled_frames_dir / f"{j:04d}.png"))
                    upscaled_marker.touch()
                    print(f"           {len(frames)} frames upscaled to {frames[0].size}")
                except Exception as e:
                    logger.warning(f"  Shot {i}: Upscale failed ({e}), using original")

        shot_frame_lists.append(frames)

        # Narration
        narr_path = NARRATION_DIR / f"narration_{i:04d}.wav"
        narration_paths.append(str(narr_path) if narr_path.exists() else "")

        # Lip sync flag
        lip_sync_flags.append(shot.get("lip_sync", False))

        # Transition
        transitions.append(shot.get("transition", "cut"))

    if not shot_frame_lists:
        print("  ⚠️ No shot frames available. Run Phase 2 first.")
        return

    # --- Step 3.3: Create Title Card + End Card ---
    print("\n--- Creating Title & End Cards ---")

    sample_size = shot_frame_lists[0][0].size if shot_frame_lists else (1664, 960)
    title_frames = _create_title_card(
        "凡人修仙传",
        "炼气篇 · 预告",
        size=sample_size,
        num_frames=int(FPS * 3),  # 3 second title card
    )
    end_frames = _create_title_card(
        "凡人修仙传",
        "即将到来",
        size=sample_size,
        num_frames=int(FPS * 3),  # 3 second end card
    )

    # Assemble: title + shots + end
    all_shot_lists = [title_frames] + shot_frame_lists + [end_frames]
    all_narrations = [""] + narration_paths + [""]
    all_transitions = ["fade"] + transitions[1:] + ["fade"]

    # --- Step 3.4: Compose Final Video ---
    print("\n--- Step 3.3: Composing Final Trailer ---")

    from animatediff.postprocess.compositor import VideoCompositor
    compositor = VideoCompositor(fps=FPS)

    final_path = str(FINAL_DIR / "fanren_trailer_v2.mp4")

    print(f"  Shots: {len(all_shot_lists)} (incl. title + end)")
    print(f"  Total frames: {sum(len(f) for f in all_shot_lists)}")
    print(f"  Narration clips: {sum(1 for n in all_narrations if n)}")
    print(f"  BGM: {bgm_path}")
    print(f"  Output: {final_path}")
    print()

    compositor.compose_with_timed_audio(
        shot_frame_lists=all_shot_lists,
        output_path=final_path,
        transitions=all_transitions,
        narration_paths=all_narrations,
        bgm_path=str(bgm_path) if bgm_path and bgm_path.exists() else None,
        bgm_volume=0.15,
    )

    # Also create a version without upscale for quick preview
    print(f"\n{'=' * 60}")
    print(f"DONE! Final trailer: {final_path}")
    total_frames = sum(len(f) for f in all_shot_lists)
    print(f"Duration: {total_frames / FPS:.1f}s ({total_frames} frames @ {FPS}fps)")
    print(f"{'=' * 60}")


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
    import numpy as np
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


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="凡人修仙传 Trailer V2 — Full Production Pipeline",
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which phase to run (1=narration, 2=GPU gen, 3=postprocess, all)")

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
