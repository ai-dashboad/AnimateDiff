#!/usr/bin/env python3
"""
凡人修仙传 Trailer V3 — A14B + Mixed T2V/I2V + RIFE + Color Grading

Upgrades from V2:
- Wan 2.2 A14B (14B active MoE) on CUDA, 5B fallback on MPS
- Mixed T2V + I2V per-shot (T2V for scenes/action, I2V for character close-ups)
- RIFE 2x frame interpolation (16fps → 32fps)
- Xianxia blue-gold color grading LUT
- Flash-white transitions
- 1280x720 native (→ 2560x1440 after upscale)

Three-phase production:
  Phase 1 (local Mac MPS): Generate narration audio, compute frame counts
  Phase 2 (cloud GPU):     Generate portraits + T2V/I2V shots
  Phase 3 (local Mac MPS): Upscale + RIFE + color grading + compose

Usage:
  # Run Phase 1 only (local, ~2h)
  python scripts/produce_trailer_v3.py --phase 1

  # Run Phase 2 only (needs CUDA GPU)
  python scripts/produce_trailer_v3.py --phase 2

  # Run Phase 3 only (local, ~30min)
  python scripts/produce_trailer_v3.py --phase 3

  # Run all phases
  python scripts/produce_trailer_v3.py --phase all
"""

import argparse
import gc
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
STORYBOARD_PATH = PROJECT_ROOT / "examples" / "fanren_trailer_v3.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "fanren-trailer-v3"

NARRATION_DIR = OUTPUT_DIR / "narration"
PORTRAIT_DIR = OUTPUT_DIR / "portraits"
SHOTS_DIR = OUTPUT_DIR / "shots"
FINAL_DIR = OUTPUT_DIR / "final"

TRANSITION_PAD = 0.3  # seconds of padding after narration

# BGM (Creative Commons)
BGM_URL = "https://peritune.com/wp-content/uploads/2024/03/PeriTune-Wuxia3.mp3"
BGM_FILENAME = "PeriTune-Wuxia3.mp3"


def _load_storyboard() -> dict:
    """Load and return the storyboard JSON."""
    with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
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
    """Extract model_params from storyboard with A14B defaults."""
    mp = storyboard.get("model_params", {})
    return {
        "width": mp.get("width", 1280),
        "height": mp.get("height", 720),
        "fps": mp.get("fps", 16),
        "max_frames": mp.get("max_frames", 81),
        "guidance_scale": mp.get("guidance_scale", 4.0),
        "guidance_scale_2": mp.get("guidance_scale_2", 3.0),
        "num_inference_steps": mp.get("num_inference_steps", 40),
    }


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
    print(f"Shots: {len(shots)}")
    print(f"Voice profiles: {list(voice_profiles.keys())}")
    print(f"Model params: fps={model_params['fps']}, max_frames={model_params['max_frames']}")
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
            print(f"  Shot {i:2d}: [cached] {duration:.2f}s -- {voice_id}: {narration[:40]}...")
            generated.append(out_path)
            continue

        print(f"  Shot {i:2d}: Generating ({voice_id})... {narration[:40]}...")
        t0 = time.time()

        try:
            # F5-TTS-MLX: generation_text is the parameter name
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
        narration = shot.get("narration", "")
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
    """Generate character portraits + T2V/I2V video shots on GPU."""
    print("\n" + "=" * 60)
    print("PHASE 2: GPU Generation (Portraits + Mixed T2V/I2V Shots)")
    print("=" * 60 + "\n")

    import torch
    from animatediff.core.vram_manager import get_vram_manager
    from animatediff.backends import get_backend

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

    # Load storyboard (computed if available, else original)
    storyboard = _load_computed_storyboard()
    model_params = _get_model_params(storyboard)

    # Determine model variant and generation params based on device
    if device == "cuda":
        model_variant = "A14B"
        gen_width = model_params["width"]
        gen_height = model_params["height"]
        gen_fps = model_params["fps"]
        max_frames = model_params["max_frames"]
        guidance_scale = model_params["guidance_scale"]
        guidance_scale_2 = model_params["guidance_scale_2"]
        num_inference_steps = model_params["num_inference_steps"]
        offload = "model_cpu"
        portrait_width = 1280
        portrait_height = 720
    else:
        # MPS fallback to TI2V-5B
        model_variant = "5B"
        gen_width, gen_height = 832, 480
        gen_fps = 24
        max_frames = 121
        guidance_scale = 5.0
        guidance_scale_2 = None  # no dual transformer on 5B
        num_inference_steps = 50
        offload = "none"
        portrait_width = 832
        portrait_height = 480

    print(f"\nModel: Wan 2.2 {model_variant}")
    print(f"Generation: {gen_width}x{gen_height}, {gen_fps}fps, max {max_frames} frames")
    print(f"Guidance: scale={guidance_scale}, scale_2={guidance_scale_2}")
    print(f"Steps: {num_inference_steps}")
    print(f"Offload: {offload}")

    BackendClass = get_backend("wan22")

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
        quantization=rec.quantization,
        offload_strategy=offload,
        enable_vae_slicing=True,
        enable_vae_tiling=True,
    )
    print("  T2V pipeline loaded.")

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
            best_frame.save(str(portrait_path))
            portrait_results[name] = str(portrait_path)
            print(f"    -> Best: {portrait_path} (seed={best_seed}, score={best_score:.2f})")

    # ---------------------------------------------------------------
    # Step 2.2: Generate T2V shots (no image input)
    # ---------------------------------------------------------------
    print("\n--- Step 2.2: Generating T2V Video Shots ---")
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    shots = storyboard["shots"]
    t2v_shots = [(i, s) for i, s in enumerate(shots) if s.get("mode") == "t2v"]
    print(f"  T2V shots to generate: {len(t2v_shots)}")

    for i, shot in t2v_shots:
        shot_path = SHOTS_DIR / f"shot_{i:04d}.mp4"
        if shot_path.exists():
            print(f"  Shot {i:2d}: [cached] {shot_path}")
            continue

        num_frames = shot.get("num_frames", 0)
        if num_frames == 0:
            num_frames = _duration_to_frames(shot.get("duration_seconds", 5.0), gen_fps)
            num_frames = min(num_frames, max_frames)

        prompt = shot["prompt"]
        scene = shot.get("scene", "")
        print(f"  Shot {i:2d}: T2V — {scene} ({num_frames}f)")

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
        # A14B dual-transformer guidance
        if guidance_scale_2 is not None:
            gen_kwargs["guidance_scale_2"] = guidance_scale_2

        output = pipeline_t2v.generate(**gen_kwargs)
        elapsed = time.time() - t0

        pipeline_t2v.save(output, str(shot_path), fps=gen_fps)
        print(f"           Done: {elapsed:.1f}s -> {shot_path}")

    # ---------------------------------------------------------------
    # Step 2.3: Reload as I2V pipeline, generate I2V shots
    # ---------------------------------------------------------------
    i2v_shots = [(i, s) for i, s in enumerate(shots) if s.get("mode") == "i2v"]
    print(f"\n--- Step 2.3: Generating I2V Video Shots ({len(i2v_shots)} shots) ---")

    if i2v_shots:
        # Free T2V pipeline memory
        del pipeline_t2v
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

        print(f"\nLoading Wan 2.2 {model_variant} pipeline (I2V mode)...")
        pipeline_i2v = BackendClass.load(
            model_variant=model_variant,
            mode="i2v",
            torch_dtype=torch_dtype,
            device=device,
            quantization=rec.quantization,
            offload_strategy=offload,
            enable_vae_slicing=True,
            enable_vae_tiling=True,
        )
        print("  I2V pipeline loaded.")

        from PIL import Image

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

            prompt = shot["prompt"]
            scene = shot.get("scene", "")
            print(f"           {scene} ({num_frames}f)")

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
            # A14B dual-transformer guidance
            if guidance_scale_2 is not None:
                gen_kwargs["guidance_scale_2"] = guidance_scale_2

            output = pipeline_i2v.generate(**gen_kwargs)
            elapsed = time.time() - t0

            pipeline_i2v.save(output, str(shot_path), fps=gen_fps)
            print(f"           Done: {elapsed:.1f}s -> {shot_path}")

        del pipeline_i2v
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
    else:
        # No I2V shots — free T2V pipeline
        del pipeline_t2v
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        print("  No I2V shots to generate.")

    # ---------------------------------------------------------------
    # Step 2.4: Generate title_card / end_card shots in code
    # ---------------------------------------------------------------
    card_shots = [(i, s) for i, s in enumerate(shots)
                  if s.get("mode") in ("title_card", "end_card")]
    if card_shots:
        print(f"\n--- Step 2.4: Generating Title/End Cards ({len(card_shots)}) ---")
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

            # Save as MP4 via ffmpeg
            _save_frames_as_mp4(frames, str(shot_path), gen_fps)
            print(f"  Shot {i:2d}: {shot.get('mode')} -> {shot_path} ({num_frames}f)")

    print("\nPhase 2 complete!")


def _save_frames_as_mp4(frames: list, output_path: str, fps: int):
    """Save a list of PIL Image frames as an MP4 using ffmpeg."""
    import subprocess
    import tempfile
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
# Phase 3: Local Post-Processing (Mac MPS)
# ============================================================================

def phase3_postprocess():
    """Upscale + RIFE interpolation + color grading + compose final trailer."""
    print("\n" + "=" * 60)
    print("PHASE 3: Post-Processing (Upscale + RIFE + Color Grade + Compose)")
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

        # Transition (supports flash_white in addition to cut, fade, dissolve)
        transitions.append(shot.get("transition", "cut"))

    if not shot_frame_lists:
        print("  No shot frames available. Run Phase 2 first.")
        return

    # --- Step 3.3: RIFE Frame Interpolation ---
    print("\n--- Step 3.3: RIFE Frame Interpolation ---")

    if rife_mult > 1:
        from animatediff.postprocess.interpolation import FrameInterpolator
        interpolator = FrameInterpolator(backend="auto", device=device)

        for i, frames in enumerate(shot_frame_lists):
            if rife_mult > 1 and len(frames) > 1:
                original_count = len(frames)
                # Expected output: (N-1) * mult + 1 frames
                expected_count = (original_count - 1) * rife_mult + 1
                print(f"  Shot {i:2d}: RIFE {rife_mult}x interpolation ({original_count} -> ~{expected_count} frames)")
                frames = interpolator.interpolate(frames, multiplier=rife_mult)
                shot_frame_lists[i] = frames
                print(f"           -> {len(frames)} frames")
    else:
        print("  RIFE disabled (multiplier=1), skipping interpolation.")

    # --- Step 3.4: Color Grading ---
    print("\n--- Step 3.4: Color Grading ---")

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

    # --- Step 3.5: Create Title Card + End Card ---
    print("\n--- Step 3.5: Title & End Cards ---")

    # Determine target resolution from upscaled shots
    target_size = None
    for frames in shot_frame_lists:
        if frames:
            target_size = frames[0].size
            break
    if target_size is None:
        # Fallback: 2x of generation resolution
        mp = _get_model_params(storyboard)
        target_size = (mp["width"] * 2, mp["height"] * 2)

    print(f"  Target size: {target_size}")

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

    # Handle flash_white transitions
    _apply_flash_white_transitions(all_shot_lists, all_transitions)

    # --- Step 3.6: Compose Final Video ---
    print("\n--- Step 3.6: Composing Final Trailer ---")

    from animatediff.postprocess.compositor import VideoCompositor
    compositor = VideoCompositor(fps=output_fps)

    final_path = str(FINAL_DIR / "fanren_trailer_v3.mp4")

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
    print(f"{'=' * 60}")


def _apply_flash_white_transitions(
    shot_frame_lists: list,
    transitions: list,
):
    """Apply flash_white transitions in-place by modifying frame lists.

    A flash_white transition appends white fade-out frames to the end of the
    previous shot and prepends white fade-in frames to the start of the next shot.
    This creates a brief white flash between shots.
    """
    import numpy as np
    from PIL import Image

    flash_duration = 6  # frames per side (6 fade-out + 6 fade-in = 12 total)

    for idx in range(1, len(transitions)):
        if transitions[idx] != "flash_white":
            continue
        if idx >= len(shot_frame_lists) or idx - 1 < 0:
            continue

        prev_frames = shot_frame_lists[idx]      # shot after the transition marker
        # transitions[idx] is between shot_frame_lists[idx-1] and shot_frame_lists[idx]
        # (the compositor indexes transitions as: transitions[i] = transition BEFORE shot i)
        # We apply: append white fade to end of shot (idx-1), prepend white fade to start of shot (idx)

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
        description="凡人修仙传 Trailer V3 — A14B + Mixed T2V/I2V + RIFE + Color Grading",
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
