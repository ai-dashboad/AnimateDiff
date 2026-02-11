"""
AnimateDiff V4 StoryEngine CLI — multi-shot story-level video generation.

End-to-end pipeline:
  Script → Storyboard → Shot Generation → Post-processing → Final Video

Usage:
  # From a natural language script
  python scripts/story.py --script "韩立站在山巅，风吹衣袂。远处传来一声龙吟。"

  # From a structured storyboard JSON/YAML
  python scripts/story.py --storyboard story.json

  # With character references and LoRAs
  python scripts/story.py --storyboard story.json --characters chars.json

  # With post-processing (interpolation + upscale + audio)
  python scripts/story.py --storyboard story.json --interpolate 2x --upscale 2x --tts

  # Single shot quick mode (no storyboard needed)
  python scripts/story.py --prompt "a girl dancing in anime style" --backend wan22
"""

import argparse
import datetime
import logging
import os
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_story(args):
    """Main story generation pipeline."""
    from animatediff.core.vram_manager import get_vram_manager
    from animatediff.core.story_engine import StoryEngine, StoryBoard, ShotSpec, save_storyboard
    from animatediff.core.character_manager import CharacterManager
    from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig
    from animatediff.backends import get_backend

    t_start = time.time()

    # ---- GPU detection ----
    vm = get_vram_manager()
    print(f"\n{'='*60}")
    print(vm.summary())
    print(f"{'='*60}\n")

    # ---- Resolve backend ----
    backend_name = args.backend
    if backend_name == "auto":
        backend_name = vm.best_backend()
    print(f"Backend: {backend_name}")

    rec = vm.recommend(backend_name)
    print(f"Recommended: variant={rec.model_variant}, quant={rec.quantization}, "
          f"max={rec.max_width}x{rec.max_height}x{rec.max_frames}")

    # ---- Build storyboard ----
    engine = StoryEngine(
        style=args.style or "",
        negative_prompt=args.negative_prompt or "blurry, low quality, distorted",
    )

    if args.storyboard:
        ext = os.path.splitext(args.storyboard)[1].lower()
        if ext == ".json":
            board = engine.from_json(args.storyboard)
        elif ext in (".yaml", ".yml"):
            board = engine.from_yaml(args.storyboard)
        else:
            raise ValueError(f"Unsupported storyboard format: {ext}")
        print(f"Loaded storyboard: {board.title} ({board.num_shots} shots, ~{board.total_duration:.0f}s)")
    elif args.script:
        # Parse natural language script
        if args.use_llm:
            board = engine.from_script_llm(args.script, model_name=args.llm_model)
        else:
            board = engine.from_script(args.script)
        print(f"Parsed script into {board.num_shots} shots (~{board.total_duration:.0f}s)")
    elif args.prompt:
        # Single shot mode
        board = StoryBoard(
            title="single_shot",
            style=args.style or "",
            shots=[ShotSpec(
                shot_id=0,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt or "",
                duration_seconds=args.duration or 5.0,
                seed=args.seed,
            )],
        )
    else:
        raise ValueError("Provide --storyboard, --script, or --prompt")

    # ---- Load characters ----
    char_mgr = CharacterManager()
    if args.characters:
        char_mgr.load(args.characters)
    if board.characters:
        char_mgr_board = CharacterManager.from_storyboard(board)
        for name, profile in char_mgr_board.characters.items():
            if name not in char_mgr.characters:
                char_mgr.characters[name] = profile

    # ---- Prepare output ----
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    outdir = args.output_dir or f"output/story-{time_str}"
    os.makedirs(outdir, exist_ok=True)

    # Save storyboard for reference
    save_storyboard(board, os.path.join(outdir, "storyboard.json"))

    # ---- Load video generation backend ----
    torch_dtype = rec.torch_dtype
    if args.device == "mps":
        torch_dtype = torch.float32

    BackendClass = get_backend(backend_name)
    load_kwargs = dict(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        device=args.device,
        quantization=args.quantization or rec.quantization,
        offload_strategy=args.offload or rec.offload_strategy,
        enable_vae_slicing=True,
        enable_vae_tiling=rec.enable_vae_tiling,
    )

    # Backend-specific kwargs
    if backend_name in ("wan22", "wan"):
        load_kwargs["model_variant"] = args.model_variant or rec.model_variant

    # Collect all LoRA paths from characters + explicit args
    lora_paths = list(args.lora_paths) if args.lora_paths else []
    lora_scales = list(args.lora_scales) if args.lora_scales else []
    if backend_name == "wan22" and lora_paths:
        load_kwargs["lora_paths"] = lora_paths
        load_kwargs["lora_scales"] = lora_scales

    print(f"\nLoading {backend_name} backend...")
    pipeline = BackendClass.load(**load_kwargs)

    # ---- Generate shots ----
    scheduler_config = SchedulerConfig(
        fps=args.fps,
        default_width=args.W or rec.max_width,
        default_height=args.H or rec.max_height,
        max_frames_per_shot=rec.max_frames,
        output_dir=os.path.join(outdir, "shots"),
        output_format="mp4",
        quality=args.quality,
        save_intermediate=True,
    )

    scheduler = ShotScheduler(pipeline, char_mgr, scheduler_config)

    def progress_cb(current, total, msg):
        print(f"  [{current}/{total}] {msg}")

    print(f"\nGenerating {board.num_shots} shots...")
    results = scheduler.generate_all(board, progress_callback=progress_cb)

    # Report generation stats
    success = sum(1 for r in results if r.error is None)
    total_gen_time = sum(r.generation_time for r in results)
    print(f"\nGeneration complete: {success}/{board.num_shots} shots, {total_gen_time:.1f}s total")

    # ---- Post-processing ----
    all_frame_lists = []
    for r in results:
        if r.output and r.output.frames:
            all_frame_lists.append(r.output.frames)

    if not all_frame_lists:
        print("No successful shots to compose.")
        return

    # Frame interpolation
    if args.interpolate and args.interpolate > 1:
        from animatediff.postprocess.interpolation import FrameInterpolator
        interp = FrameInterpolator(device=args.device)
        print(f"\nInterpolating frames ({args.interpolate}x)...")
        all_frame_lists = [
            interp.interpolate(frames, multiplier=args.interpolate)
            for frames in all_frame_lists
        ]
        effective_fps = args.fps * args.interpolate
    else:
        effective_fps = args.fps

    # Upscaling
    if args.upscale and args.upscale > 1:
        from animatediff.postprocess.upscale import VideoUpscaler
        upscaler = VideoUpscaler(
            model_name=args.upscale_model,
            scale=args.upscale,
            device=args.device,
        )
        print(f"\nUpscaling frames ({args.upscale}x with {args.upscale_model})...")
        all_frame_lists = [upscaler.upscale_frames(frames) for frames in all_frame_lists]

    # Audio generation
    audio_paths = None
    if args.tts and args.narration:
        from animatediff.postprocess.audio import AudioGenerator
        audio_gen = AudioGenerator(
            device=args.device,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
        print("\nGenerating narration audio...")
        narration_texts = args.narration.split("|")
        audio_paths = audio_gen.generate_narration(
            narration_texts,
            output_dir=os.path.join(outdir, "audio"),
        )

    # ---- Compose final video ----
    from animatediff.postprocess.compositor import VideoCompositor
    compositor = VideoCompositor(fps=effective_fps)

    transitions = [shot.transition for shot in board.shots[1:]]
    final_path = os.path.join(outdir, f"final.mp4")

    print(f"\nComposing final video ({len(all_frame_lists)} shots)...")
    compositor.compose(
        shot_frame_lists=all_frame_lists,
        output_path=final_path,
        transitions=transitions,
        audio_paths=audio_paths,
        bgm_path=args.bgm,
        bgm_volume=args.bgm_volume,
    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! Final video: {final_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="AnimateDiff V4 StoryEngine — multi-shot story video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Natural language script
  python scripts/story.py --script "韩立站在山巅。远处龙吟。他纵身跃下。"

  # Structured storyboard
  python scripts/story.py --storyboard story.json --style "anime, xianxia"

  # Single shot with Wan 2.2
  python scripts/story.py --prompt "a girl dancing" --backend wan22

  # Full pipeline with post-processing
  python scripts/story.py --storyboard story.json --interpolate 2 --upscale 2 --tts
        """,
    )

    # Input modes (mutually preferred, not exclusive)
    parser.add_argument("--storyboard", type=str, help="Path to storyboard JSON/YAML file")
    parser.add_argument("--script", type=str, help="Natural language script text")
    parser.add_argument("--prompt", type=str, help="Single shot prompt (quick mode)")
    parser.add_argument("--characters", type=str, help="Path to character profiles JSON")

    # Style
    parser.add_argument("--style", type=str, default="", help="Global style prompt (e.g., 'anime, xianxia')")
    parser.add_argument("--negative-prompt", type=str, default=None)

    # Backend
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "wan", "wan22", "wan22_animate", "hunyuan", "cogvideo", "ltx", "animatediff"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-variant", type=str, default=None)
    parser.add_argument("--quality", type=str, default="standard", choices=["draft", "standard", "high", "max"])
    parser.add_argument("--quantization", type=str, default=None, choices=["none", "nf4", "int8", "fp8"])
    parser.add_argument("--offload", type=str, default=None, choices=["none", "model_cpu", "sequential_cpu"])

    # LoRA
    parser.add_argument("--lora-paths", type=str, nargs="*", help="LoRA weight paths")
    parser.add_argument("--lora-scales", type=float, nargs="*", help="LoRA scales")

    # Dimensions
    parser.add_argument("--W", type=int, default=0, help="Width (0=auto)")
    parser.add_argument("--H", type=int, default=0, help="Height (0=auto)")
    parser.add_argument("--fps", type=int, default=16, help="Output FPS")
    parser.add_argument("--duration", type=float, default=None, help="Single shot duration (seconds)")
    parser.add_argument("--seed", type=int, default=-1)

    # Post-processing
    parser.add_argument("--interpolate", type=int, default=0, help="Frame interpolation multiplier (2=2x fps)")
    parser.add_argument("--upscale", type=int, default=0, help="Upscale factor (2=2x resolution)")
    parser.add_argument("--upscale-model", type=str, default="animevideov3",
                        choices=["animevideov3", "anime_6B", "general"])

    # Audio
    parser.add_argument("--tts", action="store_true", help="Enable TTS narration")
    parser.add_argument("--narration", type=str, default=None, help="Narration texts, pipe-separated per shot")
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference voice audio for cloning")
    parser.add_argument("--ref-text", type=str, default=None, help="Transcript of reference audio")
    parser.add_argument("--bgm", type=str, default=None, help="Background music file path")
    parser.add_argument("--bgm-volume", type=float, default=0.3, help="BGM volume (0.0-1.0)")

    # LLM script parsing
    parser.add_argument("--use-llm", action="store_true", help="Use local LLM for script parsing")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}")
    run_story(args)


if __name__ == "__main__":
    main()
