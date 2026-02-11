"""
Character Portrait Generator — generate reference images for I2V consistency.

Uses Wan 2.2 TI2V-5B in T2V mode to generate short clips, then selects
the best frame as a character reference portrait for I2V generation.

Usage:
    # Generate portraits for all characters in a storyboard
    python scripts/generate_portraits.py \
        --storyboard examples/fanren_trailer_v2.json \
        --output-dir output/fanren-trailer-v2/portraits \
        --num-candidates 3

    # Generate a single character portrait
    python scripts/generate_portraits.py \
        --prompt "portrait of a young Chinese boy, anime style" \
        --name "hanli_early" \
        --output-dir output/portraits \
        --num-candidates 5
"""

import argparse
import datetime
import json
import logging
import os

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default portrait generation parameters
PORTRAIT_DEFAULTS = {
    "width": 480,
    "height": 832,  # Portrait orientation for character refs
    "num_frames": 17,  # Minimum for Wan: generates a short clip
    "num_inference_steps": 30,
    "guidance_scale": 6.0,
}


@torch.no_grad()
def generate_portraits(args):
    """Generate character portraits from storyboard or single prompt."""
    from animatediff.core.vram_manager import get_vram_manager
    from animatediff.backends import get_backend

    vm = get_vram_manager()
    print(f"\n{'='*50}")
    print(vm.summary())
    print(f"{'='*50}\n")

    # Resolve device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # Resolve backend and model
    rec = vm.recommend("wan22")
    torch_dtype = rec.torch_dtype
    if args.device == "mps":
        torch_dtype = torch.float32

    BackendClass = get_backend("wan22")
    load_kwargs = dict(
        model_variant="5B",  # Always use 5B for portraits (works on MPS)
        torch_dtype=torch_dtype,
        device=args.device,
        quantization=args.quantization or rec.quantization,
        offload_strategy=args.offload or rec.offload_strategy,
        enable_vae_slicing=True,
        enable_vae_tiling=rec.enable_vae_tiling,
    )

    print("Loading Wan 2.2 TI2V-5B for portrait generation...")
    pipeline = BackendClass.load(**load_kwargs)

    # Build character list
    characters = []
    if args.storyboard:
        characters = _load_characters_from_storyboard(args.storyboard)
    elif args.prompt:
        characters = [{
            "name": args.name or "character",
            "prompt": args.prompt,
        }]
    else:
        raise ValueError("Provide --storyboard or --prompt")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for char in characters:
        name = char["name"]
        prompt = char["prompt"]
        print(f"\n--- Generating portraits for: {name} ---")
        print(f"  Prompt: {prompt[:80]}...")

        candidates = []
        for seed_idx in range(args.num_candidates):
            seed = args.base_seed + seed_idx if args.base_seed >= 0 else -1
            print(f"  Candidate {seed_idx + 1}/{args.num_candidates} (seed={seed})...")

            output = pipeline.generate(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                width=args.width or PORTRAIT_DEFAULTS["width"],
                height=args.height or PORTRAIT_DEFAULTS["height"],
                num_frames=PORTRAIT_DEFAULTS["num_frames"],
                num_inference_steps=PORTRAIT_DEFAULTS["num_inference_steps"],
                guidance_scale=PORTRAIT_DEFAULTS["guidance_scale"],
                seed=seed,
            )

            if output.frames:
                # Select the best frame (middle frame tends to be best)
                best_idx = len(output.frames) // 2
                best_frame = output.frames[best_idx]

                # Save candidate
                filename = f"{name}_candidate_{seed_idx:02d}.png"
                filepath = os.path.join(args.output_dir, filename)
                best_frame.save(filepath)
                candidates.append({
                    "path": filepath,
                    "seed": output.seed,
                    "frame_idx": best_idx,
                })
                print(f"    Saved: {filepath}")

                # Also save the full clip for review
                if args.save_clips:
                    clip_path = os.path.join(args.output_dir, f"{name}_clip_{seed_idx:02d}.mp4")
                    pipeline.save(output, clip_path, fps=24)

        # Use the first candidate as default (user can swap later)
        if candidates:
            default_path = os.path.join(args.output_dir, f"{name}.png")
            # Copy first candidate as the default
            candidates[0]["path"]
            from PIL import Image
            Image.open(candidates[0]["path"]).save(default_path)
            results[name] = {
                "default": default_path,
                "candidates": candidates,
            }
            print(f"  Default portrait: {default_path}")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "portraits_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPortrait manifest saved: {manifest_path}")

    # Print instructions
    print(f"\n{'='*50}")
    print("Portrait generation complete!")
    print(f"  Characters: {len(results)}")
    print(f"  Candidates per character: {args.num_candidates}")
    print(f"\nNext steps:")
    print(f"  1. Review candidates in: {args.output_dir}")
    print(f"  2. Pick the best portrait for each character")
    print(f"  3. Rename the chosen one to <name>.png (or update storyboard image_path)")
    print(f"  4. Run story.py with the storyboard to generate I2V shots")
    print(f"{'='*50}")


def _load_characters_from_storyboard(storyboard_path: str) -> list:
    """Load character portrait prompts from a storyboard JSON."""
    with open(storyboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    characters = []
    for name, info in data.get("characters", {}).items():
        if isinstance(info, str):
            info = {"description": info}

        # Use portrait_prompt if available, otherwise build from description
        prompt = info.get("portrait_prompt", "")
        if not prompt:
            desc = info.get("description", name)
            prompt = (
                f"portrait of {desc}, anime style, high detail, "
                f"upper body, looking at camera, clean background"
            )

        characters.append({
            "name": name,
            "prompt": prompt,
        })

    return characters


def main():
    parser = argparse.ArgumentParser(
        description="Generate character reference portraits for I2V consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument("--storyboard", type=str,
                        help="Storyboard JSON with character portrait_prompt fields")
    parser.add_argument("--prompt", type=str, help="Single portrait prompt")
    parser.add_argument("--name", type=str, default="character",
                        help="Character name (for single prompt mode)")

    # Output
    parser.add_argument("--output-dir", type=str, default="output/portraits",
                        help="Directory to save portrait images")
    parser.add_argument("--save-clips", action="store_true",
                        help="Also save full 17-frame clips for review")

    # Generation
    parser.add_argument("--num-candidates", type=int, default=3,
                        help="Number of portrait candidates per character")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Starting seed for candidates (-1 for random)")
    parser.add_argument("--width", type=int, default=0,
                        help="Portrait width (0=default 480)")
    parser.add_argument("--height", type=int, default=0,
                        help="Portrait height (0=default 832)")
    parser.add_argument("--negative-prompt", type=str,
                        default="blurry, low quality, distorted, deformed, ugly, watermark, "
                                "3d, realistic, photo, multiple people, text",
                        help="Negative prompt for portrait generation")

    # Backend
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["none", "nf4", "int8", "fp8"])
    parser.add_argument("--offload", type=str, default=None,
                        choices=["none", "model_cpu", "sequential_cpu"])

    args = parser.parse_args()
    generate_portraits(args)


if __name__ == "__main__":
    main()
