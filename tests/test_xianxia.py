"""Generate xianxia/cultivation anime style video (凡人修仙传 style)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_xianxia():
    import torch
    from animatediff.backends.wan import WanBackend

    if torch.cuda.is_available():
        device, dtype, offload = "cuda", torch.float16, "model_cpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype, offload = "mps", torch.float32, "none"
    else:
        print("SKIP: No GPU")
        return

    print(f"Device: {device}")

    backend = WanBackend.load(
        torch_dtype=dtype, device=device,
        quantization="none", offload_strategy=offload,
        model_variant="1.3B",
    )

    scenes = [
        {
            "name": "cultivation_meditation",
            "prompt": "a young man in white ancient Chinese robes meditating on a mountain peak, "
                      "golden spiritual energy swirling around him, clouds below, "
                      "xianxia cultivation anime style, Chinese anime, high quality, detailed",
            "negative": "ugly, blurry, low quality, realistic, western style",
        },
        {
            "name": "sword_flying",
            "prompt": "an immortal cultivator flying on a glowing sword through clouds above mountains, "
                      "flowing robes, long hair, golden sunset, "
                      "Chinese xianxia anime style, donghua, epic, cinematic, high quality",
            "negative": "ugly, blurry, low quality, realistic, deformed",
        },
        {
            "name": "spell_battle",
            "prompt": "two cultivators fighting with magical energy beams in a grand hall, "
                      "one in blue robes one in red robes, spiritual energy explosion, "
                      "Chinese xianxia donghua anime style, dramatic lighting, high quality",
            "negative": "ugly, blurry, low quality, realistic, western cartoon",
        },
    ]

    os.makedirs("samples/xianxia", exist_ok=True)

    for scene in scenes:
        print(f"\nGenerating: {scene['name']}...")
        t0 = time.time()
        output = backend.generate(
            prompt=scene["prompt"],
            negative_prompt=scene["negative"],
            width=480,
            height=320,
            num_frames=17,
            num_inference_steps=20,  # standard quality
            guidance_scale=6.0,
            seed=42,
        )
        gen_time = time.time() - t0

        mp4_path = f"samples/xianxia/{scene['name']}.mp4"
        gif_path = f"samples/xianxia/{scene['name']}.gif"
        backend.save(output, mp4_path, fps=8)
        backend.save(output, gif_path, fps=8)
        print(f"  Done in {gen_time:.0f}s -> {mp4_path}")

    print("\nAll videos saved to samples/xianxia/")


if __name__ == "__main__":
    generate_xianxia()
