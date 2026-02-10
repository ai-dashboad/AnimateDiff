"""
End-to-end video generation test.

Downloads the smallest available model (Wan 1.3B) and generates a short video
to verify the full pipeline works: load → generate → save.

This test is slow (downloads ~5GB model on first run) and requires a GPU or
Apple Silicon with MPS. Mark with pytest -m e2e to run explicitly.

Usage:
    .venv/bin/python tests/test_e2e_generate.py
"""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_wan_1_3b_generate():
    """Generate a short anime clip with Wan 2.1 1.3B."""
    import torch
    from animatediff.backends.wan import WanBackend

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        offload = "model_cpu"  # CUDA supports CPU offload
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS requires float32 for Wan
        offload = "none"  # MPS: load directly to GPU (no cpu offload support)
    else:
        print("SKIP: No GPU available (need CUDA or MPS)")
        return

    print(f"\n{'='*60}")
    print(f"Device: {device} | dtype: {dtype} | offload: {offload}")
    print(f"{'='*60}")

    # Load model (will download ~5GB on first run)
    print("\n[1/3] Loading Wan 2.1 1.3B...")
    t0 = time.time()
    backend = WanBackend.load(
        model_path=None,  # auto: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        torch_dtype=dtype,
        device=device,
        quantization="none",
        offload_strategy=offload,
        enable_vae_slicing=True,
        model_variant="1.3B",
    )
    print(f"   Loaded in {time.time() - t0:.1f}s")

    # Generate short video: small resolution, few frames, few steps
    print("\n[2/3] Generating anime test video...")
    prompt = "a cute anime girl with blue hair smiling, cherry blossom background, anime style, high quality"
    t0 = time.time()
    output = backend.generate(
        prompt=prompt,
        negative_prompt="ugly, blurry, low quality",
        width=480,
        height=320,
        num_frames=17,          # ~2 seconds at 8fps
        num_inference_steps=15,  # draft quality for speed
        guidance_scale=5.0,
        seed=42,
    )
    gen_time = time.time() - t0
    print(f"   Generated {len(output.frames)} frames in {gen_time:.1f}s")

    # Save output
    print("\n[3/3] Saving video...")
    os.makedirs("samples/e2e_test", exist_ok=True)

    mp4_path = "samples/e2e_test/wan_1.3b_anime.mp4"
    gif_path = "samples/e2e_test/wan_1.3b_anime.gif"
    backend.save(output, mp4_path, fps=8)
    backend.save(output, gif_path, fps=8)

    # Verify
    assert os.path.exists(mp4_path), f"MP4 not created: {mp4_path}"
    assert os.path.getsize(mp4_path) > 1000, f"MP4 too small: {os.path.getsize(mp4_path)}"
    assert os.path.exists(gif_path), f"GIF not created: {gif_path}"
    assert os.path.getsize(gif_path) > 1000, f"GIF too small: {os.path.getsize(gif_path)}"
    assert len(output.frames) == 17, f"Expected 17 frames, got {len(output.frames)}"
    assert output.seed == 42
    assert output.backend == "wan"

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"  Frames: {len(output.frames)}")
    print(f"  MP4: {mp4_path} ({os.path.getsize(mp4_path) / 1024:.0f} KB)")
    print(f"  GIF: {gif_path} ({os.path.getsize(gif_path) / 1024:.0f} KB)")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Prompt: {prompt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_wan_1_3b_generate()
