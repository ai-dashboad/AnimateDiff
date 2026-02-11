"""
AnimateDiff V3 — Unified Video Generation CLI

Supports multiple backends:
  - wan:        Wan 2.1 (1.3B-14B, best quality-to-VRAM ratio)
  - hunyuan:    HunyuanVideo (8.3B, high quality)
  - cogvideo:   CogVideoX (2B-5B, lightest)
  - ltx:        LTX-Video (real-time, 8-step)
  - animatediff: AnimateDiff legacy (SD1.5/SDXL/Lightning)
  - auto:       Auto-detect best backend for your GPU

Usage:
  # Auto-detect best backend
  python scripts/animate.py --backend auto --prompt "a cat playing"

  # Specific backend
  python scripts/animate.py --backend wan --prompt "a girl smiling, anime"

  # Legacy AnimateDiff (backward compatible)
  python scripts/animate.py --pipeline v2 --config configs/prompts/4_v2/4_1_v2_basic.yaml

  # Quality presets
  python scripts/animate.py --backend wan --quality high --prompt "sunset over ocean"
"""

import argparse
import datetime
import os
import logging

import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Quality Presets
# ============================================================================

QUALITY_PRESETS = {
    "draft": dict(num_inference_steps=10, guidance_scale=3.0),
    "standard": dict(num_inference_steps=20, guidance_scale=5.0),
    "high": dict(num_inference_steps=30, guidance_scale=6.0),
    "max": dict(num_inference_steps=50, guidance_scale=7.5),
}


# ============================================================================
# Multi-Backend Runner (new unified path)
# ============================================================================

@torch.no_grad()
def run_backend(args):
    """Run video generation using the unified backend system."""
    from animatediff.core.vram_manager import get_vram_manager
    from animatediff.backends import get_backend

    vm = get_vram_manager()
    print(f"\n{'='*50}")
    print(vm.summary())
    print(f"{'='*50}\n")

    # Auto-select backend
    backend_name = args.backend
    if backend_name == "auto":
        backend_name = vm.best_backend()
        print(f"Auto-selected backend: {backend_name}")

    # Get recommended config for this backend
    rec = vm.recommend(backend_name)
    print(f"Recommended config: variant={rec.model_variant}, quant={rec.quantization}, "
          f"offload={rec.offload_strategy}, max={rec.max_width}x{rec.max_height}x{rec.max_frames}")

    # Apply quality preset
    preset = QUALITY_PRESETS.get(args.quality, {})
    steps = args.steps or preset.get("num_inference_steps", 20)
    guidance = args.guidance_scale if args.guidance_scale is not None else preset.get("guidance_scale", 5.0)

    # Resolve dimensions (user override > recommended)
    width = args.W or rec.max_width
    height = args.H or rec.max_height
    num_frames = args.L or rec.max_frames

    # Resolve quantization (user override > recommended)
    quantization = args.quantization or rec.quantization
    offload = args.offload or rec.offload_strategy
    torch_dtype = rec.torch_dtype

    # Load backend
    BackendClass = get_backend(backend_name)
    load_kwargs = dict(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        device=args.device,
        quantization=quantization,
        offload_strategy=offload,
        enable_vae_slicing=True,
        enable_vae_tiling=rec.enable_vae_tiling,
    )

    # Backend-specific kwargs
    if backend_name == "wan":
        load_kwargs["model_variant"] = args.model_variant or rec.model_variant
    elif backend_name == "wan22":
        load_kwargs["model_variant"] = args.model_variant or rec.model_variant
    elif backend_name == "cogvideo":
        load_kwargs["model_variant"] = args.model_variant or rec.model_variant
    elif backend_name == "animatediff":
        load_kwargs["pipeline_type"] = args.pipeline or "v2"
        load_kwargs["scheduler"] = args.scheduler
        if args.motion_adapter:
            load_kwargs["motion_adapter"] = args.motion_adapter
        if args.pipeline == "lightning":
            load_kwargs["lightning_steps"] = args.lightning_steps

    print(f"\nLoading {backend_name} backend...")
    pipe = BackendClass.load(**load_kwargs)

    # Compile if beneficial
    if rec.use_compile and not args.no_compile:
        from animatediff.core.compile import compile_pipeline
        print("Applying torch.compile...")
        if hasattr(pipe, "pipe"):
            compile_pipeline(pipe.pipe)

    # Prepare output directory
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{backend_name}-{time_str}"
    os.makedirs(savedir, exist_ok=True)

    # Load prompts from config or CLI
    prompts = _load_prompts(args)

    for idx, (prompt, neg_prompt, seed) in enumerate(prompts):
        print(f"\n[{backend_name}] Generating ({idx+1}/{len(prompts)}): {prompt[:80]}... (seed={seed})")

        output = pipe.generate(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )

        path = f"{savedir}/{idx:04d}.{args.format}"
        pipe.save(output, path, fps=args.fps)
        print(f"  Saved: {path}")

    print(f"\nAll outputs saved to {savedir}/")


def _load_prompts(args):
    """Load prompts from --prompt or --config."""
    prompts = []

    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        for model_config in config:
            prompt_list = model_config.get("prompt", [])
            if isinstance(prompt_list, str):
                prompt_list = [prompt_list]
            n_prompt = model_config.get("n_prompt", [""])
            if isinstance(n_prompt, list):
                n_prompt = n_prompt[0] if n_prompt else ""
            seeds = model_config.get("seed", [-1])
            if isinstance(seeds, int):
                seeds = [seeds]

            for i, p in enumerate(prompt_list):
                seed = seeds[i] if i < len(seeds) else seeds[-1] if seeds else -1
                prompts.append((p, n_prompt, seed))
    elif args.prompt:
        prompts.append((args.prompt, args.negative_prompt or "", args.seed))
    else:
        raise ValueError("Provide either --prompt or --config")

    return prompts


# ============================================================================
# Legacy AnimateDiff Runners (backward compatible, unchanged)
# ============================================================================

@torch.no_grad()
def run_v2(args):
    """Run using the V2 pipeline (diffusers AnimateDiffPipeline wrapper)."""
    from animatediff.pipelines.pipeline_v2 import AnimateDiffV2Pipeline
    from animatediff.utils.prompt_travel import parse_prompt_travel
    from omegaconf import OmegaConf
    from PIL import Image

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/v2-{time_str}"
    os.makedirs(savedir, exist_ok=True)

    config = OmegaConf.load(args.config)

    pipe = AnimateDiffV2Pipeline.from_pretrained(
        model_path=args.pretrained_model_path,
        motion_adapter_path=args.motion_adapter or "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16 if args.half_precision else torch.float32,
        device=args.device,
        scheduler=args.scheduler,
    )

    if args.freeinit_iters > 0:
        pipe.enable_free_init(num_iters=args.freeinit_iters, method=args.freeinit_method, use_fast_sampling=True)
    if args.context_length > 0:
        pipe.enable_free_noise(context_length=args.context_length, context_stride=args.context_overlap)

    ip_image = None
    if args.ip_adapter_image:
        pipe.load_ip_adapter(scale=args.ip_adapter_scale)
        ip_image = Image.open(args.ip_adapter_image).convert("RGB")
    if args.lora_path:
        pipe.load_lora(args.lora_path, scale=args.lora_scale)

    sample_idx = 0
    for model_config in config:
        W = model_config.get("W", args.W)
        H = model_config.get("H", args.H)
        L = model_config.get("L", args.L)
        prompt = parse_prompt_travel(model_config)
        n_prompt = model_config.get("n_prompt", [""])[0] if isinstance(model_config.get("n_prompt", [""]), list) else model_config.get("n_prompt", "")
        seeds = model_config.get("seed", [-1])
        if isinstance(seeds, int):
            seeds = [seeds]

        for seed in seeds:
            print(f"[V2] Generating: {prompt} (seed={seed})")
            output = pipe.generate(
                prompt=prompt, negative_prompt=n_prompt, num_frames=L,
                height=H, width=W,
                num_inference_steps=model_config.get("steps", 25),
                guidance_scale=model_config.get("guidance_scale", 7.5),
                seed=seed, ip_adapter_image=ip_image,
            )
            path = f"{savedir}/sample/{sample_idx}.{args.format}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pipe.save(output, path)
            print(f"Saved to {path}")
            sample_idx += 1

    if args.freeinit_iters > 0:
        pipe.disable_free_init()


@torch.no_grad()
def run_sdxl(args):
    """Run using the SDXL pipeline."""
    from animatediff.pipelines.pipeline_sdxl import AnimateDiffSDXL
    from animatediff.utils.prompt_travel import parse_prompt_travel
    from omegaconf import OmegaConf

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/sdxl-{time_str}"
    os.makedirs(savedir, exist_ok=True)
    config = OmegaConf.load(args.config)

    pipe = AnimateDiffSDXL.from_pretrained(
        model_path=args.pretrained_model_path or "stabilityai/stable-diffusion-xl-base-1.0",
        motion_adapter_path=args.motion_adapter or "guoyww/animatediff-motion-adapter-sdxl-beta",
        torch_dtype=torch.float16, device=args.device, scheduler=args.scheduler,
    )

    sample_idx = 0
    for model_config in config:
        W, H, L = model_config.get("W", args.W), model_config.get("H", args.H), model_config.get("L", args.L)
        prompt = parse_prompt_travel(model_config)
        n_prompt = model_config.get("n_prompt", [""])[0] if isinstance(model_config.get("n_prompt", [""]), list) else model_config.get("n_prompt", "")
        seeds = model_config.get("seed", [-1])
        if isinstance(seeds, int):
            seeds = [seeds]
        for seed in seeds:
            print(f"[SDXL] Generating: {prompt} (seed={seed})")
            output = pipe.generate(prompt=prompt, negative_prompt=n_prompt, num_frames=L, height=H, width=W,
                                   num_inference_steps=model_config.get("steps", 20), guidance_scale=model_config.get("guidance_scale", 8.0), seed=seed)
            path = f"{savedir}/sample/{sample_idx}.{args.format}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pipe.save(output, path)
            print(f"Saved to {path}")
            sample_idx += 1


@torch.no_grad()
def run_lightning(args):
    """Run using AnimateDiff-Lightning for ultra-fast inference."""
    from animatediff.pipelines.pipeline_lightning import AnimateDiffLightning
    from animatediff.utils.prompt_travel import parse_prompt_travel
    from omegaconf import OmegaConf

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/lightning-{time_str}"
    os.makedirs(savedir, exist_ok=True)
    config = OmegaConf.load(args.config)

    pipe = AnimateDiffLightning.from_pretrained(
        model_path=args.pretrained_model_path or "emilianJR/epiCRealism",
        num_steps=args.lightning_steps, torch_dtype=torch.float16, device=args.device,
    )

    sample_idx = 0
    for model_config in config:
        W, H, L = model_config.get("W", args.W), model_config.get("H", args.H), model_config.get("L", args.L)
        prompt = parse_prompt_travel(model_config)
        seeds = model_config.get("seed", [-1])
        if isinstance(seeds, int):
            seeds = [seeds]
        for seed in seeds:
            print(f"[Lightning] Generating ({args.lightning_steps}-step): {prompt} (seed={seed})")
            output = pipe.generate(prompt=prompt, num_frames=L, height=H, width=W, seed=seed)
            path = f"{savedir}/sample/{sample_idx}.{args.format}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pipe.save(output, path)
            print(f"Saved to {path}")
            sample_idx += 1


@torch.no_grad()
def run_legacy(args):
    """Run using the original AnimateDiff pipeline (supports SparseCtrl)."""
    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    from einops import rearrange
    import torchvision.transforms as transforms
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler
    from diffusers.utils.import_utils import is_xformers_available
    from animatediff.models.unet import UNet3DConditionModel
    from animatediff.models.sparse_controlnet import SparseControlNetModel
    from animatediff.pipelines.pipeline_animation import AnimationPipeline
    from animatediff.utils.util import save_videos_grid, load_weights, auto_download

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    extension = args.format
    os.makedirs(savedir)

    config = OmegaConf.load(args.config)
    samples = []

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(args.device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(args.device)

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet",
                                                        unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(args.device)

        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            if not model_config.get("controlnet_images", ""):
                raise ValueError("controlnet_images must be specified when controlnet_path is set")
            if not model_config.get("controlnet_config", ""):
                raise ValueError("controlnet_config must be specified when controlnet_path is set")

            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            auto_download(model_config.controlnet_path, is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu", weights_only=False)
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.to(args.device)

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            if len(image_paths) > model_config.L:
                raise ValueError(f"Number of controlnet images ({len(image_paths)}) exceeds video length ({model_config.L})")

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop((model_config.H, model_config.W), (1.0, 1.0),
                                             ratio=(model_config.W/model_config.H, model_config.W/model_config.H)),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else:
                image_norm = lambda x: x

            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]
            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1, 2, 0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")
            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).to(args.device)
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None:
                controlnet.enable_xformers_memory_efficient_attention()

        scheduler_kwargs = OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
        scheduler_map = {
            "ddim": DDIMScheduler,
            "euler": EulerDiscreteScheduler,
            "euler-a": EulerAncestralDiscreteScheduler,
            "dpm++": DPMSolverMultistepScheduler,
            "dpm++-karras": lambda **kw: DPMSolverMultistepScheduler(**kw, use_karras_sigmas=True),
            "pndm": PNDMScheduler,
        }
        scheduler = scheduler_map[args.scheduler](**scheduler_kwargs)

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet, scheduler=scheduler,
        ).to(args.device)

        pipeline = load_weights(
            pipeline,
            motion_module_path=model_config.get("motion_module", ""),
            motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
            adapter_lora_path=model_config.get("adapter_lora_path", ""),
            adapter_lora_scale=model_config.get("adapter_lora_scale", 1.0),
            dreambooth_model_path=model_config.get("dreambooth_path", ""),
            lora_model_path=model_config.get("lora_model_path", ""),
            lora_alpha=model_config.get("lora_alpha", 0.8),
        ).to(args.device)

        pipeline.enable_vae_slicing()
        if args.half_precision and args.device != "cpu":
            pipeline.unet.half()
            pipeline.text_encoder.half()
            if controlnet is not None:
                controlnet.half()

        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

        config[model_idx].random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            if random_seed != -1:
                torch.manual_seed(random_seed)
            else:
                torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt, negative_prompt=n_prompt,
                num_inference_steps=model_config.steps, guidance_scale=model_config.guidance_scale,
                width=model_config.W, height=model_config.H, video_length=model_config.L,
                controlnet_images=controlnet_images,
                controlnet_image_index=model_config.get("controlnet_image_indexs", [0]),
            ).videos
            samples.append(sample)
            prompt_short = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt_short}.{extension}")
            print(f"save to {savedir}/sample/{prompt_short}.{extension}")
            sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.{extension}", n_rows=4)
    OmegaConf.save(config, f"{savedir}/config.yaml")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main_cli():
    """Entry point for `animatediff` CLI command via pip install."""
    parser = argparse.ArgumentParser(
        description="AnimateDiff V3 — Unified Video Generation (Wan/HunyuanVideo/CogVideoX/LTX/AnimateDiff)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best backend for your GPU
  python scripts/animate.py --backend auto --prompt "a cat playing with yarn"

  # Use Wan 2.1 with auto-optimized settings
  python scripts/animate.py --backend wan --prompt "a girl dancing, anime style"

  # High quality preset
  python scripts/animate.py --backend wan --quality high --prompt "sunset over ocean"

  # Legacy AnimateDiff (backward compatible)
  python scripts/animate.py --pipeline v2 --config configs/prompts/4_v2/4_1_v2_basic.yaml
        """,
    )

    # ---- NEW: Multi-backend mode ----
    parser.add_argument("--backend", type=str, default=None,
                        choices=["auto", "wan", "wan22", "wan22_animate", "hunyuan", "cogvideo", "ltx", "animatediff"],
                        help="Video generation backend (default: auto-detect or use --pipeline for legacy)")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--model-path", type=str, default=None, help="Model path or HuggingFace repo ID")
    parser.add_argument("--model-variant", type=str, default=None,
                        help="Model variant (e.g., '1.3B' or '14B' for Wan, '2B' or '5B' for CogVideo)")
    parser.add_argument("--quality", type=str, default="standard",
                        choices=["draft", "standard", "high", "max"],
                        help="Quality preset (affects steps and guidance)")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["none", "nf4", "int8", "fp8"],
                        help="Quantization method (default: auto based on VRAM)")
    parser.add_argument("--offload", type=str, default=None,
                        choices=["none", "model_cpu", "sequential_cpu"],
                        help="Offloading strategy (default: auto based on VRAM)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS")

    # ---- LEGACY: AnimateDiff mode (backward compatible) ----
    parser.add_argument("--pipeline", type=str, default=None,
                        choices=["legacy", "v2", "sdxl", "lightning"],
                        help="Legacy AnimateDiff pipeline (overrides --backend)")
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")
    parser.add_argument("--config", type=str, default=None, help="YAML config file with prompts/settings")
    parser.add_argument("--motion-adapter", type=str, default=None)

    # Dimensions
    parser.add_argument("--L", type=int, default=0, help="Number of frames (0=auto)")
    parser.add_argument("--W", type=int, default=0, help="Width (0=auto)")
    parser.add_argument("--H", type=int, default=0, help="Height (0=auto)")

    # Output
    parser.add_argument("--format", type=str, default="mp4", choices=["gif", "mp4"])

    # Scheduler (legacy)
    parser.add_argument("--scheduler", type=str, default="ddim",
                        choices=["ddim", "euler", "euler-a", "dpm++", "dpm++-karras", "pndm"])

    # Performance (legacy)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    # V2 features (legacy)
    parser.add_argument("--freeinit-iters", type=int, default=0)
    parser.add_argument("--freeinit-method", type=str, default="butterworth", choices=["butterworth", "ideal", "gaussian"])
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--context-overlap", type=int, default=4)
    parser.add_argument("--ip-adapter-image", type=str, default=None)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--lightning-steps", type=int, default=4, choices=[1, 2, 4, 8])

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

    # Route: if --pipeline is set, use legacy mode; if --backend is set, use new mode
    if args.pipeline is not None:
        # Legacy AnimateDiff mode
        if not args.config:
            raise ValueError("Legacy mode (--pipeline) requires --config")
        print(f"Pipeline: {args.pipeline} (legacy mode)")
        if args.W == 0:
            args.W = 512
        if args.H == 0:
            args.H = 512
        if args.L == 0:
            args.L = 16
        {"legacy": run_legacy, "v2": run_v2, "sdxl": run_sdxl, "lightning": run_lightning}[args.pipeline](args)
    elif args.backend is not None:
        # New multi-backend mode
        run_backend(args)
    else:
        # No --pipeline or --backend: default to new backend auto mode
        if args.config and not args.prompt:
            args.pipeline = "legacy"
            if args.W == 0:
                args.W = 512
            if args.H == 0:
                args.H = 512
            if args.L == 0:
                args.L = 16
            run_legacy(args)
        else:
            args.backend = "auto"
            run_backend(args)


if __name__ == "__main__":
    main_cli()
