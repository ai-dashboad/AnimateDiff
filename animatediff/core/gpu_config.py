"""
GPU Configuration — auto-detect GPU tier and return optimal generation parameters.

Builds on top of vram_manager.GPUProfile for hardware detection, but provides a
single self-contained GPUConfig dataclass that scripts can use directly without
manually wiring device/dtype/resolution/offload/quantization decisions.

Tier definitions:
    consumer_24gb  — RTX 3090/4090/5090 (~24 GB VRAM)
    pro_40gb       — A6000/L40/A40 (~40-48 GB VRAM)
    datacenter_80gb — A100/H100/H200 (~80 GB VRAM)
    mps            — Apple Silicon unified memory
    cpu            — CPU-only fallback
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU tier classification helpers
# ---------------------------------------------------------------------------

_TIER_THRESHOLDS = [
    # (min_vram_gb, tier_name) — evaluated in order, first match wins
    (60.0, "datacenter_80gb"),
    (30.0, "pro_40gb"),
    (18.0, "consumer_24gb"),
]


def _classify_cuda_tier(vram_gb: float) -> str:
    """Classify a CUDA GPU into a named tier based on VRAM."""
    for threshold, tier in _TIER_THRESHOLDS:
        if vram_gb >= threshold:
            return tier
    # Under 18 GB — still consumer, but very constrained.
    # We treat it the same as consumer_24gb with tighter limits handled
    # by the VRAM manager's tiered recommendations.
    return "consumer_24gb"


# ---------------------------------------------------------------------------
# Per-tier default generation parameters
# ---------------------------------------------------------------------------

@dataclass
class _TierDefaults:
    model_variant: str
    width: int
    height: int
    fps: int
    max_frames: int
    num_inference_steps: int
    guidance_scale: float
    offload: str          # "none" | "model_cpu" | "sequential_cpu"
    quantization: str     # "none" | "fp8" | "nf4" | "int8"
    torch_dtype: torch.dtype
    compile: bool
    portrait_width: int = 832
    portrait_height: int = 480


_TIER_CONFIGS = {
    "datacenter_80gb": _TierDefaults(
        model_variant="A14B",
        width=1280,
        height=720,
        fps=24,
        max_frames=81,
        num_inference_steps=50,
        guidance_scale=5.0,
        offload="none",
        quantization="none",
        torch_dtype=torch.bfloat16,
        compile=True,
        portrait_width=1280,
        portrait_height=720,
    ),
    "pro_40gb": _TierDefaults(
        model_variant="A14B",
        width=1280,
        height=720,
        fps=24,
        max_frames=81,
        num_inference_steps=50,
        guidance_scale=5.0,
        offload="model_cpu",
        quantization="nf4",
        torch_dtype=torch.bfloat16,
        compile=True,
        portrait_width=1280,
        portrait_height=720,
    ),
    "consumer_24gb": _TierDefaults(
        model_variant="5B",
        width=832,
        height=480,
        fps=24,
        max_frames=81,
        num_inference_steps=50,
        guidance_scale=5.0,
        offload="model_cpu",
        quantization="none",
        torch_dtype=torch.bfloat16,
        compile=False,
        portrait_width=832,
        portrait_height=480,
    ),
    "mps": _TierDefaults(
        model_variant="5B",
        width=832,
        height=480,
        fps=24,
        max_frames=121,
        num_inference_steps=50,
        guidance_scale=5.0,
        offload="none",
        quantization="none",
        torch_dtype=torch.float32,
        compile=False,
        portrait_width=832,
        portrait_height=480,
    ),
    "cpu": _TierDefaults(
        model_variant="5B",
        width=480,
        height=320,
        fps=24,
        max_frames=17,
        num_inference_steps=30,
        guidance_scale=5.0,
        offload="sequential_cpu",
        quantization="nf4",
        torch_dtype=torch.float32,
        compile=False,
        portrait_width=480,
        portrait_height=320,
    ),
}


# ---------------------------------------------------------------------------
# GPUConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class GPUConfig:
    """Auto-configured generation parameters for the detected GPU.

    Usage::

        gpu = GPUConfig.auto_detect()
        print(gpu.summary())

        # Use fields directly:
        pipe = Backend.load(
            model_variant=gpu.model_variant,
            torch_dtype=gpu.torch_dtype,
            device=gpu.device,
            ...
        )
    """

    # Hardware identity
    device: str = "cpu"               # "cuda", "mps", "cpu"
    gpu_name: str = "CPU"             # e.g. "NVIDIA RTX 4090"
    vram_gb: float = 0.0              # e.g. 24.0
    tier: str = "cpu"                 # "consumer_24gb", "pro_40gb", etc.
    compute_capability: tuple = (0, 0)

    # Generation parameters (auto-configured by tier)
    model_variant: str = "5B"         # "5B" or "A14B"
    width: int = 832
    height: int = 480
    fps: int = 24
    max_frames: int = 81
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    offload: str = "none"             # "none", "model_cpu", "sequential_cpu"
    quantization: str = "none"        # "none", "fp8", "nf4", "int8"
    torch_dtype: torch.dtype = torch.float32
    compile: bool = False             # whether to use torch.compile

    # Portrait generation
    portrait_width: int = 832
    portrait_height: int = 480

    # Wan 2.2 specifics (always None for 5B; used for A14B dual transformer)
    guidance_scale_2: Optional[float] = None

    # Distillation / acceleration (default: no distillation)
    distillation: str = "none"        # "none", "speed", "balanced", "quality"

    @classmethod
    def auto_detect(cls) -> "GPUConfig":
        """Detect GPU hardware and return an optimally configured GPUConfig."""

        # --- 1. Detect hardware ---
        device = "cpu"
        gpu_name = "CPU"
        vram_gb = 0.0
        compute_capability = (0, 0)
        tier = "cpu"

        if torch.cuda.is_available():
            device = "cuda"
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = getattr(props, "total_memory",
                              getattr(props, "total_mem", 0)) / (1024 ** 3)
            compute_capability = (props.major, props.minor)
            tier = _classify_cuda_tier(vram_gb)

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            gpu_name = "Apple Silicon (MPS)"
            tier = "mps"
            # Estimate unified memory available for GPU (~70%)
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, check=True,
                )
                total_ram_gb = int(result.stdout.strip()) / (1024 ** 3)
                vram_gb = total_ram_gb * 0.7
            except Exception:
                vram_gb = 8.0

        # --- 2. Lookup tier defaults ---
        defaults = _TIER_CONFIGS.get(tier, _TIER_CONFIGS["cpu"])

        # --- 3. Apply hardware-specific overrides ---
        torch_dtype = defaults.torch_dtype
        quantization = defaults.quantization
        compile_enabled = defaults.compile

        # FP8 quantization is only available on compute >= 8.9 (Ada / Hopper)
        if quantization == "fp8" and compute_capability < (8, 9):
            quantization = "nf4"
            logger.info("GPU does not support FP8; falling back to NF4 quantization")

        # torch.compile only benefits Ampere+ (compute >= 8.0)
        if compile_enabled and compute_capability < (8, 0):
            compile_enabled = False

        # bfloat16 needs Ampere+ on CUDA
        if device == "cuda" and torch_dtype == torch.bfloat16 and compute_capability < (8, 0):
            torch_dtype = torch.float16
            logger.info("GPU does not support bfloat16; using float16")

        # MPS must use float32 for Wan pipelines
        if device == "mps":
            torch_dtype = torch.float32

        # A14B MoE uses FP8 internally -> blocked on MPS
        model_variant = defaults.model_variant
        if model_variant == "A14B" and device == "mps":
            model_variant = "5B"
            logger.info("A14B MoE blocked on MPS; falling back to TI2V-5B")

        # --- 4. Consumer 24GB with Ada Lovelace (4090) can use FP8 for A14B ---
        if (tier == "consumer_24gb"
                and compute_capability >= (8, 9)
                and vram_gb >= 22):
            # 4090 can run A14B with FP8 quantization
            try:
                from animatediff.core.quantization import check_torchao  # noqa: F811
                if check_torchao():
                    model_variant = "A14B"
                    quantization = "fp8"
                    logger.info(
                        "Consumer 24GB with FP8 support detected; "
                        "upgrading to A14B + FP8"
                    )
            except ImportError:
                pass

        cfg = cls(
            device=device,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            tier=tier,
            compute_capability=compute_capability,
            model_variant=model_variant,
            width=defaults.width,
            height=defaults.height,
            fps=defaults.fps,
            max_frames=defaults.max_frames,
            num_inference_steps=defaults.num_inference_steps,
            guidance_scale=defaults.guidance_scale,
            offload=defaults.offload,
            quantization=quantization,
            torch_dtype=torch_dtype,
            compile=compile_enabled,
            portrait_width=defaults.portrait_width,
            portrait_height=defaults.portrait_height,
            guidance_scale_2=None,  # 5B has no dual transformer
        )

        logger.info("GPU auto-detect: %s", cfg._one_line_summary())
        return cfg

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _one_line_summary(self) -> str:
        return (
            f"{self.gpu_name} ({self.vram_gb:.1f}GB) -> "
            f"tier={self.tier}, model={self.model_variant}, "
            f"{self.width}x{self.height}@{self.fps}fps, "
            f"quant={self.quantization}, offload={self.offload}"
        )

    def summary(self) -> str:
        """Human-readable multi-line summary for logging."""
        lines = [
            "=" * 55,
            "  GPU Configuration (auto-detected)",
            "=" * 55,
            f"  Device:        {self.device}",
            f"  GPU:           {self.gpu_name}",
            f"  VRAM:          {self.vram_gb:.1f} GB",
            f"  Tier:          {self.tier}",
        ]
        if self.device == "cuda":
            lines.append(
                f"  Compute:       {self.compute_capability[0]}."
                f"{self.compute_capability[1]}"
            )
        lines += [
            "",
            f"  Model:         Wan 2.2 {self.model_variant}",
            f"  Resolution:    {self.width}x{self.height} @ {self.fps} fps",
            f"  Max frames:    {self.max_frames}",
            f"  Steps:         {self.num_inference_steps}",
            f"  Guidance:      {self.guidance_scale}"
            + (f" / {self.guidance_scale_2}" if self.guidance_scale_2 else ""),
            f"  Dtype:         {self.torch_dtype}",
            f"  Quantization:  {self.quantization}",
            f"  Offload:       {self.offload}",
            f"  torch.compile: {self.compile}",
            f"  Portrait:      {self.portrait_width}x{self.portrait_height}",
            "=" * 55,
        ]
        return "\n".join(lines)
