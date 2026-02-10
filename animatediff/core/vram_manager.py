"""
Smart VRAM Manager — auto-detects GPU hardware and selects optimal configurations.

Detects GPU model, VRAM, compute capability, and recommends:
- Best model variant for available VRAM
- Quantization strategy (none / INT8 / NF4 / FP8)
- Offloading strategy (none / model_cpu / sequential_cpu)
- Resolution and frame count limits
- Whether torch.compile is beneficial
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU profile dataclass
# ---------------------------------------------------------------------------

@dataclass
class GPUProfile:
    name: str = "cpu"
    vram_gb: float = 0.0
    compute_capability: tuple = (0, 0)
    device: str = "cpu"
    is_cuda: bool = False
    is_mps: bool = False
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False  # compute >= 8.9
    supports_compile: bool = False  # Ampere+ (compute >= 8.0)


@dataclass
class InferenceConfig:
    """Recommended inference configuration for a given GPU + backend."""
    model_variant: str = ""          # e.g. "1.3B", "5B", "14B"
    quantization: str = "none"       # none / int8 / nf4 / fp8
    torch_dtype: torch.dtype = torch.float16
    offload_strategy: str = "none"   # none / model_cpu / sequential_cpu
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    max_width: int = 512
    max_height: int = 512
    max_frames: int = 16
    use_compile: bool = False


# ---------------------------------------------------------------------------
# VRAM tier recommendations per backend
# ---------------------------------------------------------------------------

# (min_vram_gb, config_overrides)
WAN_TIERS = [
    (20.0, InferenceConfig(model_variant="14B", quantization="none", max_width=720, max_height=480, max_frames=81, offload_strategy="none")),
    (12.0, InferenceConfig(model_variant="14B", quantization="nf4", max_width=720, max_height=480, max_frames=49, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (8.0,  InferenceConfig(model_variant="1.3B", quantization="none", max_width=480, max_height=320, max_frames=49, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (6.0,  InferenceConfig(model_variant="1.3B", quantization="nf4", max_width=480, max_height=320, max_frames=33, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (0.0,  InferenceConfig(model_variant="1.3B", quantization="nf4", max_width=480, max_height=320, max_frames=17, offload_strategy="sequential_cpu", enable_vae_tiling=True, torch_dtype=torch.float32)),
]

HUNYUAN_TIERS = [
    (20.0, InferenceConfig(model_variant="default", quantization="none", max_width=720, max_height=480, max_frames=49, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (12.0, InferenceConfig(model_variant="default", quantization="nf4", max_width=720, max_height=480, max_frames=33, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (8.0,  InferenceConfig(model_variant="default", quantization="nf4", max_width=480, max_height=320, max_frames=25, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (0.0,  InferenceConfig(model_variant="default", quantization="nf4", max_width=480, max_height=320, max_frames=17, offload_strategy="sequential_cpu", enable_vae_tiling=True, torch_dtype=torch.float32)),
]

COGVIDEO_TIERS = [
    (16.0, InferenceConfig(model_variant="5B", quantization="none", max_width=720, max_height=480, max_frames=49, offload_strategy="none")),
    (10.0, InferenceConfig(model_variant="5B", quantization="nf4", max_width=720, max_height=480, max_frames=49, offload_strategy="model_cpu")),
    (6.0,  InferenceConfig(model_variant="2B", quantization="none", max_width=480, max_height=320, max_frames=49, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (0.0,  InferenceConfig(model_variant="2B", quantization="nf4", max_width=480, max_height=320, max_frames=25, offload_strategy="sequential_cpu", enable_vae_tiling=True, torch_dtype=torch.float32)),
]

LTX_TIERS = [
    (16.0, InferenceConfig(model_variant="default", quantization="none", max_width=768, max_height=512, max_frames=97, offload_strategy="none")),
    (10.0, InferenceConfig(model_variant="default", quantization="nf4", max_width=768, max_height=512, max_frames=65, offload_strategy="model_cpu")),
    (6.0,  InferenceConfig(model_variant="default", quantization="nf4", max_width=512, max_height=320, max_frames=41, offload_strategy="model_cpu", enable_vae_tiling=True)),
    (0.0,  InferenceConfig(model_variant="default", quantization="nf4", max_width=512, max_height=320, max_frames=25, offload_strategy="sequential_cpu", enable_vae_tiling=True, torch_dtype=torch.float32)),
]

ANIMATEDIFF_TIERS = [
    (8.0,  InferenceConfig(model_variant="sd15", quantization="none", max_width=512, max_height=512, max_frames=64, offload_strategy="none")),
    (4.0,  InferenceConfig(model_variant="sd15", quantization="none", max_width=512, max_height=512, max_frames=16, offload_strategy="model_cpu", torch_dtype=torch.float16)),
    (0.0,  InferenceConfig(model_variant="sd15", quantization="none", max_width=512, max_height=512, max_frames=16, offload_strategy="sequential_cpu", torch_dtype=torch.float32)),
]

BACKEND_TIERS = {
    "wan": WAN_TIERS,
    "hunyuan": HUNYUAN_TIERS,
    "cogvideo": COGVIDEO_TIERS,
    "ltx": LTX_TIERS,
    "animatediff": ANIMATEDIFF_TIERS,
}


# ---------------------------------------------------------------------------
# VRAMManager
# ---------------------------------------------------------------------------

class VRAMManager:
    """Detects GPU capabilities and recommends inference configurations."""

    def __init__(self):
        self.profile = self._detect_gpu()

    def _detect_gpu(self) -> GPUProfile:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cc = (props.major, props.minor)
            vram_gb = props.total_mem / (1024 ** 3)
            return GPUProfile(
                name=props.name,
                vram_gb=vram_gb,
                compute_capability=cc,
                device="cuda",
                is_cuda=True,
                supports_fp16=True,
                supports_bf16=cc >= (8, 0),
                supports_fp8=cc >= (8, 9),
                supports_compile=cc >= (8, 0),
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose VRAM directly; estimate from system
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, check=True,
                )
                total_ram_gb = int(result.stdout.strip()) / (1024 ** 3)
                # Apple unified memory: estimate ~70% available for GPU
                vram_gb = total_ram_gb * 0.7
            except Exception:
                vram_gb = 8.0  # conservative fallback
            return GPUProfile(
                name="Apple Silicon (MPS)",
                vram_gb=vram_gb,
                device="mps",
                is_mps=True,
                supports_fp16=True,
                supports_bf16=False,
                supports_fp8=False,
                supports_compile=False,
            )
        else:
            return GPUProfile()

    def recommend(self, backend: str) -> InferenceConfig:
        """Return the best InferenceConfig for a backend given current GPU."""
        tiers = BACKEND_TIERS.get(backend, ANIMATEDIFF_TIERS)
        for min_vram, config in tiers:
            if self.profile.vram_gb >= min_vram:
                # Adjust dtype based on GPU capabilities
                if not self.profile.supports_fp16 and config.torch_dtype == torch.float16:
                    config.torch_dtype = torch.float32
                if self.profile.supports_compile and config.quantization == "none":
                    config.use_compile = True
                # FP8 only on supported hardware
                if config.quantization == "fp8" and not self.profile.supports_fp8:
                    config.quantization = "nf4"
                return config
        return tiers[-1][1]  # lowest tier

    def best_backend(self) -> str:
        """Auto-select the best backend for current GPU."""
        vram = self.profile.vram_gb
        if vram >= 12:
            return "wan"
        elif vram >= 8:
            return "wan"
        elif vram >= 6:
            return "cogvideo"
        else:
            return "animatediff"

    def summary(self) -> str:
        p = self.profile
        lines = [
            f"GPU: {p.name}",
            f"VRAM: {p.vram_gb:.1f} GB",
            f"Device: {p.device}",
        ]
        if p.is_cuda:
            lines.append(f"Compute: {p.compute_capability[0]}.{p.compute_capability[1]}")
            lines.append(f"FP16: {p.supports_fp16} | BF16: {p.supports_bf16} | FP8: {p.supports_fp8}")
            lines.append(f"torch.compile: {p.supports_compile}")
        return "\n".join(lines)


# Singleton
_manager: Optional[VRAMManager] = None

def get_vram_manager() -> VRAMManager:
    global _manager
    if _manager is None:
        _manager = VRAMManager()
    return _manager
