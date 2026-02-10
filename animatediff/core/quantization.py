"""
Quantization utilities — unified interface for model quantization.

Supports:
- bitsandbytes NF4 / INT8 (broadest compatibility)
- torchao FP8 (requires compute >= 8.9)
- Automatic selection based on hardware
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def check_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def check_torchao() -> bool:
    try:
        import torchao  # noqa: F401
        return True
    except ImportError:
        return False


def best_quantization(vram_gb: float, compute_capability: tuple = (0, 0)) -> str:
    """Choose the best quantization strategy for given hardware.

    Returns one of: "none", "fp8", "nf4", "int8"
    """
    if vram_gb >= 20:
        return "none"

    # FP8 is fastest but needs compute >= 8.9 (RTX 4090, H100)
    if compute_capability >= (8, 9) and check_torchao():
        return "fp8"

    # NF4 gives best compression (4x), works everywhere
    if check_bitsandbytes():
        return "nf4"

    # INT8 as fallback
    if check_bitsandbytes():
        return "int8"

    logger.warning("No quantization backend available. Install bitsandbytes or torchao.")
    return "none"


def get_quantization_config(quantization: str, components: Optional[list] = None):
    """Build a PipelineQuantizationConfig for diffusers.

    Args:
        quantization: One of "none", "nf4", "int8", "fp8"
        components: List of components to quantize, e.g. ["transformer"]

    Returns:
        A PipelineQuantizationConfig or None
    """
    if quantization == "none":
        return None

    components = components or ["transformer"]

    try:
        from diffusers import PipelineQuantizationConfig
    except ImportError:
        logger.warning("PipelineQuantizationConfig not available. Upgrade diffusers>=0.31")
        return None

    if quantization == "nf4":
        if not check_bitsandbytes():
            logger.warning("bitsandbytes not installed, cannot use NF4 quantization")
            return None
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=components,
        )
    elif quantization == "int8":
        if not check_bitsandbytes():
            logger.warning("bitsandbytes not installed, cannot use INT8 quantization")
            return None
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=components,
        )
    elif quantization == "fp8":
        if not check_torchao():
            logger.warning("torchao not installed, cannot use FP8 quantization")
            return None
        return PipelineQuantizationConfig(
            quant_backend="torchao",
            quant_kwargs={"quant_type": "float8_e4m3fn"},
            components_to_quantize=components,
        )
    else:
        logger.warning(f"Unknown quantization type: {quantization}")
        return None
