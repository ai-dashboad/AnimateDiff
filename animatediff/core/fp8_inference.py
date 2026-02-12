"""
FP8 Inference Optimizer -- hardware-aware FP8 quantization for video generation.

FP8 (float8_e4m3fn) inference provides the best speed/memory/quality trade-off
for video diffusion models on compatible GPUs:
- Ada Lovelace (compute >= 8.9): RTX 4090, RTX 4080, L40
- Hopper (compute >= 9.0): H100, H200

How it works:
- Weights are stored in FP8 (1 byte per param vs 2 for FP16/BF16)
- Computation happens in higher precision (FP16/BF16) via dynamic casting
- Normalization layers are kept in full precision (sensitive to quantization)
- ~2x memory reduction with <1% quality loss on most models

This module provides two approaches:
1. Post-load quantization (torchao): Convert an already-loaded pipeline to FP8
2. Load-time quantization (diffusers PipelineQuantizationConfig): FP8 during loading

Combining FP8 with torch.compile gives up to 3.3x speedup on H100.

Hardware requirements:
- Compute capability >= 8.9 for Ada (weight-only FP8)
- Compute capability >= 9.0 for Hopper (full FP8 compute with tensor cores)

References:
- https://huggingface.co/docs/diffusers/en/quantization/torchao
- https://github.com/sayakpaul/diffusers-torchao
- https://github.com/pytorch/ao
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_compute_capability() -> tuple:
    """Return (major, minor) compute capability of the current CUDA device.

    Returns (0, 0) if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return (0, 0)
    try:
        props = torch.cuda.get_device_properties(0)
        return (props.major, props.minor)
    except Exception:
        return (0, 0)


def is_fp8_available() -> bool:
    """Check if FP8 inference is supported on the current GPU.

    FP8 requires:
    - CUDA GPU with compute capability >= 8.9 (Ada Lovelace / Hopper)
    - torchao library installed
    """
    cc = get_compute_capability()
    if cc < (8, 9):
        return False
    return _check_torchao()


def get_fp8_tier() -> str:
    """Classify the FP8 support tier for the current GPU.

    Returns:
        "hopper"  -- compute >= 9.0 (H100/H200, full FP8 tensor core support)
        "ada"     -- compute 8.9 (RTX 4090/4080, FP8 weight-only optimal)
        "none"    -- no FP8 support
    """
    cc = get_compute_capability()
    if cc >= (9, 0):
        return "hopper"
    elif cc >= (8, 9):
        return "ada"
    return "none"


# ---------------------------------------------------------------------------
# FP8 quantization schemes
# ---------------------------------------------------------------------------

# Available FP8 schemes in torchao, ordered by quality (best first)
FP8_SCHEMES = {
    # Row-wise dynamic quantization: most scales, least error, best quality
    "float8dq_e4m3_row": {
        "shorthand": "float8dq_e4m3_row",
        "description": "Row-wise FP8 dynamic quantization (best quality, needs Hopper)",
        "min_compute": (9, 0),
        "memory_reduction": "~40%",
        "quality_impact": "minimal (<0.5% degradation)",
    },
    # Tensor-wise dynamic quantization: good balance
    "float8dq": {
        "shorthand": "float8dq",
        "description": "Tensor-wise FP8 dynamic quantization (good balance)",
        "min_compute": (8, 9),
        "memory_reduction": "~45%",
        "quality_impact": "low (<1% degradation)",
    },
    # Weight-only FP8: broadest compatibility, simplest
    "float8wo": {
        "shorthand": "float8wo",
        "description": "FP8 weight-only quantization (broadest FP8 compatibility)",
        "min_compute": (8, 9),
        "memory_reduction": "~50%",
        "quality_impact": "low (<1% degradation)",
    },
}


def best_fp8_scheme() -> Optional[str]:
    """Select the best FP8 scheme for the current GPU.

    Returns the torchao shorthand string, or None if FP8 is unavailable.
    """
    tier = get_fp8_tier()
    if tier == "hopper":
        return "float8dq_e4m3_row"
    elif tier == "ada":
        return "float8dq"
    return None


# ---------------------------------------------------------------------------
# FP8InferenceOptimizer
# ---------------------------------------------------------------------------

@dataclass
class FP8Result:
    """Result from FP8 optimization."""
    optimized_components: List[str] = field(default_factory=list)
    scheme: str = ""
    optimization_time_s: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def memory_saved_mb(self) -> float:
        if self.memory_before_mb > 0 and self.memory_after_mb > 0:
            return self.memory_before_mb - self.memory_after_mb
        return 0.0

    def summary(self) -> str:
        lines = [
            f"FP8 Optimization ({self.scheme}):",
            f"  Components: {self.optimized_components}",
            f"  Time: {self.optimization_time_s:.1f}s",
        ]
        if self.memory_before_mb > 0:
            lines.append(f"  Memory: {self.memory_before_mb:.0f} MB -> {self.memory_after_mb:.0f} MB")
            lines.append(f"  Saved: {self.memory_saved_mb:.0f} MB ({self.memory_saved_mb / self.memory_before_mb * 100:.0f}%)")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)


class FP8InferenceOptimizer:
    """Enable FP8 inference on Hopper/Ada GPUs for faster video generation.

    This optimizer converts pipeline components to FP8 after they have been
    loaded, using torchao's quantization API. It automatically selects the
    best FP8 scheme based on GPU capabilities.

    For load-time FP8 quantization (during from_pretrained), use
    animatediff.core.quantization.get_quantization_config(quantization="fp8")
    instead.

    Usage:
        optimizer = FP8InferenceOptimizer()
        if optimizer.is_available:
            result = optimizer.optimize_pipeline(pipe)
            print(result.summary())
    """

    def __init__(self, device: str = "cuda", scheme: Optional[str] = None):
        """Initialize the FP8 optimizer.

        Args:
            device: Target device ("cuda"). FP8 is CUDA-only.
            scheme: Force a specific FP8 scheme (torchao shorthand).
                If None, auto-selects the best scheme for the GPU.
                Options: "float8wo", "float8dq", "float8dq_e4m3_row"
        """
        self.device = device
        self.compute_capability = get_compute_capability()
        self.tier = get_fp8_tier()
        self.is_available = is_fp8_available()

        # Select scheme
        if scheme is not None:
            if scheme not in FP8_SCHEMES:
                logger.warning(
                    f"Unknown FP8 scheme '{scheme}'. "
                    f"Available: {list(FP8_SCHEMES.keys())}. Falling back to auto."
                )
                self.scheme = best_fp8_scheme()
            else:
                self.scheme = scheme
        else:
            self.scheme = best_fp8_scheme()

        if self.is_available:
            logger.info(
                f"FP8 optimizer initialized: tier={self.tier}, "
                f"compute={self.compute_capability}, scheme={self.scheme}"
            )
        else:
            logger.info(
                f"FP8 not available: compute={self.compute_capability} "
                f"(requires >= 8.9). Using standard precision."
            )

    def optimize_pipeline(
        self,
        pipeline,
        components: Optional[List[str]] = None,
    ) -> FP8Result:
        """Convert pipeline components to FP8 inference mode.

        This applies post-load FP8 quantization using torchao. The model
        weights are quantized in-place, reducing memory and improving
        throughput on compatible hardware.

        Args:
            pipeline: A diffusers pipeline instance.
            components: Which components to quantize. Default: ["transformer"]
                (and "transformer_2" if present). The VAE should generally NOT
                be quantized -- it is sensitive to precision loss.

        Returns:
            FP8Result with optimization details.
        """
        result = FP8Result(scheme=self.scheme or "none")

        if not self.is_available:
            result.errors.append(
                f"FP8 not available (compute={self.compute_capability}, "
                f"need >= 8.9; torchao={_check_torchao()})"
            )
            logger.warning(result.errors[-1])
            return result

        if self.scheme is None:
            result.errors.append("No FP8 scheme selected")
            return result

        # Record baseline memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result.memory_before_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        # Auto-detect components
        if components is None:
            components = self._default_components(pipeline)

        t0 = time.monotonic()

        for component_name in components:
            try:
                self._quantize_component(pipeline, component_name)
                result.optimized_components.append(component_name)
            except Exception as e:
                msg = f"Failed to quantize {component_name} to FP8: {e}"
                logger.warning(msg)
                result.errors.append(msg)

        result.optimization_time_s = time.monotonic() - t0

        # Record post-optimization memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result.memory_after_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        if result.optimized_components:
            logger.info(result.summary())

        return result

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply FP8 quantization to a standalone model.

        Lower-level method for quantizing a model outside a pipeline context.

        Args:
            model: A torch.nn.Module to quantize.

        Returns:
            The quantized model (modified in-place and returned).
        """
        if not self.is_available or self.scheme is None:
            logger.warning("FP8 not available, returning model unchanged")
            return model

        return self._apply_torchao_fp8(model)

    def get_load_config(self, components: Optional[List[str]] = None):
        """Get a PipelineQuantizationConfig for FP8 load-time quantization.

        Use this when loading a pipeline with from_pretrained() to load
        weights directly in FP8, avoiding the need for post-load conversion.

        Args:
            components: Components to quantize during loading.
                Default: ["transformer"]

        Returns:
            A PipelineQuantizationConfig or None if FP8 is unavailable.
        """
        if not self.is_available or self.scheme is None:
            return None

        components = components or ["transformer"]

        try:
            from diffusers import PipelineQuantizationConfig, TorchAoConfig
        except ImportError:
            logger.warning("PipelineQuantizationConfig or TorchAoConfig not available")
            # Fall back to legacy API
            return self._get_legacy_load_config(components)

        try:
            quant_mapping = {}
            for comp in components:
                quant_mapping[comp] = TorchAoConfig(self.scheme)

            return PipelineQuantizationConfig(quant_mapping=quant_mapping)
        except Exception as e:
            logger.warning(f"Failed to create TorchAoConfig, falling back to legacy: {e}")
            return self._get_legacy_load_config(components)

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def is_fp8_available() -> bool:
        """Check if FP8 is supported on the current hardware.

        Requires compute capability >= 8.9 (Ada Lovelace/Hopper) and torchao.
        """
        return is_fp8_available()

    @staticmethod
    def get_compute_capability() -> tuple:
        """Return (major, minor) compute capability of the current GPU."""
        return get_compute_capability()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_components(self, pipeline) -> List[str]:
        """Determine which components to quantize by default.

        Quantizes transformer(s) but NOT the VAE (precision-sensitive).
        """
        components = []
        if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
            components.append("transformer")
        if hasattr(pipeline, "transformer_2") and pipeline.transformer_2 is not None:
            components.append("transformer_2")
        if hasattr(pipeline, "unet") and pipeline.unet is not None:
            components.append("unet")
        if not components:
            logger.warning("No quantizable components found in pipeline")
        return components

    def _quantize_component(self, pipeline, component_name: str):
        """Quantize a single pipeline component to FP8."""
        component = getattr(pipeline, component_name, None)
        if component is None:
            raise ValueError(f"Pipeline has no component '{component_name}'")

        logger.info(f"  Quantizing {component_name} ({type(component).__name__}) to FP8 ({self.scheme})")
        self._apply_torchao_fp8(component)

    def _apply_torchao_fp8(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply torchao FP8 quantization to a model.

        Uses torchao's quantize_ API which modifies the model in-place,
        replacing nn.Linear layers with quantized equivalents.
        """
        try:
            from torchao.quantization import quantize_, Float8WeightOnlyConfig
        except ImportError:
            raise ImportError(
                "torchao is required for FP8 quantization. "
                "Install with: pip install torchao"
            )

        # Select the appropriate config based on scheme
        quant_config = self._make_torchao_config()

        logger.info(f"  Applying torchao quantize_ with {self.scheme}")
        quantize_(model, quant_config)
        return model

    def _make_torchao_config(self):
        """Create the appropriate torchao quantization config object."""
        try:
            import torchao.quantization as taq
        except ImportError:
            raise ImportError("torchao is required for FP8 quantization")

        scheme = self.scheme

        # Map scheme names to torchao config classes
        if scheme == "float8wo" or scheme == "float8_weight_only":
            if hasattr(taq, "Float8WeightOnlyConfig"):
                return taq.Float8WeightOnlyConfig()
            # Fallback for older torchao
            return taq.float8_weight_only()

        elif scheme == "float8dq" or scheme == "float8_dynamic_activation_float8_weight":
            if hasattr(taq, "Float8DynamicActivationFloat8WeightConfig"):
                return taq.Float8DynamicActivationFloat8WeightConfig()
            if hasattr(taq, "float8_dynamic_activation_float8_weight"):
                return taq.float8_dynamic_activation_float8_weight()
            # Fallback to weight-only
            logger.warning(f"Scheme '{scheme}' not available in this torchao version, using float8wo")
            return taq.Float8WeightOnlyConfig() if hasattr(taq, "Float8WeightOnlyConfig") else taq.float8_weight_only()

        elif scheme in ("float8dq_e4m3_row", "float8dq_e4m3_tensor"):
            # Row-wise or tensor-wise e4m3 dynamic quant
            if hasattr(taq, "Float8DynamicActivationFloat8WeightConfig"):
                from torchao.float8 import Float8LinearConfig
                granularity = "row" if "row" in scheme else "tensor"
                try:
                    return taq.Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
                except TypeError:
                    return taq.Float8DynamicActivationFloat8WeightConfig()
            # Fallback
            logger.warning(f"Scheme '{scheme}' not fully supported, using float8dq")
            return self._make_fallback_config()

        else:
            logger.warning(f"Unknown FP8 scheme '{scheme}', using float8wo")
            if hasattr(taq, "Float8WeightOnlyConfig"):
                return taq.Float8WeightOnlyConfig()
            return taq.float8_weight_only()

    def _make_fallback_config(self):
        """Create a fallback FP8 config for older torchao versions."""
        import torchao.quantization as taq
        if hasattr(taq, "Float8WeightOnlyConfig"):
            return taq.Float8WeightOnlyConfig()
        return taq.float8_weight_only()

    def _get_legacy_load_config(self, components: List[str]):
        """Create a legacy PipelineQuantizationConfig for older diffusers."""
        try:
            from diffusers import PipelineQuantizationConfig
            return PipelineQuantizationConfig(
                quant_backend="torchao",
                quant_kwargs={"quant_type": "float8_e4m3fn"},
                components_to_quantize=components,
            )
        except ImportError:
            logger.warning("PipelineQuantizationConfig not available")
            return None


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

@dataclass
class FP8BenchmarkResult:
    """Result from FP8 benchmark comparison."""
    model_name: str = ""
    scheme: str = ""
    baseline_time_s: float = 0.0
    fp8_time_s: float = 0.0
    speedup: float = 1.0
    baseline_memory_mb: float = 0.0
    fp8_memory_mb: float = 0.0
    memory_reduction_pct: float = 0.0

    def summary(self) -> str:
        lines = [
            f"FP8 Benchmark: {self.model_name} ({self.scheme})",
            f"  Baseline:  {self.baseline_time_s:.2f}s, {self.baseline_memory_mb:.0f} MB",
            f"  FP8:       {self.fp8_time_s:.2f}s, {self.fp8_memory_mb:.0f} MB",
            f"  Speedup:   {self.speedup:.2f}x",
            f"  Memory:    -{self.memory_reduction_pct:.0f}%",
        ]
        return "\n".join(lines)


def benchmark_fp8(
    pipeline_factory,
    model_name: str = "unknown",
    scheme: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    num_frames: int = 16,
    num_steps: int = 5,
    num_repeats: int = 3,
) -> FP8BenchmarkResult:
    """Benchmark FP8 vs standard precision on a pipeline.

    Args:
        pipeline_factory: Callable that returns a fresh pipeline instance.
            Called twice: once for baseline, once for FP8.
        model_name: Name for reporting.
        scheme: FP8 scheme to benchmark (None = auto).
        width, height, num_frames: Generation dimensions.
        num_steps: Inference steps per run.
        num_repeats: Number of timed runs to average.

    Returns:
        FP8BenchmarkResult with timing and memory comparisons.
    """
    result = FP8BenchmarkResult(model_name=model_name)
    optimizer = FP8InferenceOptimizer(scheme=scheme)
    result.scheme = optimizer.scheme or "none"

    if not optimizer.is_available:
        result.speedup = 1.0
        logger.warning("FP8 not available, cannot benchmark")
        return result

    gen_kwargs = {
        "prompt": "a cinematic shot of a mountain landscape at sunset",
        "width": width,
        "height": height,
        "num_inference_steps": num_steps,
        "output_type": "latent",
    }

    # Check if pipeline supports num_frames
    test_pipe = pipeline_factory()
    import inspect
    try:
        sig = inspect.signature(test_pipe.__call__)
        if "num_frames" in sig.parameters:
            gen_kwargs["num_frames"] = num_frames
    except (ValueError, TypeError):
        pass
    del test_pipe

    # -- Baseline --
    logger.info(f"FP8 benchmark: baseline ({num_repeats} runs)...")
    pipe = pipeline_factory()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    baseline_times = []
    for _ in range(num_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            pipe(**gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_times.append(time.monotonic() - t0)

    result.baseline_time_s = sum(baseline_times) / len(baseline_times)
    if torch.cuda.is_available():
        result.baseline_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- FP8 --
    logger.info(f"FP8 benchmark: quantized ({num_repeats} runs)...")
    pipe = pipeline_factory()
    optimizer.optimize_pipeline(pipe)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    fp8_times = []
    for _ in range(num_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            pipe(**gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fp8_times.append(time.monotonic() - t0)

    result.fp8_time_s = sum(fp8_times) / len(fp8_times)
    if torch.cuda.is_available():
        result.fp8_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    if result.fp8_time_s > 0:
        result.speedup = result.baseline_time_s / result.fp8_time_s
    if result.baseline_memory_mb > 0:
        result.memory_reduction_pct = (
            (result.baseline_memory_mb - result.fp8_memory_mb) / result.baseline_memory_mb * 100
        )

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_torchao() -> bool:
    """Check if torchao is installed."""
    try:
        import torchao  # noqa: F401
        return True
    except ImportError:
        return False
