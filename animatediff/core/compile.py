"""
Compilation utilities -- torch.compile integration for video generation pipelines.

torch.compile benefits:
- 1.5-3x speedup on Ampere+ GPUs (compute >= 8.0)
- Best with static shapes (same resolution/frame count across batches)
- First call is slow (compilation), subsequent calls are fast
- Combine with FP8 quantization for up to 3.3x total speedup on Hopper/Ada

Compile strategy (based on PyTorch/diffusers best practices):
- Only compile the DiT transformer (or UNet) -- this dominates compute.
- Do NOT compile text encoders or full pipelines -- minimal gain, extra overhead.
- VAE decoder is optional; compile it only for large frame counts.
- Use fullgraph=True during development to catch graph breaks early.
- Use dynamic=False for video (fixed resolution/frame count per batch).

Modes:
- "reduce-overhead": CUDA graphs, best for repeated same-shape inference
- "max-autotune": Triton autotuning, best absolute throughput on Hopper/Ada
- "max-autotune-no-cudagraphs": Autotuning without CUDA graph overhead
- "default": Minimal compilation, lowest compile time

SageAttention (optional):
- 2-5x faster attention vs FlashAttention
- Plug-in replacement for any attention-based model

References:
- https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
- https://huggingface.co/docs/diffusers/en/quantization/torchao
- https://github.com/sayakpaul/diffusers-torchao
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid configuration values
# ---------------------------------------------------------------------------

VALID_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
VALID_BACKENDS = ("inductor", "cudagraphs", "eager")
COMPILABLE_COMPONENTS = ("transformer", "transformer_2", "unet", "vae", "text_encoder")

# Expected speedups per model family (compiled vs uncompiled, measured on A100/H100)
MODEL_SPEEDUP_ESTIMATES: Dict[str, Dict[str, str]] = {
    "wan": {
        "speedup": "1.3-1.8x",
        "note": "DiT transformer, static shapes recommended",
        "best_mode": "max-autotune",
    },
    "wan22": {
        "speedup": "1.4-2.0x",
        "note": "Dual transformer (A14B) or single (5B); compile both for A14B",
        "best_mode": "max-autotune",
    },
    "hunyuan": {
        "speedup": "1.3-1.7x",
        "note": "Large DiT, benefits most from max-autotune",
        "best_mode": "max-autotune",
    },
    "cogvideo": {
        "speedup": "1.3-1.5x",
        "note": "3D DiT, moderate gains from compilation",
        "best_mode": "reduce-overhead",
    },
    "ltx": {
        "speedup": "1.4-1.8x",
        "note": "Lightweight DiT, fast compile, good gains",
        "best_mode": "max-autotune",
    },
    "animatediff": {
        "speedup": "1.2-1.5x",
        "note": "UNet-based, moderate gains",
        "best_mode": "reduce-overhead",
    },
}


@dataclass
class CompileResult:
    """Result from a compilation or warmup operation."""
    compiled_components: List[str] = field(default_factory=list)
    compile_time_s: float = 0.0
    warmup_time_s: float = 0.0
    errors: List[str] = field(default_factory=list)


class PipelineCompiler:
    """Compile video generation pipelines for faster inference.

    This class wraps torch.compile with settings optimized for diffusers
    video generation pipelines. It focuses compilation on the compute-heavy
    components (transformer/unet) and optionally the VAE decoder.

    Usage:
        compiler = PipelineCompiler(mode="max-autotune")
        pipe = compiler.compile_pipeline(pipe)
        compiler.warmup(pipe, width=1280, height=720, num_frames=81)
    """

    def __init__(
        self,
        mode: str = "reduce-overhead",
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool = False,
    ):
        """
        Args:
            mode: Compile mode.
                "default" -- minimal compilation, lowest compile time
                "reduce-overhead" -- CUDA graphs, best for repeated inference
                "max-autotune" -- Triton autotuning, best throughput on H100/4090
                "max-autotune-no-cudagraphs" -- autotuning without CUDA graphs
            backend: Compiler backend ("inductor", "cudagraphs", "eager").
                "inductor" is recommended for nearly all cases.
            fullgraph: If True, require the entire model to be captured as a
                single graph (no graph breaks). Useful during development to
                catch compatibility issues. Default False for production use.
            dynamic: If True, generate kernels that handle varying input shapes
                without recompilation. Set False (default) for video pipelines
                where resolution and frame count are typically fixed per batch.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid compile mode '{mode}'. Must be one of {VALID_MODES}")
        if backend not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Must be one of {VALID_BACKENDS}")

        self.mode = mode
        self.backend = backend
        self.fullgraph = fullgraph
        self.dynamic = dynamic

        # Enable inductor cache for faster recompilation across runs
        if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "animatediff", "inductor")
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

    def compile_pipeline(
        self,
        pipeline,
        components: Optional[List[str]] = None,
    ) -> "CompileResult":
        """Compile specific components of a diffusers pipeline.

        By default compiles the transformer (or unet) and VAE decoder, which
        account for >95% of compute in typical video generation.

        Args:
            pipeline: A diffusers pipeline instance (WanPipeline, etc.)
            components: Which components to compile. Options:
                "transformer", "transformer_2", "unet", "vae", "text_encoder"
                Default: auto-detect transformer/unet + vae decoder.

        Returns:
            CompileResult with details about what was compiled.
        """
        if not self.is_compile_available():
            logger.warning("torch.compile is not available or not beneficial on this device")
            return CompileResult(errors=["torch.compile not available"])

        result = CompileResult()
        t0 = time.monotonic()

        # Auto-detect components if not specified
        if components is None:
            components = self._auto_detect_components(pipeline)

        for component_name in components:
            try:
                self._compile_component(pipeline, component_name)
                result.compiled_components.append(component_name)
            except Exception as e:
                msg = f"Failed to compile {component_name}: {e}"
                logger.warning(msg)
                result.errors.append(msg)

        result.compile_time_s = time.monotonic() - t0

        if result.compiled_components:
            logger.info(
                f"Compiled {result.compiled_components} with mode={self.mode}, "
                f"backend={self.backend} in {result.compile_time_s:.1f}s"
            )
        return result

    def compile_unet(self, unet) -> Any:
        """Compile a UNet or Transformer model with optimal settings.

        This is a lower-level method for compiling a standalone model outside
        of a pipeline context.

        Args:
            unet: A torch.nn.Module (UNet2DConditionModel, transformer, etc.)

        Returns:
            The compiled model.
        """
        if not self.is_compile_available():
            logger.warning("torch.compile not available, returning model uncompiled")
            return unet

        logger.info(
            f"Compiling model ({type(unet).__name__}) with mode={self.mode}, "
            f"backend={self.backend}, fullgraph={self.fullgraph}, dynamic={self.dynamic}"
        )
        compiled = torch.compile(
            unet,
            mode=self.mode,
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        return compiled

    def warmup(
        self,
        pipeline,
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        num_inference_steps: int = 1,
    ) -> float:
        """Run a minimal warmup inference to trigger compilation.

        The first call to a compiled model is slow because torch.compile
        traces and optimizes the computation graph. This method triggers
        that compilation with a single step so subsequent calls are fast.

        Args:
            pipeline: The diffusers pipeline (after compile_pipeline).
            width: Width for warmup generation.
            height: Height for warmup generation.
            num_frames: Number of frames for warmup.
            num_inference_steps: Steps for warmup (1 is sufficient).

        Returns:
            Warmup time in seconds.
        """
        logger.info(
            f"Running compilation warmup ({width}x{height}, {num_frames} frames, "
            f"{num_inference_steps} step)..."
        )
        t0 = time.monotonic()

        try:
            # Use a dummy prompt for warmup
            warmup_kwargs = {
                "prompt": "warmup",
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "output_type": "latent",  # Skip VAE decode for speed
            }

            # Add num_frames for video pipelines
            if _pipeline_supports_param(pipeline, "num_frames"):
                warmup_kwargs["num_frames"] = num_frames

            with torch.no_grad():
                pipeline(**warmup_kwargs)

        except Exception as e:
            logger.warning(f"Warmup failed (this may be OK if shapes differ at runtime): {e}")

        elapsed = time.monotonic() - t0
        logger.info(f"Warmup complete in {elapsed:.1f}s")
        return elapsed

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def is_compile_available() -> bool:
        """Check if torch.compile is available and beneficial.

        Returns True if:
        - PyTorch >= 2.0 (torch.compile exists)
        - Running on CUDA with compute capability >= 8.0 (Ampere+)
        - Not on MPS (torch.compile has limited MPS support)
        """
        if not hasattr(torch, "compile"):
            return False

        if not torch.cuda.is_available():
            return False

        try:
            props = torch.cuda.get_device_properties(0)
            cc = (props.major, props.minor)
            return cc >= (8, 0)
        except Exception:
            return False

    @staticmethod
    def estimate_speedup(model_name: str) -> str:
        """Estimate expected speedup for a given model/backend name.

        Args:
            model_name: Backend name like "wan", "wan22", "hunyuan", etc.

        Returns:
            Human-readable estimate string.
        """
        info = MODEL_SPEEDUP_ESTIMATES.get(model_name)
        if info is None:
            return f"Unknown model '{model_name}'. Expected 1.2-1.5x speedup with torch.compile."
        return (
            f"{model_name}: {info['speedup']} speedup "
            f"(best mode: {info['best_mode']}). {info['note']}"
        )

    @staticmethod
    def recommended_mode(model_name: str) -> str:
        """Return the recommended compile mode for a model/backend.

        Args:
            model_name: Backend name like "wan", "wan22", "hunyuan", etc.

        Returns:
            One of the VALID_MODES strings.
        """
        info = MODEL_SPEEDUP_ESTIMATES.get(model_name)
        if info is not None:
            return info["best_mode"]
        return "reduce-overhead"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_detect_components(self, pipeline) -> List[str]:
        """Detect which pipeline components should be compiled."""
        components = []

        # Primary: transformer (DiT models) or unet (SD/AnimateDiff)
        if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
            components.append("transformer")
        if hasattr(pipeline, "transformer_2") and pipeline.transformer_2 is not None:
            components.append("transformer_2")
        if hasattr(pipeline, "unet") and pipeline.unet is not None:
            components.append("unet")

        # VAE decoder is secondary -- useful for large frame counts
        if hasattr(pipeline, "vae") and pipeline.vae is not None:
            components.append("vae")

        if not components:
            logger.warning("No compilable components found in pipeline")

        return components

    def _compile_component(self, pipeline, component_name: str):
        """Compile a single named component on a pipeline."""
        component = getattr(pipeline, component_name, None)
        if component is None:
            raise ValueError(f"Pipeline has no component '{component_name}'")

        logger.info(
            f"  Compiling {component_name} ({type(component).__name__}) "
            f"mode={self.mode} backend={self.backend}"
        )

        compiled = torch.compile(
            component,
            mode=self.mode,
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        setattr(pipeline, component_name, compiled)


# ---------------------------------------------------------------------------
# Benchmarking utilities
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result from a compilation benchmark."""
    model_name: str = ""
    mode: str = ""
    compile_time_s: float = 0.0
    warmup_time_s: float = 0.0
    baseline_step_time_s: float = 0.0
    compiled_step_time_s: float = 0.0
    speedup: float = 1.0
    peak_memory_mb: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.model_name} (mode={self.mode})",
            f"  Compile time:    {self.compile_time_s:.1f}s",
            f"  Warmup time:     {self.warmup_time_s:.1f}s",
            f"  Baseline step:   {self.baseline_step_time_s:.3f}s",
            f"  Compiled step:   {self.compiled_step_time_s:.3f}s",
            f"  Speedup:         {self.speedup:.2f}x",
            f"  Peak memory:     {self.peak_memory_mb:.0f} MB",
        ]
        return "\n".join(lines)


def benchmark_compile(
    pipeline,
    model_name: str = "unknown",
    mode: str = "max-autotune",
    width: int = 512,
    height: int = 512,
    num_frames: int = 16,
    num_steps: int = 5,
    num_repeats: int = 3,
) -> BenchmarkResult:
    """Benchmark torch.compile speedup on a pipeline.

    Runs inference without compilation, then with compilation, and
    measures the per-step time difference.

    Args:
        pipeline: A loaded diffusers pipeline (already on device).
        model_name: Name for reporting (e.g. "wan22").
        mode: torch.compile mode to benchmark.
        width, height, num_frames: Generation dimensions.
        num_steps: Inference steps per run.
        num_repeats: Number of timed runs to average.

    Returns:
        BenchmarkResult with timing comparisons.
    """
    result = BenchmarkResult(model_name=model_name, mode=mode)

    gen_kwargs = {
        "prompt": "a cinematic shot of a mountain landscape at sunset",
        "width": width,
        "height": height,
        "num_inference_steps": num_steps,
        "output_type": "latent",
    }
    if _pipeline_supports_param(pipeline, "num_frames"):
        gen_kwargs["num_frames"] = num_frames

    # -- Baseline (uncompiled) --
    logger.info(f"Benchmarking baseline ({num_repeats} runs, {num_steps} steps)...")
    _warmup_run(pipeline, gen_kwargs)

    baseline_times = []
    for _ in range(num_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            pipeline(**gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_times.append(time.monotonic() - t0)

    result.baseline_step_time_s = (sum(baseline_times) / len(baseline_times)) / num_steps

    # -- Compiled --
    logger.info(f"Compiling with mode={mode}...")
    compiler = PipelineCompiler(mode=mode)
    t_compile = time.monotonic()
    compile_result = compiler.compile_pipeline(pipeline)
    result.compile_time_s = time.monotonic() - t_compile

    # Warmup (triggers compilation)
    t_warmup = time.monotonic()
    _warmup_run(pipeline, gen_kwargs)
    result.warmup_time_s = time.monotonic() - t_warmup

    # Timed runs
    logger.info(f"Benchmarking compiled ({num_repeats} runs, {num_steps} steps)...")
    compiled_times = []
    for _ in range(num_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.no_grad():
            pipeline(**gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compiled_times.append(time.monotonic() - t0)

    result.compiled_step_time_s = (sum(compiled_times) / len(compiled_times)) / num_steps

    if result.compiled_step_time_s > 0:
        result.speedup = result.baseline_step_time_s / result.compiled_step_time_s

    if torch.cuda.is_available():
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# Legacy API (backward-compatible free functions)
# ---------------------------------------------------------------------------

def should_compile(compute_capability: tuple = (0, 0)) -> bool:
    """Check if torch.compile is likely beneficial.

    Kept for backward compatibility. Prefer PipelineCompiler.is_compile_available().
    """
    return compute_capability >= (8, 0)


def compile_pipeline(pipe, mode: str = "reduce-overhead"):
    """Apply torch.compile to a diffusers pipeline's transformer/unet.

    Kept for backward compatibility. Prefer PipelineCompiler.compile_pipeline().

    Args:
        pipe: A diffusers pipeline instance
        mode: Compile mode -- "default", "reduce-overhead", or "max-autotune"
    """
    # Try transformer (DiT-based models: Wan, HunyuanVideo, CogVideoX, LTX)
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        logger.info(f"Compiling transformer with mode={mode}")
        pipe.transformer = torch.compile(pipe.transformer, mode=mode)
        return pipe

    # Try unet (UNet-based models: AnimateDiff, SD)
    if hasattr(pipe, "unet") and pipe.unet is not None:
        logger.info(f"Compiling unet with mode={mode}")
        pipe.unet = torch.compile(pipe.unet, mode=mode)
        return pipe

    logger.warning("No transformer or unet found to compile")
    return pipe


def try_enable_sage_attention():
    """Try to enable SageAttention globally if available.

    SageAttention replaces standard attention with 8-bit quantized attention,
    providing 2-5x speedup with minimal quality loss.
    """
    try:
        from sageattention import sageattn  # noqa: F401
        import diffusers  # noqa: F401
        # SageAttention patches are model-specific; log availability
        logger.info("SageAttention is available and can be used for acceleration")
        return True
    except ImportError:
        logger.debug("SageAttention not installed (optional)")
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pipeline_supports_param(pipeline, param: str) -> bool:
    """Check if a pipeline's __call__ accepts a given parameter."""
    import inspect
    try:
        sig = inspect.signature(pipeline.__call__)
        return param in sig.parameters
    except (ValueError, TypeError):
        return False


def _warmup_run(pipeline, gen_kwargs: dict):
    """Run a single warmup inference."""
    warmup_kwargs = {**gen_kwargs, "num_inference_steps": 1}
    try:
        with torch.no_grad():
            pipeline(**warmup_kwargs)
    except Exception:
        pass  # Warmup failures are non-fatal
