"""
Acceleration Manager — unified one-line acceleration for inference pipelines.

Wires together existing distillation, FP8 quantization, and torch.compile
into preset-based profiles:

  - "fast":      Max speed. LCM/Lightning distillation + FP8 + compile.
                 ~3-5x speedup, ~10-15% quality loss.
  - "balanced":  Good tradeoff. Distillation (if available) + compile.
                 ~2-3x speedup, ~5% quality loss.
  - "quality":   Full quality. No shortcuts. Optional compile for mild speedup.
                 ~1-1.5x speedup, no quality loss.

Also provides benchmarking to measure actual speedup and memory savings.

Usage::

    from animatediff.core.acceleration import AccelerationManager

    accel = AccelerationManager()
    report = accel.optimize(pipeline, preset="balanced")
    print(f"Speedup: {report.speedup_estimate}x")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["AccelerationReport", "BenchmarkResult", "AccelerationManager"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AccelerationReport:
    """Report from applying acceleration optimizations."""
    preset: str = ""
    optimizations_applied: List[str] = field(default_factory=list)
    speedup_estimate: float = 1.0  # Estimated speedup multiplier
    memory_saved_mb: float = 0.0
    quality_impact: str = "none"  # "none", "minimal", "moderate"
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        opts = ", ".join(self.optimizations_applied) or "none"
        return (
            f"preset={self.preset}, optimizations=[{opts}], "
            f"speedup~{self.speedup_estimate:.1f}x, "
            f"memory_saved={self.memory_saved_mb:.0f}MB, "
            f"quality_impact={self.quality_impact}"
        )


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    time_per_step_ms: float = 0.0
    total_time_s: float = 0.0
    memory_peak_mb: float = 0.0
    quality_score: float = 0.0
    num_frames: int = 0
    resolution: str = ""
    details: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_PRESET_CONFIGS = {
    "fast": {
        "distillation": True,
        "fp8": True,
        "compile": True,
        "compile_mode": "max-autotune",
        "quality_impact": "moderate",
        "speedup_base": 3.0,
    },
    "balanced": {
        "distillation": True,
        "fp8": False,
        "compile": True,
        "compile_mode": "reduce-overhead",
        "quality_impact": "minimal",
        "speedup_base": 2.0,
    },
    "quality": {
        "distillation": False,
        "fp8": False,
        "compile": True,
        "compile_mode": "default",
        "quality_impact": "none",
        "speedup_base": 1.2,
    },
}


# ---------------------------------------------------------------------------
# Hardware detection helpers
# ---------------------------------------------------------------------------

def _detect_gpu_capabilities() -> Dict[str, Any]:
    """Detect GPU capabilities for optimization decisions."""
    caps: Dict[str, Any] = {
        "device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "compute_capability": (0, 0),
        "gpu_name": "",
        "vram_gb": 0.0,
        "supports_fp8": False,
        "supports_compile": False,
    }

    try:
        import torch
        if torch.cuda.is_available():
            caps["device"] = "cuda"
            caps["cuda_available"] = True
            caps["gpu_name"] = torch.cuda.get_device_name(0)
            caps["vram_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            cc = torch.cuda.get_device_capability(0)
            caps["compute_capability"] = cc
            caps["supports_fp8"] = cc[0] >= 9 or (cc[0] == 8 and cc[1] >= 9)
            caps["supports_compile"] = cc[0] >= 8
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            caps["device"] = "mps"
            caps["mps_available"] = True
    except ImportError:
        pass

    return caps


# ---------------------------------------------------------------------------
# Acceleration Manager
# ---------------------------------------------------------------------------

class AccelerationManager:
    """One-line acceleration for video generation pipelines.

    Applies preset-based optimizations: distillation, FP8, torch.compile.
    Hardware-aware — skips unsupported optimizations gracefully.
    """

    def __init__(self):
        self._gpu_caps = _detect_gpu_capabilities()
        logger.info(
            f"AccelerationManager: device={self._gpu_caps['device']}, "
            f"gpu={self._gpu_caps['gpu_name'] or 'N/A'}, "
            f"vram={self._gpu_caps['vram_gb']:.1f}GB, "
            f"fp8={self._gpu_caps['supports_fp8']}, "
            f"compile={self._gpu_caps['supports_compile']}"
        )

    @property
    def device(self) -> str:
        return self._gpu_caps["device"]

    @property
    def gpu_name(self) -> str:
        return self._gpu_caps["gpu_name"]

    @property
    def vram_gb(self) -> float:
        return self._gpu_caps["vram_gb"]

    def optimize(
        self,
        pipeline: Any,
        preset: str = "balanced",
        backend_name: str = "",
    ) -> AccelerationReport:
        """Apply acceleration preset to a pipeline.

        Args:
            pipeline: A diffusers pipeline or BasePipeline instance.
            preset: "fast", "balanced", or "quality".
            backend_name: Backend name for distillation config lookup.

        Returns:
            AccelerationReport with details of what was applied.
        """
        if preset not in _PRESET_CONFIGS:
            logger.warning(f"Unknown preset '{preset}', falling back to 'balanced'")
            preset = "balanced"

        config = _PRESET_CONFIGS[preset]
        report = AccelerationReport(
            preset=preset,
            quality_impact=config["quality_impact"],
        )

        speedup = 1.0
        memory_saved = 0.0

        # 1. Distillation
        if config["distillation"]:
            result = self._apply_distillation(pipeline, backend_name)
            if result:
                report.optimizations_applied.append("distillation")
                speedup *= result.get("speedup", 1.5)
                report.details["distillation"] = result

        # 2. FP8 quantization
        if config["fp8"] and self._gpu_caps["supports_fp8"]:
            result = self._apply_fp8(pipeline)
            if result:
                report.optimizations_applied.append("fp8")
                speedup *= result.get("speedup", 1.3)
                memory_saved += result.get("memory_saved_mb", 0)
                report.details["fp8"] = result
        elif config["fp8"] and not self._gpu_caps["supports_fp8"]:
            logger.info("FP8 not supported on this GPU, skipping")

        # 3. torch.compile
        if config["compile"] and self._gpu_caps["supports_compile"]:
            result = self._apply_compile(pipeline, config["compile_mode"])
            if result:
                report.optimizations_applied.append("torch.compile")
                speedup *= result.get("speedup", 1.2)
                report.details["compile"] = result
        elif config["compile"] and not self._gpu_caps["supports_compile"]:
            logger.info("torch.compile not supported on this device, skipping")

        report.speedup_estimate = round(speedup, 1)
        report.memory_saved_mb = memory_saved

        logger.info(f"Acceleration applied: {report.summary()}")
        return report

    def _apply_distillation(
        self, pipeline: Any, backend_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply distillation (LCM/Lightning) configuration."""
        try:
            from animatediff.core.distillation import (
                DistillationManager,
                get_distillation_manager,
                recommended_distillation,
            )

            manager = get_distillation_manager()
            config = recommended_distillation(backend_name or "wan22")

            if config is None:
                logger.info(f"No distillation config available for {backend_name}")
                return None

            # Apply distillation config to the pipeline's scheduler
            if hasattr(pipeline, "scheduler"):
                result = manager.apply(pipeline, config)
                logger.info(f"Distillation applied: {config.name}")
                return {
                    "name": config.name,
                    "steps_reduction": config.steps_reduction if hasattr(config, "steps_reduction") else 0,
                    "speedup": 1.5,
                }

        except (ImportError, Exception) as e:
            logger.debug(f"Distillation not applied: {e}")

        return None

    def _apply_fp8(self, pipeline: Any) -> Optional[Dict[str, Any]]:
        """Apply FP8 quantization."""
        try:
            from animatediff.core.fp8_inference import FP8InferenceOptimizer

            optimizer = FP8InferenceOptimizer()
            if not optimizer.is_available:
                return None

            result = optimizer.optimize_pipeline(pipeline)

            if result.optimized_components:
                return {
                    "scheme": result.scheme,
                    "components": result.optimized_components,
                    "memory_saved_mb": result.memory_saved_mb,
                    "speedup": 1.3,
                }

        except (ImportError, Exception) as e:
            logger.debug(f"FP8 not applied: {e}")

        return None

    def _apply_compile(
        self, pipeline: Any, compile_mode: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply torch.compile."""
        try:
            from animatediff.core.compile import PipelineCompiler

            compiler = PipelineCompiler(mode=compile_mode)
            if not compiler.is_compile_available():
                return None

            result = compiler.compile_pipeline(pipeline)

            if result.compiled_components:
                return {
                    "mode": compile_mode,
                    "components": result.compiled_components,
                    "speedup": 1.3 if compile_mode == "max-autotune" else 1.15,
                }

        except (ImportError, Exception) as e:
            logger.debug(f"torch.compile not applied: {e}")

        return None

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        pipeline: Any,
        prompt: str = "a scenic mountain landscape at sunset",
        width: int = 512,
        height: int = 320,
        num_frames: int = 17,
        steps: int = 20,
        warmup: bool = True,
    ) -> BenchmarkResult:
        """Benchmark a pipeline's generation speed and memory usage.

        Args:
            pipeline: The pipeline to benchmark.
            prompt: Test prompt.
            width, height: Test resolution.
            num_frames: Number of test frames.
            steps: Number of inference steps.
            warmup: Whether to run a warmup pass first.

        Returns:
            BenchmarkResult with timing and memory statistics.
        """
        import torch

        gen_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "seed": 42,
        }

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        if warmup:
            logger.info("Running warmup...")
            try:
                pipeline.generate(**gen_kwargs)
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        # Timed run
        logger.info("Running benchmark...")
        t0 = time.time()
        try:
            output = pipeline.generate(**gen_kwargs)
        except Exception as e:
            logger.error(f"Benchmark generation failed: {e}")
            return BenchmarkResult()

        total_time = time.time() - t0

        # Memory stats
        peak_mem_mb = 0.0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Quality score (optional)
        quality = 0.0
        if hasattr(output, "frames") and output.frames:
            try:
                from animatediff.core.quality_scorer import VideoQualityScorer
                scorer = VideoQualityScorer(
                    backends=["aesthetic", "technical"],
                    device=self.device,
                )
                vs = scorer.score_video(output.frames, prompt)
                quality = vs.overall
            except Exception:
                pass

        result = BenchmarkResult(
            time_per_step_ms=(total_time / steps) * 1000,
            total_time_s=total_time,
            memory_peak_mb=peak_mem_mb,
            quality_score=quality,
            num_frames=num_frames,
            resolution=f"{width}x{height}",
        )

        logger.info(
            f"Benchmark: {result.total_time_s:.1f}s total, "
            f"{result.time_per_step_ms:.0f}ms/step, "
            f"peak_mem={result.memory_peak_mb:.0f}MB, "
            f"quality={result.quality_score:.3f}"
        )
        return result

    def recommend_preset(self, backend_name: str = "") -> str:
        """Recommend an acceleration preset based on hardware.

        Returns:
            "fast", "balanced", or "quality".
        """
        vram = self._gpu_caps["vram_gb"]
        device = self._gpu_caps["device"]

        if device == "cpu":
            return "quality"  # No acceleration available on CPU
        if device == "mps":
            return "quality"  # Limited acceleration on MPS

        # CUDA tiers
        if vram >= 40:
            return "quality"  # Plenty of VRAM, maximize quality
        elif vram >= 20:
            return "balanced"
        else:
            return "fast"  # Low VRAM, need all the speedup we can get
