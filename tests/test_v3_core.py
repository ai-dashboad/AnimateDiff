"""Tests for V3 core modules: VRAMManager, BasePipeline, VideoOutput, quantization, compile."""
import pytest
import torch
from unittest.mock import patch, MagicMock
from dataclasses import asdict


# ============================================================================
# VideoOutput
# ============================================================================

class TestVideoOutput:
    def test_creation(self):
        from animatediff.core.base_pipeline import VideoOutput
        vo = VideoOutput(frames=["frame1", "frame2"], fps=16, seed=42, backend="wan")
        assert vo.frames == ["frame1", "frame2"]
        assert vo.fps == 16
        assert vo.seed == 42
        assert vo.backend == "wan"
        assert vo.metadata == {}

    def test_defaults(self):
        from animatediff.core.base_pipeline import VideoOutput
        vo = VideoOutput(frames=[])
        assert vo.fps == 8
        assert vo.seed == -1
        assert vo.backend == ""

    def test_metadata(self):
        from animatediff.core.base_pipeline import VideoOutput
        vo = VideoOutput(frames=[], metadata={"model_variant": "1.3B"})
        assert vo.metadata["model_variant"] == "1.3B"


# ============================================================================
# GPUProfile & InferenceConfig
# ============================================================================

class TestGPUProfile:
    def test_defaults(self):
        from animatediff.core.vram_manager import GPUProfile
        p = GPUProfile()
        assert p.name == "cpu"
        assert p.vram_gb == 0.0
        assert p.device == "cpu"
        assert not p.is_cuda
        assert not p.is_mps
        assert not p.supports_fp16

    def test_cuda_profile(self):
        from animatediff.core.vram_manager import GPUProfile
        p = GPUProfile(
            name="RTX 4090",
            vram_gb=24.0,
            compute_capability=(8, 9),
            device="cuda",
            is_cuda=True,
            supports_fp16=True,
            supports_bf16=True,
            supports_fp8=True,
            supports_compile=True,
        )
        assert p.vram_gb == 24.0
        assert p.supports_fp8
        assert p.supports_compile


class TestInferenceConfig:
    def test_defaults(self):
        from animatediff.core.vram_manager import InferenceConfig
        ic = InferenceConfig()
        assert ic.model_variant == ""
        assert ic.quantization == "none"
        assert ic.torch_dtype == torch.float16
        assert ic.offload_strategy == "none"
        assert ic.enable_vae_slicing
        assert not ic.enable_vae_tiling
        assert ic.max_width == 512
        assert ic.max_height == 512
        assert ic.max_frames == 16
        assert not ic.use_compile


# ============================================================================
# VRAMManager
# ============================================================================

class TestVRAMManager:
    def test_cpu_fallback(self):
        """On a CPU-only machine, VRAMManager should work and return cpu profile."""
        from animatediff.core.vram_manager import VRAMManager
        vm = VRAMManager()
        assert vm.profile is not None
        assert vm.profile.device in ("cpu", "mps", "cuda")

    def test_summary_not_empty(self):
        from animatediff.core.vram_manager import VRAMManager
        vm = VRAMManager()
        s = vm.summary()
        assert "GPU:" in s
        assert "VRAM:" in s

    def test_recommend_returns_config(self):
        from animatediff.core.vram_manager import VRAMManager
        vm = VRAMManager()
        for backend in ["wan", "hunyuan", "cogvideo", "ltx", "animatediff"]:
            rec = vm.recommend(backend)
            assert rec.model_variant != ""
            assert rec.quantization in ("none", "nf4", "int8", "fp8")
            assert rec.offload_strategy in ("none", "model_cpu", "sequential_cpu")

    def test_recommend_unknown_backend_defaults(self):
        from animatediff.core.vram_manager import VRAMManager
        vm = VRAMManager()
        rec = vm.recommend("unknown_backend")
        # Should fall back to ANIMATEDIFF_TIERS
        assert rec.model_variant == "sd15"

    def test_best_backend_returns_valid(self):
        from animatediff.core.vram_manager import VRAMManager
        vm = VRAMManager()
        best = vm.best_backend()
        assert best in ("wan", "cogvideo", "animatediff")

    def test_get_vram_manager_singleton(self):
        from animatediff.core.vram_manager import get_vram_manager
        import animatediff.core.vram_manager as vm_mod
        vm_mod._manager = None  # reset singleton
        m1 = get_vram_manager()
        m2 = get_vram_manager()
        assert m1 is m2
        vm_mod._manager = None  # cleanup

    def test_tier_ordering(self):
        """VRAM tiers should be in descending order of min_vram."""
        from animatediff.core.vram_manager import BACKEND_TIERS
        for backend, tiers in BACKEND_TIERS.items():
            vrams = [t[0] for t in tiers]
            assert vrams == sorted(vrams, reverse=True), f"{backend} tiers not in descending VRAM order"

    @patch("torch.cuda.is_available", return_value=False)
    def test_mocked_cpu_detection(self, mock_cuda):
        """With CUDA unavailable, falls back to MPS or CPU."""
        from animatediff.core.vram_manager import VRAMManager
        vm = VRAMManager()
        assert vm.profile.device in ("cpu", "mps")

    def test_recommend_high_vram(self):
        """Simulate a 24GB GPU recommending the top tier."""
        from animatediff.core.vram_manager import VRAMManager, GPUProfile
        vm = VRAMManager()
        vm.profile = GPUProfile(
            name="RTX 4090", vram_gb=24.0, compute_capability=(8, 9),
            device="cuda", is_cuda=True, supports_fp16=True,
            supports_bf16=True, supports_fp8=True, supports_compile=True,
        )
        rec = vm.recommend("wan")
        assert rec.model_variant == "14B"
        assert rec.quantization == "none"
        assert rec.max_width == 720
        assert rec.use_compile  # should be enabled for high VRAM + Ampere+

    def test_recommend_low_vram(self):
        """Simulate a 6GB GPU."""
        from animatediff.core.vram_manager import VRAMManager, GPUProfile
        vm = VRAMManager()
        vm.profile = GPUProfile(
            name="RTX 3060", vram_gb=6.0, compute_capability=(8, 6),
            device="cuda", is_cuda=True, supports_fp16=True,
            supports_bf16=True, supports_compile=True,
        )
        rec = vm.recommend("wan")
        assert rec.model_variant == "1.3B"
        assert rec.quantization == "nf4"
        assert rec.offload_strategy == "model_cpu"


# ============================================================================
# BasePipeline
# ============================================================================

class TestBasePipeline:
    def test_cannot_instantiate(self):
        from animatediff.core.base_pipeline import BasePipeline
        with pytest.raises(TypeError):
            BasePipeline()

    def test_make_generator_positive_seed(self):
        from animatediff.core.base_pipeline import BasePipeline
        # Create a concrete subclass to test protected methods
        class DummyPipeline(BasePipeline):
            @classmethod
            def load(cls, **kw): pass
            def generate(self, prompt, **kw): pass

        dp = DummyPipeline.__new__(DummyPipeline)
        gen = dp._make_generator(42, "cpu")
        assert gen is not None
        assert isinstance(gen, torch.Generator)

    def test_make_generator_negative_seed(self):
        from animatediff.core.base_pipeline import BasePipeline
        class DummyPipeline(BasePipeline):
            @classmethod
            def load(cls, **kw): pass
            def generate(self, prompt, **kw): pass
        dp = DummyPipeline.__new__(DummyPipeline)
        gen = dp._make_generator(-1, "cpu")
        assert gen is None

    def test_apply_offloading_model_cpu(self):
        from animatediff.core.base_pipeline import BasePipeline
        class DummyPipeline(BasePipeline):
            @classmethod
            def load(cls, **kw): pass
            def generate(self, prompt, **kw): pass
        dp = DummyPipeline.__new__(DummyPipeline)
        mock_pipe = MagicMock()
        dp._apply_offloading(mock_pipe, "model_cpu")
        mock_pipe.enable_model_cpu_offload.assert_called_once()

    def test_apply_offloading_sequential(self):
        from animatediff.core.base_pipeline import BasePipeline
        class DummyPipeline(BasePipeline):
            @classmethod
            def load(cls, **kw): pass
            def generate(self, prompt, **kw): pass
        dp = DummyPipeline.__new__(DummyPipeline)
        mock_pipe = MagicMock()
        dp._apply_offloading(mock_pipe, "sequential_cpu")
        mock_pipe.enable_sequential_cpu_offload.assert_called_once()

    def test_apply_vae_opts(self):
        from animatediff.core.base_pipeline import BasePipeline
        class DummyPipeline(BasePipeline):
            @classmethod
            def load(cls, **kw): pass
            def generate(self, prompt, **kw): pass
        dp = DummyPipeline.__new__(DummyPipeline)
        mock_pipe = MagicMock()
        dp._apply_vae_opts(mock_pipe, slicing=True, tiling=True)
        mock_pipe.enable_vae_slicing.assert_called_once()
        mock_pipe.enable_vae_tiling.assert_called_once()

    def test_save_mp4(self, tmp_path):
        """Test save dispatches to export_to_video for .mp4."""
        from animatediff.core.base_pipeline import BasePipeline, VideoOutput
        class DummyPipeline(BasePipeline):
            @classmethod
            def load(cls, **kw): pass
            def generate(self, prompt, **kw): pass
        dp = DummyPipeline.__new__(DummyPipeline)

        from PIL import Image
        frames = [Image.new("RGB", (64, 64), "red") for _ in range(4)]
        vo = VideoOutput(frames=frames, backend="test")
        path = str(tmp_path / "test.mp4")
        dp.save(vo, path, fps=8)
        assert os.path.exists(path)

    def test_save_gif(self, tmp_path):
        from animatediff.core.base_pipeline import BasePipeline, VideoOutput
        class DummyPipeline(BasePipeline):
            @classmethod
            def load(cls, **kw): pass
            def generate(self, prompt, **kw): pass
        dp = DummyPipeline.__new__(DummyPipeline)

        from PIL import Image
        frames = [Image.new("RGB", (64, 64), "blue") for _ in range(4)]
        vo = VideoOutput(frames=frames, backend="test")
        path = str(tmp_path / "test.gif")
        dp.save(vo, path, fps=8)
        assert os.path.exists(path)


import os


# ============================================================================
# Quantization
# ============================================================================

class TestQuantization:
    def test_get_quantization_config_none(self):
        from animatediff.core.quantization import get_quantization_config
        assert get_quantization_config("none") is None

    def test_best_quantization_high_vram(self):
        from animatediff.core.quantization import best_quantization
        assert best_quantization(24.0) == "none"

    def test_best_quantization_no_backends(self):
        from animatediff.core.quantization import best_quantization
        with patch("animatediff.core.quantization.check_torchao", return_value=False), \
             patch("animatediff.core.quantization.check_bitsandbytes", return_value=False):
            assert best_quantization(8.0) == "none"

    def test_best_quantization_fp8(self):
        from animatediff.core.quantization import best_quantization
        with patch("animatediff.core.quantization.check_torchao", return_value=True):
            result = best_quantization(12.0, compute_capability=(8, 9))
            assert result == "fp8"

    def test_best_quantization_nf4(self):
        from animatediff.core.quantization import best_quantization
        with patch("animatediff.core.quantization.check_torchao", return_value=False), \
             patch("animatediff.core.quantization.check_bitsandbytes", return_value=True):
            result = best_quantization(12.0, compute_capability=(7, 5))
            assert result == "nf4"

    def test_get_quantization_config_unknown(self):
        from animatediff.core.quantization import get_quantization_config
        result = get_quantization_config("unknown_quant")
        assert result is None

    def test_check_bitsandbytes(self):
        from animatediff.core.quantization import check_bitsandbytes
        # Should return bool regardless of whether it's installed
        assert isinstance(check_bitsandbytes(), bool)

    def test_check_torchao(self):
        from animatediff.core.quantization import check_torchao
        assert isinstance(check_torchao(), bool)


# ============================================================================
# Compile
# ============================================================================

class TestCompile:
    def test_should_compile_ampere(self):
        from animatediff.core.compile import should_compile
        assert should_compile((8, 0)) is True
        assert should_compile((8, 6)) is True
        assert should_compile((8, 9)) is True
        assert should_compile((9, 0)) is True

    def test_should_not_compile_pre_ampere(self):
        from animatediff.core.compile import should_compile
        assert should_compile((7, 5)) is False
        assert should_compile((0, 0)) is False

    def test_compile_pipeline_with_transformer(self):
        from animatediff.core.compile import compile_pipeline
        mock_pipe = MagicMock()
        transformer = MagicMock()
        mock_pipe.transformer = transformer
        mock_pipe.unet = None
        with patch("torch.compile", return_value=MagicMock()) as mock_compile:
            result = compile_pipeline(mock_pipe, mode="reduce-overhead")
            mock_compile.assert_called_once_with(transformer, mode="reduce-overhead")
            assert result is mock_pipe

    def test_compile_pipeline_with_unet(self):
        from animatediff.core.compile import compile_pipeline
        mock_pipe = MagicMock()
        mock_pipe.transformer = None
        unet = MagicMock()
        mock_pipe.unet = unet
        with patch("torch.compile", return_value=MagicMock()) as mock_compile:
            result = compile_pipeline(mock_pipe, mode="default")
            mock_compile.assert_called_once_with(unet, mode="default")

    def test_compile_pipeline_no_model(self):
        from animatediff.core.compile import compile_pipeline
        mock_pipe = MagicMock(spec=[])  # no transformer or unet
        result = compile_pipeline(mock_pipe)
        assert result is mock_pipe

    def test_try_enable_sage_attention(self):
        from animatediff.core.compile import try_enable_sage_attention
        # SageAttention likely not installed; should return False gracefully
        result = try_enable_sage_attention()
        assert isinstance(result, bool)
