"""Tests for V3 backend registry and backend class structure."""
import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Backend Registry
# ============================================================================

class TestBackendRegistry:
    def test_list_backends(self):
        from animatediff.backends import list_backends
        backends = list_backends()
        assert "wan" in backends
        assert "hunyuan" in backends
        assert "cogvideo" in backends
        assert "ltx" in backends
        assert "animatediff" in backends

    def test_registry_has_seven_backends(self):
        from animatediff.backends import BACKEND_REGISTRY
        assert len(BACKEND_REGISTRY) == 7

    def test_get_backend_wan(self):
        from animatediff.backends import get_backend
        cls = get_backend("wan")
        assert cls.__name__ == "WanBackend"

    def test_get_backend_hunyuan(self):
        from animatediff.backends import get_backend
        cls = get_backend("hunyuan")
        assert cls.__name__ == "HunyuanBackend"

    def test_get_backend_cogvideo(self):
        from animatediff.backends import get_backend
        cls = get_backend("cogvideo")
        assert cls.__name__ == "CogVideoBackend"

    def test_get_backend_ltx(self):
        from animatediff.backends import get_backend
        cls = get_backend("ltx")
        assert cls.__name__ == "LTXBackend"

    def test_get_backend_animatediff(self):
        from animatediff.backends import get_backend
        cls = get_backend("animatediff")
        assert cls.__name__ == "AnimateDiffBackend"

    def test_get_backend_unknown_raises(self):
        from animatediff.backends import get_backend
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")


# ============================================================================
# Backend Class Structure
# ============================================================================

class TestBackendStructure:
    """Verify all backends inherit from BasePipeline and have required attributes."""

    @pytest.mark.parametrize("backend_name", ["wan", "hunyuan", "cogvideo", "ltx", "animatediff"])
    def test_inherits_base_pipeline(self, backend_name):
        from animatediff.backends import get_backend
        from animatediff.core.base_pipeline import BasePipeline
        cls = get_backend(backend_name)
        assert issubclass(cls, BasePipeline)

    @pytest.mark.parametrize("backend_name", ["wan", "hunyuan", "cogvideo", "ltx", "animatediff"])
    def test_has_backend_name(self, backend_name):
        from animatediff.backends import get_backend
        cls = get_backend(backend_name)
        assert hasattr(cls, "backend_name")
        assert isinstance(cls.backend_name, str)
        assert len(cls.backend_name) > 0

    @pytest.mark.parametrize("backend_name", ["wan", "hunyuan", "cogvideo", "ltx", "animatediff"])
    def test_has_load_method(self, backend_name):
        from animatediff.backends import get_backend
        cls = get_backend(backend_name)
        assert hasattr(cls, "load")
        assert callable(cls.load)

    @pytest.mark.parametrize("backend_name", ["wan", "hunyuan", "cogvideo", "ltx", "animatediff"])
    def test_has_generate_method(self, backend_name):
        from animatediff.backends import get_backend
        cls = get_backend(backend_name)
        assert hasattr(cls, "generate")
        assert callable(cls.generate)

    @pytest.mark.parametrize("backend_name", ["wan", "hunyuan", "cogvideo", "ltx", "animatediff"])
    def test_has_save_method(self, backend_name):
        from animatediff.backends import get_backend
        cls = get_backend(backend_name)
        assert hasattr(cls, "save")
        assert callable(cls.save)


# ============================================================================
# Wan Backend specifics
# ============================================================================

class TestWanBackend:
    def test_model_registry(self):
        from animatediff.backends.wan import WAN_MODELS, WAN_I2V_MODELS
        assert "1.3B" in WAN_MODELS
        assert "14B" in WAN_MODELS
        assert "14B" in WAN_I2V_MODELS

    def test_backend_name(self):
        from animatediff.backends.wan import WanBackend
        assert WanBackend.backend_name == "wan"


# ============================================================================
# HunyuanVideo Backend specifics
# ============================================================================

class TestHunyuanBackend:
    def test_model_registry(self):
        from animatediff.backends.hunyuan import HUNYUAN_MODELS
        assert "default" in HUNYUAN_MODELS

    def test_backend_name(self):
        from animatediff.backends.hunyuan import HunyuanBackend
        assert HunyuanBackend.backend_name == "hunyuan"


# ============================================================================
# CogVideoX Backend specifics
# ============================================================================

class TestCogVideoBackend:
    def test_model_registry(self):
        from animatediff.backends.cogvideo import COGVIDEO_MODELS
        assert "2B" in COGVIDEO_MODELS
        assert "5B" in COGVIDEO_MODELS

    def test_backend_name(self):
        from animatediff.backends.cogvideo import CogVideoBackend
        assert CogVideoBackend.backend_name == "cogvideo"


# ============================================================================
# LTX Backend specifics
# ============================================================================

class TestLTXBackend:
    def test_model_registry(self):
        from animatediff.backends.ltx import LTX_MODELS
        assert "default" in LTX_MODELS

    def test_backend_name(self):
        from animatediff.backends.ltx import LTXBackend
        assert LTXBackend.backend_name == "ltx"


# ============================================================================
# AnimateDiff Legacy Backend specifics
# ============================================================================

class TestAnimateDiffBackend:
    def test_backend_name(self):
        from animatediff.backends.animatediff_legacy import AnimateDiffBackend
        assert AnimateDiffBackend.backend_name == "animatediff"
