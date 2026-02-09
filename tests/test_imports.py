"""Test that all core modules can be imported without error."""
import pytest


def test_import_motion_module():
    from animatediff.models.motion_module import VanillaTemporalModule, TemporalTransformer3DModel

def test_import_unet():
    from animatediff.models.unet import UNet3DConditionModel

def test_import_attention():
    from animatediff.models.attention import Transformer3DModel, BasicTransformerBlock

def test_import_pipeline():
    from animatediff.pipelines.pipeline_animation import AnimationPipeline

def test_import_sparse_controlnet():
    from animatediff.models.sparse_controlnet import SparseControlNetModel

def test_import_util():
    from animatediff.utils.util import save_videos_grid, load_weights, auto_download

def test_import_dataset():
    from animatediff.data.dataset import WebVid10M

def test_import_diffusers_compat():
    """Verify diffusers compatibility imports resolve correctly."""
    try:
        from diffusers.models.attention import CrossAttention, FeedForward
    except ImportError:
        from diffusers.models.attention_processor import Attention as CrossAttention
        from diffusers.models.attention import FeedForward

    try:
        from diffusers.modeling_utils import ModelMixin
    except ImportError:
        from diffusers.models.modeling_utils import ModelMixin

    try:
        from diffusers.pipeline_utils import DiffusionPipeline
    except ImportError:
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    assert CrossAttention is not None
    assert FeedForward is not None
    assert ModelMixin is not None
    assert DiffusionPipeline is not None
