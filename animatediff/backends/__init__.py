"""
Video generation backends — unified access to multiple model families.

Available backends:
- wan: Wan 2.1 (Alibaba, 1.3B-14B, best quality-to-VRAM ratio)
- hunyuan: HunyuanVideo (Tencent, 8.3B, high quality)
- cogvideo: CogVideoX (THU, 2B-5B, lightest)
- ltx: LTX-Video (Lightricks, real-time capable)
- animatediff: AnimateDiff legacy (SD1.5/SDXL/Lightning)
"""

from typing import Dict, Type

BACKEND_REGISTRY: Dict[str, str] = {
    "wan": "animatediff.backends.wan.WanBackend",
    "hunyuan": "animatediff.backends.hunyuan.HunyuanBackend",
    "cogvideo": "animatediff.backends.cogvideo.CogVideoBackend",
    "ltx": "animatediff.backends.ltx.LTXBackend",
    "animatediff": "animatediff.backends.animatediff_legacy.AnimateDiffBackend",
}


def get_backend(name: str):
    """Lazily import and return a backend class by name."""
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}")

    module_path, class_name = BACKEND_REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def list_backends():
    return list(BACKEND_REGISTRY.keys())
