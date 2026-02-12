"""
Video generation backends — unified access to multiple model families.

Available backends:
- wan: Wan 2.1 (Alibaba, 1.3B-14B, best quality-to-VRAM ratio)
- wan22: Wan 2.2 (MoE dual-path, LoRA support, TI2V-5B for MPS)
- wan22_animate: Wan 2.2 Animate (character animation + replacement)
- wan22_s2v: Wan 2.2 S2V (audio-driven cinematic video, lip sync)
- wan22_vace: Wan VACE (reference-to-video, continuation, inpainting)
- skyreels_v3: SkyReels-V3 (Skywork, 14B, multi-reference character consistency)
- hunyuan: HunyuanVideo (Tencent, 8.3B, high quality)
- cogvideo: CogVideoX (THU, 2B-5B, lightest)
- ltx: LTX-Video (Lightricks, real-time capable)
- ltx2: LTX-2 (Lightricks, 19B, joint audio-video generation)
- animatediff: AnimateDiff legacy (SD1.5/SDXL/Lightning)
"""

from typing import Dict, Type

BACKEND_REGISTRY: Dict[str, str] = {
    "wan": "animatediff.backends.wan.WanBackend",
    "wan22": "animatediff.backends.wan22.Wan22Backend",
    "wan22_animate": "animatediff.backends.wan22_animate.Wan22AnimateBackend",
    "wan22_s2v": "animatediff.backends.wan22_s2v.Wan22S2VBackend",
    "wan22_vace": "animatediff.backends.wan22_vace.Wan22VACEBackend",
    "skyreels_v3": "animatediff.backends.skyreels_v3.SkyReelsV3Backend",
    "hunyuan": "animatediff.backends.hunyuan.HunyuanBackend",
    "cogvideo": "animatediff.backends.cogvideo.CogVideoBackend",
    "ltx": "animatediff.backends.ltx.LTXBackend",
    "ltx2": "animatediff.backends.ltx2.LTX2Backend",
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
