"""
Post-processing pipeline for video generation.

Modules:
- interpolation: RIFE frame interpolation for smoother motion
- upscale: Real-ESRGAN anime super-resolution
- audio: F5-TTS voice generation + BGM
- compositor: shot assembly, transitions, audio mixing
"""

from animatediff.postprocess.interpolation import FrameInterpolator
from animatediff.postprocess.upscale import VideoUpscaler
from animatediff.postprocess.audio import AudioGenerator, VoiceProfile
from animatediff.postprocess.compositor import VideoCompositor
from animatediff.postprocess.lipsync import LipSyncProcessor, apply_lipsync_to_shots
