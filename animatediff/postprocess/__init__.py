"""
Post-processing pipeline for video generation.

Modules:
- interpolation: RIFE frame interpolation for smoother motion
- upscale: Real-ESRGAN anime super-resolution
- audio: F5-TTS voice generation + BGM + per-character voice cloning
- compositor: shot assembly, transitions, audio mixing
- deflicker: temporal flicker removal within and across shots
- beat_sync: beat detection and shot-to-music alignment (requires librosa)
- mtv_sync: multi-track audio separation + frame-level video sync
- video_extend: extend short video clips to longer duration
- style_harmonize: cross-shot visual consistency
- lip_sync: multi-backend lip synchronization (MuseTalk, Wav2Lip, SadTalker, Hallo2)
- phoneme_sync: phoneme-level lip-sync alignment (requires whisper)
"""

from animatediff.postprocess.interpolation import FrameInterpolator
from animatediff.postprocess.upscale import VideoUpscaler
from animatediff.postprocess.audio import (
    AudioGenerator,
    VoiceProfile,
    generate_narration_with_voice,
    normalize_audio,
    add_reverb,
    crossfade_audio,
    concatenate_audio,
    get_audio_duration,
    prepare_reference_audio,
)
from animatediff.postprocess.compositor import VideoCompositor
from animatediff.postprocess.lipsync import LipSyncProcessor, apply_lipsync_to_shots
from animatediff.postprocess.lip_sync import LipSyncer, FaceDetector
from animatediff.postprocess.deflicker import VideoDeflicker
from animatediff.postprocess.beat_sync import (
    BeatAnalyzer,
    ShotBeatAligner,
    BeatInfo,
    generate_beat_synced_shots,
    energy_matched_transitions,
)
from animatediff.postprocess.mtv_sync import (
    AudioStreamAnalyzer,
    VideoAudioSync,
    AudioStreams,
    TimeSegment,
    SyncMap,
)
from animatediff.postprocess.video_extend import VideoExtender
from animatediff.postprocess.style_harmonize import StyleHarmonizer, StyleDescriptor
from animatediff.postprocess.phoneme_sync import PhonemeAligner, PhonemeTimeline, Viseme
