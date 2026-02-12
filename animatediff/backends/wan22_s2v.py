"""
Wan 2.2 S2V Backend -- audio-driven cinematic video generation.

Model: Wan-AI/Wan2.2-S2V-14B (27B total / 14B active MoE)
Audio encoder: wav2vec2-large-xlsr-53-english (bundled in HF repo)

Architecture:
- MoE dual-transformer: high-noise (layout) + low-noise (detail) experts
- 40 layers, 40 heads, dim=5120
- Audio injected at 12 transformer layers via cross-attention
- Wav2Vec features extracted at 50 fps, interpolated to video rate, bucketed per frame

Inputs: text prompt + reference image + audio file (WAV/MP3/FLAC)
Optional: pose_video for body-motion guidance

NOTE: The diffusers pipeline (WanSpeechToVideoPipeline) has not been merged yet
      as of diffusers 0.36.0. This backend provides a standalone implementation
      that wraps the original Wan2.2 S2V inference logic and will automatically
      switch to the diffusers pipeline once it becomes available.

NOTE: S2V-14B uses FP8 MoE experts internally. MPS (Apple Silicon) is NOT
      supported -- CUDA only.
"""

import logging
import math
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from animatediff.core.base_pipeline import BasePipeline, VideoOutput
from animatediff.core.quantization import get_quantization_config

logger = logging.getLogger(__name__)

# Default HuggingFace model repo for S2V
WAN22_S2V_MODEL = "Wan-AI/Wan2.2-S2V-14B"

# Audio encoder bundled inside the S2V model repo
WAV2VEC_SUBFOLDER = "wav2vec2-large-xlsr-53-english"

# S2V generation defaults (from the Wan2.2 reference config)
S2V_DEFAULTS = dict(
    width=1024,
    height=704,
    num_frames=81,
    fps=16,
    guidance_scale=4.5,
    steps=40,
    shift=3.0,
    infer_frames=80,         # frames generated per clip (num_frames = infer_frames + 1)
    motion_frames=5,         # temporal context frames for motion continuity
    audio_sample_rate=16000,
)

# S2V transformer config (from config.json in the HF repo)
S2V_MODEL_CONFIG = dict(
    dim=5120,
    ffn_dim=13824,
    num_heads=40,
    num_layers=40,
    audio_dim=1024,
    num_audio_token=4,
    audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
    enable_adain=True,
    adain_mode="attn_norm",
    enable_framepack=True,
    motion_token_num=1024,
    text_len=512,
    in_dim=16,
    out_dim=16,
    cond_dim=16,
    freq_dim=256,
)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and resample to target_sr.

    Tries torchaudio first (no extra deps if torch is installed),
    falls back to librosa, then to ffmpeg + numpy as last resort.

    Returns:
        1-D float32 numpy array of audio samples at target_sr.
    """
    audio_path = str(audio_path)

    # Strategy 1: librosa (the Wan2.2 reference implementation uses this)
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio
    except ImportError:
        pass

    # Strategy 2: torchaudio
    try:
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy()
    except ImportError:
        pass

    # Strategy 3: ffmpeg + raw numpy
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(target_sr), "-ac", "1",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            return np.frombuffer(proc.stdout, dtype=np.float32)

    raise ImportError(
        "Cannot load audio. Install one of: librosa, torchaudio, or have ffmpeg on PATH."
    )


def _linear_interpolation(features: torch.Tensor, input_fps: float, output_fps: float,
                           output_len: Optional[int] = None) -> torch.Tensor:
    """Resample temporal features from input_fps to output_fps via linear interpolation.

    Args:
        features: (1, T, D) or (L, T, D) tensor of audio features.
        input_fps: Source temporal rate (wav2vec outputs at ~50 fps).
        output_fps: Target temporal rate (video fps, typically 30).
        output_len: Exact output length (overrides fps-based calculation).

    Returns:
        Resampled features with same batch/layer dims but new temporal length.
    """
    # (*, T, D) -> (*, D, T) for F.interpolate
    feat = features.transpose(-2, -1)
    if output_len is None:
        seq_len = feat.shape[-1] / float(input_fps)
        output_len = int(seq_len * output_fps)
    feat = F.interpolate(feat, size=output_len, align_corners=True, mode="linear")
    return feat.transpose(-2, -1)


def _get_sample_indices(original_fps: float, total_frames: int, target_fps: float,
                        num_sample: int, fixed_start: int = 0) -> np.ndarray:
    """Compute frame indices for sampling from original_fps to target_fps."""
    required_duration = num_sample / target_fps
    end_time = fixed_start / original_fps + required_duration
    time_points = np.linspace(fixed_start / original_fps, end_time, num_sample, endpoint=False)
    frame_indices = np.round(time_points * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


class AudioFeatureExtractor:
    """Extract and bucket Wav2Vec features for S2V generation.

    This mirrors the AudioEncoder from the Wan2.2 reference implementation:
    1. Load audio at 16 kHz
    2. Run through Wav2Vec2ForCTC to get hidden states from all layers
    3. Interpolate from wav2vec output rate (~50 fps) to video rate (30 fps)
    4. Bucket features by frame for the diffusion transformer
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        logger.info(f"Loading Wav2Vec audio encoder from {model_path}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval()
        self.video_rate = 30  # intermediate video rate for feature alignment
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def extract_features(self, audio_path: str, return_all_layers: bool = True) -> torch.Tensor:
        """Extract wav2vec features from an audio file.

        Args:
            audio_path: Path to WAV/MP3/FLAC audio file.
            return_all_layers: If True, stack all hidden layer outputs (L, T, D).
                               If False, return only the last layer (1, T, D).

        Returns:
            Tensor of shape (L, T, D) where L=num_layers, T=time_steps, D=feature_dim.
            Features are interpolated to self.video_rate fps.
        """
        audio = _load_audio(audio_path, target_sr=16000)
        input_values = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(device=self.device, dtype=self.dtype)

        result = self.model(input_values, output_hidden_states=True)

        if return_all_layers:
            # Stack all hidden states: tuple of (1, T, D) -> (L, T, D)
            feat = torch.cat(result.hidden_states, dim=0)  # (L, T, D)
        else:
            feat = result.hidden_states[-1]  # (1, T, D)

        # Interpolate from wav2vec rate (~50 fps) to video rate (30 fps)
        feat = _linear_interpolation(feat, input_fps=50, output_fps=self.video_rate)
        return feat.to(self.dtype)

    def bucket_features(
        self,
        audio_embed: torch.Tensor,
        fps: int = 16,
        batch_frames: int = 81,
        m: int = 0,
    ) -> tuple:
        """Bucket audio features to align with video frames at the target fps.

        Maps wav2vec features (at video_rate=30 fps) onto the video generation
        frame grid (at target fps, typically 16). Each bucketed frame gets a
        window of (2*m + 1) neighboring audio features, flattened.

        Args:
            audio_embed: (L, T, D) tensor from extract_features.
            fps: Target video frame rate.
            batch_frames: Number of video frames per generation batch.
            m: Context window half-size for each frame (0 = single frame).

        Returns:
            (bucketed_embeddings, num_repeats):
            - bucketed_embeddings: (bucket_num, D') tensor
            - num_repeats: number of full clips that fit in the audio
        """
        num_layers, audio_frame_num, audio_dim = audio_embed.shape
        return_all_layers = num_layers > 1

        scale = self.video_rate / fps
        min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
        bucket_num = min_batch_num * batch_frames

        padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * self.video_rate) - audio_frame_num
        batch_idx = _get_sample_indices(
            original_fps=self.video_rate,
            total_frames=audio_frame_num + padd_audio_num,
            target_fps=fps,
            num_sample=bucket_num,
            fixed_start=0,
        )

        audio_sample_stride = int(self.video_rate / fps)
        batch_audio_eb = []

        for bi in batch_idx:
            if bi < audio_frame_num:
                chosen_idx = list(range(
                    bi - m * audio_sample_stride,
                    bi + (m + 1) * audio_sample_stride,
                    audio_sample_stride,
                ))
                chosen_idx = [max(0, c) for c in chosen_idx]
                chosen_idx = [min(audio_frame_num - 1, c) for c in chosen_idx]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                # Zero-pad for frames beyond audio duration
                if return_all_layers:
                    frame_audio_embed = torch.zeros(
                        num_layers, audio_dim * (2 * m + 1),
                        device=audio_embed.device, dtype=audio_embed.dtype,
                    )
                else:
                    frame_audio_embed = torch.zeros(
                        audio_dim * (2 * m + 1),
                        device=audio_embed.device, dtype=audio_embed.dtype,
                    )
            batch_audio_eb.append(frame_audio_embed.unsqueeze(0))

        bucketed = torch.cat(batch_audio_eb, dim=0)
        return bucketed, min_batch_num


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_reference_image(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    """Resize and center-crop a reference image to (width, height).

    Uses the 'resize shortest edge + center crop' strategy from the Wan2.2
    reference implementation to avoid distortion.
    """
    img = image.convert("RGB")

    # Resize so the shorter edge matches the target, preserving aspect ratio
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new_w = int(src_w * scale + 0.5)
    new_h = int(src_h * scale + 0.5)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to exact target size
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))
    return img


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr
    except ImportError:
        pass

    try:
        import torchaudio
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except (ImportError, RuntimeError):
        pass

    # ffprobe fallback
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())

    logger.warning("Cannot determine audio duration; defaulting to single-clip generation")
    return 5.0


# ---------------------------------------------------------------------------
# S2V Backend
# ---------------------------------------------------------------------------

class Wan22S2VBackend(BasePipeline):
    """Backend for Wan 2.2 S2V -- audio-driven cinematic video generation.

    Generates lip-synced, rhythm-aware video from a text prompt, reference image,
    and audio file. Supports minute-level generation via multi-clip chaining.

    Usage:
        backend = Wan22S2VBackend.load(offload_strategy="model_cpu")
        output = backend.generate(
            prompt="A young woman speaking confidently at a podium",
            audio_path="speech.wav",
            image=PIL.Image.open("speaker.jpg"),
        )
        backend.save(output, "output.mp4")
    """

    backend_name = "wan22_s2v"

    def __init__(
        self,
        pipe,
        audio_encoder: Optional[AudioFeatureExtractor] = None,
        use_diffusers_pipeline: bool = False,
    ):
        self.pipe = pipe
        self.audio_encoder = audio_encoder
        self.use_diffusers_pipeline = use_diffusers_pipeline
        self._defaults = S2V_DEFAULTS.copy()

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        quantization: str = "none",
        offload_strategy: str = "none",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        audio_encoder_path: Optional[str] = None,
        **kwargs,
    ) -> "Wan22S2VBackend":
        """Load the Wan 2.2 S2V model and audio encoder.

        The loader first tries the diffusers WanSpeechToVideoPipeline (for when
        the community PR is merged). If that import fails, it falls back to
        loading the raw transformer + VAE + text encoder from the HuggingFace
        checkpoint and wrapping them in a minimal inference pipeline.

        Args:
            model_path: HuggingFace model ID or local path.
                        Defaults to 'Wan-AI/Wan2.2-S2V-14B'.
            torch_dtype: Compute dtype (bfloat16 recommended for CUDA).
            device: Target device. Must be 'cuda' -- MPS not supported.
            quantization: 'none', 'nf4', 'int8', or 'fp8'.
            offload_strategy: 'none', 'model_cpu', or 'sequential_cpu'.
            enable_vae_slicing: Reduce VAE memory via slice processing.
            enable_vae_tiling: Reduce VAE memory via tiled processing.
            audio_encoder_path: Override path to wav2vec model.
                                Defaults to the one bundled in the S2V repo.

        Returns:
            Wan22S2VBackend instance ready for generation.
        """
        model_path = model_path or WAN22_S2V_MODEL

        # MPS check: S2V-14B uses FP8 MoE, which MPS cannot run
        if device == "mps":
            raise RuntimeError(
                "Wan 2.2 S2V-14B uses FP8 MoE experts internally and requires CUDA. "
                "MPS (Apple Silicon) is not supported for this model. "
                "Use device='cuda' with model_cpu offloading for 24 GB GPUs."
            )

        logger.info(f"Loading Wan 2.2 S2V from {model_path} (dtype={torch_dtype}, quant={quantization})")

        # --- Attempt diffusers pipeline first ---
        use_diffusers = False
        pipe = None

        try:
            from diffusers import WanSpeechToVideoPipeline, AutoencoderKLWan
            logger.info("Found WanSpeechToVideoPipeline in diffusers -- using native pipeline")

            vae = AutoencoderKLWan.from_pretrained(
                model_path, subfolder="vae", torch_dtype=torch.float32
            )

            quant_config = get_quantization_config(
                quantization, components=["transformer", "transformer_2"]
            )
            load_kwargs = dict(torch_dtype=torch_dtype, vae=vae)
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config

            pipe = WanSpeechToVideoPipeline.from_pretrained(model_path, **load_kwargs)
            use_diffusers = True

        except (ImportError, AttributeError):
            logger.info(
                "WanSpeechToVideoPipeline not available in this diffusers version. "
                "Falling back to standalone S2V pipeline."
            )

        # --- Fallback: standalone pipeline ---
        if pipe is None:
            pipe = _load_standalone_pipeline(
                model_path=model_path,
                torch_dtype=torch_dtype,
                quantization=quantization,
            )

        # Fix: transformers 5.x UMT5 embed_tokens zero-weight bug
        if hasattr(pipe, "text_encoder"):
            te = pipe.text_encoder
            if (hasattr(te, "shared") and hasattr(te, "encoder")
                    and hasattr(te.encoder, "embed_tokens")
                    and te.encoder.embed_tokens.weight.abs().sum().item() == 0
                    and te.shared.weight.abs().sum().item() > 0):
                logger.warning("Fixing UMT5 embed_tokens: binding shared.weight -> encoder.embed_tokens.weight")
                te.encoder.embed_tokens.weight = te.shared.weight

        # --- Load audio encoder ---
        if audio_encoder_path is None:
            # The wav2vec model is bundled inside the S2V repo
            audio_encoder_path = _resolve_audio_encoder_path(model_path)

        audio_encoder = AudioFeatureExtractor(
            model_path=audio_encoder_path,
            device="cpu",  # Always on CPU to save VRAM; features are small
            dtype=torch.float32,
        )

        instance = cls(
            pipe=pipe,
            audio_encoder=audio_encoder,
            use_diffusers_pipeline=use_diffusers,
        )

        # Apply offloading / device placement
        if offload_strategy != "none":
            instance._apply_offloading(pipe, offload_strategy, device=device)
        else:
            pipe.to(device)

        instance._apply_vae_opts(pipe, slicing=enable_vae_slicing, tiling=enable_vae_tiling)

        return instance

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        num_frames: int = 0,
        num_inference_steps: int = 0,
        guidance_scale: float = 0,
        seed: int = -1,
        image: Optional[Image.Image] = None,
        # S2V-specific
        audio_path: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        sampling_rate: int = 16000,
        num_clips: int = 0,
        shift: float = 0,
        pose_video: Optional[List] = None,
        **kwargs,
    ) -> VideoOutput:
        """Generate audio-driven video.

        Combines a text prompt, reference image, and audio input to produce
        lip-synced video with rhythm-aware motion.

        Args:
            prompt: Text description of the desired video scene/style.
            negative_prompt: What to avoid in generation.
            width: Output width in pixels (default 1024).
            height: Output height in pixels (default 704).
            num_frames: Total frames to generate. If 0, auto-calculated from
                        audio duration. Must satisfy (N - 1) % 4 == 0.
            num_inference_steps: Denoising steps per clip (default 40).
            guidance_scale: Classifier-free guidance strength (default 4.5).
            seed: Random seed (-1 for random).
            image: Reference image (required). The starting frame for the video.
            audio_path: Path to audio file (WAV/MP3/FLAC). Required if audio
                        tensor is not provided.
            audio: Pre-loaded audio tensor (1-D float32). Alternative to
                   audio_path.
            sampling_rate: Sample rate of the audio tensor (ignored when
                          audio_path is used -- resampled to 16 kHz internally).
            num_clips: Number of clips to generate. If 0, auto-calculated
                       from audio duration.
            shift: Flow-matching shift parameter (default 3.0).
            pose_video: Optional preprocessed pose video for body-motion guidance.

        Returns:
            VideoOutput with all generated frames.
        """
        if image is None:
            raise ValueError(
                "Wan 2.2 S2V requires a reference image (image= argument). "
                "This serves as the starting frame for audio-driven generation."
            )
        if audio_path is None and audio is None:
            raise ValueError(
                "Wan 2.2 S2V requires audio input. Provide either audio_path= "
                "(path to WAV/MP3/FLAC file) or audio= (pre-loaded tensor)."
            )

        d = self._defaults
        width = width or d["width"]
        height = height or d["height"]
        num_inference_steps = num_inference_steps or d["steps"]
        guidance_scale = guidance_scale or d["guidance_scale"]
        shift = shift or d["shift"]
        infer_frames = d["infer_frames"]
        fps = d["fps"]

        # Preprocess reference image
        ref_image = _preprocess_reference_image(image, width, height)

        # Determine audio duration and number of clips
        if audio_path is not None:
            audio_duration = _get_audio_duration(audio_path)
        else:
            audio_duration = len(audio) / sampling_rate

        if num_clips == 0:
            clip_duration = infer_frames / fps  # ~5 seconds per clip
            num_clips = max(1, math.ceil(audio_duration / clip_duration))

        if num_frames == 0:
            num_frames = infer_frames + 1  # 81 frames per clip (must be 4N+1)

        logger.info(
            f"S2V generation: {audio_duration:.1f}s audio -> {num_clips} clip(s), "
            f"{num_frames} frames/clip @ {fps} fps, {width}x{height}, "
            f"{num_inference_steps} steps, guidance={guidance_scale}"
        )

        # --- Diffusers pipeline path ---
        if self.use_diffusers_pipeline:
            return self._generate_diffusers(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=ref_image,
                audio_path=audio_path,
                audio=audio,
                sampling_rate=sampling_rate,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_clips=num_clips,
                pose_video=pose_video,
                **kwargs,
            )

        # --- Standalone pipeline path ---
        return self._generate_standalone(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=ref_image,
            audio_path=audio_path,
            audio=audio,
            sampling_rate=sampling_rate,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_clips=num_clips,
            shift=shift,
            infer_frames=infer_frames,
            fps=fps,
            pose_video=pose_video,
            **kwargs,
        )

    def _generate_diffusers(
        self,
        prompt: str,
        negative_prompt: str,
        image: Image.Image,
        audio_path: Optional[str],
        audio: Optional[torch.Tensor],
        sampling_rate: int,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        num_clips: int,
        pose_video: Optional[List],
        **kwargs,
    ) -> VideoOutput:
        """Generate via the diffusers WanSpeechToVideoPipeline."""
        gen_device = "cpu" if self.pipe.device.type == "cpu" else self.pipe.device
        generator = self._make_generator(seed, gen_device)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        # Audio input: path or tensor
        if audio_path is not None:
            audio_data = _load_audio(audio_path, target_sr=sampling_rate)
            pipe_kwargs["audio"] = audio_data
            pipe_kwargs["sampling_rate"] = sampling_rate
        elif audio is not None:
            pipe_kwargs["audio"] = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            pipe_kwargs["sampling_rate"] = sampling_rate

        # Multi-clip support
        if num_clips > 1:
            pipe_kwargs["num_frames_per_chunk"] = num_frames

        # Optional pose conditioning
        if pose_video is not None:
            pipe_kwargs["pose_video"] = pose_video

        output = self.pipe(**pipe_kwargs)
        frames = output.frames[0]

        return VideoOutput(
            frames=frames,
            fps=self._defaults["fps"],
            seed=seed,
            backend=self.backend_name,
            metadata={
                "num_clips": num_clips,
                "audio_path": audio_path,
                "diffusers_pipeline": True,
            },
        )

    def _generate_standalone(
        self,
        prompt: str,
        negative_prompt: str,
        image: Image.Image,
        audio_path: Optional[str],
        audio: Optional[torch.Tensor],
        sampling_rate: int,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        num_clips: int,
        shift: float,
        infer_frames: int,
        fps: int,
        pose_video: Optional[List],
        **kwargs,
    ) -> VideoOutput:
        """Generate using standalone pipeline components (no diffusers S2V pipeline).

        This implements the core S2V inference loop:
        1. Encode text prompt via T5
        2. Extract and bucket audio features via Wav2Vec
        3. Encode reference image as first-frame latent
        4. For each clip: denoise latents with audio conditioning
        5. Decode latents via VAE
        6. Chain clips with motion-frame overlap
        """
        components = self.pipe  # StandaloneS2VPipeline namespace

        device = next(iter(components.transformer.parameters())).device
        dtype = next(iter(components.transformer.parameters())).dtype

        # --- 1. Encode text ---
        text_encoder = components.text_encoder
        tokenizer = components.tokenizer
        te_device = next(iter(text_encoder.parameters())).device

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=S2V_MODEL_CONFIG["text_len"],
            truncation=True,
            return_tensors="pt",
        ).to(te_device)
        context = text_encoder(text_inputs.input_ids).last_hidden_state.to(device=device, dtype=dtype)

        # Null context for classifier-free guidance
        null_inputs = tokenizer(
            "",
            padding="max_length",
            max_length=S2V_MODEL_CONFIG["text_len"],
            truncation=True,
            return_tensors="pt",
        ).to(te_device)
        context_null = text_encoder(null_inputs.input_ids).last_hidden_state.to(device=device, dtype=dtype)

        # --- 2. Extract audio features ---
        if audio_path is not None:
            audio_embed = self.audio_encoder.extract_features(audio_path, return_all_layers=True)
        else:
            # Write tensor to temp file and extract (simpler than re-implementing wav2vec processing)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                import torchaudio
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio)
                torchaudio.save(tmp_path, audio.unsqueeze(0), sampling_rate)
                audio_embed = self.audio_encoder.extract_features(tmp_path, return_all_layers=True)
            finally:
                os.unlink(tmp_path)

        audio_bucketed, num_repeats_available = self.audio_encoder.bucket_features(
            audio_embed, fps=fps, batch_frames=num_frames, m=0
        )
        audio_bucketed = audio_bucketed.to(device=device, dtype=dtype)

        # Clamp num_clips to what audio supports
        num_clips = min(num_clips, num_repeats_available)
        logger.info(f"Audio supports {num_repeats_available} clip(s), generating {num_clips}")

        # --- 3. Encode reference image ---
        vae = components.vae
        vae_device = next(iter(vae.parameters())).device

        img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0  # [-1, 1]
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
        img_tensor = img_tensor.to(device=vae_device, dtype=torch.float32)

        ref_latent = vae.encode(img_tensor).latent_dist.sample()  # (1, C, 1, h, w)
        ref_latent = ref_latent.to(device=device, dtype=dtype)

        # --- 4. Generate clips ---
        all_frames = []
        transformer = components.transformer
        scheduler = components.scheduler

        for clip_idx in range(num_clips):
            logger.info(f"Generating clip {clip_idx + 1}/{num_clips}")

            # Slice audio features for this clip
            clip_start = clip_idx * num_frames
            clip_end = clip_start + num_frames
            clip_audio = audio_bucketed[clip_start:clip_end]

            # Seed generator per clip for reproducibility
            if seed >= 0:
                clip_seed = seed + clip_idx
                generator = torch.Generator(device=device).manual_seed(clip_seed)
            else:
                generator = None

            # Initialize noise latents
            latent_h = height // 8
            latent_w = width // 8
            latent_t = (num_frames - 1) // 4 + 1
            latent_c = S2V_MODEL_CONFIG["in_dim"]

            noise = torch.randn(
                1, latent_c, latent_t, latent_h, latent_w,
                device=device, dtype=dtype, generator=generator,
            )

            # Scheduler setup
            scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = scheduler.timesteps

            # Apply shift to timesteps if the scheduler supports it
            if hasattr(scheduler, "shift"):
                scheduler.shift = shift

            latents = noise * scheduler.init_noise_sigma

            # Denoising loop
            for i, t in enumerate(timesteps):
                latent_input = scheduler.scale_model_input(latents, t)

                # Prepare model input
                model_kwargs = dict(
                    hidden_states=latent_input,
                    timestep=t.unsqueeze(0).to(device),
                    encoder_hidden_states=context,
                    audio_embeds=clip_audio.unsqueeze(0),
                    ref_latents=ref_latent,
                )

                if pose_video is not None:
                    model_kwargs["pose_cond"] = pose_video

                # Conditional prediction
                noise_pred_cond = transformer(**model_kwargs).sample

                # Unconditional prediction for CFG
                model_kwargs["encoder_hidden_states"] = context_null
                noise_pred_uncond = transformer(**model_kwargs).sample

                # Classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # --- 5. VAE decode ---
            latents_decode = latents.to(device=vae_device, dtype=torch.float32)
            video_tensor = vae.decode(latents_decode).sample  # (1, C, T, H, W)
            video_tensor = video_tensor.clamp(-1, 1)

            # Convert to PIL frames
            frames = video_tensor.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
            frames = ((frames + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()

            clip_frames = [Image.fromarray(f) for f in frames]

            # For multi-clip continuity: skip first frame of subsequent clips
            # (it overlaps with the last frame of the previous clip)
            if clip_idx > 0 and len(clip_frames) > 1:
                clip_frames = clip_frames[1:]

            all_frames.extend(clip_frames)

        return VideoOutput(
            frames=all_frames,
            fps=fps,
            seed=seed,
            backend=self.backend_name,
            metadata={
                "num_clips": num_clips,
                "audio_path": audio_path,
                "diffusers_pipeline": False,
                "total_frames": len(all_frames),
                "duration_seconds": len(all_frames) / fps,
            },
        )

    def save(self, output: VideoOutput, path: str, fps: int = 0):
        """Save video output, optionally muxing with the source audio.

        If the output metadata contains an audio_path and the output path
        ends with .mp4, the audio is muxed into the video file.
        """
        fps = fps or output.fps
        audio_path = output.metadata.get("audio_path")

        if path.endswith(".mp4") and audio_path and shutil.which("ffmpeg"):
            self._save_mp4_with_audio(output.frames, path, fps, audio_path)
        else:
            super().save(output, path, fps)

    @staticmethod
    def _save_mp4_with_audio(frames: list, path: str, fps: int, audio_path: str):
        """Save video frames + audio to MP4 using ffmpeg."""
        import tempfile

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # First render video-only to a temp file
        w, h = frames[0].size
        video_tmp = tempfile.mktemp(suffix=".mp4")

        try:
            # Encode frames to raw video pipe
            cmd_video = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-an",
                video_tmp,
            ]
            proc = subprocess.Popen(cmd_video, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(np.array(frame).tobytes())
            proc.stdin.close()
            proc.wait()

            if proc.returncode != 0:
                logger.warning("Video encoding failed, falling back to BasePipeline.save()")
                from diffusers.utils import export_to_video
                export_to_video(frames, path, fps=fps)
                return

            # Mux video + audio
            video_duration = len(frames) / fps
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", video_tmp,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{video_duration:.3f}",
                "-shortest",
                path,
            ]
            result = subprocess.run(cmd_mux, capture_output=True)
            if result.returncode != 0:
                logger.warning("Audio muxing failed, saving video-only")
                shutil.copy(video_tmp, path)
        finally:
            if os.path.exists(video_tmp):
                os.unlink(video_tmp)

        logger.info(f"Saved S2V video with audio to {path}")

    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Public utility: get audio duration in seconds."""
        return _get_audio_duration(audio_path)


# ---------------------------------------------------------------------------
# Standalone pipeline loader (fallback when diffusers S2V pipeline is missing)
# ---------------------------------------------------------------------------

class _StandaloneS2VPipeline:
    """Minimal namespace to hold model components when the diffusers
    WanSpeechToVideoPipeline is not available.

    This is NOT a full pipeline -- it just holds the transformer, VAE,
    text encoder, tokenizer, and scheduler so the backend can orchestrate
    inference directly.
    """

    def __init__(self, transformer, vae, text_encoder, tokenizer, scheduler):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    def to(self, device):
        device = torch.device(device)
        self._device = device
        self.transformer.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        return self

    def enable_model_cpu_offload(self, gpu_id: int = 0):
        """Move components to CPU and auto-transfer to GPU during forward pass.

        This requires accelerate. Each component is wrapped so that its
        forward() method moves it to GPU, runs, then moves it back.
        """
        try:
            from accelerate import cpu_offload_with_hook
        except ImportError:
            logger.warning("accelerate not installed -- cannot use model_cpu_offload. Moving to CUDA.")
            self.to(f"cuda:{gpu_id}")
            return

        device = torch.device(f"cuda:{gpu_id}")
        self._device = torch.device("cpu")

        # Offload in order of usage: text_encoder -> transformer -> vae
        self.text_encoder, hook_te = cpu_offload_with_hook(self.text_encoder, device)
        self.transformer, hook_t = cpu_offload_with_hook(self.transformer, device, prev_module_hook=hook_te)
        self.vae, hook_v = cpu_offload_with_hook(self.vae, device, prev_module_hook=hook_t)

    def enable_sequential_cpu_offload(self, gpu_id: int = 0):
        """Offload each sub-module sequentially for minimal VRAM usage."""
        try:
            from accelerate import cpu_offload
        except ImportError:
            logger.warning("accelerate not installed -- cannot use sequential_cpu_offload")
            self.to(f"cuda:{gpu_id}")
            return

        device = torch.device(f"cuda:{gpu_id}")
        self._device = torch.device("cpu")

        for component in [self.text_encoder, self.transformer, self.vae]:
            cpu_offload(component, device)

    def enable_vae_slicing(self):
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()

    def enable_vae_tiling(self):
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()


def _resolve_audio_encoder_path(model_path: str) -> str:
    """Resolve the path to the bundled wav2vec audio encoder.

    For HuggingFace Hub models, the wav2vec subfolder is downloaded with
    the main model. For local paths, we look for the subfolder directly.
    """
    local_path = Path(model_path) / WAV2VEC_SUBFOLDER
    if local_path.exists():
        return str(local_path)

    # For HF Hub, try to download just the audio encoder subfolder
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            model_path,
            allow_patterns=[f"{WAV2VEC_SUBFOLDER}/*"],
        )
        resolved = Path(path) / WAV2VEC_SUBFOLDER
        if resolved.exists():
            return str(resolved)
    except Exception as e:
        logger.warning(f"Could not download audio encoder from {model_path}: {e}")

    # Last resort: use the HF model ID directly (transformers will download)
    logger.info(f"Using model_path as audio encoder path: {model_path}/{WAV2VEC_SUBFOLDER}")
    return f"{model_path}/{WAV2VEC_SUBFOLDER}"


def _load_standalone_pipeline(
    model_path: str,
    torch_dtype: torch.dtype,
    quantization: str,
) -> _StandaloneS2VPipeline:
    """Load S2V model components individually when no diffusers pipeline exists.

    Loads:
    - WanS2VTransformer3DModel (or falls back to loading safetensors directly)
    - AutoencoderKLWan (VAE)
    - UMT5 text encoder + tokenizer
    - FlowMatchEulerDiscreteScheduler
    """
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler

    logger.info("Loading S2V components individually (standalone mode)")

    # --- VAE (always float32 for Wan) ---
    vae = _try_load_vae(model_path)

    # --- Text encoder (UMT5-XXL) ---
    text_encoder, tokenizer = _try_load_text_encoder(model_path, torch_dtype)

    # --- Transformer (S2V DiT) ---
    transformer = _try_load_transformer(model_path, torch_dtype, quantization)

    # --- Scheduler ---
    scheduler = _try_load_scheduler(model_path)

    return _StandaloneS2VPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )


def _try_load_vae(model_path: str):
    """Load the Wan VAE, trying diffusers subfolder first, then standalone .pth."""
    from diffusers import AutoencoderKLWan

    # Try diffusers-format subfolder
    try:
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        logger.info("Loaded VAE from diffusers subfolder")
        return vae
    except (OSError, ValueError):
        pass

    # Try loading standalone VAE .pth
    vae_path = Path(model_path) / "Wan2.1_VAE.pth"
    if vae_path.exists():
        logger.info(f"Loading VAE from {vae_path}")
        vae = AutoencoderKLWan.from_single_file(str(vae_path), torch_dtype=torch.float32)
        return vae

    raise FileNotFoundError(
        f"Cannot find VAE in {model_path}. Expected either a 'vae' subfolder "
        f"(diffusers format) or 'Wan2.1_VAE.pth' (original format)."
    )


def _try_load_text_encoder(model_path: str, torch_dtype: torch.dtype):
    """Load the UMT5-XXL text encoder and tokenizer."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    # Try diffusers-format subfolder
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch_dtype
        )
        logger.info("Loaded text encoder from diffusers subfolder")
        return text_encoder, tokenizer
    except (OSError, ValueError):
        pass

    # Try loading from google/umt5-xxl directly
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
        te_path = Path(model_path) / "models_t5_umt5-xxl-enc-bf16.pth"
        if te_path.exists():
            logger.info(f"Loading UMT5-XXL from {te_path}")
            text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl", torch_dtype=torch_dtype)
            # Load custom weights
            state_dict = torch.load(str(te_path), map_location="cpu", weights_only=True)
            text_encoder.load_state_dict(state_dict, strict=False)
            return text_encoder, tokenizer
        else:
            # Just use the pretrained model
            text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl", torch_dtype=torch_dtype)
            return text_encoder, tokenizer
    except Exception as e:
        logger.warning(f"Failed to load UMT5-XXL: {e}")

    raise FileNotFoundError(
        f"Cannot find text encoder in {model_path}. Expected either 'text_encoder' subfolder "
        f"(diffusers format) or 'models_t5_umt5-xxl-enc-bf16.pth' (original format)."
    )


def _try_load_transformer(model_path: str, torch_dtype: torch.dtype, quantization: str):
    """Load the S2V transformer model.

    Tries several strategies:
    1. Diffusers WanS2VTransformer3DModel from subfolder
    2. Diffusers from_single_file with safetensors shards
    3. Raw safetensors loading with manual config
    """
    # Strategy 1: Try diffusers model class
    try:
        from diffusers import WanS2VTransformer3DModel
        transformer = WanS2VTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch_dtype
        )
        logger.info("Loaded S2V transformer via diffusers WanS2VTransformer3DModel")
        return transformer
    except (ImportError, OSError, ValueError):
        pass

    # Strategy 2: Load as generic WanTransformer3DModel with S2V config
    try:
        from diffusers import WanTransformer3DModel
        # Check if there's a transformer subfolder with config
        transformer_path = Path(model_path) / "transformer"
        if transformer_path.exists() and (transformer_path / "config.json").exists():
            transformer = WanTransformer3DModel.from_pretrained(
                model_path, subfolder="transformer", torch_dtype=torch_dtype
            )
            logger.info("Loaded S2V transformer via WanTransformer3DModel")
            return transformer
    except (ImportError, OSError, ValueError):
        pass

    # Strategy 3: Load safetensors directly into a minimal transformer wrapper
    logger.info("Attempting direct safetensors loading for S2V transformer")
    safetensor_files = sorted(Path(model_path).glob("diffusion_pytorch_model*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No transformer weights found in {model_path}. "
            f"Expected 'diffusion_pytorch_model*.safetensors' files."
        )

    # Load sharded safetensors
    from safetensors.torch import load_file
    state_dict = {}
    for sf in safetensor_files:
        if sf.name.endswith(".index.json"):
            continue
        logger.info(f"Loading transformer shard: {sf.name}")
        state_dict.update(load_file(str(sf)))

    # Create a generic wrapper that holds the state dict and config
    # This is a placeholder until the proper diffusers class is available
    transformer = _RawTransformerWrapper(state_dict, S2V_MODEL_CONFIG, torch_dtype)
    logger.info(f"Loaded S2V transformer ({len(state_dict)} tensors) via raw safetensors")
    return transformer


def _try_load_scheduler(model_path: str):
    """Load the flow-matching scheduler."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    # Try diffusers-format subfolder
    try:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        logger.info("Loaded scheduler from diffusers subfolder")
        return scheduler
    except (OSError, ValueError):
        pass

    # Default scheduler config for S2V
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=S2V_DEFAULTS["shift"],
    )
    logger.info("Using default FlowMatchEulerDiscreteScheduler")
    return scheduler


class _RawTransformerWrapper(torch.nn.Module):
    """Minimal wrapper for raw safetensors state dict.

    This is a stopgap for when neither WanS2VTransformer3DModel nor
    WanTransformer3DModel is available in diffusers. It stores the
    loaded weights and config, but cannot perform inference.

    When this wrapper is encountered during generation, the backend will
    raise an informative error directing the user to upgrade diffusers.
    """

    def __init__(self, state_dict: dict, config: dict, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self._dtype = dtype
        # Store parameters so .to() and .parameters() work
        for name, tensor in state_dict.items():
            safe_name = name.replace(".", "__DOT__")
            self.register_buffer(safe_name, tensor.to(dtype))

    def forward(self, **kwargs):
        raise NotImplementedError(
            "Raw transformer weights loaded but no compatible model class found. "
            "The S2V transformer requires either:\n"
            "  1. diffusers with WanS2VTransformer3DModel (install latest diffusers from git)\n"
            "  2. The Wan2.2 repo's WanModel_S2V class\n"
            "Please upgrade diffusers: pip install git+https://github.com/huggingface/diffusers.git"
        )
