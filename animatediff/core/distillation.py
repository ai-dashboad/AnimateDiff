"""
Distillation Manager -- accelerated inference via step/CFG distillation.

Supports multiple acceleration strategies for video diffusion models:

1. **LCM LoRA** (Latent Consistency Models)
   - Universal plug-and-play acceleration for SD 1.5 / SDXL based models
   - Swaps scheduler to LCMScheduler, loads LCM LoRA weights
   - 4-8 steps, guidance_scale ~1.0-2.0
   - Models: latent-consistency/lcm-lora-sdv1-5, latent-consistency/lcm-lora-sdxl

2. **Lightning** (AnimateDiff-Lightning)
   - ByteDance's cross-model distillation for AnimateDiff
   - Replaces the motion adapter with a distilled variant
   - 1/2/4/8 steps with EulerDiscreteScheduler (trailing spacing)
   - Model: ByteDance/AnimateDiff-Lightning

3. **Step Distillation** (LightX2V / FastVideo)
   - Full-model or LoRA-based step distillation for Wan 2.1/2.2, HunyuanVideo
   - Wan 2.2: dual LoRA (high-noise + low-noise) for 4-step MoE inference
   - HunyuanVideo: FastHunyuan 6-step PCM distillation
   - Requires custom sigma schedules / timestep remapping

4. **Native Distilled** (pre-distilled checkpoints)
   - LTX-2 19b-distilled: ships with 8-step sigma schedule, CFG=1.0
   - No LoRA loading needed -- the checkpoint IS the distilled model
   - Just override num_steps + guidance_scale + sigma schedule

5. **Turbo / Hyper** (placeholder for future methods)
   - SDXL-Turbo, SD-Turbo adversarial distillation
   - HyperSD: 1-step generation via progressive distillation
   - Not yet applicable to video pipelines (image-only as of 2025)

Scheduler compatibility notes:
- LCMScheduler: works with UNet-based models (SD 1.5, SDXL, AnimateDiff)
- FlowMatchEulerDiscreteScheduler: used by DiT models (Wan, HunyuanVideo, LTX, CogVideoX)
  -- LCMScheduler is NOT compatible with flow-matching models
  -- Step distillation for DiT models uses the native flow-match scheduler
     with modified sigma/timestep schedules instead
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """Configuration for distilled/accelerated inference.

    Attributes:
        method: Distillation strategy. One of:
            "lcm"        -- LCM LoRA + LCMScheduler (UNet models only)
            "lightning"   -- AnimateDiff-Lightning distilled motion adapter
            "step_distill" -- LightX2V / FastVideo step + CFG distillation
            "native"      -- Pre-distilled checkpoint (e.g. LTX-2 19b-distilled)
            "turbo"       -- Adversarial distillation (SDXL-Turbo family)
            "hyper"       -- HyperSD progressive distillation
            "none"        -- No distillation (full inference)
        num_steps: Target number of denoising steps (4-8 typical for distilled).
        lora_path: HuggingFace repo ID or local path for distillation LoRA.
            For Wan 2.2 dual-transformer, this is a list encoded as
            "high_noise_path|low_noise_path".
        scheduler_class: Scheduler to use. "LCMScheduler" for LCM,
            "EulerDiscreteScheduler" for Lightning, or "" to keep the
            pipeline's existing scheduler (typical for step distillation).
        scheduler_kwargs: Extra kwargs passed to scheduler.from_config().
        guidance_scale: Recommended CFG scale (distilled models use lower
            values, typically 1.0-2.0).
        lora_scale: LoRA adapter weight (default 1.0).
        sigmas: Custom sigma schedule for step-distilled models.
        description: Human-readable description of this config.
    """
    method: str = "none"
    num_steps: int = 25
    lora_path: str = ""
    scheduler_class: str = ""
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    guidance_scale: float = 7.5
    lora_scale: float = 1.0
    sigmas: Optional[List[float]] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Known distillation registries
# ---------------------------------------------------------------------------

# LCM LoRA models (UNet / SD-based pipelines only)
LCM_LORAS = {
    "sd15": "latent-consistency/lcm-lora-sdv1-5",
    "sdxl": "latent-consistency/lcm-lora-sdxl",
    "ssd1b": "latent-consistency/lcm-lora-ssd-1b",
}

# AnimateDiff-Lightning step-specific checkpoints
LIGHTNING_REPO = "ByteDance/AnimateDiff-Lightning"
LIGHTNING_STEPS = {1, 2, 4, 8}

# Wan 2.2 distillation LoRAs (LightX2V)
# Dual-transformer architecture requires separate high-noise and low-noise LoRAs
WAN22_DISTILL_LORAS = {
    "i2v_A14B_4step": {
        "high_noise": "lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "low_noise": "lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
    },
}

# Wan 2.1 distillation (LightX2V full-model distilled checkpoints)
WAN21_DISTILL_MODELS = {
    "t2v_14B": "lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill",
}

# FastVideo / FastHunyuan distilled model
FAST_HUNYUAN = {
    "6step": "FastVideo/FastHunyuan",
}

# LTX-2 pre-distilled variant (no LoRA needed)
LTX2_DISTILLED = {
    "8step": "rootonchair/LTX-2-19b-distilled",
}

# Mapping from backend name to the scheduler architecture type
# "unet" = traditional noise prediction (LCMScheduler compatible)
# "dit_flow" = flow matching (FlowMatchEulerDiscreteScheduler)
# "dit_ddpm" = DiT with DDPM-style scheduling
BACKEND_SCHEDULER_TYPE = {
    "animatediff": "unet",
    "animatediff_legacy": "unet",
    "wan": "dit_flow",
    "wan22": "dit_flow",
    "wan22_animate": "dit_flow",
    "wan22_s2v": "dit_flow",
    "wan22_vace": "dit_flow",
    "hunyuan": "dit_flow",
    "cogvideo": "dit_ddpm",
    "ltx": "dit_flow",
    "ltx2": "dit_flow",
    "skyreels_v3": "dit_flow",
}


# ---------------------------------------------------------------------------
# Pre-built distillation configs per backend x quality tier
# ---------------------------------------------------------------------------

def _build_configs() -> Dict[str, Dict[str, DistillationConfig]]:
    """Build the registry of known distillation configs.

    Returns:
        Nested dict: {backend_name: {quality_tier: DistillationConfig}}
        quality_tier is one of "speed", "balanced", "quality".
    """
    configs: Dict[str, Dict[str, DistillationConfig]] = {}

    # ── AnimateDiff (SD 1.5 UNet) ──────────────────────────────────────
    configs["animatediff"] = {
        "speed": DistillationConfig(
            method="lightning",
            num_steps=2,
            lora_path=LIGHTNING_REPO,
            scheduler_class="EulerDiscreteScheduler",
            scheduler_kwargs={"timestep_spacing": "trailing", "beta_schedule": "linear"},
            guidance_scale=1.0,
            description="AnimateDiff-Lightning 2-step (fastest, minor quality loss)",
        ),
        "balanced": DistillationConfig(
            method="lightning",
            num_steps=4,
            lora_path=LIGHTNING_REPO,
            scheduler_class="EulerDiscreteScheduler",
            scheduler_kwargs={"timestep_spacing": "trailing", "beta_schedule": "linear"},
            guidance_scale=1.0,
            description="AnimateDiff-Lightning 4-step (good speed/quality balance)",
        ),
        "quality": DistillationConfig(
            method="lcm",
            num_steps=8,
            lora_path=LCM_LORAS["sd15"],
            scheduler_class="LCMScheduler",
            guidance_scale=1.5,
            description="LCM LoRA 8-step (near-original quality, 3x faster)",
        ),
    }

    # ── AnimateDiff Legacy (same as AnimateDiff) ────────────────────────
    configs["animatediff_legacy"] = configs["animatediff"]

    # ── Wan 2.2 (MoE DiT, flow matching) ──────────────────────────────
    # NOTE: LCMScheduler is NOT compatible with Wan's flow-matching scheduler.
    # Step distillation via LightX2V LoRAs is the correct approach.
    configs["wan22"] = {
        "speed": DistillationConfig(
            method="step_distill",
            num_steps=4,
            lora_path=(
                "lightx2v/Wan2.2-Distill-Loras|"
                "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors|"
                "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
            ),
            guidance_scale=1.0,  # CFG-distilled, no classifier-free guidance needed
            sigmas=[1000, 750, 500, 250],
            description="LightX2V 4-step distillation (A14B MoE, ~10x faster, CUDA only)",
        ),
        "balanced": DistillationConfig(
            method="step_distill",
            num_steps=8,
            lora_path="",  # No community 8-step LoRA yet; use full model with fewer steps
            guidance_scale=3.0,
            description="Reduced-step inference (8 steps, no distillation LoRA, moderate speedup)",
        ),
        "quality": DistillationConfig(
            method="none",
            num_steps=20,
            guidance_scale=4.0,
            description="Reduced from default 40 steps; still good quality with flow matching",
        ),
    }

    # Also covers wan22_animate, wan22_s2v, wan22_vace (same model family)
    for variant in ("wan22_animate", "wan22_s2v", "wan22_vace"):
        configs[variant] = configs["wan22"]

    # ── Wan 2.1 (DiT, flow matching) ──────────────────────────────────
    configs["wan"] = {
        "speed": DistillationConfig(
            method="step_distill",
            num_steps=4,
            lora_path="",  # LightX2V provides full-model distilled checkpoints
            guidance_scale=1.0,
            description="Step-distilled (requires lightx2v/Wan2.1-Distill-Models checkpoint)",
        ),
        "balanced": DistillationConfig(
            method="none",
            num_steps=12,
            guidance_scale=5.0,
            description="Reduced steps (12 vs default 25-50), moderate speedup",
        ),
        "quality": DistillationConfig(
            method="none",
            num_steps=20,
            guidance_scale=6.0,
            description="Reduced from default; near-original quality",
        ),
    }

    # ── HunyuanVideo (8.3B DiT, flow matching) ────────────────────────
    configs["hunyuan"] = {
        "speed": DistillationConfig(
            method="step_distill",
            num_steps=6,
            lora_path="FastVideo/FastHunyuan",
            guidance_scale=6.0,
            description="FastHunyuan PCM distillation (6 steps, ~8x faster)",
        ),
        "balanced": DistillationConfig(
            method="step_distill",
            num_steps=12,
            lora_path="FastVideo/FastHunyuan",
            guidance_scale=6.0,
            description="FastHunyuan with 12 steps (better quality, ~4x faster)",
        ),
        "quality": DistillationConfig(
            method="none",
            num_steps=20,
            guidance_scale=6.0,
            description="Reduced from default 30-50 steps; near-original quality",
        ),
    }

    # ── CogVideoX (Expert Transformer, DDPM-style) ────────────────────
    # No known distillation LoRAs for CogVideoX as of 2025.
    # Can still reduce steps for moderate speedup.
    configs["cogvideo"] = {
        "speed": DistillationConfig(
            method="none",
            num_steps=10,
            guidance_scale=6.0,
            description="Reduced steps only (no distillation available for CogVideoX)",
        ),
        "balanced": DistillationConfig(
            method="none",
            num_steps=20,
            guidance_scale=6.0,
            description="Moderate step reduction (20 vs default 50)",
        ),
        "quality": DistillationConfig(
            method="none",
            num_steps=30,
            guidance_scale=6.0,
            description="Slight step reduction (30 vs default 50)",
        ),
    }

    # ── LTX (v1) ──────────────────────────────────────────────────────
    configs["ltx"] = {
        "speed": DistillationConfig(
            method="none",
            num_steps=10,
            guidance_scale=3.0,
            description="Reduced steps (LTX-Video v1 has no distilled variant)",
        ),
        "balanced": DistillationConfig(
            method="none",
            num_steps=20,
            guidance_scale=3.5,
            description="Moderate step reduction",
        ),
        "quality": DistillationConfig(
            method="none",
            num_steps=30,
            guidance_scale=4.0,
            description="Slight step reduction",
        ),
    }

    # ── LTX-2 (19B dual-stream DiT) ──────────────────────────────────
    # The distilled variant is a separate checkpoint, not a LoRA.
    configs["ltx2"] = {
        "speed": DistillationConfig(
            method="native",
            num_steps=8,
            lora_path="",  # Pre-distilled checkpoint: rootonchair/LTX-2-19b-distilled
            guidance_scale=1.0,
            description="LTX-2 19b-distilled (8 steps, CFG=1.0, native distilled checkpoint)",
        ),
        "balanced": DistillationConfig(
            method="native",
            num_steps=8,
            lora_path="",
            guidance_scale=1.0,
            description="LTX-2 19b-distilled (same as speed -- 8 steps is the distilled default)",
        ),
        "quality": DistillationConfig(
            method="none",
            num_steps=20,
            guidance_scale=4.0,
            description="Full LTX-2 19b with reduced steps",
        ),
    }

    # ── SkyReels V3 ───────────────────────────────────────────────────
    configs["skyreels_v3"] = {
        "speed": DistillationConfig(
            method="none",
            num_steps=10,
            guidance_scale=3.0,
            description="Reduced steps only (no distillation available)",
        ),
        "balanced": DistillationConfig(
            method="none",
            num_steps=20,
            guidance_scale=4.0,
            description="Moderate step reduction",
        ),
        "quality": DistillationConfig(
            method="none",
            num_steps=30,
            guidance_scale=5.0,
            description="Slight step reduction",
        ),
    }

    return configs


_DISTILLATION_REGISTRY = _build_configs()


# ---------------------------------------------------------------------------
# DistillationManager
# ---------------------------------------------------------------------------

class DistillationManager:
    """Manage distilled inference for video generation pipelines.

    This manager handles the full lifecycle of distillation acceleration:
    - Querying available distillation options for a given backend/model
    - Applying distillation (LoRA loading, scheduler swapping, param adjustment)
    - Removing distillation and restoring original pipeline state

    Usage::

        from animatediff.core.distillation import DistillationManager

        manager = DistillationManager()

        # See what's available
        configs = manager.get_available_distillations("wan22")
        for name, cfg in configs.items():
            print(f"  {name}: {cfg.description}")

        # Apply the fastest option
        config = manager.recommended_config("wan22", quality="speed")
        pipeline = manager.apply_distillation(pipeline, config)

        # Generate with distilled settings
        output = pipeline.generate(
            prompt="...",
            num_inference_steps=config.num_steps,
            guidance_scale=config.guidance_scale,
        )

        # Restore original pipeline
        pipeline = manager.remove_distillation(pipeline)
    """

    def __init__(self):
        self._original_state: Dict[str, Any] = {}

    def get_available_distillations(
        self, backend_name: str,
    ) -> Dict[str, DistillationConfig]:
        """Return available distillation options for a backend.

        Args:
            backend_name: One of the registered backend names
                (e.g. "wan22", "hunyuan", "animatediff", "ltx2").

        Returns:
            Dict mapping quality tier names ("speed", "balanced", "quality")
            to their DistillationConfig. Returns empty dict if backend
            is not recognized.
        """
        return _DISTILLATION_REGISTRY.get(backend_name, {})

    def apply_distillation(self, pipeline, config: DistillationConfig):
        """Apply distillation to a loaded pipeline.

        This method performs up to three operations depending on the
        distillation method:

        1. Load distillation LoRA weights (LCM, Lightning, step_distill)
        2. Swap the scheduler (LCM -> LCMScheduler, Lightning -> Euler trailing)
        3. Log recommended num_steps and guidance_scale

        The caller is responsible for passing config.num_steps and
        config.guidance_scale to the pipeline's generate() call.

        Args:
            pipeline: A BasePipeline instance (must have a .pipe attribute
                pointing to the underlying diffusers pipeline).
            config: The DistillationConfig to apply.

        Returns:
            The pipeline (modified in-place). Original state is saved
            internally for remove_distillation().
        """
        if config.method == "none":
            logger.info("Distillation method is 'none'; no changes applied.")
            return pipeline

        pipe = self._get_diffusers_pipe(pipeline)
        if pipe is None:
            logger.warning("Could not find diffusers pipeline on this backend. Skipping distillation.")
            return pipeline

        # Save original state for restoration
        self._save_original_state(pipe)

        method = config.method
        logger.info(
            f"Applying distillation: method={method}, steps={config.num_steps}, "
            f"cfg={config.guidance_scale}, desc='{config.description}'"
        )

        if method == "lcm":
            self._apply_lcm(pipe, config)
        elif method == "lightning":
            self._apply_lightning(pipe, config)
        elif method == "step_distill":
            self._apply_step_distill(pipe, config)
        elif method == "native":
            self._apply_native(pipe, config)
        elif method in ("turbo", "hyper"):
            logger.warning(
                f"Distillation method '{method}' is recognized but not yet "
                f"implemented for video pipelines. Applying step/CFG reduction only."
            )
        else:
            logger.warning(f"Unknown distillation method: {method}")

        return pipeline

    def remove_distillation(self, pipeline):
        """Remove distillation and restore original pipeline settings.

        Restores the original scheduler and unloads any distillation LoRAs
        that were loaded by apply_distillation().

        Args:
            pipeline: The pipeline that was previously distilled.

        Returns:
            The pipeline with original settings restored.
        """
        pipe = self._get_diffusers_pipe(pipeline)
        if pipe is None:
            return pipeline

        state = self._original_state

        # Restore scheduler
        if "scheduler" in state:
            pipe.scheduler = state["scheduler"]
            logger.info("Restored original scheduler")

        # Unload distillation LoRAs
        distill_adapters = state.get("distill_adapters", [])
        if distill_adapters and hasattr(pipe, "delete_adapters"):
            try:
                pipe.delete_adapters(distill_adapters)
                logger.info(f"Unloaded distillation adapters: {distill_adapters}")
            except Exception as e:
                logger.warning(f"Failed to unload distillation adapters: {e}")

        # Restore previously active adapters
        prev_adapters = state.get("previous_adapters")
        if prev_adapters and hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters(prev_adapters)
                logger.info(f"Restored previous adapters: {prev_adapters}")
            except Exception:
                pass

        self._original_state = {}
        return pipeline

    @staticmethod
    def recommended_config(
        backend_name: str,
        quality: str = "balanced",
    ) -> DistillationConfig:
        """Get the recommended distillation config for a backend.

        Args:
            backend_name: The video generation backend name.
            quality: Quality/speed tradeoff preference:
                "speed"    -- Fastest (4 steps typical, may lose quality)
                "balanced" -- Good tradeoff (6-8 steps, minor quality loss)
                "quality"  -- Best quality (12-20 steps, moderate speedup)

        Returns:
            DistillationConfig for the requested tier. If the backend or
            quality tier is not found, returns a no-op config.
        """
        configs = _DISTILLATION_REGISTRY.get(backend_name, {})
        if quality in configs:
            return copy.deepcopy(configs[quality])

        # Fallback: return a sensible default based on quality tier
        step_map = {"speed": 8, "balanced": 15, "quality": 25}
        cfg_map = {"speed": 1.5, "balanced": 4.0, "quality": 6.0}
        return DistillationConfig(
            method="none",
            num_steps=step_map.get(quality, 15),
            guidance_scale=cfg_map.get(quality, 4.0),
            description=f"Generic fallback for {backend_name} ({quality} tier)",
        )

    @staticmethod
    def supports_distillation(backend_name: str) -> bool:
        """Check if a backend has any non-trivial distillation options.

        Returns True if at least one quality tier uses a real distillation
        method (not just step reduction).
        """
        configs = _DISTILLATION_REGISTRY.get(backend_name, {})
        return any(
            c.method not in ("none", "")
            for c in configs.values()
        )

    @staticmethod
    def list_backends_with_distillation() -> List[str]:
        """Return backend names that have real distillation support."""
        return [
            name for name, configs in _DISTILLATION_REGISTRY.items()
            if any(c.method not in ("none", "") for c in configs.values())
        ]

    # -----------------------------------------------------------------------
    # Internal: method-specific application logic
    # -----------------------------------------------------------------------

    def _apply_lcm(self, pipe, config: DistillationConfig):
        """Apply LCM LoRA distillation (UNet-based models only).

        Steps:
        1. Load LCM LoRA weights
        2. Swap scheduler to LCMScheduler
        3. Set guidance_scale to config value (typically 1.0-2.0)
        """
        # Load LCM LoRA
        if config.lora_path:
            adapter_name = "_distill_lcm"
            try:
                pipe.load_lora_weights(config.lora_path, adapter_name=adapter_name)
                pipe.set_adapters([adapter_name], adapter_weights=[config.lora_scale])
                self._original_state.setdefault("distill_adapters", []).append(adapter_name)
                logger.info(f"Loaded LCM LoRA from {config.lora_path}")
            except Exception as e:
                logger.warning(f"Failed to load LCM LoRA from {config.lora_path}: {e}")

        # Swap to LCMScheduler
        if config.scheduler_class == "LCMScheduler":
            try:
                from diffusers import LCMScheduler
                new_scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                pipe.scheduler = new_scheduler
                logger.info("Switched to LCMScheduler")
            except ImportError:
                logger.warning(
                    "LCMScheduler not available in this diffusers version. "
                    "Keeping current scheduler (results may be suboptimal)."
                )

    def _apply_lightning(self, pipe, config: DistillationConfig):
        """Apply AnimateDiff-Lightning distillation.

        Lightning uses a distilled motion adapter loaded from safetensors
        and EulerDiscreteScheduler with trailing timestep spacing.
        """
        # Load the step-specific distilled motion adapter
        if config.lora_path and config.num_steps in LIGHTNING_STEPS:
            try:
                from safetensors.torch import load_file
                from huggingface_hub import hf_hub_download

                ckpt_name = f"animatediff_lightning_{config.num_steps}step_diffusers.safetensors"
                ckpt_path = hf_hub_download(config.lora_path, ckpt_name)

                if hasattr(pipe, "motion_adapter"):
                    pipe.motion_adapter.load_state_dict(load_file(ckpt_path, device="cpu"))
                    logger.info(f"Loaded Lightning {config.num_steps}-step motion adapter")
                else:
                    logger.warning(
                        "Pipeline has no motion_adapter attribute. "
                        "Lightning distillation requires AnimateDiffPipeline."
                    )
            except Exception as e:
                logger.warning(f"Failed to load Lightning adapter: {e}")

        # Swap to EulerDiscreteScheduler with trailing spacing
        if config.scheduler_class == "EulerDiscreteScheduler":
            try:
                from diffusers import EulerDiscreteScheduler
                new_scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config,
                    **config.scheduler_kwargs,
                )
                pipe.scheduler = new_scheduler
                logger.info("Switched to EulerDiscreteScheduler (trailing)")
            except Exception as e:
                logger.warning(f"Failed to set EulerDiscreteScheduler: {e}")

    def _apply_step_distill(self, pipe, config: DistillationConfig):
        """Apply step distillation via LoRA weights (LightX2V, FastVideo).

        For Wan 2.2 MoE models, this loads separate high-noise and low-noise
        LoRAs into transformer and transformer_2 respectively.

        For HunyuanVideo (FastHunyuan), loads PCM distilled weights.

        NOTE: LightX2V distillation LoRAs for Wan 2.2 have known compatibility
        issues with the standard diffusers load_lora_weights() API as of
        diffusers 0.36. If loading fails, the manager logs a warning and
        falls back to step reduction without LoRA.
        """
        if not config.lora_path:
            logger.info(
                f"No distillation LoRA path specified for step_distill. "
                f"Applying step reduction only ({config.num_steps} steps)."
            )
            return

        # Parse Wan 2.2 dual-LoRA format: "repo|high_noise_file|low_noise_file"
        parts = config.lora_path.split("|")
        if len(parts) == 3:
            self._apply_wan22_dual_lora(pipe, parts[0], parts[1], parts[2], config)
        else:
            # Single LoRA path (FastHunyuan, generic)
            self._apply_single_distill_lora(pipe, config.lora_path, config)

    def _apply_wan22_dual_lora(
        self,
        pipe,
        repo_id: str,
        high_noise_file: str,
        low_noise_file: str,
        config: DistillationConfig,
    ):
        """Load Wan 2.2 dual-transformer distillation LoRAs.

        The MoE architecture uses two transformers:
        - transformer (high-noise denoising stages)
        - transformer_2 (low-noise refinement stages)

        Each gets its own distilled LoRA.
        """
        adapters_loaded = []

        # High-noise LoRA -> transformer
        try:
            adapter_name = "_distill_high_noise"
            pipe.load_lora_weights(
                repo_id,
                weight_name=high_noise_file,
                adapter_name=adapter_name,
            )
            adapters_loaded.append(adapter_name)
            logger.info(f"Loaded high-noise distillation LoRA: {high_noise_file}")
        except Exception as e:
            logger.warning(
                f"Failed to load Wan 2.2 high-noise distillation LoRA: {e}. "
                f"This may be due to diffusers compatibility issues with LightX2V LoRAs. "
                f"Consider using the LightX2V framework directly for 4-step inference."
            )

        # Low-noise LoRA -> transformer_2
        if hasattr(pipe, "load_lora_weights"):
            try:
                adapter_name = "_distill_low_noise"
                load_kwargs = dict(
                    adapter_name=adapter_name,
                )
                # Wan 2.2 dual-transformer LoRA loading
                if hasattr(pipe, "transformer_2"):
                    load_kwargs["load_into_transformer_2"] = True

                pipe.load_lora_weights(repo_id, weight_name=low_noise_file, **load_kwargs)
                adapters_loaded.append(adapter_name)
                logger.info(f"Loaded low-noise distillation LoRA: {low_noise_file}")
            except Exception as e:
                logger.warning(f"Failed to load Wan 2.2 low-noise distillation LoRA: {e}")

        if adapters_loaded:
            try:
                scales = [config.lora_scale] * len(adapters_loaded)
                pipe.set_adapters(adapters_loaded, adapter_weights=scales)
            except Exception as e:
                logger.warning(f"Failed to activate distillation adapters: {e}")

            self._original_state.setdefault("distill_adapters", []).extend(adapters_loaded)

    def _apply_single_distill_lora(self, pipe, lora_path: str, config: DistillationConfig):
        """Load a single distillation LoRA (FastHunyuan, generic)."""
        adapter_name = "_distill_step"
        try:
            # Handle "repo_id/weight_name.safetensors" format
            if "/" in lora_path and "." in lora_path.rsplit("/", 1)[-1]:
                repo_id, weight_name = lora_path.rsplit("/", 1)
                pipe.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name)
            else:
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name)

            pipe.set_adapters([adapter_name], adapter_weights=[config.lora_scale])
            self._original_state.setdefault("distill_adapters", []).append(adapter_name)
            logger.info(f"Loaded distillation LoRA from {lora_path}")
        except Exception as e:
            logger.warning(
                f"Failed to load distillation LoRA from {lora_path}: {e}. "
                f"Falling back to step reduction only."
            )

    def _apply_native(self, pipe, config: DistillationConfig):
        """Apply native distilled checkpoint settings.

        For pre-distilled models (e.g. LTX-2 19b-distilled), the checkpoint
        itself is already distilled. We only need to adjust inference params
        and optionally set the sigma schedule.
        """
        if config.sigmas:
            logger.info(f"Native distilled model: using custom sigma schedule ({len(config.sigmas)} values)")
        else:
            logger.info(
                f"Native distilled model: set num_steps={config.num_steps}, "
                f"guidance_scale={config.guidance_scale}"
            )

    # -----------------------------------------------------------------------
    # Internal: state management
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_diffusers_pipe(pipeline):
        """Extract the underlying diffusers pipeline from a BasePipeline wrapper.

        Handles both direct diffusers pipelines and our BasePipeline subclasses
        which store the diffusers pipe as self.pipe.
        """
        if hasattr(pipeline, "pipe"):
            return pipeline.pipe
        # Maybe it's already a raw diffusers pipeline
        if hasattr(pipeline, "scheduler"):
            return pipeline
        return None

    def _save_original_state(self, pipe):
        """Save the original pipeline state before modification."""
        self._original_state = {
            "scheduler": copy.deepcopy(pipe.scheduler),
        }

        # Save active LoRA adapters if any
        if hasattr(pipe, "get_active_adapters"):
            try:
                active = pipe.get_active_adapters()
                if active:
                    self._original_state["previous_adapters"] = active
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_distillation_manager() -> DistillationManager:
    """Get a DistillationManager instance.

    Unlike VRAMManager, DistillationManager is stateless enough that we
    don't need a strict singleton. Each call returns a fresh instance.
    """
    return DistillationManager()


def recommended_distillation(
    backend_name: str,
    quality: str = "balanced",
) -> DistillationConfig:
    """Shortcut to get a recommended distillation config.

    Args:
        backend_name: The video generation backend.
        quality: "speed", "balanced", or "quality".

    Returns:
        DistillationConfig with recommended settings.
    """
    return DistillationManager.recommended_config(backend_name, quality)


def apply_distillation_to_pipeline(pipeline, backend_name: str, quality: str = "balanced"):
    """One-shot convenience: get recommended config and apply it.

    Returns:
        Tuple of (pipeline, config) so the caller can use config.num_steps
        and config.guidance_scale when calling generate().
    """
    manager = DistillationManager()
    config = manager.recommended_config(backend_name, quality)
    pipeline = manager.apply_distillation(pipeline, config)
    return pipeline, config
