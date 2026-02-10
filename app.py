
import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, load_weights, auto_download, MOTION_MODULES, BACKUP_DREAMBOOTH_MODELS
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora


sample_idx = 0
scheduler_dict = {
    "DDIM":      DDIMScheduler,
    "Euler":     EulerDiscreteScheduler,
    "Euler A":   EulerAncestralDiscreteScheduler,
    "DPM++ 2M":  DPMSolverMultistepScheduler,
    "DPM++ 2M Karras": lambda **kwargs: DPMSolverMultistepScheduler(**kwargs, use_karras_sigmas=True),
    "PNDM":      PNDMScheduler,
}

css = """
.toolbutton {
    margin-bottom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

PRETRAINED_SD = "runwayml/stable-diffusion-v1-5"

default_motion_module = "v3_sd15_mm.ckpt"
default_inference_config = "configs/inference/inference-v3.yaml"
default_dreambooth_model = "realisticVisionV60B1_v51VAE.safetensors"
default_prompt = "b&w photo of 42 y.o man in black clothes, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
default_n_prompt = "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
default_seed = 8893659352891878017

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ============================================================================
# V3 Backend Controller (new multi-backend)
# ============================================================================

class V3Controller:
    """Controller for the new multi-backend video generation."""

    BACKENDS = ["auto", "wan", "hunyuan", "cogvideo", "ltx", "animatediff"]
    QUALITY_PRESETS = ["draft", "standard", "high", "max"]

    def __init__(self):
        from animatediff.core.vram_manager import get_vram_manager
        self.vm = get_vram_manager()
        self.pipe = None
        self.current_backend = None
        self.savedir = os.path.join(os.getcwd(), "samples", datetime.now().strftime("Gradio-V3-%Y-%m-%dT%H-%M-%S"))
        os.makedirs(self.savedir, exist_ok=True)

    def get_gpu_info(self):
        return self.vm.summary()

    def load_backend(self, backend_name, progress=gr.Progress()):
        from animatediff.backends import get_backend

        if backend_name == "auto":
            backend_name = self.vm.best_backend()

        rec = self.vm.recommend(backend_name)
        progress(0.1, desc=f"Loading {backend_name} (variant={rec.model_variant}, quant={rec.quantization})...")

        BackendClass = get_backend(backend_name)
        self.pipe = BackendClass.load(
            torch_dtype=rec.torch_dtype,
            device=device,
            quantization=rec.quantization,
            offload_strategy=rec.offload_strategy,
            enable_vae_slicing=rec.enable_vae_slicing,
            enable_vae_tiling=rec.enable_vae_tiling,
            model_variant=rec.model_variant,
        )
        self.current_backend = backend_name

        progress(1.0, desc="Ready!")
        return f"Loaded: {backend_name} (variant={rec.model_variant}, quant={rec.quantization})"

    @torch.no_grad()
    def generate(self, prompt, negative_prompt, backend_name, quality, width, height,
                 num_frames, steps, guidance, seed_text, progress=gr.Progress()):
        global sample_idx

        if self.pipe is None or self.current_backend != backend_name:
            status = self.load_backend(backend_name, progress=progress)
            print(status)

        quality_map = {
            "draft": (10, 3.0), "standard": (20, 5.0), "high": (30, 6.0), "max": (50, 7.5),
        }
        q_steps, q_guidance = quality_map.get(quality, (20, 5.0))
        final_steps = steps if steps > 0 else q_steps
        final_guidance = guidance if guidance > 0 else q_guidance

        seed = int(seed_text)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        progress(0.3, desc="Generating video...")
        output = self.pipe.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            num_inference_steps=int(final_steps),
            guidance_scale=float(final_guidance),
            seed=seed,
        )

        progress(0.9, desc="Saving...")
        save_path = os.path.join(self.savedir, f"{sample_idx:04d}.mp4")
        self.pipe.save(output, save_path, fps=8)
        sample_idx += 1

        return gr.Video(value=save_path)


# ============================================================================
# Legacy AnimateDiff Controller (unchanged)
# ============================================================================

class AnimateController:
    def __init__(self):
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = [PRETRAINED_SD]
        self.motion_module_list = MOTION_MODULES
        self.personalized_model_list = BACKUP_DREAMBOOTH_MODELS
        self.pipeline = None

        self.refresh_stable_diffusion()
        self.refresh_personalized_model()

        self.update_pipeline(
            stable_diffusion_dropdown=PRETRAINED_SD,
            motion_module_dropdown=default_motion_module,
            base_model_dropdown=default_dreambooth_model,
            sampler_dropdown="DDIM",
        )

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = [PRETRAINED_SD] + glob(os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = BACKUP_DREAMBOOTH_MODELS + [os.path.basename(p) for p in personalized_model_list if os.path.basename(p) not in BACKUP_DREAMBOOTH_MODELS]

    def update_pipeline(self, stable_diffusion_dropdown, motion_module_dropdown,
                        base_model_dropdown="", lora_model_dropdown="none",
                        lora_alpha_dropdown="0.6", sampler_dropdown="DDIM"):
        if "v2" in motion_module_dropdown:
            inference_config = "configs/inference/inference-v2.yaml"
        elif "v3" in motion_module_dropdown:
            inference_config = "configs/inference/inference-v3.yaml"
        else:
            inference_config = "configs/inference/inference-v1.yaml"

        unet = UNet3DConditionModel.from_pretrained_2d(
            stable_diffusion_dropdown, subfolder="unet",
            unet_additional_kwargs=OmegaConf.load(inference_config).unet_additional_kwargs
        )
        if is_xformers_available() and torch.cuda.is_available():
            unet.enable_xformers_memory_efficient_attention()

        noise_scheduler_cls = scheduler_dict[sampler_dropdown]
        noise_scheduler_kwargs = OmegaConf.load(inference_config).noise_scheduler_kwargs
        if noise_scheduler_cls == EulerDiscreteScheduler:
            noise_scheduler_kwargs.pop("steps_offset")
            noise_scheduler_kwargs.pop("clip_sample")
        elif noise_scheduler_cls == PNDMScheduler:
            noise_scheduler_kwargs.pop("clip_sample")

        pipeline = AnimationPipeline(
            unet=unet,
            vae=AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae"),
            text_encoder=CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder"),
            tokenizer=CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer"),
            scheduler=noise_scheduler_cls(**noise_scheduler_kwargs),
        )

        pipeline = load_weights(
            pipeline,
            motion_module_path=os.path.join(self.motion_module_dir, motion_module_dropdown),
            dreambooth_model_path=os.path.join(self.personalized_model_dir, base_model_dropdown) if base_model_dropdown != "" else "",
            lora_model_path=os.path.join(self.personalized_model_dir, lora_model_dropdown) if lora_model_dropdown != "none" else "",
            lora_alpha=float(lora_alpha_dropdown),
        )

        pipeline.to(device)
        self.pipeline = pipeline
        print("done.")
        return gr.Dropdown()

    def update_pipeline_alpha(self, stable_diffusion_dropdown, motion_module_dropdown,
                              base_model_dropdown="", lora_model_dropdown="none",
                              lora_alpha_dropdown="0.6", sampler_dropdown="DDIM"):
        if lora_model_dropdown == "none":
            return gr.Slider()
        self.update_pipeline(stable_diffusion_dropdown=stable_diffusion_dropdown,
                             motion_module_dropdown=motion_module_dropdown,
                             base_model_dropdown=base_model_dropdown,
                             lora_model_dropdown=lora_model_dropdown,
                             lora_alpha_dropdown=lora_alpha_dropdown,
                             sampler_dropdown=sampler_dropdown)
        return gr.Slider()

    @torch.no_grad()
    def animate(self, prompt_textbox, negative_prompt_textbox, sampler_dropdown,
                sample_step_slider, width_slider, length_slider, height_slider,
                cfg_scale_slider, seed_textbox):
        global sample_idx
        if int(seed_textbox) != -1:
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()

        sample = self.pipeline(
            prompt_textbox, negative_prompt=negative_prompt_textbox,
            num_inference_steps=sample_step_slider, guidance_scale=cfg_scale_slider,
            width=width_slider, height=height_slider, video_length=length_slider,
        ).videos

        save_sample_path = os.path.join(self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)

        json_str = json.dumps({
            "prompt": prompt_textbox, "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown, "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider, "width": width_slider,
            "height": height_slider, "video_length": length_slider, "seed": seed
        }, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str + "\n\n")

        sample_idx += 1
        return gr.Video(value=save_sample_path)


# ============================================================================
# UI
# ============================================================================

v3_controller = V3Controller()
legacy_controller = AnimateController()


def ui():
    with gr.Blocks(css=css, title="AnimateDiff V3 — Unified Video Generation") as demo:
        gr.Markdown(
            """
            # AnimateDiff V3: Unified Video Generation
            **Backends**: Wan 2.1 | HunyuanVideo | CogVideoX | LTX-Video | AnimateDiff (SD1.5/SDXL/Lightning)<br>
            **Smart VRAM**: Auto-detects your GPU and selects optimal model, quantization, and resolution<br>
            [Github](https://github.com/ai-dashboad/AnimateDiff) | [Paper](https://arxiv.org/abs/2307.04725)
            """
        )

        with gr.Tabs():
            # ====================== V3 Tab ======================
            with gr.TabItem("V3 Multi-Backend"):
                gr.Markdown(f"**GPU**: {v3_controller.get_gpu_info()}")

                with gr.Row():
                    v3_backend = gr.Dropdown(label="Backend", choices=V3Controller.BACKENDS, value="auto")
                    v3_quality = gr.Dropdown(label="Quality Preset", choices=V3Controller.QUALITY_PRESETS, value="standard")
                    v3_load_btn = gr.Button("Load Model", variant="secondary")
                    v3_status = gr.Textbox(label="Status", interactive=False, value="Not loaded")

                v3_prompt = gr.Textbox(label="Prompt", lines=2, value="a cat playing with a ball of yarn, photorealistic, 4k")
                v3_n_prompt = gr.Textbox(label="Negative Prompt", lines=1, value="bad quality, worst quality, low resolution")

                with gr.Row():
                    v3_width = gr.Slider(label="Width", value=480, minimum=256, maximum=1280, step=16)
                    v3_height = gr.Slider(label="Height", value=320, minimum=256, maximum=1280, step=16)
                    v3_frames = gr.Slider(label="Frames", value=33, minimum=8, maximum=128, step=1)

                with gr.Row():
                    v3_steps = gr.Slider(label="Steps (0=auto)", value=0, minimum=0, maximum=100, step=1)
                    v3_guidance = gr.Slider(label="Guidance (0=auto)", value=0, minimum=0, maximum=20, step=0.5)
                    v3_seed = gr.Textbox(label="Seed (-1=random)", value="-1")
                    v3_seed_btn = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")

                v3_gen_btn = gr.Button("Generate", variant="primary")
                v3_result = gr.Video(label="Generated Video", interactive=False)

                v3_seed_btn.click(fn=lambda: str(random.randint(1, int(1e8))), outputs=[v3_seed])
                v3_load_btn.click(fn=v3_controller.load_backend, inputs=[v3_backend], outputs=[v3_status])
                v3_gen_btn.click(
                    fn=v3_controller.generate,
                    inputs=[v3_prompt, v3_n_prompt, v3_backend, v3_quality, v3_width, v3_height, v3_frames, v3_steps, v3_guidance, v3_seed],
                    outputs=[v3_result],
                )

            # ====================== Legacy Tab ======================
            with gr.TabItem("AnimateDiff Legacy"):
                with gr.Column(variant="panel"):
                    gr.Markdown("### Model Checkpoints")
                    with gr.Row():
                        stable_diffusion_dropdown = gr.Dropdown(label="Pretrained Model Path", choices=legacy_controller.stable_diffusion_list, value=PRETRAINED_SD, interactive=True)
                    with gr.Row():
                        motion_module_dropdown = gr.Dropdown(label="Motion module", choices=legacy_controller.motion_module_list, value=default_motion_module, interactive=True)
                        base_model_dropdown = gr.Dropdown(label="Base Dreambooth model", choices=legacy_controller.personalized_model_list, value=default_dreambooth_model, interactive=True)
                        lora_model_dropdown = gr.Dropdown(label="LoRA model (optional)", choices=["none"] + legacy_controller.personalized_model_list, value="none", interactive=True)
                        lora_alpha_dropdown = gr.Dropdown(label="LoRA alpha", choices=["0.", "0.2", "0.4", "0.6", "0.8", "1.0"], value="0.6", interactive=True)
                        personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")

                        def update_personalized_model():
                            legacy_controller.refresh_stable_diffusion()
                            legacy_controller.refresh_personalized_model()
                            return [
                                gr.Dropdown(choices=legacy_controller.stable_diffusion_list),
                                gr.Dropdown(choices=legacy_controller.personalized_model_list),
                                gr.Dropdown(choices=["none"] + legacy_controller.personalized_model_list)
                            ]
                        personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[stable_diffusion_dropdown, base_model_dropdown, lora_model_dropdown])

                with gr.Column(variant="panel"):
                    gr.Markdown("### Generation Settings")
                    prompt_textbox = gr.Textbox(label="Prompt", lines=2, value=default_prompt)
                    negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value=default_n_prompt)

                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                sampler_dropdown = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                                sample_step_slider = gr.Slider(label="Sampling steps", value=25, minimum=10, maximum=100, step=1)
                            width_slider = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=64)
                            height_slider = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=64)
                            length_slider = gr.Slider(label="Animation length", value=16, minimum=8, maximum=64, step=1)
                            cfg_scale_slider = gr.Slider(label="CFG Scale", value=8.0, minimum=0, maximum=20)
                            with gr.Row():
                                seed_textbox = gr.Textbox(label="Seed (-1 for random)", value=default_seed)
                                seed_button = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                                seed_button.click(fn=lambda: gr.Textbox(value=random.randint(1, int(1e8))), inputs=[], outputs=[seed_textbox])
                            generate_button = gr.Button(value="Generate", variant='primary')

                        result_video = gr.Video(label="Generated Animation", interactive=False)

                    stable_diffusion_dropdown.change(fn=legacy_controller.update_pipeline, inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[stable_diffusion_dropdown])
                    motion_module_dropdown.change(fn=legacy_controller.update_pipeline, inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[motion_module_dropdown])
                    base_model_dropdown.change(fn=legacy_controller.update_pipeline, inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[base_model_dropdown])
                    lora_model_dropdown.change(fn=legacy_controller.update_pipeline, inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[lora_model_dropdown])
                    lora_alpha_dropdown.change(fn=legacy_controller.update_pipeline_alpha, inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[lora_alpha_dropdown])

                    generate_button.click(
                        fn=legacy_controller.animate,
                        inputs=[prompt_textbox, negative_prompt_textbox, sampler_dropdown, sample_step_slider,
                                width_slider, length_slider, height_slider, cfg_scale_slider, seed_textbox],
                        outputs=[result_video]
                    )

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)
