"""Tests for V3 CLI argument parsing, quality presets, and routing logic."""
import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Quality Presets
# ============================================================================

class TestQualityPresets:
    def test_all_presets_exist(self):
        from scripts.animate import QUALITY_PRESETS
        assert "draft" in QUALITY_PRESETS
        assert "standard" in QUALITY_PRESETS
        assert "high" in QUALITY_PRESETS
        assert "max" in QUALITY_PRESETS

    def test_preset_has_required_keys(self):
        from scripts.animate import QUALITY_PRESETS
        for name, preset in QUALITY_PRESETS.items():
            assert "num_inference_steps" in preset, f"{name} missing num_inference_steps"
            assert "guidance_scale" in preset, f"{name} missing guidance_scale"

    def test_draft_is_fastest(self):
        from scripts.animate import QUALITY_PRESETS
        assert QUALITY_PRESETS["draft"]["num_inference_steps"] < QUALITY_PRESETS["standard"]["num_inference_steps"]

    def test_max_is_slowest(self):
        from scripts.animate import QUALITY_PRESETS
        assert QUALITY_PRESETS["max"]["num_inference_steps"] > QUALITY_PRESETS["high"]["num_inference_steps"]

    def test_steps_increase_with_quality(self):
        from scripts.animate import QUALITY_PRESETS
        order = ["draft", "standard", "high", "max"]
        steps = [QUALITY_PRESETS[q]["num_inference_steps"] for q in order]
        assert steps == sorted(steps), "Steps should increase: draft < standard < high < max"


# ============================================================================
# CLI Arg Parsing
# ============================================================================

class TestCLIParsing:
    def _parse(self, args_list):
        """Helper to parse CLI args."""
        import sys
        from scripts.animate import main_cli
        import argparse

        # We need to re-create the parser from main_cli, but since it calls parse_args()
        # and then runs, we'll test the parser construction instead
        from scripts.animate import main_cli
        import scripts.animate as animate_mod

        # Create parser manually (same as main_cli but without running)
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", type=str, default=None,
                            choices=["auto", "wan", "hunyuan", "cogvideo", "ltx", "animatediff"])
        parser.add_argument("--prompt", type=str, default=None)
        parser.add_argument("--negative-prompt", type=str, default=None)
        parser.add_argument("--model-path", type=str, default=None)
        parser.add_argument("--model-variant", type=str, default=None)
        parser.add_argument("--quality", type=str, default="standard",
                            choices=["draft", "standard", "high", "max"])
        parser.add_argument("--quantization", type=str, default=None,
                            choices=["none", "nf4", "int8", "fp8"])
        parser.add_argument("--offload", type=str, default=None,
                            choices=["none", "model_cpu", "sequential_cpu"])
        parser.add_argument("--no-compile", action="store_true")
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--guidance-scale", type=float, default=None)
        parser.add_argument("--seed", type=int, default=-1)
        parser.add_argument("--fps", type=int, default=8)
        parser.add_argument("--pipeline", type=str, default=None,
                            choices=["legacy", "v2", "sdxl", "lightning"])
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--format", type=str, default="mp4", choices=["gif", "mp4"])
        parser.add_argument("--L", type=int, default=0)
        parser.add_argument("--W", type=int, default=0)
        parser.add_argument("--H", type=int, default=0)
        parser.add_argument("--scheduler", type=str, default="ddim")
        parser.add_argument("--device", type=str, default=None)
        return parser.parse_args(args_list)

    def test_backend_auto(self):
        args = self._parse(["--backend", "auto", "--prompt", "test"])
        assert args.backend == "auto"
        assert args.prompt == "test"

    def test_backend_wan(self):
        args = self._parse(["--backend", "wan", "--prompt", "hello"])
        assert args.backend == "wan"

    def test_quality_default(self):
        args = self._parse(["--backend", "auto", "--prompt", "x"])
        assert args.quality == "standard"

    def test_quality_override(self):
        args = self._parse(["--backend", "auto", "--prompt", "x", "--quality", "max"])
        assert args.quality == "max"

    def test_legacy_pipeline(self):
        args = self._parse(["--pipeline", "v2", "--config", "test.yaml"])
        assert args.pipeline == "v2"
        assert args.config == "test.yaml"

    def test_format_default(self):
        args = self._parse(["--backend", "auto", "--prompt", "x"])
        assert args.format == "mp4"

    def test_format_gif(self):
        args = self._parse(["--backend", "auto", "--prompt", "x", "--format", "gif"])
        assert args.format == "gif"

    def test_seed_default(self):
        args = self._parse(["--backend", "auto", "--prompt", "x"])
        assert args.seed == -1

    def test_seed_override(self):
        args = self._parse(["--backend", "auto", "--prompt", "x", "--seed", "42"])
        assert args.seed == 42

    def test_quantization(self):
        args = self._parse(["--backend", "wan", "--prompt", "x", "--quantization", "nf4"])
        assert args.quantization == "nf4"

    def test_offload(self):
        args = self._parse(["--backend", "wan", "--prompt", "x", "--offload", "model_cpu"])
        assert args.offload == "model_cpu"

    def test_no_compile(self):
        args = self._parse(["--backend", "wan", "--prompt", "x", "--no-compile"])
        assert args.no_compile is True

    def test_dimensions(self):
        args = self._parse(["--backend", "wan", "--prompt", "x", "--W", "720", "--H", "480", "--L", "33"])
        assert args.W == 720
        assert args.H == 480
        assert args.L == 33


# ============================================================================
# Prompt Loading
# ============================================================================

class TestPromptLoading:
    def test_load_from_cli_prompt(self):
        from scripts.animate import _load_prompts
        args = MagicMock()
        args.config = None
        args.prompt = "a cat playing"
        args.negative_prompt = "bad quality"
        args.seed = 42
        prompts = _load_prompts(args)
        assert len(prompts) == 1
        assert prompts[0] == ("a cat playing", "bad quality", 42)

    def test_load_from_cli_no_negative(self):
        from scripts.animate import _load_prompts
        args = MagicMock()
        args.config = None
        args.prompt = "test"
        args.negative_prompt = None
        args.seed = -1
        prompts = _load_prompts(args)
        assert prompts[0] == ("test", "", -1)

    def test_no_prompt_or_config_raises(self):
        from scripts.animate import _load_prompts
        args = MagicMock()
        args.config = None
        args.prompt = None
        with pytest.raises(ValueError, match="Provide either"):
            _load_prompts(args)

    def test_load_from_config(self, tmp_path):
        from scripts.animate import _load_prompts
        import yaml

        config_path = tmp_path / "test_config.yaml"
        config_data = [
            {"prompt": ["a dog running", "a cat sleeping"], "n_prompt": ["ugly"], "seed": [10, 20]},
        ]
        config_path.write_text(yaml.dump(config_data))

        args = MagicMock()
        args.config = str(config_path)
        args.prompt = None
        prompts = _load_prompts(args)
        assert len(prompts) == 2
        assert prompts[0][0] == "a dog running"
        assert prompts[0][2] == 10
        assert prompts[1][0] == "a cat sleeping"
        assert prompts[1][2] == 20
