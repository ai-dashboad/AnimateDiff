"""Test that all config files are valid YAML and have required fields."""
import pytest
import glob
from omegaconf import OmegaConf


def get_inference_configs():
    return glob.glob("configs/inference/*.yaml") + glob.glob("configs/inference/**/*.yaml")


def get_prompt_configs():
    configs = []
    for pattern in ["configs/prompts/**/*.yaml", "configs/prompts/*.yaml"]:
        configs.extend(glob.glob(pattern, recursive=True))
    return configs


class TestInferenceConfigs:
    @pytest.mark.parametrize("config_path", get_inference_configs())
    def test_config_loads(self, config_path):
        config = OmegaConf.load(config_path)
        assert config is not None


class TestPromptConfigs:
    @pytest.mark.parametrize("config_path", get_prompt_configs())
    def test_config_loads(self, config_path):
        configs = OmegaConf.load(config_path)
        assert configs is not None
        assert len(configs) > 0

    @pytest.mark.parametrize("config_path", get_prompt_configs())
    def test_required_fields(self, config_path):
        configs = OmegaConf.load(config_path)
        for i, cfg in enumerate(configs):
            assert "prompt" in cfg, f"Config entry {i} in {config_path} missing 'prompt'"
            assert "inference_config" in cfg, f"Config entry {i} in {config_path} missing 'inference_config'"
