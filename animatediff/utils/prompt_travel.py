"""
Prompt Travel utility — enables frame-varying prompts for AnimateDiff.

Supports two formats:
1. Dict format (used with FreeNoise/diffusers): {0: "prompt A", 8: "prompt B"}
2. YAML config format: prompt_travel: [{frame: 0, prompt: "A"}, {frame: 8, prompt: "B"}]
"""

from typing import Dict, List, Union
from omegaconf import DictConfig, ListConfig


def parse_prompt_travel(config) -> Union[str, Dict[int, str]]:
    """Parse prompt travel config into format compatible with diffusers FreeNoise.

    Args:
        config: Either a model config with prompt_travel field, or a simple prompt string/list.

    Returns:
        str for single prompt, Dict[int, str] for prompt travel.
    """
    if isinstance(config, (DictConfig, dict)):
        prompt_travel = config.get("prompt_travel", None)
        if prompt_travel is not None:
            return _parse_travel_entries(prompt_travel)

        prompt = config.get("prompt", "")
        if isinstance(prompt, (list, ListConfig)):
            return prompt[0] if len(prompt) == 1 else prompt[0]
        return prompt

    if isinstance(config, str):
        return config

    if isinstance(config, (list, ListConfig)):
        return config[0] if len(config) == 1 else config[0]

    return str(config)


def _parse_travel_entries(entries) -> Dict[int, str]:
    """Parse prompt travel entries into {frame_idx: prompt} dict.

    Supports:
    - Dict/DictConfig: {0: "prompt A", 8: "prompt B"}
    - List of dicts: [{frame: 0, prompt: "A"}, {frame: 8, prompt: "B"}]
    """
    if isinstance(entries, (dict, DictConfig)):
        return {int(k): str(v) for k, v in entries.items()}

    if isinstance(entries, (list, ListConfig)):
        result = {}
        for entry in entries:
            if isinstance(entry, (dict, DictConfig)):
                frame = int(entry.get("frame", 0))
                prompt = str(entry.get("prompt", ""))
                result[frame] = prompt
        return result

    raise ValueError(f"Unsupported prompt_travel format: {type(entries)}")
