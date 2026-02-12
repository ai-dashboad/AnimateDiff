"""
Camera movement preset utilities for AnimateDiff V4 StoryEngine.

Provides functions to load, query, and apply cinematic camera movement
presets designed for xianxia (仙侠) video generation with AI video models
(Wan 2.2, CogVideoX, LTX-Video, etc.).

Usage:
    from animatediff.data.camera_utils import load_presets, apply_preset

    presets = load_presets()
    prompt = apply_preset("a cultivator meditating on a cliff", "qi_gathering_orbit")
"""

import json
from pathlib import Path
from typing import Optional

_PRESETS_PATH = Path(__file__).parent / "camera_presets.json"

# Module-level cache so we only read the file once per process.
_cache: Optional[dict] = None


def _ensure_loaded() -> dict:
    """Load and cache the presets file. Returns the full JSON structure."""
    global _cache
    if _cache is None:
        with open(_PRESETS_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def load_presets() -> dict[str, dict]:
    """
    Load all camera presets keyed by preset ID.

    Returns:
        dict mapping preset_id -> preset dict.
    """
    data = _ensure_loaded()
    return {p["id"]: p for p in data["presets"]}


def get_preset(preset_id: str) -> dict:
    """
    Retrieve a single preset by its ID.

    Args:
        preset_id: The unique identifier (e.g. "sword_flight_fpv").

    Returns:
        The preset dict.

    Raises:
        KeyError: If the preset_id does not exist.
    """
    presets = load_presets()
    if preset_id not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise KeyError(
            f"Camera preset '{preset_id}' not found. "
            f"Available presets: {available}"
        )
    return presets[preset_id]


def apply_preset(
    prompt: str,
    preset_id: str,
    *,
    include_negative: bool = False,
    separator: str = ", ",
) -> str | tuple[str, str]:
    """
    Append a camera movement suffix to a text prompt.

    Args:
        prompt: The base text prompt for the video shot.
        preset_id: The camera preset to apply.
        include_negative: If True, also return the negative prompt suffix
            as the second element of a tuple.
        separator: String used to join the prompt and suffix.

    Returns:
        If include_negative is False:  the augmented prompt string.
        If include_negative is True:   a tuple of (augmented_prompt, negative_suffix).
    """
    preset = get_preset(preset_id)
    augmented = f"{prompt.rstrip(', ')}{separator}{preset['prompt_suffix']}"
    if include_negative:
        return augmented, preset["negative_prompt_suffix"]
    return augmented


def list_by_category(category: str) -> list[dict]:
    """
    Return all presets belonging to a given category.

    Args:
        category: One of "aerial", "dynamic", "dramatic", "intimate",
            "establishing", or "xianxia_specific".

    Returns:
        List of preset dicts in that category.

    Raises:
        ValueError: If the category name is not recognized.
    """
    data = _ensure_loaded()
    valid_categories = set(data["categories"].keys())
    if category not in valid_categories:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid categories: {', '.join(sorted(valid_categories))}"
        )
    return [p for p in data["presets"] if p["category"] == category]


def list_categories() -> dict[str, dict]:
    """
    Return the category metadata (Chinese/English names).

    Returns:
        dict mapping category_id -> {"name_zh": ..., "name_en": ...}
    """
    data = _ensure_loaded()
    return data["categories"]


def recommend_for_shot(
    camera_hint: str | None = None,
    emotion: str | None = None,
) -> list[dict]:
    """
    Suggest camera presets based on a shot's camera hint and/or emotion.

    This performs a simple keyword match against preset IDs, names, and
    prompt suffixes. Useful for auto-suggesting a preset when building a
    storyboard.

    Args:
        camera_hint: Free-text camera description from a storyboard shot
            (e.g. "slow dolly in", "crane up", "dynamic tracking").
        emotion: Emotional tone of the shot (e.g. "epic", "melancholy",
            "mysterious", "determined").

    Returns:
        List of matching presets sorted by relevance (best match first).
    """
    presets = load_presets()
    scored: list[tuple[int, dict]] = []

    # Normalise search terms.
    hint_tokens = set((camera_hint or "").lower().split())
    emotion_lower = (emotion or "").lower()

    # Emotion -> motion intensity heuristic.
    emotion_intensity_map = {
        "epic": "high",
        "fearful": "high",
        "happy": "medium",
        "determined": "medium",
        "mysterious": "low",
        "melancholy": "low",
    }

    for preset in presets.values():
        score = 0
        searchable = " ".join([
            preset["id"],
            preset["name_en"].lower(),
            preset["prompt_suffix"].lower(),
        ])

        # Score by keyword overlap with the camera hint.
        for token in hint_tokens:
            if len(token) < 3:
                continue
            if token in searchable:
                score += 10

        # Bonus if the emotion maps to the same motion intensity.
        if emotion_lower in emotion_intensity_map:
            if preset["motion_intensity"] == emotion_intensity_map[emotion_lower]:
                score += 3

        if score > 0:
            scored.append((score, preset))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def format_preset_table(presets: list[dict] | None = None) -> str:
    """
    Format presets as a human-readable ASCII table for CLI display.

    Args:
        presets: List of preset dicts. If None, shows all presets.

    Returns:
        Formatted table string.
    """
    if presets is None:
        presets = list(load_presets().values())

    header = f"{'ID':<30} {'Category':<18} {'Name':<30} {'Intensity':<10}"
    separator = "-" * len(header)
    lines = [header, separator]
    for p in presets:
        lines.append(
            f"{p['id']:<30} {p['category']:<18} "
            f"{p['name_en']:<30} {p['motion_intensity']:<10}"
        )
    return "\n".join(lines)
