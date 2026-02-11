"""
Character Manager — manages character reference images and LoRAs for consistent identity.

Provides:
- Character profile storage (name, reference images, LoRA paths)
- Prompt building with character descriptions
- LoRA scheduling for multi-character shots
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CharacterProfile:
    """Complete profile for a character."""
    name: str
    description: str = ""
    reference_images: List[str] = field(default_factory=list)  # paths to ref images
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    trigger_word: str = ""  # LoRA trigger word, e.g., "hanli_character"

    @property
    def primary_image(self) -> Optional[str]:
        return self.reference_images[0] if self.reference_images else None


class CharacterManager:
    """Manage characters for multi-shot video generation."""

    def __init__(self):
        self.characters: Dict[str, CharacterProfile] = {}

    def add(self, name: str, **kwargs) -> CharacterProfile:
        """Register a character."""
        profile = CharacterProfile(name=name, **kwargs)
        self.characters[name] = profile
        logger.info(f"Registered character: {name}")
        return profile

    def get(self, name: str) -> Optional[CharacterProfile]:
        return self.characters.get(name)

    def get_loras_for_shot(self, character_names: List[str]) -> List[Tuple[str, float]]:
        """Get LoRA paths and scales for characters in a shot."""
        loras = []
        for name in character_names:
            char = self.characters.get(name)
            if char and char.lora_path:
                loras.append((char.lora_path, char.lora_scale))
        return loras

    def build_character_prompt(self, character_names: List[str]) -> str:
        """Build a prompt fragment describing the characters in a shot."""
        parts = []
        for name in character_names:
            char = self.characters.get(name)
            if char:
                desc = char.trigger_word or char.description or char.name
                parts.append(desc)
        return ", ".join(parts)

    def get_reference_image(self, name: str) -> Optional[Image.Image]:
        """Load the primary reference image for a character."""
        char = self.characters.get(name)
        if char and char.primary_image:
            return Image.open(char.primary_image).convert("RGB")
        return None

    def save(self, path: str):
        """Save character profiles to JSON."""
        data = {}
        for name, char in self.characters.items():
            data[name] = {
                "description": char.description,
                "reference_images": char.reference_images,
                "lora_path": char.lora_path,
                "lora_scale": char.lora_scale,
                "trigger_word": char.trigger_word,
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        """Load character profiles from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, info in data.items():
            self.add(name, **info)
        logger.info(f"Loaded {len(data)} characters from {path}")

    @classmethod
    def from_storyboard(cls, board) -> "CharacterManager":
        """Build a CharacterManager from a StoryBoard's character definitions."""
        mgr = cls()
        for name, char_ref in board.characters.items():
            mgr.add(
                name=name,
                description=char_ref.description,
                reference_images=[char_ref.image_path] if char_ref.image_path else [],
                lora_path=char_ref.lora_path,
                lora_scale=char_ref.lora_scale,
            )
        return mgr
