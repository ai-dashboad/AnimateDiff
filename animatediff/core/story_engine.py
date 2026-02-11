"""
Story Engine — parse scripts into structured shot lists for multi-shot video generation.

Supports two input modes:
1. Structured JSON/YAML: explicit shot definitions
2. Natural language script: parsed via LLM into shots (optional, requires transformers)

Each shot is a ShotSpec with prompt, camera, duration, characters, etc.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CharacterRef:
    """A character referenced in a shot."""
    name: str
    image_path: Optional[str] = None  # reference image
    lora_path: Optional[str] = None  # character LoRA
    lora_scale: float = 1.0
    description: str = ""


@dataclass
class ShotSpec:
    """Specification for a single video shot/clip."""
    shot_id: int = 0
    prompt: str = ""
    negative_prompt: str = "blurry, low quality, distorted, deformed"
    duration_seconds: float = 5.0
    camera: str = ""  # e.g., "pan left", "zoom in", "static", "dolly forward"
    characters: List[str] = field(default_factory=list)  # character names
    emotion: str = ""  # e.g., "determined", "sad", "excited"
    scene: str = ""  # e.g., "mountain peak at sunset"
    transition: str = "cut"  # cut, fade, dissolve, wipe
    width: int = 0  # 0 = use default
    height: int = 0
    num_frames: int = 0
    seed: int = -1
    narration: str = ""  # narration/dialogue text for this shot (used by TTS)
    lip_sync: bool = False  # whether to apply lip sync post-processing
    voice_id: str = ""  # voice profile key for TTS (e.g., "narrator", "moDoctor")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryBoard:
    """A complete storyboard: characters + ordered shots."""
    title: str = ""
    characters: Dict[str, CharacterRef] = field(default_factory=dict)
    shots: List[ShotSpec] = field(default_factory=list)
    style: str = ""  # global style prompt, e.g., "anime, xianxia, high detail"
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly"

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.shots)

    @property
    def num_shots(self) -> int:
        return len(self.shots)


class StoryEngine:
    """Parse and manage storyboards from various input formats."""

    def __init__(self, style: str = "", negative_prompt: str = ""):
        self.default_style = style
        self.default_negative = negative_prompt

    def from_json(self, path: str) -> StoryBoard:
        """Load a storyboard from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._parse_dict(data)

    def from_yaml(self, path: str) -> StoryBoard:
        """Load a storyboard from a YAML file."""
        try:
            from omegaconf import OmegaConf
            data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except ImportError:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        return self._parse_dict(data)

    def from_dict(self, data: dict) -> StoryBoard:
        """Create a storyboard from a dictionary."""
        return self._parse_dict(data)

    def from_script(self, script: str, num_shots: int = 0) -> StoryBoard:
        """Parse a natural language script into shots.

        Uses simple heuristic parsing: splits on sentence boundaries,
        paragraph breaks, or explicit markers like [SHOT 1], [CUT TO], etc.
        For higher quality, use from_script_llm() with a local LLM.
        """
        board = StoryBoard(style=self.default_style, negative_prompt=self.default_negative)

        # Try to detect explicit shot markers
        shot_pattern = re.compile(
            r'(?:\[(?:SHOT|CUT|SCENE|镜头|场景)\s*\d*\]|'
            r'(?:SHOT|CUT TO|SCENE|镜头|场景)\s*\d+\s*[:\-])',
            re.IGNORECASE
        )

        if shot_pattern.search(script):
            # Split by explicit markers
            parts = shot_pattern.split(script)
            parts = [p.strip() for p in parts if p.strip()]
        else:
            # Split by double newlines (paragraphs) or periods for long text
            parts = [p.strip() for p in re.split(r'\n\s*\n', script) if p.strip()]
            if len(parts) == 1 and len(parts[0]) > 200:
                # Single block: split by sentences
                sentences = re.split(r'[。.！!？?]+', parts[0])
                parts = [s.strip() for s in sentences if s.strip()]

        # Limit number of shots
        if num_shots > 0:
            parts = parts[:num_shots]

        for i, text in enumerate(parts):
            camera = self._detect_camera(text)
            emotion = self._detect_emotion(text)

            shot = ShotSpec(
                shot_id=i,
                prompt=self._build_prompt(text, board.style),
                negative_prompt=board.negative_prompt,
                duration_seconds=self._estimate_duration(text),
                camera=camera,
                emotion=emotion,
                scene=text[:100],
            )
            board.shots.append(shot)

        logger.info(f"Parsed script into {len(board.shots)} shots, total ~{board.total_duration:.1f}s")
        return board

    def from_script_llm(self, script: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> StoryBoard:
        """Parse a script using a local LLM for higher quality shot decomposition.

        Requires transformers library. Uses the LLM to generate structured JSON
        which is then parsed into a StoryBoard.
        """
        system_prompt = """You are a professional storyboard artist and anime director.
Given a script or story description, break it down into individual video shots.

Output a JSON object with this structure:
{
  "title": "story title",
  "style": "anime style description",
  "characters": {
    "name": {"description": "character description"}
  },
  "shots": [
    {
      "prompt": "detailed visual description for AI video generation",
      "duration_seconds": 5.0,
      "camera": "camera movement (static/pan left/zoom in/dolly forward/etc)",
      "characters": ["character names in this shot"],
      "emotion": "emotional tone",
      "scene": "brief scene description",
      "transition": "cut/fade/dissolve"
    }
  ]
}

Rules:
- Each shot should be 3-8 seconds
- Prompts should be vivid, visual descriptions suitable for AI video generation
- Include camera movements for dynamic shots
- Keep character descriptions consistent across shots
- Output ONLY valid JSON, no markdown or explanation"""

        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading LLM: {model_name}")
            pipe = hf_pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Break this script into storyboard shots:\n\n{script}"},
            ]

            result = pipe(messages, max_new_tokens=2048, temperature=0.7, do_sample=True)
            text = result[0]["generated_text"][-1]["content"]

            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                return self._parse_dict(data)
            else:
                logger.warning("LLM did not return valid JSON, falling back to heuristic parsing")
                return self.from_script(script)

        except ImportError:
            logger.warning("transformers not available for LLM parsing, using heuristic")
            return self.from_script(script)
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}, falling back to heuristic")
            return self.from_script(script)

    def _parse_dict(self, data: dict) -> StoryBoard:
        """Parse a dictionary into a StoryBoard."""
        board = StoryBoard(
            title=data.get("title", ""),
            style=data.get("style", self.default_style),
            negative_prompt=data.get("negative_prompt", self.default_negative),
        )

        # Parse characters
        for name, info in data.get("characters", {}).items():
            if isinstance(info, str):
                info = {"description": info}
            board.characters[name] = CharacterRef(
                name=name,
                image_path=info.get("image_path"),
                lora_path=info.get("lora_path"),
                lora_scale=info.get("lora_scale", 1.0),
                description=info.get("description", ""),
            )

        # Parse shots
        for i, shot_data in enumerate(data.get("shots", [])):
            shot = ShotSpec(
                shot_id=i,
                prompt=self._build_prompt(shot_data.get("prompt", ""), board.style),
                negative_prompt=shot_data.get("negative_prompt", board.negative_prompt),
                duration_seconds=shot_data.get("duration_seconds", 5.0),
                camera=shot_data.get("camera", ""),
                characters=shot_data.get("characters", []),
                emotion=shot_data.get("emotion", ""),
                scene=shot_data.get("scene", ""),
                transition=shot_data.get("transition", "cut"),
                width=shot_data.get("width", 0),
                height=shot_data.get("height", 0),
                num_frames=shot_data.get("num_frames", 0),
                seed=shot_data.get("seed", -1),
                narration=shot_data.get("narration", ""),
                lip_sync=shot_data.get("lip_sync", False),
                voice_id=shot_data.get("voice_id", ""),
            )
            board.shots.append(shot)

        return board

    def _build_prompt(self, text: str, style: str) -> str:
        """Combine shot text with global style."""
        parts = []
        if style:
            parts.append(style)
        parts.append(text.strip())
        return ", ".join(parts)

    def _detect_camera(self, text: str) -> str:
        """Detect camera movement hints from text."""
        text_lower = text.lower()
        camera_keywords = {
            "pan left": ["pan left", "向左平移", "左移"],
            "pan right": ["pan right", "向右平移", "右移"],
            "zoom in": ["zoom in", "推进", "拉近", "close up", "特写"],
            "zoom out": ["zoom out", "拉远", "远景"],
            "dolly forward": ["dolly", "推镜", "前进"],
            "tilt up": ["tilt up", "仰拍", "向上"],
            "tilt down": ["tilt down", "俯拍", "向下"],
            "orbit": ["orbit", "环绕", "旋转"],
            "static": ["static", "静止", "定镜"],
        }
        for camera, keywords in camera_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return camera
        return "static"

    def _detect_emotion(self, text: str) -> str:
        """Detect emotional tone from text."""
        text_lower = text.lower()
        emotion_keywords = {
            "determined": ["determined", "坚定", "决心"],
            "sad": ["sad", "悲", "哀", "泪"],
            "happy": ["happy", "笑", "喜", "开心", "高兴"],
            "angry": ["angry", "怒", "愤"],
            "fearful": ["fear", "恐", "惧", "害怕"],
            "serene": ["serene", "calm", "peace", "宁静", "平静"],
            "epic": ["epic", "壮", "宏大", "震撼"],
            "mysterious": ["myster", "秘", "幽"],
        }
        for emotion, keywords in emotion_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return emotion
        return ""

    def _estimate_duration(self, text: str) -> float:
        """Estimate shot duration based on text length and content."""
        # Short descriptions → shorter shots, long descriptions → longer
        word_count = len(text.split())
        char_count = len(text)
        # Use the larger of word or character-based estimate (handles CJK)
        duration = max(word_count * 0.3, char_count * 0.1)
        return max(3.0, min(8.0, duration))  # Clamp to 3-8 seconds


def save_storyboard(board: StoryBoard, path: str):
    """Save a storyboard to JSON."""
    data = {
        "title": board.title,
        "style": board.style,
        "negative_prompt": board.negative_prompt,
        "characters": {
            name: {
                "description": c.description,
                "image_path": c.image_path,
                "lora_path": c.lora_path,
                "lora_scale": c.lora_scale,
            }
            for name, c in board.characters.items()
        },
        "shots": [
            {
                "shot_id": s.shot_id,
                "prompt": s.prompt,
                "duration_seconds": s.duration_seconds,
                "camera": s.camera,
                "characters": s.characters,
                "emotion": s.emotion,
                "scene": s.scene,
                "transition": s.transition,
                **({"narration": s.narration} if s.narration else {}),
                **({"lip_sync": s.lip_sync} if s.lip_sync else {}),
                **({"voice_id": s.voice_id} if s.voice_id else {}),
            }
            for s in board.shots
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved storyboard to {path}")
