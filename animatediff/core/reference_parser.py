"""
Reference Parser — parse typed references from natural language or storyboard JSON.

Implements the "@" reference system inspired by SeedAnce 2.0's multi-reference approach.
References bind external files (images, videos, audio) to semantic roles (character,
camera, style, soundtrack, voice) for use by the DirectorEngine.

Supported syntaxes:
  - "@portrait.png as 韩立"           → Reference(type=image, role=character, tag="韩立")
  - "@camera_orbit.mp4 as camera"      → Reference(type=video, role=camera)
  - "@bgm.mp3 as soundtrack"           → Reference(type=audio, role=soundtrack)
  - "@narration.wav as voice:narrator" → Reference(type=audio, role=voice, tag="narrator")
  - "@style_ref.jpg as style"          → Reference(type=image, role=style)

Also parses from storyboard JSON "references" field.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

logger = logging.getLogger(__name__)

__all__ = ["Reference", "ReferenceSet", "ReferenceParser"]

# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})
VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".webm", ".gif", ".mkv"})
AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"})

ROLE_KEYWORDS = {
    "character": {"character", "char", "人物", "角色"},
    "camera": {"camera", "cam", "motion", "镜头", "运镜"},
    "style": {"style", "reference", "ref", "风格", "参考"},
    "soundtrack": {"soundtrack", "bgm", "music", "背景音乐", "配乐"},
    "voice": {"voice", "narration", "tts", "配音", "旁白"},
    "sfx": {"sfx", "sound", "effect", "音效"},
    "pose": {"pose", "skeleton", "姿势", "骨骼"},
    "face": {"face", "portrait", "脸", "肖像"},
}


def _detect_file_type(path: str) -> Literal["image", "video", "audio"]:
    """Detect file type from extension."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    raise ValueError(f"Unknown file type for '{path}'. Supported: {IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS}")


def _normalize_role(role_str: str) -> str:
    """Map role aliases to canonical role names."""
    role_lower = role_str.lower().strip()
    for canonical, aliases in ROLE_KEYWORDS.items():
        if role_lower in aliases:
            return canonical
    # If role string matches a canonical name directly, return it
    if role_lower in ROLE_KEYWORDS:
        return role_lower
    return role_lower


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """A typed reference binding a file to a semantic role.

    Attributes:
        type: File type — "image", "video", or "audio".
        path: Path to the reference file (absolute or relative).
        role: Semantic role — "character", "camera", "style", "soundtrack",
              "voice", "sfx", "pose", "face", or any custom string.
        tag: Optional sub-identifier (e.g., character name, voice profile key).
        metadata: Extra key-value metadata for backend-specific params.
    """
    type: Literal["image", "video", "audio"] = "image"
    path: str = ""
    role: str = "character"
    tag: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()

    def resolve(self, base_dir: str = "") -> "Reference":
        """Resolve relative path against a base directory."""
        p = Path(self.path)
        if not p.is_absolute() and base_dir:
            p = Path(base_dir) / p
        return Reference(
            type=self.type, path=str(p), role=self.role,
            tag=self.tag, metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict:
        d = {"type": self.type, "path": self.path, "role": self.role}
        if self.tag:
            d["tag"] = self.tag
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class ReferenceSet:
    """A collection of references with role-based querying.

    Enforces SeedAnce 2.0-like limits: up to 12 references total
    (9 image + 3 video + 3 audio).
    """
    refs: List[Reference] = field(default_factory=list)

    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIO = 3

    def add(self, ref: Reference):
        """Add a reference, enforcing capacity limits."""
        counts = self._counts()
        if ref.type == "image" and counts["image"] >= self.MAX_IMAGES:
            logger.warning(f"Image reference limit ({self.MAX_IMAGES}) reached, skipping: {ref.path}")
            return
        if ref.type == "video" and counts["video"] >= self.MAX_VIDEOS:
            logger.warning(f"Video reference limit ({self.MAX_VIDEOS}) reached, skipping: {ref.path}")
            return
        if ref.type == "audio" and counts["audio"] >= self.MAX_AUDIO:
            logger.warning(f"Audio reference limit ({self.MAX_AUDIO}) reached, skipping: {ref.path}")
            return
        self.refs.append(ref)

    def by_role(self, role: str) -> List[Reference]:
        return [r for r in self.refs if r.role == role]

    def by_type(self, type_: str) -> List[Reference]:
        return [r for r in self.refs if r.type == type_]

    def characters(self) -> List[Reference]:
        return self.by_role("character") + self.by_role("face")

    def cameras(self) -> List[Reference]:
        return self.by_role("camera")

    def audio_refs(self) -> List[Reference]:
        return self.by_type("audio")

    def style_refs(self) -> List[Reference]:
        return self.by_role("style")

    def _counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {"image": 0, "video": 0, "audio": 0}
        for r in self.refs:
            c[r.type] = c.get(r.type, 0) + 1
        return c

    def __len__(self) -> int:
        return len(self.refs)

    def __bool__(self) -> bool:
        return len(self.refs) > 0

    def summary(self) -> str:
        c = self._counts()
        roles = {}
        for r in self.refs:
            roles.setdefault(r.role, []).append(r.tag or Path(r.path).stem)
        parts = [f"{k}: {', '.join(v)}" for k, v in roles.items()]
        return f"{len(self)} refs ({c['image']}img/{c['video']}vid/{c['audio']}aud) — {'; '.join(parts)}"

    def to_list(self) -> List[dict]:
        return [r.to_dict() for r in self.refs]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Pattern: @<path> as <role>[:tag]
_AT_PATTERN = re.compile(
    r'@([^\s]+)\s+as\s+([^\s,;]+?)(?::([^\s,;]+))?(?:\s|$|,|;)',
    re.IGNORECASE,
)


class ReferenceParser:
    """Parse references from natural language strings or storyboard JSON dicts."""

    def __init__(self, base_dir: str = ""):
        """
        Args:
            base_dir: Base directory for resolving relative file paths.
        """
        self.base_dir = base_dir

    def parse_text(self, text: str) -> ReferenceSet:
        """Parse "@file as role" syntax from a text string.

        Examples:
            "@portrait.png as 韩立"
            "@orbit.mp4 as camera, @bgm.mp3 as soundtrack"
            "@voice.wav as voice:narrator"
        """
        ref_set = ReferenceSet()
        for match in _AT_PATTERN.finditer(text):
            path_str, role_str, tag_str = match.group(1), match.group(2), match.group(3)
            try:
                file_type = _detect_file_type(path_str)
            except ValueError as e:
                logger.warning(f"Skipping reference: {e}")
                continue

            role = _normalize_role(role_str)
            # If role is a character name (not a known keyword), treat as character
            if role == role_str.lower().strip() and role not in ROLE_KEYWORDS:
                tag_str = tag_str or role_str
                role = "character"

            ref = Reference(
                type=file_type,
                path=path_str,
                role=role,
                tag=tag_str or "",
            )
            if self.base_dir:
                ref = ref.resolve(self.base_dir)
            ref_set.add(ref)

        if ref_set:
            logger.info(f"Parsed {len(ref_set)} references from text: {ref_set.summary()}")
        return ref_set

    def parse_dict_list(self, refs_data: List[dict]) -> ReferenceSet:
        """Parse references from a storyboard JSON "references" field.

        Expected format per entry:
            {
                "path": "output/portraits/韩立.png",
                "role": "character",
                "tag": "韩立",
                "type": "image"  // optional, auto-detected from extension
            }
        """
        ref_set = ReferenceSet()
        for entry in refs_data:
            path_str = entry.get("path", "")
            if not path_str:
                continue

            file_type = entry.get("type")
            if not file_type:
                try:
                    file_type = _detect_file_type(path_str)
                except ValueError as e:
                    logger.warning(f"Skipping reference: {e}")
                    continue

            role = _normalize_role(entry.get("role", "character"))
            tag = entry.get("tag", "")
            metadata = {k: v for k, v in entry.items() if k not in ("path", "role", "tag", "type")}

            ref = Reference(type=file_type, path=path_str, role=role, tag=tag, metadata=metadata)
            if self.base_dir:
                ref = ref.resolve(self.base_dir)
            ref_set.add(ref)

        if ref_set:
            logger.info(f"Parsed {len(ref_set)} references from dict: {ref_set.summary()}")
        return ref_set

    def parse_characters(self, characters: Dict[str, dict]) -> ReferenceSet:
        """Convert storyboard character definitions into character references.

        This bridges the existing CharacterRef format to the new Reference system.
        """
        ref_set = ReferenceSet()
        for name, info in characters.items():
            if isinstance(info, str):
                info = {"description": info}

            image_path = info.get("image_path", "")
            if image_path:
                ref = Reference(
                    type="image",
                    path=image_path,
                    role="character",
                    tag=name,
                    metadata={
                        "description": info.get("description", ""),
                        "lora_path": info.get("lora_path"),
                        "lora_scale": info.get("lora_scale", 1.0),
                    },
                )
                if self.base_dir:
                    ref = ref.resolve(self.base_dir)
                ref_set.add(ref)

        return ref_set
