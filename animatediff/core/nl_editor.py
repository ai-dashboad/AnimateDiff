"""
Natural Language Video Editor — text instruction → mask → regeneration.

Implements SeedAnce 2.0-style NL video editing by parsing text instructions
into edit operations and routing them through the VACE backend:

  1. Parse instruction → edit type (replace, remove, add, restyle)
  2. Generate mask:
     - "remove the sword" → object detection → binary mask
     - "change sky to night" → semantic segmentation → sky mask
     - "make it rain" → full-frame mask (restyle)
  3. Route to VACE backend with mask + new prompt

Also supports video extension with narrative awareness.

Requires: wan22_vace backend for mask-based editing.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

__all__ = ["EditInstruction", "NLVideoEditor"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EditInstruction:
    """Parsed video editing instruction."""
    edit_type: Literal["replace", "remove", "add", "restyle", "extend"] = "restyle"
    target_object: str = ""  # Object to edit (e.g., "sword", "sky")
    new_description: str = ""  # What to replace with (e.g., "night sky")
    mask_mode: str = "full"  # "object", "region", "semantic", "full"
    region: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2) normalized
    raw_instruction: str = ""


# ---------------------------------------------------------------------------
# Instruction parsing
# ---------------------------------------------------------------------------

# Pattern matches for edit types
_REMOVE_PATTERNS = [
    re.compile(r'(?:remove|delete|erase|get rid of)\s+(?:the\s+)?(.+)', re.IGNORECASE),
    re.compile(r'(?:去掉|删除|移除|擦除)\s*(.+)', re.IGNORECASE),
]

_REPLACE_PATTERNS = [
    re.compile(r'(?:change|replace|turn|convert)\s+(?:the\s+)?(.+?)\s+(?:to|into|with)\s+(.+)', re.IGNORECASE),
    re.compile(r'(?:把|将)\s*(.+?)\s*(?:变成|改为|换成)\s*(.+)', re.IGNORECASE),
]

_ADD_PATTERNS = [
    re.compile(r'(?:add|put|place|insert)\s+(.+?)(?:\s+(?:to|on|in|at)\s+(.+))?$', re.IGNORECASE),
    re.compile(r'(?:添加|加入|放上)\s*(.+?)(?:\s*(?:到|在)\s*(.+))?$', re.IGNORECASE),
]

_EXTEND_PATTERNS = [
    re.compile(r'(?:extend|continue|lengthen|make.+longer)', re.IGNORECASE),
    re.compile(r'(?:延长|继续|接下去)', re.IGNORECASE),
]

_RESTYLE_KEYWORDS = {
    "make it", "change style", "restyle", "artistic",
    "cinematic", "anime style", "watercolor", "oil painting",
    "change mood", "darker", "brighter", "warmer", "cooler",
    "让它", "风格化", "变暗", "变亮",
}


def parse_instruction(text: str) -> EditInstruction:
    """Parse a natural language editing instruction.

    Examples:
        "remove the sword" → EditInstruction(type=remove, target="sword")
        "change the sky to night" → EditInstruction(type=replace, target="sky", new="night")
        "make it rain" → EditInstruction(type=restyle, new="rain")
        "add a dragon in the sky" → EditInstruction(type=add, target="dragon", new="in the sky")

    Args:
        text: Natural language editing instruction.

    Returns:
        Parsed EditInstruction.
    """
    text = text.strip()

    # Check extend patterns
    for pattern in _EXTEND_PATTERNS:
        if pattern.search(text):
            return EditInstruction(
                edit_type="extend",
                new_description=text,
                mask_mode="full",
                raw_instruction=text,
            )

    # Check remove patterns
    for pattern in _REMOVE_PATTERNS:
        match = pattern.search(text)
        if match:
            target = match.group(1).strip()
            return EditInstruction(
                edit_type="remove",
                target_object=target,
                mask_mode="object",
                raw_instruction=text,
            )

    # Check replace patterns
    for pattern in _REPLACE_PATTERNS:
        match = pattern.search(text)
        if match:
            target = match.group(1).strip()
            replacement = match.group(2).strip()
            return EditInstruction(
                edit_type="replace",
                target_object=target,
                new_description=replacement,
                mask_mode="semantic",
                raw_instruction=text,
            )

    # Check add patterns
    for pattern in _ADD_PATTERNS:
        match = pattern.search(text)
        if match:
            obj = match.group(1).strip()
            location = match.group(2).strip() if match.group(2) else ""
            return EditInstruction(
                edit_type="add",
                target_object=obj,
                new_description=location,
                mask_mode="region",
                raw_instruction=text,
            )

    # Check restyle keywords
    text_lower = text.lower()
    for keyword in _RESTYLE_KEYWORDS:
        if keyword in text_lower:
            return EditInstruction(
                edit_type="restyle",
                new_description=text,
                mask_mode="full",
                raw_instruction=text,
            )

    # Default: treat as restyle
    return EditInstruction(
        edit_type="restyle",
        new_description=text,
        mask_mode="full",
        raw_instruction=text,
    )


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def _generate_full_mask(width: int, height: int) -> Image.Image:
    """Generate a full-frame white mask (edit everything)."""
    return Image.new("L", (width, height), 255)


def _generate_region_mask(
    width: int,
    height: int,
    region: Optional[Tuple[float, float, float, float]] = None,
) -> Image.Image:
    """Generate a rectangular region mask.

    Args:
        width, height: Image dimensions.
        region: Normalized (x1, y1, x2, y2) in [0, 1]. Defaults to center.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if region is None:
        # Default: center region (30% to 70%)
        region = (0.2, 0.2, 0.8, 0.8)

    x1 = int(region[0] * width)
    y1 = int(region[1] * height)
    x2 = int(region[2] * width)
    y2 = int(region[3] * height)

    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def _generate_semantic_mask(
    frame: Image.Image,
    target: str,
) -> Image.Image:
    """Generate a semantic segmentation mask for a target object.

    Uses a simple heuristic: common targets like "sky", "ground", "background"
    are mapped to image regions. For specific objects, falls back to
    CLIPSeg if available, otherwise uses a center-biased mask.
    """
    w, h = frame.size

    # Common semantic regions
    target_lower = target.lower()

    if any(kw in target_lower for kw in ["sky", "天空", "cloud", "云"]):
        # Upper portion of the image
        return _generate_region_mask(w, h, (0.0, 0.0, 1.0, 0.45))

    if any(kw in target_lower for kw in ["ground", "floor", "地面", "地板"]):
        # Lower portion
        return _generate_region_mask(w, h, (0.0, 0.6, 1.0, 1.0))

    if any(kw in target_lower for kw in ["background", "背景", "backdrop"]):
        # Everything except center
        mask = _generate_full_mask(w, h)
        draw = ImageDraw.Draw(mask)
        # Cut out center (subject area)
        cx, cy = w // 2, h // 2
        rw, rh = w // 4, h // 3
        draw.rectangle([cx - rw, cy - rh, cx + rw, cy + rh], fill=0)
        return mask

    # Try CLIPSeg for arbitrary objects
    try:
        return _clipseg_mask(frame, target)
    except (ImportError, Exception) as e:
        logger.debug(f"CLIPSeg not available for '{target}': {e}")

    # Fallback: center-biased mask
    logger.info(f"Using center-biased mask for target '{target}'")
    return _generate_region_mask(w, h, (0.2, 0.2, 0.8, 0.8))


def _clipseg_mask(frame: Image.Image, target: str) -> Image.Image:
    """Generate mask using CLIPSeg (CLIP-based segmentation)."""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    import torch

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(
        text=[target],
        images=[frame],
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Sigmoid to get probability map
    logits = outputs.logits[0]
    prob = torch.sigmoid(logits).cpu().numpy()

    # Resize to frame size
    mask_arr = (prob * 255).astype(np.uint8)
    mask = Image.fromarray(mask_arr).resize(frame.size, Image.BILINEAR)

    # Threshold
    mask_arr = np.array(mask)
    mask_arr = np.where(mask_arr > 128, 255, 0).astype(np.uint8)

    return Image.fromarray(mask_arr, mode="L")


# ---------------------------------------------------------------------------
# NL Video Editor
# ---------------------------------------------------------------------------

class NLVideoEditor:
    """Natural language video editing via VACE mask + regeneration.

    Parses text instructions into edit operations, generates appropriate
    masks, and routes to the VACE backend for inpainting/regeneration.

    Usage::

        editor = NLVideoEditor()

        # Edit a video
        result = editor.edit(
            video_frames=frames,
            instruction="remove the sword",
            backend_loader=my_loader,
        )

        # Extend a video
        result = editor.extend(
            video_frames=frames,
            instruction="the character walks into the forest",
            target_duration=10.0,
            backend_loader=my_loader,
        )
    """

    def __init__(self, default_backend: str = "wan22_vace"):
        """
        Args:
            default_backend: Default backend for editing (wan22_vace preferred).
        """
        self.default_backend = default_backend

    def parse(self, instruction: str) -> EditInstruction:
        """Parse a natural language editing instruction.

        Args:
            instruction: Text instruction (e.g., "remove the sword").

        Returns:
            Parsed EditInstruction.
        """
        return parse_instruction(instruction)

    def generate_mask(
        self,
        frame: Image.Image,
        instruction: EditInstruction,
    ) -> Image.Image:
        """Generate an edit mask from a parsed instruction.

        Args:
            frame: Reference frame for mask generation.
            instruction: Parsed editing instruction.

        Returns:
            Binary mask (PIL Image, mode "L") — white = edit region.
        """
        w, h = frame.size

        if instruction.mask_mode == "full":
            return _generate_full_mask(w, h)

        if instruction.mask_mode == "region":
            return _generate_region_mask(w, h, instruction.region)

        if instruction.mask_mode == "object":
            return _generate_semantic_mask(frame, instruction.target_object)

        if instruction.mask_mode == "semantic":
            return _generate_semantic_mask(frame, instruction.target_object)

        return _generate_full_mask(w, h)

    def edit(
        self,
        video_frames: List[Image.Image],
        instruction: str,
        backend_loader: Optional[Any] = None,
        fps: int = 24,
        **gen_kwargs,
    ) -> List[Image.Image]:
        """Edit a video using a natural language instruction.

        Args:
            video_frames: Input video frames.
            instruction: Text instruction (e.g., "change the sky to sunset").
            backend_loader: Callable to load backend by name.
            fps: Video frame rate.
            **gen_kwargs: Extra generation kwargs.

        Returns:
            Edited video frames.
        """
        if not video_frames:
            return []

        # Parse instruction
        edit = self.parse(instruction)
        logger.info(
            f"NL Edit: type={edit.edit_type}, target='{edit.target_object}', "
            f"mask_mode={edit.mask_mode}"
        )

        # Generate mask from first frame (apply to all frames)
        mask = self.generate_mask(video_frames[0], edit)

        # Build prompt for regeneration
        if edit.edit_type == "remove":
            edit_prompt = f"clean background without {edit.target_object}"
        elif edit.edit_type == "replace":
            edit_prompt = edit.new_description
        elif edit.edit_type == "add":
            edit_prompt = f"scene with {edit.target_object}"
            if edit.new_description:
                edit_prompt += f" {edit.new_description}"
        else:
            edit_prompt = edit.new_description

        # Route to VACE backend
        try:
            backend = self._load_backend(backend_loader)

            result = backend.generate(
                prompt=edit_prompt,
                source_frames=video_frames,
                mask=mask,
                mode="inpainting",
                num_frames=len(video_frames),
                **gen_kwargs,
            )
            logger.info(f"NL Edit complete: {len(result.frames)} frames")
            return result.frames

        except Exception as e:
            logger.error(f"NL Edit failed: {e}")
            return video_frames

    def extend(
        self,
        video_frames: List[Image.Image],
        instruction: str = "",
        target_duration: float = 0.0,
        backend_loader: Optional[Any] = None,
        fps: int = 24,
        **gen_kwargs,
    ) -> List[Image.Image]:
        """Extend a video with narrative-aware continuation.

        Uses the last frame as I2V reference and the instruction as
        the prompt for what should happen next.

        Args:
            video_frames: Input video frames.
            instruction: What should happen next (e.g., "walks into forest").
            target_duration: Target total duration in seconds (0 = one segment).
            backend_loader: Callable to load backend by name.
            fps: Video frame rate.
            **gen_kwargs: Extra generation kwargs.

        Returns:
            Extended video frames (original + new).
        """
        if not video_frames:
            return []

        last_frame = video_frames[-1]
        prompt = instruction or "continuation of the scene"

        # Calculate frames to generate
        if target_duration > 0:
            current_duration = len(video_frames) / fps
            extra_duration = max(0, target_duration - current_duration)
            extra_frames = int(extra_duration * fps)
        else:
            extra_frames = len(video_frames)  # Generate same length again

        if extra_frames <= 0:
            return video_frames

        # Try VACE continuation first, then I2V
        try:
            backend = self._load_backend(backend_loader)

            if hasattr(backend, "generate") and self.default_backend == "wan22_vace":
                result = backend.generate(
                    prompt=prompt,
                    source_frames=video_frames,
                    mode="continuation",
                    num_frames=extra_frames,
                    **gen_kwargs,
                )
            else:
                result = backend.generate(
                    prompt=prompt,
                    image=last_frame,
                    num_frames=extra_frames,
                    **gen_kwargs,
                )

            extended = list(video_frames) + list(result.frames)
            logger.info(
                f"Video extended: {len(video_frames)} → {len(extended)} frames"
            )
            return extended

        except Exception as e:
            logger.error(f"Video extension failed: {e}")
            return video_frames

    def _load_backend(self, loader: Optional[Any]):
        """Load the editing backend."""
        if loader:
            return loader(self.default_backend)

        from animatediff.backends import get_backend
        backend_cls = get_backend(self.default_backend)
        return backend_cls.load()
