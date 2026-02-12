"""
Identity Keeper — cross-shot character identity preservation.

Approximates SeedAnce 2.0's training-time character consistency by using
face embeddings (InsightFace/ArcFace), clothing descriptors (CLIP), and
identity enforcement via IP-Adapter, LoRA, or reference-image conditioning.

Workflow:
  1. extract_identity(image) → IdentityEmbedding (face + clothing + body)
  2. verify_identity(frame, identity) → similarity score
  3. enforce_identity(shot_spec, identity) → generation kwargs with ID conditioning
  4. build_character_sheet(storyboard) → per-character identities for reuse
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "IdentityEmbedding",
    "IdentityScore",
    "IdentityKeeper",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IdentityEmbedding:
    """Character identity representation for cross-shot consistency.

    Stores face embedding, clothing descriptor, and body proportions
    extracted from a reference image.
    """
    character_name: str = ""
    face_embedding: Optional[np.ndarray] = None   # (512,) ArcFace embedding
    clothing_embedding: Optional[np.ndarray] = None  # (512,) CLIP embedding of clothing
    body_proportions: Dict[str, float] = field(default_factory=dict)
    reference_image: Optional[Image.Image] = None  # Original reference
    reference_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_face(self) -> bool:
        return self.face_embedding is not None

    @property
    def has_clothing(self) -> bool:
        return self.clothing_embedding is not None


@dataclass
class IdentityScore:
    """Identity verification score between a generated frame and a reference."""
    overall: float = 0.0
    face_similarity: float = 0.0
    clothing_similarity: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)

    @property
    def is_match(self) -> bool:
        """Whether the identity is considered a match (>0.6 overall)."""
        return self.overall > 0.6

    def summary(self) -> str:
        return (
            f"identity={self.overall:.3f} "
            f"face={self.face_similarity:.3f} "
            f"clothing={self.clothing_similarity:.3f}"
        )


# ---------------------------------------------------------------------------
# Face embedding extraction
# ---------------------------------------------------------------------------

def _try_extract_face_insightface(image: Image.Image) -> Optional[np.ndarray]:
    """Extract face embedding using InsightFace/ArcFace.

    Returns (512,) float32 embedding or None.
    """
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        img_arr = np.array(image.convert("RGB"))
        # InsightFace expects BGR
        img_bgr = img_arr[:, :, ::-1]
        faces = app.get(img_bgr)

        if not faces:
            logger.debug("InsightFace: no face detected")
            return None

        # Return embedding of the largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = largest.embedding
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    except ImportError:
        logger.debug("InsightFace not available")
        return None
    except Exception as e:
        logger.debug(f"InsightFace extraction failed: {e}")
        return None


def _try_extract_face_clip(image: Image.Image) -> Optional[np.ndarray]:
    """Fallback: extract face region embedding using CLIP.

    Less discriminative than ArcFace but works without InsightFace.
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch

        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).eval()

        # Crop face region (center 60% of image, upper half)
        w, h = image.size
        face_region = image.crop((
            int(w * 0.2), int(h * 0.05),
            int(w * 0.8), int(h * 0.55),
        ))

        inputs = processor(images=[face_region], return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            embedding = features[0].cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    except (ImportError, Exception) as e:
        logger.debug(f"CLIP face extraction failed: {e}")
        return None


def _extract_clothing_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """Extract clothing descriptor using CLIP on the body region."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch

        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).eval()

        # Crop body region (center, lower portion — below face)
        w, h = image.size
        body_region = image.crop((
            int(w * 0.15), int(h * 0.35),
            int(w * 0.85), int(h * 0.95),
        ))

        inputs = processor(images=[body_region], return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            embedding = features[0].cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    except (ImportError, Exception) as e:
        logger.debug(f"Clothing embedding extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Identity Keeper
# ---------------------------------------------------------------------------

class IdentityKeeper:
    """Cross-shot character identity preservation.

    Extracts, stores, verifies, and enforces character identity across
    shots in a multi-shot storyboard.

    Usage::

        keeper = IdentityKeeper()

        # Extract identity from reference image
        identity = keeper.extract_identity(portrait_image, name="韩立")

        # Verify identity in a generated frame
        score = keeper.verify_identity(generated_frame, identity)
        print(f"Match: {score.is_match}, score: {score.overall:.3f}")

        # Enforce identity in generation kwargs
        kwargs = keeper.enforce_identity(shot_spec, identity, method="reference")

        # Build character sheet from storyboard
        sheet = keeper.build_character_sheet(storyboard)
    """

    def __init__(
        self,
        face_backend: str = "auto",
        similarity_threshold: float = 0.6,
    ):
        """
        Args:
            face_backend: "insightface", "clip", or "auto" (try insightface first).
            similarity_threshold: Minimum score to consider identity a match.
        """
        self.face_backend = face_backend
        self.similarity_threshold = similarity_threshold
        self._character_sheet: Dict[str, IdentityEmbedding] = {}

        logger.info(
            f"IdentityKeeper: face_backend={face_backend}, "
            f"threshold={similarity_threshold}"
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_identity(
        self,
        image: Image.Image,
        name: str = "",
    ) -> IdentityEmbedding:
        """Extract identity features from a reference image.

        Extracts face embedding (InsightFace or CLIP), clothing descriptor
        (CLIP), and basic body proportions.

        Args:
            image: Reference portrait/full-body image (PIL).
            name: Character name for tracking.

        Returns:
            IdentityEmbedding with extracted features.
        """
        identity = IdentityEmbedding(
            character_name=name,
            reference_image=image.copy(),
        )

        # Face embedding
        face_emb = None
        if self.face_backend in ("insightface", "auto"):
            face_emb = _try_extract_face_insightface(image)
        if face_emb is None and self.face_backend in ("clip", "auto"):
            face_emb = _try_extract_face_clip(image)

        identity.face_embedding = face_emb
        if face_emb is not None:
            logger.info(f"Face embedding extracted for '{name}' (dim={face_emb.shape[0]})")
        else:
            logger.warning(f"No face embedding extracted for '{name}'")

        # Clothing embedding
        identity.clothing_embedding = _extract_clothing_embedding(image)
        if identity.clothing_embedding is not None:
            logger.debug(f"Clothing embedding extracted for '{name}'")

        # Body proportions (simple aspect ratio + face-to-body ratio)
        w, h = image.size
        identity.body_proportions = {
            "aspect_ratio": w / max(h, 1),
            "image_width": w,
            "image_height": h,
        }

        return identity

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_identity(
        self,
        frame: Image.Image,
        identity: IdentityEmbedding,
    ) -> IdentityScore:
        """Verify whether a generated frame matches a character identity.

        Compares face and clothing embeddings between the frame and the
        stored identity reference.

        Args:
            frame: Generated video frame (PIL).
            identity: Reference identity embedding.

        Returns:
            IdentityScore with per-dimension similarity.
        """
        score = IdentityScore()
        details: Dict[str, float] = {}

        # Face similarity
        if identity.has_face:
            frame_face = None
            if self.face_backend in ("insightface", "auto"):
                frame_face = _try_extract_face_insightface(frame)
            if frame_face is None and self.face_backend in ("clip", "auto"):
                frame_face = _try_extract_face_clip(frame)

            if frame_face is not None:
                sim = float(np.dot(identity.face_embedding, frame_face))
                score.face_similarity = max(0.0, sim)
                details["face_cosine_sim"] = sim
            else:
                details["face_note"] = "no face detected in frame"

        # Clothing similarity
        if identity.has_clothing:
            frame_clothing = _extract_clothing_embedding(frame)
            if frame_clothing is not None:
                sim = float(np.dot(identity.clothing_embedding, frame_clothing))
                score.clothing_similarity = max(0.0, sim)
                details["clothing_cosine_sim"] = sim

        # Overall: weighted combination
        if identity.has_face and score.face_similarity > 0:
            score.overall = 0.7 * score.face_similarity + 0.3 * score.clothing_similarity
        elif score.clothing_similarity > 0:
            score.overall = score.clothing_similarity
        else:
            score.overall = 0.0

        score.details = details
        logger.debug(f"Identity verification for '{identity.character_name}': {score.summary()}")
        return score

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def enforce_identity(
        self,
        gen_kwargs: Dict[str, Any],
        identity: IdentityEmbedding,
        method: Literal["reference", "ip_adapter", "lora"] = "reference",
    ) -> Dict[str, Any]:
        """Add identity conditioning to generation kwargs.

        Methods:
          - "reference": Use character reference image as I2V input (simplest)
          - "ip_adapter": Inject face embedding via IP-Adapter (if available)
          - "lora": Apply character-specific LoRA weights

        Args:
            gen_kwargs: Base generation kwargs to augment.
            identity: Character identity to enforce.
            method: Enforcement method.

        Returns:
            Updated gen_kwargs with identity conditioning.
        """
        kwargs = dict(gen_kwargs)

        if method == "reference":
            # Use reference image as I2V conditioning
            if identity.reference_image is not None:
                kwargs["image"] = identity.reference_image
                logger.debug(
                    f"Identity enforcement (reference): "
                    f"using {identity.character_name}'s portrait as I2V ref"
                )

        elif method == "ip_adapter":
            # IP-Adapter: inject face embedding into cross-attention
            if identity.face_embedding is not None:
                kwargs["_ip_adapter_embedding"] = identity.face_embedding
                if identity.reference_image is not None:
                    kwargs["ip_adapter_image"] = identity.reference_image
                logger.debug(f"Identity enforcement (ip_adapter): injected face embedding")

        elif method == "lora":
            # LoRA: per-character fine-tuned weights
            lora_path = identity.metadata.get("lora_path")
            if lora_path and Path(lora_path).exists():
                kwargs["_lora_path"] = lora_path
                kwargs["_lora_scale"] = identity.metadata.get("lora_scale", 1.0)
                logger.debug(f"Identity enforcement (lora): {lora_path}")
            elif identity.reference_image is not None:
                # Fallback to reference method if no LoRA available
                kwargs["image"] = identity.reference_image
                logger.debug("Identity enforcement: no LoRA found, falling back to reference")

        return kwargs

    # ------------------------------------------------------------------
    # Character sheet management
    # ------------------------------------------------------------------

    def build_character_sheet(
        self,
        storyboard: Any,
    ) -> Dict[str, IdentityEmbedding]:
        """Build identity embeddings for all characters in a storyboard.

        Loads each character's reference image, extracts identity features,
        and stores them for cross-shot reuse.

        Args:
            storyboard: StoryBoard instance with characters dict.

        Returns:
            Dict mapping character name → IdentityEmbedding.
        """
        sheet: Dict[str, IdentityEmbedding] = {}

        characters = getattr(storyboard, "characters", {})
        for name, char_ref in characters.items():
            image_path = getattr(char_ref, "image_path", None)
            if not image_path or not Path(image_path).exists():
                logger.warning(f"Character '{name}': no image at {image_path}")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                identity = self.extract_identity(image, name=name)
                identity.reference_path = str(image_path)

                # Store LoRA info in metadata
                lora_path = getattr(char_ref, "lora_path", None)
                if lora_path:
                    identity.metadata["lora_path"] = lora_path
                    identity.metadata["lora_scale"] = getattr(char_ref, "lora_scale", 1.0)

                sheet[name] = identity
                logger.info(
                    f"Character sheet: '{name}' — "
                    f"face={'yes' if identity.has_face else 'no'}, "
                    f"clothing={'yes' if identity.has_clothing else 'no'}"
                )

            except Exception as e:
                logger.warning(f"Failed to extract identity for '{name}': {e}")

        self._character_sheet = sheet
        logger.info(f"Character sheet built: {len(sheet)} characters")
        return sheet

    def get_identity(self, character_name: str) -> Optional[IdentityEmbedding]:
        """Get a character's identity from the built sheet."""
        return self._character_sheet.get(character_name)

    def verify_shot_identity(
        self,
        frames: List[Image.Image],
        character_names: List[str],
        sample_count: int = 3,
    ) -> Dict[str, IdentityScore]:
        """Verify character identity across a set of frames.

        Samples keyframes and verifies each expected character.

        Args:
            frames: Generated video frames.
            character_names: Expected character names in this shot.
            sample_count: Number of frames to sample.

        Returns:
            Dict mapping character name → average IdentityScore.
        """
        if not frames or not character_names:
            return {}

        # Sample keyframes
        indices = [0, len(frames) // 2, len(frames) - 1][:sample_count]
        indices = [i for i in indices if i < len(frames)]

        results: Dict[str, IdentityScore] = {}
        for name in character_names:
            identity = self.get_identity(name)
            if identity is None:
                logger.debug(f"No identity stored for '{name}', skipping verification")
                continue

            scores = []
            for idx in indices:
                score = self.verify_identity(frames[idx], identity)
                scores.append(score)

            # Average scores
            avg = IdentityScore(
                overall=float(np.mean([s.overall for s in scores])),
                face_similarity=float(np.mean([s.face_similarity for s in scores])),
                clothing_similarity=float(np.mean([s.clothing_similarity for s in scores])),
            )
            results[name] = avg

            logger.info(
                f"Shot identity check '{name}': "
                f"overall={avg.overall:.3f} "
                f"(face={avg.face_similarity:.3f}, cloth={avg.clothing_similarity:.3f})"
            )

        return results
