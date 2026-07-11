"""Frozen image encoders (DINOv2 / CLIP) for the visual + environment features.

The only feature module that imports torch. :class:`Embedder` maps PIL images to L2-normalised float32
vectors; :class:`EmbeddingCache` persists them keyed by ``(encoder, mask_crop)`` so re-runs skip the
forward pass. DINOv2 is loaded via ``timm`` (weights from the HF cache), CLIP via ``open_clip`` — both
download once on first use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import timm
import torch
from PIL import Image

from src.config import Config

_DINOV2_MODEL = "vit_base_patch14_dinov2"
_CLIP_MODEL = ("ViT-B-32", "openai")


class Embedder:
    """A lazily-loaded, cached frozen encoder: ``embed(images) -> (N, D)`` L2-normalised float32."""

    def __init__(self, encoder: str | None = None, config: Config | None = None) -> None:
        """Initialize (the model is loaded on first ``embed``).

        Args:
            encoder: ``"dinov2"`` or ``"clip"`` (defaults to ``features.visual_encoder``).
            config: Project config (``features.embed_device`` / ``embed_batch``).
        """
        self.config = config or Config()
        self.encoder = encoder or self.config.features.visual_encoder
        self._model = None
        self._transform = None
        self._encode = None
        self._device = "cpu"

    def _resolve_device(self) -> str:
        want = self.config.features.embed_device
        if want == "cuda" and torch.cuda.is_available():
            return "cuda"
        if want == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        self._device = self._resolve_device()
        if self.encoder == "dinov2":
            model = timm.create_model(_DINOV2_MODEL, pretrained=True, num_classes=0)
            data_cfg = timm.data.resolve_model_data_config(model)
            self._transform = timm.data.create_transform(**data_cfg, is_training=False)
            self._encode = model.forward
        elif self.encoder == "clip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                _CLIP_MODEL[0], pretrained=_CLIP_MODEL[1]
            )
            self._transform = preprocess
            self._encode = model.encode_image
        else:
            raise ValueError(f"unknown encoder {self.encoder!r} (expected 'dinov2' or 'clip')")
        model.eval().to(self._device)
        self._model = model

    def _encode_batch(self, batch: list[Image.Image]) -> np.ndarray:
        """Encode one batch to a float32 array, falling back to CPU if an MPS op is unsupported."""
        x = torch.stack([self._transform(im) for im in batch]).to(self._device)
        try:
            feats = self._encode(x)
        except (RuntimeError, NotImplementedError):  # MPS op unsupported -> CPU fallback
            self._device = "cpu"
            self._model.to("cpu")
            feats = self._encode(x.to("cpu"))
        return feats.float().cpu().numpy()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Embed images to L2-normalised float32 vectors, shape ``(len(images), D)``."""
        if not images:
            return np.zeros((0, 0), dtype="float32")
        self._load()
        batch_size = self.config.features.embed_batch
        with torch.no_grad():
            chunks = [
                self._encode_batch(images[start : start + batch_size])
                for start in range(0, len(images), batch_size)
            ]
        vecs = np.concatenate(chunks, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / np.clip(norms, 1e-12, None)).astype("float32")


class EmbeddingCache:
    """A disk-backed ``key -> vector`` cache, one file per ``(encoder, mask_crop)``."""

    def __init__(self, config: Config, encoder: str, mask_crop: bool) -> None:
        """Load any existing cache for this encoder/crop setting into memory."""
        self.path: Path = (
            config.paths.outputs_root
            / config.features.embeddings_subdir
            / f"{encoder}_crop{int(mask_crop)}.npz"
        )
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=True) as data:
                self._vectors = dict(zip(data["keys"].tolist(), data["vecs"], strict=False))

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached vector for ``key`` (or ``None``)."""
        return self._vectors.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        """Store a vector under ``key``."""
        self._vectors[key] = vector.astype("float32")

    def save(self) -> Path:
        """Persist the cache to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._vectors.keys()), dtype=object)
        vecs = (
            np.stack(list(self._vectors.values())) if self._vectors else np.zeros((0, 0), "float32")
        )
        np.savez(self.path, keys=keys, vecs=vecs)
        return self.path
