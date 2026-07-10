"""Visual distance (T2.2).

Goal: how unfamiliar the animal *looks* to a frozen vision encoder.
Input: an analysis :class:`~src.splits.Partition`, ``data/frames/``, and the GT masks (to crop).
Output: ``visual_distance`` per probe species (``category_id``).
Method: embed MASK-CROPPED animal frames with DINOv2 (§9 DO); represent each species by a prototype
    (re-normalised mean of L2-normalised crop embeddings); distance = ``1 - max cosine`` to the nearest
    reference-species prototype, with leave-one-species-out self-exclusion on Split A.
Done when: distances are stable under resampling; cropped-vs-uncropped changes the ranking (background is
    not the dominant signal). ``NaN`` where a species yields no usable crops.
Depends on: T0.3, frames on disk (fetch-or-skip). Uses GT masks — no SAM 3.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.features.embed import Embedder, EmbeddingCache
from src.features.frames import animal_crop, ensure_frames, load_frame, sample_frame_indices
from src.splits import probe_records, reference_records

if TYPE_CHECKING:
    from src.splits import Partition


def _prototype(vectors: list[np.ndarray]) -> np.ndarray | None:
    """Re-normalised mean of L2-normalised vectors (the ``nearest_prototype`` representation)."""
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


class VisualDistance:
    """DINOv2 mask-crop embedding distance from each probe species to the nearest reference prototype."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.visual_encoder`` / ``mask_crop`` / sampling caps).
        """
        self.config = config or Config()
        self.encoder = self.config.features.visual_encoder
        self._safari = {"train": SAFARI("train", self.config), "test": SAFARI("test", self.config)}

    def _masklets(self, record: VideoRecord) -> list[dict]:
        """The record's own-species annotations (a video may carry multiple category_ids)."""
        origin, _, raw_id = record.video_id.partition(":")
        return [
            a
            for a in self._safari[origin].annotations_by_video().get(raw_id, [])
            if str(a["category_id"]) == record.category_id
        ]

    def _prototypes(
        self, records: list[VideoRecord], embedder: Embedder, cache: EmbeddingCache
    ) -> dict[str, np.ndarray]:
        """Build one prototype per species (``category_id``) from mask-cropped animal embeddings."""
        feat = self.config.features
        keys_by_species: dict[str, list[str]] = defaultdict(list)
        masklets_seen: dict[str, int] = defaultdict(int)
        pending: list[tuple[str, object]] = []  # (key, crop) for cache misses

        for record in records:
            origin = record.origin
            for m_idx, ann in enumerate(self._masklets(record)):
                if masklets_seen[record.category_id] >= feat.max_masklets_per_species:
                    break
                masklets_seen[record.category_id] += 1
                frame_idxs = [
                    fi
                    for fi in sample_frame_indices(ann, feat.n_frames_per_masklet, self.config.seed)
                    if fi < len(record.file_names)
                ]
                keys = [f"{record.video_id}__{m_idx}__{fi}" for fi in frame_idxs]
                keys_by_species[record.category_id].extend(keys)

                misses = [
                    (k, fi) for k, fi in zip(keys, frame_idxs, strict=True) if cache.get(k) is None
                ]
                if not misses:
                    continue
                ensure_frames([record.file_names[fi] for _, fi in misses], origin, self.config)
                for key, fi in misses:
                    frame = load_frame(record.file_names[fi], self.config)
                    if frame is None:
                        continue
                    mask = self._safari[origin].mask_at(ann, fi)
                    crop = animal_crop(frame, mask, feat.mask_crop, feat.min_mask_pixels)
                    if crop is not None:
                        pending.append((key, crop))

        if pending:
            vecs = embedder.embed([crop for _, crop in pending])
            for (key, _), vec in zip(pending, vecs, strict=True):
                cache.put(key, vec)
        cache.save()

        prototypes: dict[str, np.ndarray] = {}
        for cid, keys in keys_by_species.items():
            proto = _prototype([v for k in keys if (v := cache.get(k)) is not None])
            if proto is not None:
                prototypes[cid] = proto
        return prototypes

    def compute(self, partition: Partition) -> pd.Series:
        """Return ``visual_distance`` per probe species (``category_id``).

        Args:
            partition: The active split (its ``loso`` flag drives self-exclusion).

        Returns:
            A Series indexed by probe ``category_id``; ``NaN`` where no usable crops exist.
        """
        embedder = Embedder(self.encoder, self.config)
        cache = EmbeddingCache(self.config, self.encoder, self.config.features.mask_crop)
        ref = self._prototypes(reference_records(partition, self.config), embedder, cache)
        # Split A: reference and probe are the same present set — reuse the prototypes.
        probe = (
            ref
            if partition.held_axis == "species"
            else self._prototypes(probe_records(partition, self.config), embedder, cache)
        )

        rows: dict[str, float] = {}
        for cid in partition.probe_species:
            if cid not in probe:
                rows[cid] = float("nan")
                continue
            candidates = [p for other, p in ref.items() if not (partition.loso and other == cid)]
            rows[cid] = (
                1.0 - max(float(probe[cid] @ p) for p in candidates) if candidates else float("nan")
            )
        return pd.Series(rows, name="visual_distance", dtype="float64")
