"""Visual distance — how unfamiliar the animal *looks* to a frozen vision encoder.

DINOv2 fingerprints of mask-cropped animals are averaged into a per-species prototype; the distance is
the cosine gap to the nearest *other* species' prototype (leave-one-species-out on the species split).
NaN where a species yields no usable crops. Uses ground-truth masks — no SAM 3.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.features.embed import Embedder, EmbeddingCache
from src.features.frames import animal_crop, sample_frame_indices
from src.features.pipeline import (
    CropFn,
    embed_crops,
    nearest_distance,
    prototype,
    record_annotations,
    safari_by_origin,
)
from src.splits import probe_records, reference_records

if TYPE_CHECKING:
    from src.splits import Partition


def _animal_crop_fn(
    safari: SAFARI, ann: dict, frame_index: int, mask_crop: bool, min_px: int
) -> CropFn:
    """A crop function that decodes the mask lazily (only for cache misses)."""
    return lambda frame: animal_crop(frame, safari.mask_at(ann, frame_index), mask_crop, min_px)


class VisualDistance:
    """Mask-crop embedding distance from each probe species to the nearest reference species."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize."""
        self.config = config or Config()
        self.encoder = self.config.features.visual_encoder
        self.safari = safari_by_origin(self.config)

    def _prototypes(
        self, records: list[VideoRecord], embedder: Embedder, cache: EmbeddingCache
    ) -> dict[str, np.ndarray]:
        """One prototype per species (``category_id``) from its mask-cropped animal embeddings."""
        feat = self.config.features
        items = []
        keys_by_species: dict[str, list[str]] = defaultdict(list)
        masklets_seen: dict[str, int] = defaultdict(int)
        for record in records:
            safari = self.safari[record.origin]
            for m_idx, ann in enumerate(
                record_annotations(record, self.safari, record.category_id)
            ):
                if masklets_seen[record.category_id] >= feat.max_masklets_per_species:
                    break
                masklets_seen[record.category_id] += 1
                for fi in sample_frame_indices(ann, feat.n_frames_per_masklet):
                    if fi >= len(record.file_names):
                        continue
                    key = f"{record.video_id}__{m_idx}__{fi}"
                    keys_by_species[record.category_id].append(key)
                    items.append(
                        (
                            key,
                            record,
                            fi,
                            _animal_crop_fn(safari, ann, fi, feat.mask_crop, feat.min_mask_pixels),
                        )
                    )

        vectors = embed_crops(items, self.config, embedder, cache)
        return {
            cid: proto
            for cid, keys in keys_by_species.items()
            if (proto := prototype([vectors[k] for k in keys if k in vectors])) is not None
        }

    def compute(self, partition: Partition) -> pd.Series:
        """Return ``visual_distance`` per probe species (``category_id``); ``NaN`` where no crops exist."""
        embedder = Embedder(self.encoder, self.config)
        cache = EmbeddingCache(self.config, self.encoder, self.config.features.mask_crop)
        ref = self._prototypes(reference_records(partition, self.config), embedder, cache)
        # On the species split reference and probe are the same set — reuse the prototypes.
        probe = (
            ref
            if partition.held_axis == "species"
            else self._prototypes(probe_records(partition, self.config), embedder, cache)
        )
        rows = {
            cid: (
                nearest_distance(probe[cid], ref, exclude=cid if partition.loso else None)
                if cid in probe
                else float("nan")
            )
            for cid in partition.probe_species
        }
        return pd.Series(rows, name="visual_distance", dtype="float64")
