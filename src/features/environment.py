"""Environment distance — how unfamiliar the *place* looks.

DINOv2 fingerprints of the animal-masked-out background are averaged into a per-location scene
prototype; the distance is the cosine gap to the nearest reference location. Colour statistics give a
night/IR flag and a clutter proxy. Uses ground-truth masks — no SAM 3.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import SAFARI, VideoRecord, _is_real
from src.features.embed import Embedder, EmbeddingCache
from src.features.frames import (
    annotated_frame_indices,
    ensure_frames,
    frame_achromatic,
    laplacian_variance,
    load_frame,
    masked_background,
    sample_evenly,
)
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


def _masks_at(safari: SAFARI, anns: list[dict], frame_index: int) -> list[np.ndarray]:
    """Decoded masks of every masklet present on a frame."""
    return [
        safari.mask_at(a, frame_index)
        for a in anns
        if frame_index < len(a.get("segmentations") or [])
        and a["segmentations"][frame_index] is not None
    ]


def _background_crop_fn(safari: SAFARI, anns: list[dict], frame_index: int, fill: str) -> CropFn:
    """A crop function that masks out the animals lazily (only for cache misses)."""
    return lambda frame: masked_background(frame, _masks_at(safari, anns, frame_index), fill)


class EnvironmentDistance:
    """Scene-embedding distance per probe location + night/IR and clutter covariates."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize."""
        self.config = config or Config()
        self.encoder = self.config.features.scene_encoder
        self.safari = safari_by_origin(self.config)

    def _location_features(
        self,
        records: list[VideoRecord],
        embedder: Embedder,
        scene_cache: EmbeddingCache,
        stats_cache: EmbeddingCache,
    ) -> dict[str, dict]:
        """Per ``location_id``: scene prototype + mean achromatic fraction + mean clutter."""
        feat = self.config.features
        items = []
        stat_items = []  # (key, record, frame_index, anns)
        keys_by_loc: dict[str, list[str]] = defaultdict(list)
        count: dict[str, int] = defaultdict(int)
        for record in records:
            loc = record.location_id
            if not _is_real(loc) or count[loc] >= feat.max_frames_per_location:
                continue
            safari = self.safari[record.origin]
            anns = record_annotations(record, self.safari)
            annotated = sorted({i for a in anns for i in annotated_frame_indices(a)})
            for fi in sample_evenly(annotated, feat.n_frames_per_masklet):
                if fi >= len(record.file_names) or count[loc] >= feat.max_frames_per_location:
                    continue
                count[loc] += 1
                key = f"{record.video_id}__{fi}"
                keys_by_loc[loc].append(key)
                items.append(
                    (key, record, fi, _background_crop_fn(safari, anns, fi, feat.background_fill))
                )
                stat_items.append((key, record, fi, anns))

        vectors = embed_crops(items, self.config, embedder, scene_cache)
        stats = self._stats(stat_items, stats_cache)

        out: dict[str, dict] = {}
        for loc, keys in keys_by_loc.items():
            proto = prototype([vectors[k] for k in keys if k in vectors])
            if proto is None:
                continue
            achro = [stats[k][0] for k in keys if k in stats]
            clutter = [stats[k][1] for k in keys if k in stats]
            out[loc] = {
                "proto": proto,
                "achromatic_fraction": float(np.mean(achro)) if achro else float("nan"),
                "clutter": float(np.mean(clutter)) if clutter else float("nan"),
            }
        return out

    def _stats(self, stat_items: list[tuple], stats_cache: EmbeddingCache) -> dict[str, np.ndarray]:
        """Per-frame ``[achromatic, clutter]`` colour stats, cached (frames already local after embedding)."""
        misses = [it for it in stat_items if stats_cache.get(it[0]) is None]
        by_record: dict[str, list[tuple]] = defaultdict(list)
        for item in misses:
            by_record[item[1].video_id].append(item)
        for group in by_record.values():
            record = group[0][1]
            ensure_frames([record.file_names[it[2]] for it in group], record.origin, self.config)

        for key, record, fi, anns in misses:
            frame = load_frame(record.file_names[fi], self.config)
            if frame is None:
                continue
            background = masked_background(
                frame,
                _masks_at(self.safari[record.origin], anns, fi),
                self.config.features.background_fill,
            )
            stats_cache.put(
                key,
                np.array(
                    [
                        float(frame_achromatic(frame, self.config.features.night_ir_threshold)),
                        laplacian_variance(background),
                    ],
                    dtype="float32",
                ),
            )
        stats_cache.save()
        return {it[0]: s for it in stat_items if (s := stats_cache.get(it[0])) is not None}

    def compute(self, partition: Partition) -> pd.DataFrame:
        """Return per-probe-location ``environment_distance`` + colour covariates (indexed by location)."""
        embedder = Embedder(self.encoder, self.config)
        scene_cache = EmbeddingCache(self.config, f"{self.encoder}_bg", mask_crop=False)
        stats_cache = EmbeddingCache(self.config, "colorstats", mask_crop=False)

        ref = self._location_features(
            reference_records(partition, self.config), embedder, scene_cache, stats_cache
        )
        probe = self._location_features(
            probe_records(partition, self.config), embedder, scene_cache, stats_cache
        )
        ref_protos = {loc: info["proto"] for loc, info in ref.items()}

        rows = []
        for loc, info in probe.items():
            night = (
                bool(info["achromatic_fraction"] > 0.5)
                if self.config.features.night_ir_from_color
                and not np.isnan(info["achromatic_fraction"])
                else False
            )
            rows.append(
                {
                    "location_id": loc,
                    "environment_distance": nearest_distance(
                        info["proto"], ref_protos, exclude=loc
                    ),
                    "achromatic_fraction": info["achromatic_fraction"],
                    "is_night_ir": night,
                    "clutter": info["clutter"],
                }
            )
        return pd.DataFrame(rows).set_index("location_id")
