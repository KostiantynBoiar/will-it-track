"""Environment distance (T2.3).

Goal: the scene/appearance gap between a probe location and the reference locations.
Input: an analysis :class:`~src.splits.Partition`, ``data/frames/``, and GT masks (to remove the animal).
Output: per ``location_id`` — ``environment_distance`` + ``is_night_ir`` + ``achromatic_fraction`` + a
    ``clutter`` proxy.
Method: embed the animal-masked-out background scene with DINOv2, average per location into a scene
    prototype; ``environment_distance`` = ``1 - max cosine`` to the nearest reference-location prototype
    (self-``location_id`` excluded). Night/IR is derived from colour statistics.
Done when: distances separate obviously different habitats (large on Split B, where probe locations are
    unseen); the night/IR flag is validated on samples.
Depends on: T0.3, frames on disk. Uses GT masks — no SAM 3.
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
)
from src.splits import probe_records, reference_records

if TYPE_CHECKING:
    from src.splits import Partition


def _sample(indices: list[int], n: int, n_frames: int) -> list[int]:
    """Up to ``n`` evenly-spaced indices that are in range for a video with ``n_frames`` frames."""
    picks = (
        indices
        if len(indices) <= n
        else [indices[p] for p in np.linspace(0, len(indices) - 1, n).round().astype(int)]
    )
    return sorted({p for p in picks if p < n_frames})


class EnvironmentDistance:
    """Scene-embedding distance per probe location + interpretable colour covariates."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.scene_encoder`` / ``night_ir_from_color`` / caps).
        """
        self.config = config or Config()
        self.encoder = self.config.features.scene_encoder
        self._safari = {"train": SAFARI("train", self.config), "test": SAFARI("test", self.config)}

    def _location_features(
        self,
        records: list[VideoRecord],
        embedder: Embedder,
        scene_cache: EmbeddingCache,
        stats_cache: EmbeddingCache,
    ) -> dict[str, dict]:
        """Per ``location_id``: scene prototype + achromatic fraction + mean clutter."""
        feat = self.config.features
        vecs: dict[str, list[np.ndarray]] = defaultdict(list)
        achro: dict[str, list[float]] = defaultdict(list)
        clutter: dict[str, list[float]] = defaultdict(list)
        count: dict[str, int] = defaultdict(int)
        pending: list[tuple[str, str, VideoRecord, int, list[dict], str]] = []

        for record in records:
            loc = record.location_id
            if not _is_real(loc) or count[loc] >= feat.max_frames_per_location:
                continue
            origin, _, raw_id = record.video_id.partition(":")
            anns = self._safari[origin].annotations_by_video().get(raw_id, [])
            annotated = sorted({i for a in anns for i in annotated_frame_indices(a)})
            for fi in _sample(annotated, feat.n_frames_per_masklet, len(record.file_names)):
                if count[loc] >= feat.max_frames_per_location:
                    break
                count[loc] += 1
                key = f"{record.video_id}__{fi}"
                sv, stat = scene_cache.get(key), stats_cache.get(key)
                if sv is not None and stat is not None:
                    vecs[loc].append(sv)
                    achro[loc].append(float(stat[0]))
                    clutter[loc].append(float(stat[1]))
                else:
                    pending.append((key, loc, record, fi, anns, origin))

        self._embed_pending(pending, embedder, scene_cache, stats_cache, vecs, achro, clutter)

        out: dict[str, dict] = {}
        for loc, vlist in vecs.items():
            if not vlist:
                continue
            mean = np.mean(vlist, axis=0)
            out[loc] = {
                "proto": mean / max(float(np.linalg.norm(mean)), 1e-12),
                "achromatic_fraction": float(np.mean(achro[loc])) if achro[loc] else float("nan"),
                "clutter": float(np.mean(clutter[loc])) if clutter[loc] else float("nan"),
            }
        return out

    def _embed_pending(
        self, pending, embedder, scene_cache, stats_cache, vecs, achro, clutter
    ) -> None:
        """Load + mask + embed the cache-miss frames, then fill the per-location accumulators."""
        by_record: dict[str, list] = defaultdict(list)
        for item in pending:
            by_record[item[2].video_id].append(item)
        for group in by_record.values():
            record = group[0][2]
            ensure_frames([record.file_names[p[3]] for p in group], record.origin, self.config)

        images: list = []
        meta: list = []
        fill = self.config.features.background_fill
        for key, loc, record, fi, anns, origin in pending:
            frame = load_frame(record.file_names[fi], self.config)
            if frame is None:
                continue
            masks = [
                self._safari[origin].mask_at(a, fi)
                for a in anns
                if fi < len(a.get("segmentations") or []) and (a["segmentations"][fi] is not None)
            ]
            background = masked_background(frame, masks, fill)
            images.append(background)
            meta.append(
                (
                    key,
                    loc,
                    float(frame_achromatic(frame, self.config.features.night_ir_threshold)),
                    laplacian_variance(background),
                )
            )
        if not images:
            return
        embedded = embedder.embed(images)
        for (key, loc, is_achro, clut), vec in zip(meta, embedded, strict=True):
            scene_cache.put(key, vec)
            stats_cache.put(key, np.array([is_achro, clut], dtype="float32"))
            vecs[loc].append(vec)
            achro[loc].append(is_achro)
            clutter[loc].append(clut)
        scene_cache.save()
        stats_cache.save()

    def compute(self, partition: Partition) -> pd.DataFrame:
        """Return per-probe-location ``environment_distance`` + colour covariates.

        Args:
            partition: The active split (probe locations are scored against the reference locations).

        Returns:
            A DataFrame indexed by ``location_id``.
        """
        embedder = Embedder(self.encoder, self.config)
        scene_cache = EmbeddingCache(self.config, f"{self.encoder}_bg", mask_crop=False)
        stats_cache = EmbeddingCache(self.config, "colorstats", mask_crop=False)

        ref = self._location_features(
            reference_records(partition, self.config), embedder, scene_cache, stats_cache
        )
        probe = self._location_features(
            probe_records(partition, self.config), embedder, scene_cache, stats_cache
        )

        rows = []
        for loc, info in probe.items():
            candidates = [r["proto"] for other, r in ref.items() if other != loc]
            distance = (
                1.0 - max(float(info["proto"] @ p) for p in candidates)
                if candidates
                else float("nan")
            )
            rows.append(
                {
                    "location_id": loc,
                    "environment_distance": distance,
                    "achromatic_fraction": info["achromatic_fraction"],
                    "is_night_ir": (
                        bool(info["achromatic_fraction"] > 0.5)
                        if self.config.features.night_ir_from_color
                        and not np.isnan(info["achromatic_fraction"])
                        else False
                    ),
                    "clutter": info["clutter"],
                }
            )
        return pd.DataFrame(rows).set_index("location_id")
