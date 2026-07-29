"""SAM 3 familiarity proxy (T2.5).

Hedges against the unknown true pretraining set by measuring representational familiarity directly: how
separable/typical each test species is in **SAM 3's own feature space**. Mask-cropped animals are embedded with
SAM 3's vision encoder (:class:`~src.features.sam3_embed.Sam3Embedder`) exactly as the visual distance embeds
them with DINOv2; the per-species vectors are then reduced to one ``familiarity_proxy`` value per species by one
of three metrics (``features.familiarity_metric``), keyed by ``category_id`` and NaN where a species yields no
usable crops.

Sign convention: **higher = more separable / more familiar** (SAM 3 represents the species as a distinct,
well-formed cluster). Metrics:

* ``silhouette`` --- ``(inter - intra) / max(inter, intra)``: a tight own cluster (small ``intra``) that is far
  from the nearest other species (large ``inter``) scores high. The genuine separability signal; least
  reference-dependent (``intra`` needs no reference).
* ``nearest_prototype`` --- the cosine gap to the nearest *other* species' prototype (i.e. the visual distance
  computed with SAM 3's encoder instead of DINOv2); most directly comparable to ``visual_distance``.
* ``mahalanobis`` --- typicality of the species' prototype under the reference feature Gaussian.

This is **before-running** (an image property, no SAM 3 tracking output --- non-circular, unlike the ATC
confidence) and **label-free**. It is expected to *harden* H0: separability confounds with the animal-size
effect that already dissolved ``visual_distance``. The experiment (:mod:`src.analysis.familiarity_experiment`)
controls for size and runs a head-to-head against ``visual_distance`` at the same leave-species-out bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.features.embed import EmbeddingCache
from src.features.pipeline import (
    embed_crops,
    gaussian_stats,
    nearest_distance,
    prototype,
    record_annotations,
    safari_by_origin,
)
from src.features.sam3_embed import Sam3Embedder
from src.features.visual import _animal_crop_fn

if TYPE_CHECKING:
    from src.splits import Partition


def silhouette(target_vecs: list[np.ndarray], prototypes: dict[str, np.ndarray],
               self_cid: str, exclude: str | None, eps: float = 1e-12) -> float:
    """``(inter - intra) / max(inter, intra)`` separability of a species in embedding space.

    ``inter`` is the cosine distance from the species' prototype to the nearest *other* species' prototype
    (``1 - max cosine``); ``intra`` is the mean cosine distance of the species' vectors to their own prototype.
    High = tight, well-isolated cluster = "SAM 3 represents this species distinctly". ``NaN`` if the species has
    fewer than two vectors or has no other species to compare against.
    """
    proto = prototypes.get(self_cid)
    if proto is None or len(target_vecs) < 2:
        return float("nan")
    inter = nearest_distance(proto, prototypes, exclude=exclude)
    if not np.isfinite(inter):
        return float("nan")
    intra = float(np.mean([1.0 - float(np.asarray(v) @ proto) for v in target_vecs]))
    return (inter - intra) / max(inter, intra, eps)


def mahalanobis(proto: np.ndarray | None, pool: list[np.ndarray], eps: float = 1e-6) -> float:
    """Mahalanobis distance of a species' prototype to the reference feature Gaussian (``NaN`` if degenerate).

    Low = typical of the reference distribution = familiar. A pseudo-inverse with ``eps*I`` regularises the
    high-dimensional, rank-deficient covariance.
    """
    if proto is None or len(pool) < 2:
        return float("nan")
    mu, cov = gaussian_stats(np.asarray(pool))
    inv = np.linalg.pinv(cov + eps * np.eye(cov.shape[0]))
    diff = np.asarray(proto, dtype=np.float64) - mu
    return float(np.sqrt(max(float(diff @ inv @ diff), 0.0)))


class FamiliarityProxy:
    """Separability of each test species in SAM 3's own feature space (``familiarity_proxy``)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (SAFARI loaders are opened lazily on first ``compute``)."""
        self.config = config or Config()
        self._safari: dict[str, SAFARI] | None = None

    @property
    def safari(self) -> dict[str, SAFARI]:
        """Cached per-origin SAFARI loaders (opened on first access)."""
        if self._safari is None:
            self._safari = safari_by_origin(self.config)
        return self._safari

    def _vectors_by_species(
        self, records: list[VideoRecord], embedder: Sam3Embedder, cache: EmbeddingCache
    ) -> dict[str, list[np.ndarray]]:
        """Per-species (``category_id``) list of mask-cropped SAM 3 embeddings (mirrors ``VisualDistance``)."""
        from collections import defaultdict

        from src.features.frames import sample_frame_indices

        feat = self.config.features
        items = []
        keys_by_species: dict[str, list[str]] = defaultdict(list)
        masklets_seen: dict[str, int] = defaultdict(int)
        for record in records:
            safari = self.safari[record.origin]
            for m_idx, ann in enumerate(record_annotations(record, self.safari, record.category_id)):
                if masklets_seen[record.category_id] >= feat.max_masklets_per_species:
                    break
                masklets_seen[record.category_id] += 1
                for fi in sample_frame_indices(ann, feat.n_frames_per_masklet):
                    if fi >= len(record.file_names):
                        continue
                    key = f"{record.video_id}__{m_idx}__{fi}"
                    keys_by_species[record.category_id].append(key)
                    items.append(
                        (key, record, fi, _animal_crop_fn(safari, ann, fi, feat.mask_crop, feat.min_mask_pixels))
                    )
        vectors = embed_crops(items, self.config, embedder, cache)
        return {cid: [vectors[k] for k in keys if k in vectors] for cid, keys in keys_by_species.items()}

    def _scores(
        self,
        probe_vecs: dict[str, list[np.ndarray]],
        ref_vecs: dict[str, list[np.ndarray]],
        probe_species: list[str],
        loso: bool,
    ) -> pd.Series:
        """Reduce per-species embeddings to ``familiarity_proxy`` per probe species (metric-agnostic core)."""
        metric = self.config.features.familiarity_metric
        ref_protos = {cid: p for cid, vecs in ref_vecs.items() if (p := prototype(vecs)) is not None}
        rows: dict[str, float] = {}
        for cid in probe_species:
            target = probe_vecs.get(cid)
            if not target:
                rows[cid] = float("nan")
                continue
            proto = prototype(target)
            if metric == "nearest_prototype":  # visual distance with SAM 3's encoder: loso-gated self-exclude
                excl = cid if loso else None
                rows[cid] = nearest_distance(proto, ref_protos, exclude=excl) if proto is not None else float("nan")
            elif metric == "silhouette":  # inter = nearest OTHER species by definition -> always self-exclude
                rows[cid] = silhouette(target, {**ref_protos, cid: proto}, cid, exclude=cid)
            elif metric == "mahalanobis":  # typicality vs the reference Gaussian (drop self only under loso)
                pool = [v for c, vecs in ref_vecs.items() if not (loso and c == cid) for v in vecs]
                rows[cid] = mahalanobis(proto, pool)
            else:
                raise ValueError(
                    f"unknown familiarity_metric {metric!r} "
                    "(expected 'silhouette', 'nearest_prototype' or 'mahalanobis')"
                )
        for cid in probe_species:
            rows.setdefault(cid, float("nan"))
        return pd.Series(rows, name="familiarity_proxy", dtype="float64")

    def compute(self, partition: Partition) -> pd.Series:
        """Return ``familiarity_proxy`` per probe species (``category_id``); ``NaN`` where no crops exist."""
        from src.splits import probe_records, reference_records

        embedder = Sam3Embedder(self.config)
        cache = EmbeddingCache(
            self.config, self.config.features.familiarity_encoder, self.config.features.mask_crop
        )
        ref_vecs = self._vectors_by_species(reference_records(partition, self.config), embedder, cache)
        probe_vecs = (
            ref_vecs
            if partition.held_axis == "species"
            else self._vectors_by_species(probe_records(partition, self.config), embedder, cache)
        )
        return self._scores(probe_vecs, ref_vecs, list(partition.probe_species), partition.loso)
