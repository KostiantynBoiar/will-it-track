"""Shared pipeline for the image-embedding distance features (visual + environment).

Both features do the same thing — turn video records into per-group embedding prototypes and compare
probes against references — differing only in *which crop* they embed and *what they group by*. This
module holds the parts they share; each feature supplies its crop function and grouping key.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.features.embed import Embedder, EmbeddingCache
from src.features.frames import ensure_frames, load_frame

CropFn = Callable[[Image.Image], "Image.Image | None"]
Item = tuple[str, VideoRecord, int, CropFn]


def safari_by_origin(config: Config) -> dict[str, SAFARI]:
    """One cached loader per split, for annotation/mask lookup on pooled (origin-tagged) records."""
    return {"train": SAFARI("train", config), "test": SAFARI("test", config)}


def record_annotations(
    record: VideoRecord, safari: dict[str, SAFARI], category_id: str | None = None
) -> list[dict]:
    """A record's video annotations, optionally filtered to one ``category_id``."""
    origin, _, raw_id = record.video_id.partition(":")
    anns = safari[origin].annotations_by_video().get(raw_id, [])
    if category_id is None:
        return anns
    return [a for a in anns if str(a["category_id"]) == category_id]


def prototype(vectors: list[np.ndarray]) -> np.ndarray | None:
    """Re-normalised mean of L2-normalised vectors (``None`` if there are none)."""
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def nearest_distance(
    vector: np.ndarray, references: dict[str, np.ndarray], exclude: str | None = None
) -> float:
    """``1 - max cosine`` to the nearest reference prototype (skipping ``exclude``); ``NaN`` if none."""
    candidates = [proto for key, proto in references.items() if key != exclude]
    if not candidates:
        return float("nan")
    return 1.0 - max(float(vector @ proto) for proto in candidates)


def gaussian_stats(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix of a stack of embeddings ``(n, d)`` (cov = ``0`` if ``n < 2``)."""
    x = np.asarray(vectors, dtype=np.float64)
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False) if x.shape[0] > 1 else np.zeros((x.shape[1], x.shape[1]))
    return mu, np.atleast_2d(cov)


def frechet_distance(
    mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray, eps: float = 1e-6
) -> float:
    """Fréchet (FID-style) distance between Gaussians ``N(mu1,cov1)`` and ``N(mu2,cov2)``.

    ``||mu1-mu2||^2 + Tr(cov1 + cov2 - 2(cov1 cov2)^{1/2})`` via ``scipy.linalg.sqrtm``; a small ``eps*I``
    is added before the matrix square root for numerical stability (rank-deficient per-species covariances),
    and any imaginary residue from ``sqrtm`` is dropped. AutoEval's distributional feature distance.
    """
    from scipy.linalg import sqrtm

    diff = np.asarray(mu1, dtype=np.float64) - np.asarray(mu2, dtype=np.float64)
    offset = np.eye(cov1.shape[0]) * eps
    covmean = sqrtm((cov1 + offset) @ (cov2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(covmean))


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between rows of ``a`` and ``b`` (clamped ``>= 0``)."""
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)


def mmd_rbf(x: np.ndarray, y: np.ndarray, *, gamma: float | None = None,
            max_samples: int = 512, seed: int = 0) -> float:
    """Biased RBF-kernel MMD^2 between two embedding sets (median-heuristic bandwidth, subsampled)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_samples:
        x = x[rng.choice(x.shape[0], max_samples, replace=False)]
    if y.shape[0] > max_samples:
        y = y[rng.choice(y.shape[0], max_samples, replace=False)]
    if gamma is None:
        pooled = np.vstack([x, y])
        med = float(np.median(_sq_dists(pooled, pooled)))
        gamma = 1.0 / (2.0 * med + 1e-12)
    kxx = np.exp(-gamma * _sq_dists(x, x))
    kyy = np.exp(-gamma * _sq_dists(y, y))
    kxy = np.exp(-gamma * _sq_dists(x, y))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def distributional_distance(
    target: np.ndarray, reference: np.ndarray, variant: str, *, seed: int = 0, ref_cap: int = 4000
) -> float:
    """``frechet``/``mmd`` distance from a target species' embedding set to the pooled reference set.

    ``NaN`` if either set has fewer than two vectors. The reference pool is deterministically subsampled to
    ``ref_cap`` to bound the covariance/kernel cost. This is the distributional analogue of the
    nearest-prototype cosine distance — the whole target distribution against the whole reference.
    """
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if target.shape[0] < 2 or reference.shape[0] < 2:
        return float("nan")
    if reference.shape[0] > ref_cap:
        rng = np.random.default_rng(seed)
        reference = reference[rng.choice(reference.shape[0], ref_cap, replace=False)]
    if variant == "frechet":
        return frechet_distance(*gaussian_stats(target), *gaussian_stats(reference))
    if variant == "mmd":
        return mmd_rbf(target, reference, seed=seed)
    raise ValueError(f"unknown distance_variant {variant!r} (expected 'frechet' or 'mmd')")


def embed_crops(
    items: list[Item], config: Config, embedder: Embedder, cache: EmbeddingCache
) -> dict[str, np.ndarray]:
    """Embed each item's crop (cache-first), returning ``{cache_key: vector}``.

    Processed in chunks of ``features.embed_crop_chunk``: each chunk fetches its frames, decodes +
    crops them (in parallel across ``features.embed_load_workers`` threads, since the path is I/O-bound
    on frame reads not the GPU), embeds, caches, then drops the crops before the next chunk. Peak RAM is
    bounded by the chunk size, so full sampling caps do not exhaust memory even though the environment
    "crop" is a whole frame. Behaviour is unchanged — the result is re-read from the cache.
    """
    misses = [item for item in items if cache.get(item[0]) is None]
    chunk = max(1, config.features.embed_crop_chunk)
    workers = max(1, config.features.embed_load_workers)

    def _load_crop(item: Item) -> tuple[str, Image.Image] | None:
        key, record, frame_index, crop_fn = item
        frame = load_frame(record.file_names[frame_index], config)
        crop = crop_fn(frame) if frame is not None else None
        return (key, crop) if crop is not None else None

    for n_chunk, start in enumerate(range(0, len(misses), chunk)):
        batch = misses[start : start + chunk]
        by_record: dict[str, list[Item]] = defaultdict(list)
        for item in batch:
            by_record[item[1].video_id].append(item)
        for group in by_record.values():
            record = group[0][1]
            ensure_frames([record.file_names[fi] for _, _, fi, _ in group], record.origin, config)

        if workers == 1:
            loaded = [_load_crop(item) for item in batch]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                loaded = list(pool.map(_load_crop, batch))
        pending: list[tuple[str, Image.Image]] = [c for c in loaded if c is not None]
        if pending:
            vectors = embedder.embed([crop for _, crop in pending])
            for (key, _), vector in zip(pending, vectors, strict=True):
                cache.put(key, vector)
        del pending  # release this chunk's decoded frames before the next one
        if (n_chunk + 1) % 16 == 0:
            cache.save()  # checkpoint so a long run is resumable, not lost on a crash

    if misses:
        cache.save()
    return {item[0]: cache.get(item[0]) for item in items if cache.get(item[0]) is not None}
