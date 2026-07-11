"""Shared pipeline for the image-embedding distance features (visual + environment).

Both features do the same thing — turn video records into per-group embedding prototypes and compare
probes against references — differing only in *which crop* they embed and *what they group by*. This
module holds the parts they share; each feature supplies its crop function and grouping key.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

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


def embed_crops(
    items: list[Item], config: Config, embedder: Embedder, cache: EmbeddingCache
) -> dict[str, np.ndarray]:
    """Embed each item's crop (cache-first), returning ``{cache_key: vector}``.

    Missing frames are batch-fetched per video; any frame/crop that cannot be produced is skipped.
    """
    misses = [item for item in items if cache.get(item[0]) is None]
    by_record: dict[str, list[Item]] = defaultdict(list)
    for item in misses:
        by_record[item[1].video_id].append(item)
    for group in by_record.values():
        record = group[0][1]
        ensure_frames([record.file_names[fi] for _, _, fi, _ in group], record.origin, config)

    pending: list[tuple[str, Image.Image]] = []
    for key, record, frame_index, crop_fn in misses:
        frame = load_frame(record.file_names[frame_index], config)
        if frame is None:
            continue
        crop = crop_fn(frame)
        if crop is not None:
            pending.append((key, crop))
    if pending:
        vectors = embedder.embed([crop for _, crop in pending])
        for (key, _), vector in zip(pending, vectors, strict=True):
            cache.put(key, vector)
        cache.save()

    return {item[0]: cache.get(item[0]) for item in items if cache.get(item[0]) is not None}
