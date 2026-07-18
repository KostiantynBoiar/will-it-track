"""embed_crops chunking — hermetic (fake embedder + local frames, no DINOv2, no network)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.config import Config
from src.dataset import VideoRecord
from src.features import pipeline as P
from src.features.embed import EmbeddingCache


class _FakeEmbedder:
    """Records how many crops it was handed per call; returns the red channel as the vector."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        self.batches.append(len(images))
        return np.array([[float(im.getpixel((0, 0))[0])] for im in images], dtype="float32")


def _make_record(tmp_path, cfg: Config, n: int) -> VideoRecord:
    frame_dir = tmp_path / cfg.data.frames_subdir / "v"
    frame_dir.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (12, 12), (i * 7 % 256, 0, 0)).save(frame_dir / f"{i}.jpg")
    return VideoRecord(
        video_id="v",
        file_names=[f"v/{i}.jpg" for i in range(n)],
        category_id="1",
        species="c",
        noun_phrase="c",
        location_id="L",
        creation_datetime="2020",
        origin="test",
        num_masklets=1,
        is_hard_negative=False,
    )


def test_embed_crops_chunks_and_caches(tmp_path, monkeypatch) -> None:
    """Never more than a chunk of crops is embedded at once; everything is embedded then cached."""
    monkeypatch.setattr(P, "ensure_frames", lambda *a, **k: 0)  # frames are already local
    cfg = Config()
    cfg.paths.data_root = tmp_path
    cfg.paths.outputs_root = tmp_path
    cfg.features.embed_crop_chunk = 2
    n = 5
    record = _make_record(tmp_path, cfg, n)
    items = [(f"k{i}", record, i, lambda frame: frame) for i in range(n)]

    emb = _FakeEmbedder()
    cache = EmbeddingCache(cfg, "fake", mask_crop=False)
    out = P.embed_crops(items, cfg, emb, cache)

    assert set(out) == {f"k{i}" for i in range(n)}
    assert emb.batches and max(emb.batches) <= cfg.features.embed_crop_chunk  # bounded per chunk
    assert sum(emb.batches) == n  # all crops embedded exactly once

    emb_again = _FakeEmbedder()  # everything is cached now
    out_again = P.embed_crops(items, cfg, emb_again, cache)
    assert emb_again.batches == []  # nothing re-embedded on a cache hit
    assert set(out_again) == set(out)
