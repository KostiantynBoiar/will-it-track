"""Inference harness + scorer tests (no GPU, no SAM 3 — a fake tracker drives the plumbing)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.eval.score import Scorer
from src.inference.harness import InferenceHarness, _bbox, _cap_per_species
from src.inference.sam3_tracker import FakeTracker, encode_rle
from src.io import read_parquet, write_parquet

_CFG = Config()
_ANN_PRESENT = SAFARI("test", _CFG).ann_path.exists()
_needs_ann = pytest.mark.skipif(not _ANN_PRESENT, reason="SA-FARI annotations not fetched")


def test_encode_rle_and_bbox() -> None:
    """RLE encodes round-trips through pycocotools and the bbox matches the mask extent."""
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 6:12] = True
    rle = encode_rle(mask)
    assert (
        coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).sum() == 30
    )
    x, y, w, h = _bbox(rle)
    assert (x, y, w, h) == (6, 5, 6, 5)


def test_fake_tracker() -> None:
    """The fake tracker returns per-frame-aligned masklets; 0 masklets simulates a hard negative."""
    frames = [Image.new("RGB", (32, 24)) for _ in range(4)]
    masklets = FakeTracker(_CFG, masklets_per_call=2).track(frames, "cat")
    assert len(masklets) == 2
    assert len(masklets[0].segmentations) == 4
    assert masklets[0].segmentations[0] is not None and masklets[0].segmentations[1] is None
    assert FakeTracker(_CFG, masklets_per_call=0).track(frames, "cat") == []


def test_predict_video_hermetic(tmp_path: Path) -> None:
    """A probe with local frames + the fake tracker produces flat, VEval-shaped masklet entries."""
    cfg = Config()
    cfg.paths.data_root = tmp_path
    frame_dir = tmp_path / cfg.data.frames_subdir / "vid"
    frame_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (40, 30)).save(frame_dir / f"{i}.jpg")
    record = VideoRecord(
        video_id="7",
        file_names=[f"vid/{i}.jpg" for i in range(3)],
        category_id="1",
        species="cat",
        noun_phrase="cat",
        location_id="L0",
        creation_datetime="2020",
        origin="test",
        num_masklets=1,
        is_hard_negative=False,
    )
    harness = InferenceHarness(cfg, FakeTracker(cfg))

    preds = harness._predict_video(record)
    assert isinstance(preds, list) and len(preds) == 1  # one flat entry per masklet
    entry = preds[0]
    assert entry["video_id"] == 7 and entry["category_id"] == 1  # ints for the VEval GT join
    assert entry["score"] == 0.9
    assert len(entry["segmentations"]) == len(entry["bboxes"]) == len(entry["areas"]) == 3
    assert entry["bboxes"][0] is not None and entry["areas"][0] > 0  # object present on frame 0
    assert entry["bboxes"][1] is None and entry["areas"][1] == 0  # absent frame nulled

    empty = InferenceHarness(cfg, FakeTracker(cfg, masklets_per_call=0))._predict_video(record)
    assert empty == []  # hard-negative style: nothing found

    class _BoomTracker:
        def track(self, *_a, **_k):  # noqa: ANN002, ANN003, ANN201
            raise RuntimeError("CUDA out of memory")

    skipped = InferenceHarness(cfg, _BoomTracker())._predict_video(record)
    assert skipped == []  # a crashing clip is skipped, not fatal — the batch continues


def test_cap_per_species_samples_positives() -> None:
    """The stratified cap keeps <= N present videos per species and drops hard negatives."""

    def rec(vid: str, cid: str, masklets: int) -> VideoRecord:
        return VideoRecord(
            video_id=vid,
            file_names=[f"{vid}/0.jpg"],
            category_id=cid,
            species=f"sp{cid}",
            noun_phrase=f"sp{cid}",
            location_id="L0",
            creation_datetime="2020",
            origin="train",
            num_masklets=masklets,
            is_hard_negative=masklets == 0,
        )

    records = (
        [rec(f"a{i}", "1", 1) for i in range(5)]  # 5 positives for species 1
        + [rec(f"b{i}", "2", 1) for i in range(2)]  # 2 positives for species 2
        + [rec(f"n{i}", "1", 0) for i in range(3)]  # 3 hard negatives for species 1
    )
    kept = _cap_per_species(records, cap=3)
    by_species = Counter(r.category_id for r in kept)
    assert by_species["1"] == 3  # capped at 3
    assert by_species["2"] == 2  # fewer than the cap -> all kept
    assert all(r.num_masklets > 0 for r in kept)  # hard negatives dropped


def test_io_parquet_roundtrip(tmp_path: Path) -> None:
    """Parquet write creates parent dirs and reads back identically."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = write_parquet(df, tmp_path / "nested" / "t.parquet")
    assert path.exists()
    pd.testing.assert_frame_equal(read_parquet(path), df)


@_needs_ann
def test_score_aggregate_to_cells() -> None:
    """Aggregation joins per-probe metrics + support onto the cell grid."""
    scorer = Scorer(_CFG)
    df = scorer.aggregate({}, "test")  # no metrics → NaN scores, but support still counted
    assert {
        "category_id",
        "species",
        "location_id",
        "time",
        "pDetA",
        "pAssA",
        "pHOTA",
        "n_frames",
        "n_masklets",
        "n_videos",
    } <= set(df.columns)
    assert (df["n_videos"] > 0).all()
    assert df["pDetA"].isna().all()  # empty per_probe → all NaN metrics
