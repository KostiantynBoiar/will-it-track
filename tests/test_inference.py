"""Inference harness + scorer tests (no GPU, no SAM 3 — a fake tracker drives the plumbing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.eval.score import Scorer
from src.inference.harness import InferenceHarness, _bbox
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
    """A probe with local frames + the fake tracker produces a well-formed prediction entry."""
    cfg = Config()
    cfg.paths.data_root = tmp_path
    frame_dir = tmp_path / cfg.data.frames_subdir / "vid"
    frame_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (40, 30)).save(frame_dir / f"{i}.jpg")
    record = VideoRecord(
        video_id="vid",
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

    pred = harness._predict_video(record)
    assert pred["video_id"] == "vid" and pred["prompt"] == "cat" and pred["n_frames"] == 3
    assert len(pred["masklets"]) == 1
    assert len(pred["masklets"][0]["segmentations"]) == 3
    assert pred["masklets"][0]["bboxes"][0] is not None

    empty = InferenceHarness(cfg, FakeTracker(cfg, masklets_per_call=0))._predict_video(record)
    assert empty["masklets"] == []  # hard-negative style: nothing found


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
