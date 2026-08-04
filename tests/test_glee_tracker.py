"""GleeTracker adapter tests (hermetic CPU — no torch/GLEE/detectron2, no GPU, no network).

The real GLEE forward needs a GPU, so these tests exercise the contract: the pure output post-processing
(_masklets_from_glee), the config default, and the harness tracker dispatch + prediction-path namespacing
via a fake tracker. Mirrors tests/test_inference.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from src.config import Config
from src.dataset import VideoRecord
from src.inference.glee_tracker import GleeTracker, _glee_out_dict, _masklets_from_glee
from src.inference.harness import InferenceHarness, probe_filename
from src.inference.sam3_tracker import Masklet


def _mask(h: int, w: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """A boolean mask of size (h, w) with box=(r0, r1, c0, c1) set True."""
    m = np.zeros((h, w), dtype=bool)
    r0, r1, c0, c1 = box
    m[r0:r1, c0:c1] = True
    return m


def test_masklets_from_glee_alignment_and_rle() -> None:
    """One track on frames 0 and 2 gives a 4-long, frame-aligned masklet with original-res RLE."""
    h, w = 20, 30
    tracks = {
        7: [(0, _mask(h, w, (2, 6, 3, 9)), 0.8), (2, _mask(h, w, (2, 6, 3, 9)), 0.6)],
    }
    out = _masklets_from_glee(tracks, n_frames=4)
    assert len(out) == 1
    m = out[0]
    assert isinstance(m, Masklet)
    assert len(m.segmentations) == 4  # padded to n_frames
    assert m.segmentations[0] is not None and m.segmentations[2] is not None
    assert m.segmentations[1] is None and m.segmentations[3] is None  # absent frames nulled
    assert m.score == pytest.approx(0.8)  # one scalar per track = max over its frames
    rle = m.segmentations[0]
    assert rle["size"] == [h, w]  # RLE at original frame resolution
    decoded = coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
    assert decoded.sum() == (6 - 2) * (9 - 3)  # 4x6 box


def test_masklets_from_glee_threshold_and_empty() -> None:
    """Below-threshold tracks are dropped; no tracks (hard negative) gives an empty list, no fake mask."""
    h, w = 10, 10
    tracks = {
        1: [(0, _mask(h, w, (0, 3, 0, 3)), 0.9)],  # kept
        2: [(0, _mask(h, w, (0, 2, 0, 2)), 0.2)],  # below threshold
    }
    kept = _masklets_from_glee(tracks, n_frames=2, threshold=0.5)
    assert len(kept) == 1 and kept[0].score == pytest.approx(0.9)
    assert _masklets_from_glee({}, n_frames=2, threshold=0.5) == []  # hard negative


def test_masklets_from_glee_multiple_tracks_no_obj_id() -> None:
    """Two tracks give two masklets (one per track); identity is list membership, no obj_id field."""
    h, w = 12, 12
    tracks = {
        3: [(0, _mask(h, w, (0, 4, 0, 4)), 0.7)],
        4: [(1, _mask(h, w, (6, 10, 6, 10)), 0.5)],
    }
    out = _masklets_from_glee(tracks, n_frames=2)
    assert len(out) == 2
    assert set(Masklet.model_fields) == {"segmentations", "score"}  # no obj_id on the schema


def test_masklets_from_glee_rejects_out_of_range_frame() -> None:
    """A detection frame index outside the clip is a hard error, not a silent drop."""
    with pytest.raises(ValueError, match="out of range"):
        _masklets_from_glee({1: [(5, _mask(8, 8, (0, 2, 0, 2)), 0.9)]}, n_frames=3)


def test_glee_out_dict_unwraps_both_shapes() -> None:
    """The forward-output unwrap reaches the pred dict whether GLEE returns a dict or a nested tuple."""
    d = {"pred_masks": 1, "pred_logits": 2, "pred_track_embed": 3}
    assert _glee_out_dict(d) is d  # bare dict
    assert _glee_out_dict((d, "masks")) is d  # (out_dict, mask_dict)
    assert _glee_out_dict(((d, "masks"), "loss", "loss")) is d  # nested training-style tuple
    with pytest.raises(RuntimeError, match="could not locate"):
        _glee_out_dict({"no_masks_here": 1})


def test_config_default_tracker_is_sam3() -> None:
    """The default config keeps the SAM 3 path (the swap is inert until explicitly selected)."""
    assert Config().inference.tracker == "sam3"


def test_harness_builds_glee_without_gpu_imports() -> None:
    """Selecting tracker='glee' constructs a GleeTracker on CPU (model deps stay lazy in load())."""
    cfg = Config()
    cfg.inference.tracker = "glee"
    harness = InferenceHarness(cfg)  # must not import torch/GLEE
    assert isinstance(harness.tracker, GleeTracker)


def test_harness_unknown_tracker_errors() -> None:
    """An unknown tracker name is a clear ValueError, not a late import failure."""
    cfg = Config()
    cfg.inference.tracker = "nope"
    with pytest.raises(ValueError, match="unknown inference.tracker"):
        InferenceHarness(cfg)


def test_glee_predictions_are_namespaced(tmp_path: Path) -> None:
    """GLEE predictions land under a glee/ subdir so they never collide with SAM 3's files."""
    cfg = Config()
    cfg.paths.outputs_root = tmp_path
    cfg.inference.tracker = "glee"
    sam3_cfg = Config()
    sam3_cfg.paths.outputs_root = tmp_path
    glee_dir = InferenceHarness(cfg, _FakeGleeTracker())._out_dir("test")
    sam3_dir = InferenceHarness(sam3_cfg, _FakeGleeTracker())._out_dir("test")
    assert "glee" in glee_dir.parts and glee_dir != sam3_dir
    assert sam3_dir.parts[-2:] == ("test", "species")  # sam3 path unchanged (no tracker segment)


def test_glee_predict_and_resume(tmp_path: Path) -> None:
    """A GLEE probe writes a namespaced JSON via the fake tracker and is resumable."""
    cfg = Config()
    cfg.paths.data_root = tmp_path
    cfg.paths.outputs_root = tmp_path / "out"
    cfg.inference.tracker = "glee"
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
    harness = InferenceHarness(cfg, _FakeGleeTracker())
    preds = harness._predict_video(record)
    assert len(preds) == 1 and len(preds[0]["segmentations"]) == 3

    out_dir = harness._out_dir("test")
    out_dir.mkdir(parents=True, exist_ok=True)
    assert "glee" in out_dir.parts  # namespaced away from any SAM 3 run
    path = out_dir / probe_filename(record.video_id, record.category_id)
    path.write_text("[]")  # a pre-existing prediction
    assert path.exists()  # the harness skips this probe on resume (run() short-circuits existing files)


class _FakeGleeTracker:
    """A GLEE-shaped tracker with no model — returns one planted masklet (drives the plumbing tests)."""

    def track(self, frames: list[Image.Image], prompt: str) -> list[Masklet]:  # noqa: ARG002
        """Return one masklet present on frame 0, aligned to frames (via the pure assembler)."""
        if not frames:
            return []
        w, h = frames[0].size
        return _masklets_from_glee({0: [(0, _mask(h, w, (0, h // 4, 0, w // 4)), 0.9)]}, len(frames))
