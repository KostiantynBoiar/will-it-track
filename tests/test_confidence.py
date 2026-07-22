"""Confidence feature — hermetic (synthetic prediction JSONs + monkeypatched record loaders, no gated data)."""

from __future__ import annotations

import json

import numpy as np

from src.config import Config
from src.dataset import VideoRecord
from src.features import confidence as C
from src.features.confidence import CONF_COLS, ConfidenceFeature, calibrate_atc_threshold
from src.splits import Partition


def _rec(video_id: str, category_id: str, origin: str = "test", n_masklets: int = 1) -> VideoRecord:
    return VideoRecord(
        video_id=f"{origin}:{video_id}",
        file_names=[f"{video_id}/0.jpg"],
        category_id=category_id,
        species=f"sp{category_id}",
        noun_phrase=f"sp{category_id}",
        location_id="L1",
        creation_datetime="2020-01-01",
        origin=origin,
        num_masklets=n_masklets,
        is_hard_negative=(n_masklets == 0),
    )


def _partition(species: list[str]) -> Partition:
    return Partition(
        name="t", held_axis="location", loso=False,
        reference_species=species, probe_species=species,
        reference_locations=["L1"], probe_locations=["L1"],
        reference_years=["2020"], probe_origins=["test"],
    )


def _write_pred(root, cfg, origin: str, raw_id: str, cid: str, entries: list[dict]) -> None:
    d = root / cfg.inference.predictions_subdir / origin / cfg.inference.prompt_mode
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{raw_id}_{cid}.json").write_text(json.dumps(entries))


def test_calibrate_atc_threshold_matches_reference_pdeta() -> None:
    """t is frozen so the reference coverage (frac >= t) equals the reference mean pDetA."""
    scores = [i / 10 for i in range(1, 11)]  # 0.1 .. 1.0
    t = calibrate_atc_threshold(scores, reference_pdeta=0.5)
    assert abs(float(np.mean([s >= t for s in scores])) - 0.5) <= 0.15
    # a perfect detector (pDetA=1) counts (almost) everything; an empty reference falls back to 0.5
    assert calibrate_atc_threshold(scores, 1.0) <= min(scores)
    assert calibrate_atc_threshold([], 0.5) == 0.5


def test_compute_confidence_features(tmp_path, monkeypatch) -> None:
    """Per-cell ATC coverage / mean / median / frame-coverage aggregate the cell's probe JSONs correctly."""
    cfg = Config()
    cfg.paths.outputs_root = tmp_path
    rle = {"size": [2, 2], "counts": "ab"}  # opaque RLE stand-in (presence is by None-pattern, not decode)

    probe = [_rec("v1", "1"), _rec("v2", "1")]  # both map to the SAME cell (category 1, sp1, L1, 2020)
    ref = [_rec("r1", "1", origin="train")]
    monkeypatch.setattr(C, "probe_records", lambda partition, config=None: probe)
    monkeypatch.setattr(C, "reference_records", lambda partition, config=None: ref)

    _write_pred(tmp_path, cfg, "test", "v1", "1", [{"score": 0.9, "segmentations": [rle, None]}])  # 1/2 frames
    _write_pred(tmp_path, cfg, "test", "v2", "1", [{"score": 0.3, "segmentations": [rle, rle]}])   # 2/2 frames
    _write_pred(tmp_path, cfg, "train", "r1", "1", [{"score": 0.8, "segmentations": [rle]}])        # calibration

    df = ConfidenceFeature(cfg).compute(_partition(["1"]), reference_pdeta=0.5)

    assert list(df.columns) == CONF_COLS
    assert len(df) == 1  # v1 + v2 pooled into one cell
    row = df.iloc[0]
    assert abs(row["conf_mean_score"] - 0.6) < 1e-9        # mean(0.9, 0.3)
    assert abs(row["conf_median_score"] - 0.6) < 1e-9
    assert abs(row["conf_frame_coverage"] - 0.75) < 1e-9   # (1 + 2) / (2 + 2)
    # t calibrated on ref [0.8] at pDetA 0.5 -> t = 0.8; frac(0.9, 0.3 >= 0.8) = 0.5
    assert abs(row["conf_atc_coverage"] - 0.5) < 1e-9


def test_missing_prediction_and_total_miss(tmp_path, monkeypatch) -> None:
    """A probe with no JSON is omitted (NaN after the left merge); an empty JSON is a zero-confidence cell."""
    cfg = Config()
    cfg.paths.outputs_root = tmp_path
    probe = [_rec("v1", "1"), _rec("v9", "2")]  # v9's file is intentionally absent
    monkeypatch.setattr(C, "probe_records", lambda partition, config=None: probe)
    monkeypatch.setattr(C, "reference_records", lambda partition, config=None: [])

    _write_pred(tmp_path, cfg, "test", "v1", "1", [])  # inferred, found nothing -> zero-confidence cell

    df = ConfidenceFeature(cfg).compute(_partition(["1", "2"]), threshold=0.5).reset_index()
    assert set(df["category_id"]) == {"1"}  # species 2 (no file) omitted, not counted as a rejection
    miss = df.iloc[0]
    assert miss["conf_mean_score"] == 0.0 and miss["conf_atc_coverage"] == 0.0
    assert miss["conf_frame_coverage"] == 0.0
