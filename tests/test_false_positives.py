"""False-positive analysis — hermetic (synthetic tables + monkeypatched loaders, no gated data)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis import false_positives as FP
from src.analysis.false_positives import _weighted_pearson, fp_table, summarise
from src.config import Config
from src.dataset import VideoRecord
from src.splits import Partition


def _hardneg(video_id: str, category_id: str) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        file_names=[f"{video_id}/0.jpg"],
        category_id=category_id,
        species=f"sp{category_id}",
        noun_phrase=f"sp{category_id}",
        location_id="L",
        creation_datetime="2020",
        origin="test",
        num_masklets=0,
        is_hard_negative=True,
    )


def _partition(species: list[str]) -> Partition:
    return Partition(
        name="t", held_axis="species", loso=True, reference_species=species, probe_species=species,
        reference_locations=[], probe_locations=[], reference_years=[], probe_origins=["test"],
    )


def test_weighted_pearson_matches_unweighted_with_equal_weights() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 1.0, 4.0, 3.0])
    w = np.ones(4)
    assert abs(_weighted_pearson(x, y, w) - np.corrcoef(x, y)[0, 1]) < 1e-9


def test_fp_table_reads_predictions(tmp_path, monkeypatch) -> None:
    """A non-empty prediction above threshold is a hallucination; empty is a rejection; missing is omitted."""
    cfg = Config()
    cfg.paths.outputs_root = tmp_path
    records = [_hardneg("v1", "1"), _hardneg("v2", "1"), _hardneg("v3", "2")]

    class _FakeSafari:
        def __init__(self, *a, **k) -> None: ...
        def records(self):
            return records

    class _FakeTax:
        def __init__(self, *a, **k) -> None: ...
        def compute(self, partition):
            return pd.Series({"1": 2.0, "2": 5.0})

    class _FakeVis:
        def __init__(self, *a, **k) -> None: ...
        def compute(self, partition):
            return pd.Series({"1": 0.3})  # species "2" -> NaN (never present positively)

    monkeypatch.setattr(FP, "SAFARI", _FakeSafari)
    monkeypatch.setattr(FP, "build_species_partition", lambda config: _partition(["1", "2"]))
    monkeypatch.setattr(FP, "TaxonomicDistance", _FakeTax)
    monkeypatch.setattr(FP, "VisualDistance", _FakeVis)

    pdir = tmp_path / cfg.inference.predictions_subdir / "test" / cfg.inference.prompt_mode
    pdir.mkdir(parents=True)
    (pdir / "v1_1.json").write_text(json.dumps([{"score": 0.9}]))  # (video, species) key; hallucination
    (pdir / "v2_1.json").write_text(json.dumps([]))  # correct rejection
    # v3's file intentionally absent -> omitted

    df = fp_table("test", cfg, threshold=0.5).set_index("video_id")
    assert set(df.index) == {"v1", "v2"}  # v3 omitted (no prediction file)
    assert df.loc["v1", "fp"] == 1 and df.loc["v2", "fp"] == 0
    assert df.loc["v1", "taxonomic_distance"] == 2.0
    assert df.loc["v1", "visual_distance"] == 0.3


def test_summarise_detects_rising_fp_with_novelty() -> None:
    """When FP rate rises with taxonomic distance across species, the correlation is positive."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(8):  # 8 species, FP probability increasing with taxonomic distance
        tax = float(s)
        p = 0.05 + 0.11 * s
        for j in range(12):
            rows.append({
                "video_id": f"{s}_{j}", "category_id": str(s), "species": f"sp{s}", "location_id": "L",
                "n_pred": 1, "max_score": 1.0,
                "fp": int(rng.random() < p), "taxonomic_distance": tax,
                "visual_distance": 0.1 * s, "log_area": float((3 * s) % 8),  # not collinear with visual
            })
    df = pd.DataFrame(rows)
    out = summarise(df, Config(), threshold=0.5)
    assert out["n_species"] == 8
    assert 0.0 <= out["overall_fp_rate"] <= 1.0
    tercile = out["fp_rate_by_taxonomic_tercile"]
    assert isinstance(tercile, dict) and tercile and all(isinstance(k, str) for k in tercile)
    assert out["fp_vs_taxonomic"]["pearson_r"] > 0  # hallucination rises with novelty
    # the size-confound checks are present when log_area is available
    assert "fp_vs_size" in out and "fp_vs_visual_size_controlled" in out
    assert -1.0 <= out["fp_vs_visual_size_controlled"]["partial_r"] <= 1.0
    # the leave-species-out validation of the FP predictor is reported
    oos = out["fp_predictor_oos"]
    assert {"mae", "baseline_mae", "delta", "p_value", "significant"} <= set(oos)
    assert oos["n_species"] == 8 and 0.0 <= oos["p_value"] <= 1.0
