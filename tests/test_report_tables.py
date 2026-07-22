"""Report table builders — hermetic (synthetic JSON/CSV inputs, no gated data or model fits)."""

from __future__ import annotations

import json

import pandas as pd

from src.analysis.report import _ablation, _fp, _variance


def test_fp_table_renders_the_correlate_not_predictor_story(tmp_path) -> None:
    """The FP table shows the visual correlate as significant but the OOS row as just-missing."""
    (tmp_path / "false_positives_summary.json").write_text(json.dumps({
        "n_probes": 1486, "n_species": 113, "overall_fp_rate": 0.0983,
        "fp_vs_taxonomic": {"pearson_r": 0.06, "ci_lo": -0.17, "ci_hi": 0.23, "significant": False},
        "fp_vs_visual": {"pearson_r": -0.33, "ci_lo": -0.43, "ci_hi": -0.23, "significant": True},
        "fp_vs_size": {"pearson_r": 0.13, "ci_lo": -0.06, "ci_hi": 0.31, "significant": False},
        "fp_vs_visual_size_controlled": {"partial_r": -0.37, "ci_lo": -0.47, "ci_hi": -0.28, "significant": True},
        "fp_predictor_oos": {"delta": 0.008, "ci_lo": -0.002, "ci_hi": 0.017, "p_value": 0.053, "significant": False},
    }))
    summary, tex = _fp(tmp_path)
    assert "\\label{tab:fp}" in tex
    assert "9.8\\%" in tex and "$p=0.053$" in tex
    assert "$-0.33$" in tex and "significant" in tex  # the visual correlate
    assert summary["overall_fp_rate"] == 0.0983


def test_ablation_table_shows_only_size_moves_the_score(tmp_path) -> None:
    """The ablation table renders each design's pseudo-R2 + OOS gain; +size is the one that lifts it."""
    adir = tmp_path / "ablations"
    adir.mkdir()
    pd.DataFrame([
        {"design": "full (4 distances)", "target": "pDetA", "pseudo_r2": 0.055,
         "oos_species_delta": 0.004, "oos_species_p": 0.24, "oos_location_delta": 0.004, "oos_location_p": 0.23},
        {"design": "$+$ animal size", "target": "pDetA", "pseudo_r2": 0.119,
         "oos_species_delta": 0.018, "oos_species_p": 0.02, "oos_location_delta": 0.020, "oos_location_p": 0.00},
    ]).to_csv(adir / "factor_ablation.csv", index=False)
    _, tex = _ablation(tmp_path)
    assert "\\label{tab:ablation}" in tex
    assert "full (4 distances)" in tex and "$+$ animal size" in tex
    assert "0.119" in tex and "+0.018" in tex


def test_variance_table_reports_tiny_total_and_low_vif(tmp_path) -> None:
    """The variance table sums the per-factor LMG R2 into a small total, with VIF near 1."""
    adir = tmp_path / "ablations"
    adir.mkdir()
    pd.DataFrame([
        {"target": "pDetA", "factor": "taxonomic_distance", "lmg_r2": 0.014, "share": 0.48, "vif": 1.06},
        {"target": "pDetA", "factor": "visual_distance", "lmg_r2": 0.012, "share": 0.39, "vif": 1.06},
        {"target": "pAssA", "factor": "taxonomic_distance", "lmg_r2": 0.015, "share": 0.44, "vif": 1.06},
    ]).to_csv(adir / "variance_partition.csv", index=False)
    d, tex = _variance(tmp_path)
    assert "\\label{tab:variance}" in tex
    assert set(d["target"]) == {"pDetA"}  # pAssA row filtered out
    assert "1.06" in tex and "\\approx3\\%" in tex  # total 0.014+0.012 = 0.026 -> ~3%
