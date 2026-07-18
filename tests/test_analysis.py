"""Modelling-stack tests on a synthetic merged table — no gated data, no SAM 3, no DINOv2.

The synthetic law is deliberate: ``pAssA`` falls with ``environment_distance`` and ``pDetA`` falls with
``visual_distance``. The tests assert the pipeline recovers that structure (negative coefficients, the
right dominant factor, out-of-sample error that beats a mean predictor) and writes its artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.cross_val import GroupedCV, oos_predictions
from src.analysis.regression import TargetRegression
from src.analysis.uncertainty import Uncertainty
from src.analysis.variance import VariancePartition
from src.config import Config
from src.features.assemble import FeatureAssembler, merge_features
from src.io import read_parquet, write_parquet
from src.splits import Partition


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _synthetic(n_loc: int = 12, per_loc: int = 12) -> pd.DataFrame:
    """A merged features+scores table where env→pAssA and visual→pDetA are the true drivers."""
    rng = np.random.default_rng(0)
    species = [str(1000 + i) for i in range(8)]
    rows = []
    for loc in range(n_loc):
        for j in range(per_loc):
            cid = species[(loc + j) % len(species)]
            env, vis = rng.uniform(0, 1), rng.uniform(0, 1)
            rows.append(
                {
                    "category_id": cid,
                    "species": f"sp{cid}",
                    "location_id": f"L{loc}",
                    "time": "2020",
                    "taxonomic_distance": float(rng.integers(0, 6)),
                    "temporal_gap": int(rng.integers(0, 10)),
                    "visual_distance": vis,
                    "environment_distance": env,
                    "is_night_ir": bool(rng.integers(0, 2)),
                    "clutter": rng.uniform(0, 1),
                    "familiarity_proxy": float("nan"),
                    "pAssA": float(_sigmoid(2.5 - 4.0 * env + rng.normal(0, 0.25))),
                    "pDetA": float(_sigmoid(1.5 - 4.0 * vis + rng.normal(0, 0.25))),
                    "pHOTA": 0.5,
                    "n_frames": int(rng.integers(20, 300)),
                    "n_masklets": int(rng.integers(1, 10)),
                    "n_videos": int(rng.integers(1, 5)),
                    "prompt_mode": "species",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def table(tmp_path):
    """Synthetic table on disk + a config whose outputs_root is the tmp dir."""
    cfg = Config()
    cfg.paths.outputs_root = tmp_path
    df = _synthetic()
    path = write_parquet(df, tmp_path / "features.parquet")
    return cfg, df, path


def test_merge_features_broadcasts_each_granularity() -> None:
    """merge_features broadcasts per-species / per-cell / per-location distances onto the cell grid."""
    scores = pd.DataFrame(
        {
            "category_id": ["1", "1", "2"],
            "species": ["a", "a", "b"],
            "location_id": ["L0", "L1", "L0"],
            "time": ["2020", "2020", "2020"],
            "pDetA": [0.5, 0.6, 0.7],
            "n_frames": [10, 20, 30],
        }
    )
    taxonomic = pd.Series({"1": 2.0, "2": 4.0}, name="taxonomic_distance")
    visual = pd.Series({"1": 0.3, "2": 0.5}, name="visual_distance")
    temporal = pd.Series(
        [5, 6, 7],
        index=pd.MultiIndex.from_tuples(
            [("1", "a", "L0", "2020"), ("1", "a", "L1", "2020"), ("2", "b", "L0", "2020")],
            names=["category_id", "species", "location_id", "time"],
        ),
        name="temporal_gap",
    )
    environment = pd.DataFrame(
        {"environment_distance": [0.1, 0.9], "is_night_ir": [False, True], "clutter": [0.2, 0.8]},
        index=pd.Index(["L0", "L1"], name="location_id"),
    )

    merged = merge_features(
        scores, taxonomic=taxonomic, visual=visual, temporal=temporal, environment=environment
    )
    assert len(merged) == 3
    assert list(merged["taxonomic_distance"]) == [2.0, 2.0, 4.0]  # per-species broadcast
    assert list(merged["temporal_gap"]) == [5, 6, 7]  # per-cell join
    assert list(merged["environment_distance"]) == [0.1, 0.9, 0.1]  # per-location broadcast
    assert merged["familiarity_proxy"].isna().all()  # deferred


def test_regression_recovers_negative_slopes(table) -> None:
    """The GLM recovers the true negative env→pAssA and visual→pDetA relationships + writes artefacts."""
    cfg, _df, path = table
    for target, driver in (("pAssA", "environment_distance"), ("pDetA", "visual_distance")):
        pkl = TargetRegression(target, cfg).fit(path)
        assert pkl.exists()
        coef = pd.read_csv(cfg.paths.outputs_root / "models" / f"{target}_coef.csv", index_col=0)
        assert coef.loc[driver, "coef"] < 0  # score falls as the driving distance grows


def test_variance_partition_finds_the_driver(table) -> None:
    """Dominance analysis puts the true driver on top with finite, non-degenerate VIF."""
    cfg, _df, _path = table
    part = VariancePartition(cfg).partition("pAssA")
    assert part.iloc[0]["factor"] == "environment_distance"  # dominant for association
    assert np.isfinite(part["vif"]).all() and (part["vif"] > 0).all()
    assert (part["lmg_r2"] >= 0).all()


def test_grouped_cv_beats_baseline_and_writes(table) -> None:
    """Leave-location-out OOS prediction beats a mean predictor and the results table is written."""
    cfg, df, path = table
    oos = oos_predictions(df, "pAssA", "location_id", cfg)
    oos = oos[oos["predicted"].notna()]
    assert len(oos) > 0
    mae = float(np.mean(np.abs(oos["actual"] - oos["predicted"])))
    baseline = float(np.mean(np.abs(oos["actual"] - oos["actual"].mean())))
    assert mae < baseline  # distance genuinely predicts held-out sites

    out = GroupedCV(cfg).run(path)
    assert out.exists() and len(read_parquet(out)) > 0


def test_predictive_line_writes_png(table) -> None:
    """The predictive-line figure is produced for a target."""
    cfg, _df, _path = table
    png = Uncertainty(cfg).predictive_line("assoc")
    assert png.exists() and png.suffix == ".png"


def test_report_export_writes_dissertation_artifacts(table) -> None:
    """The exporter turns fitted outputs into results_summary.md + booktabs table fragments."""
    from src.analysis.report import export

    cfg, df, path = table
    write_parquet(df, cfg.paths.outputs_root / "scores.parquet")  # exporter's Gate-1 source
    for target in ("pDetA", "pAssA"):
        TargetRegression(target, cfg).fit(path)
    GroupedCV(cfg).run(path)

    summary = export(cfg)
    assert summary.exists()
    text = summary.read_text()
    assert "Gate 1" in text and "Gate 2" in text

    tables = cfg.paths.outputs_root.parent / "report" / "dissertation" / "tables"
    for frag in ("gate1_measurement.tex", "coefficients.tex", "cv_validation.tex"):
        assert (tables / frag).exists()
    assert "\\begin{table}" in (tables / "coefficients.tex").read_text()


def test_leakage_firewall_rejects_overlap() -> None:
    """assemble refuses to build features when the held (location) axis is not disjoint."""

    def part(probe_loc: list[str], ref_loc: list[str]) -> Partition:
        return Partition(
            name="t",
            held_axis="location",
            loso=False,
            reference_species=[],
            probe_species=[],
            reference_locations=ref_loc,
            probe_locations=probe_loc,
            reference_years=[],
            probe_origins=["test"],
        )

    FeatureAssembler._assert_no_leakage(part(["A"], ["B"]))  # disjoint → fine
    with pytest.raises(ValueError, match="location leakage"):
        FeatureAssembler._assert_no_leakage(part(["A"], ["A", "B"]))
