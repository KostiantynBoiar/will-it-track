"""Power-analysis tests: the planted effect is injected correctly and the endpoints behave.

Synthetic-only (no gated data): a tiny table with real-shaped columns. Endpoint checks use few
replicates, so they assert direction (null rarely detected, huge effect detected more often), not
exact power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.power import _plant, _resample_groups, run
from src.config import Config


@pytest.fixture()
def table(tmp_path):
    """A 96-cell table over 12 species x 4 locations with noise-only pDetA."""
    rng = np.random.default_rng(7)
    rows = []
    for s in range(12):
        for k in range(8):
            rows.append({
                "category_id": f"c{s}", "species": f"sp{s}", "location_id": f"loc{k % 4}",
                "time": "all", "pDetA": float(rng.uniform(0.2, 0.8)), "pAssA": np.nan,
                "visual_distance": float(s) / 11.0, "taxonomic_distance": np.nan,
                "environment_distance": float(rng.uniform(0, 1)), "temporal_gap": np.nan,
                "is_night_ir": 0.0, "clutter": float(rng.uniform(0, 1)),
                "n_frames": 40, "n_masklets": 1, "n_videos": 1,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    return df, tmp_path


def test_plant_moves_scores_in_the_right_direction(table):
    df, _ = table
    planted = _plant(df, beta=1.0)
    hi = df["visual_distance"] > df["visual_distance"].median()
    shift = planted["pDetA"] - df["pDetA"]
    assert shift[hi].mean() > 0 > shift[~hi].mean()
    planted0 = _plant(df, beta=0.0)
    assert np.allclose(planted0["pDetA"], df["pDetA"].clip(1e-3, 1 - 1e-3))


def test_resample_has_no_duplicate_groups(table):
    df, _ = table
    sub = _resample_groups(df, "category_id", np.random.default_rng(0))
    assert sub["category_id"].nunique() == len(set(sub["category_id"]))
    assert 2 <= sub["category_id"].nunique() <= df["category_id"].nunique()


def test_endpoints_null_vs_large_effect(table, monkeypatch, tmp_path):
    df, data_dir = table
    config = Config()
    monkeypatch.setattr(config.paths, "outputs_root", data_dir)
    monkeypatch.setattr(config.cv, "n_bootstrap", 200)
    monkeypatch.chdir(tmp_path)  # the table writer emits under report/, keep it in the sandbox
    (tmp_path / "report/dissertation/tables").mkdir(parents=True)
    csv = run(config, reps=6, betas=(0.0, 4.0))
    out = pd.read_csv(csv)
    null_power = out[out["beta"] == 0.0]["power"].max()
    big_power = out[out["beta"] == 4.0]["power"].min()
    assert null_power <= 0.35
    assert big_power >= null_power
    assert big_power >= 0.5
