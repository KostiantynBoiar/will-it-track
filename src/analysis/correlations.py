"""Raw correlations --- a leakage-free sanity check that complements the fitted GLM.

Computes the plain Pearson correlation of each label-free feature with each target (no standardisation, no
modelling), plus the ``visual_distance`` <-> ``log_area`` correlation that motivates the size-confound
check. If the raw associations are tiny, that corroborates the null; if ``visual_distance`` correlates
positively with both the score *and* animal size, that corroborates the confound. Writes
``outputs/correlations.csv``.

Run: ``PYTHONPATH=. python -m src.analysis.correlations [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.regression import DISTANCE_COLS, TARGETS
from src.config import Config
from src.io import read_parquet

_FEATURES = (*DISTANCE_COLS, "clutter", "log_area")


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Pearson r over rows where both are present (``NaN`` if too few / degenerate)."""
    a, b = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 3 or a[mask].std() == 0 or b[mask].std() == 0:
        return float("nan"), int(mask.sum())
    return float(np.corrcoef(a[mask], b[mask])[0, 1]), int(mask.sum())


def correlations(config: Config | None = None) -> Path:
    """Write ``outputs/correlations.csv`` of raw feature<->target (+ visual<->size) correlations."""
    cfg = config or Config()
    df = read_parquet(cfg.paths.outputs_root / "features.parquet")

    rows = []
    for feat in _FEATURES:
        if feat not in df.columns:
            continue
        for target in TARGETS:
            r, n = _pearson(df[feat], df[target])
            rows.append({"feature": feat, "against": target, "pearson_r": r, "n": n})
    if "visual_distance" in df.columns and "log_area" in df.columns:
        r, n = _pearson(df["visual_distance"], df["log_area"])
        rows.append({"feature": "visual_distance", "against": "log_area", "pearson_r": r, "n": n})

    out = pd.DataFrame(rows)
    path = cfg.paths.outputs_root / "correlations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if not out.empty:
        print(out.to_string(index=False))
    print(f"correlations -> {path}")
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    correlations(Config.load(args.config))


if __name__ == "__main__":
    main()
