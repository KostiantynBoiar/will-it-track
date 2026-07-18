"""Grouped cross-validation.

Turns a fitted curve into a predictor: predicts held-out species/places from distance alone
using leave-one-species-out and leave-one-location-out schemes (whole groups held out). Reports
out-of-sample error and calibration to ``outputs/validation/cv_results.parquet`` and asserts
leakage-free grouping.

Run: ``PYTHONPATH=. .venv/bin/python -m src.analysis.cross_val [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneGroupOut

from src.analysis.regression import TARGETS, DesignBuilder, _num, _weights
from src.config import Config
from src.io import read_parquet, write_parquet

# CV scheme name → the whole-group column held out.
_SCHEME_COLUMN = {"species": "category_id", "location": "location_id"}
_CELL_KEYS = ["category_id", "species", "location_id", "time"]


def oos_predictions(
    df: pd.DataFrame, target: str, group_col: str, config: Config
) -> pd.DataFrame:
    """Leave-one-group-out out-of-sample predictions for ``target`` grouped by ``group_col``.

    Each fold refits the GLM on the training groups (with its own standardisation, so no held-out
    statistic leaks) and predicts the held-out group. Returns one row per predicted cell with the
    actual score, the OOS prediction, and its group. Empty if there are fewer than two groups.
    """
    rows = df[_num(df[target]).notna() & df[group_col].astype(str).ne("")].copy()
    rows = rows.reset_index(drop=True)
    groups = rows[group_col].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        return pd.DataFrame()

    endog = _num(rows[target]).clip(0.0, 1.0).to_numpy(dtype=float)
    predicted = np.full(len(rows), np.nan)
    for train_idx, test_idx in LeaveOneGroupOut().split(rows, endog, groups):
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])  # leakage firewall
        train, test = rows.iloc[train_idx], rows.iloc[test_idx]
        builder = DesignBuilder(config).fit(train)
        try:
            result = sm.GLM(
                endog[train_idx],
                builder.transform(train),
                family=sm.families.Binomial(),
                var_weights=_weights(train, config),
            ).fit()
            predicted[test_idx] = result.predict(builder.transform(test))
        except Exception:  # noqa: BLE001 - a non-converging fold leaves NaN predictions, not a crash
            continue

    out = rows[_CELL_KEYS].copy()
    out["target"] = target
    out["group_scheme"] = {v: k for k, v in _SCHEME_COLUMN.items()}.get(group_col, group_col)
    out["group"] = groups
    out["actual"] = endog
    out["predicted"] = predicted
    return out


def _bootstrap_delta(delta: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Paired group-bootstrap of ``mean(delta)``: resample whole groups -> 95% CI + one-sided ``p``.

    ``delta`` is the per-cell ``|error_baseline| - |error_model|`` (positive => the model beats the
    mean). Cells inside a held-out group are not independent, so resampling *whole groups* (not cells)
    is what keeps the significance test honest. ``p`` is ``P(mean delta <= 0)`` across resamples.
    """
    uniq = np.unique(groups)
    if len(uniq) < 2 or n_boot <= 0:
        return {"delta_lo": float("nan"), "delta_hi": float("nan"), "p_value": float("nan"),
                "significant": False}
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        boots[b] = delta[sel].mean()
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {"delta_lo": lo, "delta_hi": hi, "p_value": float((boots <= 0).mean()),
            "significant": bool(lo > 0)}


def _summarise(cv: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Per (scheme, target): OOS MAE vs the mean baseline, with a paired group-bootstrap on the gain.

    ``delta = baseline_mae - mae`` (>0 => the model beats the mean); its 95% CI and one-sided
    ``p_value`` come from resampling whole held-out groups (a cell-level test would overstate
    significance because cells within a group share a species/location). ``significant`` is the 5% call.
    """
    cfg = config or Config()
    done = cv[cv["predicted"].notna()]
    records = []
    for (scheme, target), grp in done.groupby(["group_scheme", "target"]):
        actual, predicted = grp["actual"].to_numpy(), grp["predicted"].to_numpy()
        e_model, e_base = np.abs(actual - predicted), np.abs(actual - actual.mean())
        record = {
            "group_scheme": scheme,
            "target": target,
            "n": len(grp),
            "mae": float(e_model.mean()),
            "baseline_mae": float(e_base.mean()),
            "delta": float((e_base - e_model).mean()),
        }
        record |= _bootstrap_delta(e_base - e_model, grp["group"].to_numpy(), cfg.cv.n_bootstrap, cfg.seed)
        records.append(record)
    return pd.DataFrame(records)


class GroupedCV:
    """Leave-one-group-out CV (whole species / whole locations) for both targets."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``cv.group_schemes``).
        """
        self.config = config or Config()

    def run(self, table_path: Path) -> Path:
        """Run grouped CV and write ``outputs/validation/cv_results.parquet``.

        Args:
            table_path: Merged scores-x-features parquet.

        Returns:
            Path to the CV results parquet.
        """
        df = read_parquet(table_path)
        frames = [
            preds
            for scheme in self.config.cv.group_schemes
            if (col := _SCHEME_COLUMN.get(scheme)) in df.columns
            for target in TARGETS
            if not (preds := oos_predictions(df, target, col, self.config)).empty
        ]
        cv = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=[*_CELL_KEYS, "target", "group_scheme", "actual", "predicted"])
        )
        path = write_parquet(cv, self.config.paths.outputs_root / "validation" / "cv_results.parquet")
        summary = _summarise(cv, self.config)
        if not summary.empty:
            print(summary.to_string(index=False))
        print(f"cv results -> {path} ({len(cv)} OOS rows)")
        return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    GroupedCV(cfg).run(cfg.paths.outputs_root / "features.parquet")


if __name__ == "__main__":
    main()
