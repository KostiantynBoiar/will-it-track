"""GLEE model-swap experiment: do the four before-running distances predict GLEE's pDetA out of sample?

A controlled model-swap (docs/glee_second_model.md): SAME SA-FARI cells, SAME ground truth, SAME four
label-free distances, SAME support-weighted logit GLM + grouped-CV bar — only the frozen tracker changes
(SAM 3 -> GLEE). Tests whether the label-free null is SAM-3-specific or task-general. Detection (pDetA) is
primary; pAssA is reported for parity but the association target is near-degenerate on SA-FARI (CLAUDE.md
section 12), so treat any pAssA "signal" as a leakage red flag, not a result.

Reuses the estimation core (regression + cross_val) UNCHANGED — this is the confidence_experiment machinery
with the predictor set swapped back to the four distances and the input table pointed at the GLEE run.

Run: PYTHONPATH=. python -m src.analysis.glee_experiment --config configs/glee.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis import regression as R
from src.analysis.confidence_experiment import _log_area
from src.analysis.cross_val import _SCHEME_COLUMN, _summarise, oos_predictions
from src.config import Config
from src.io import read_parquet, write_parquet
from src.splits import build_location_partition

_CELL_KEYS = ("category_id", "species", "location_id", "time")
# One distance in isolation (the Bonferroni family), then all four jointly. Mirrors the SAM 3 distance
# experiments so the two trackers are compared like-for-like.
_PREREG = (
    ("taxonomic", ("taxonomic_distance",), True),
    ("visual", ("visual_distance",), False),
    ("environment", ("environment_distance",), False),
    ("temporal", ("temporal_gap",), False),
    ("all_four", R.DISTANCE_COLS, False),
)
_BONFERRONI_M = 4  # the four single-distance tests


def augment(config: Config) -> pd.DataFrame:
    """Add log_area to the GLEE features.parquet and write features_glee.parquet.

    The GLEE features.parquet must already exist under outputs_root (built by running assemble against the
    GLEE scores.parquet, inference.tracker='glee'). The distances are identical to the SAM 3 table (same
    GT, same partition); only pDetA/pAssA differ.
    """
    outputs = config.paths.outputs_root
    df = read_parquet(outputs / "features.parquet").copy()
    for key in _CELL_KEYS:
        df[key] = df[key].astype(str)
    if "log_area" not in df.columns or df["log_area"].isna().all():
        log_area = _log_area(config, build_location_partition(config))
        df["log_area"] = df["category_id"].map(log_area)
    write_parquet(df, outputs / "features_glee.parquet")
    n_det = int(pd.to_numeric(df.get("pDetA"), errors="coerce").notna().sum())
    print(f"features_glee -> {outputs / 'features_glee.parquet'} ({len(df)} cells, {n_det} with pDetA)")
    return df


def _cv_for_model(df: pd.DataFrame, config: Config, dist_cols: tuple[str, ...], target: str) -> pd.DataFrame:
    """OOS summary (per scheme) for the isolated model dist_cols + log_area + log(support).

    Pins regression's predictor globals to exactly dist_cols (confidence/covariate tuples emptied), forces
    the log_area size control on, runs both CV schemes through the unchanged cross_val core, then restores.
    """
    saved = (R.DISTANCE_COLS, R.CONFIDENCE_COLS, R._CONT_COVARIATES, R._BINARY_COVARIATES,
             config.model.control_size)
    R.DISTANCE_COLS, R.CONFIDENCE_COLS, R._CONT_COVARIATES, R._BINARY_COVARIATES = dist_cols, (), (), ()
    config.model.control_size = True
    try:
        frames = [
            preds
            for _scheme, col in _SCHEME_COLUMN.items()
            if col in df.columns
            for preds in [oos_predictions(df, target, col, config)]
            if not preds.empty
        ]
        cv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return _summarise(cv, config)
    finally:
        (R.DISTANCE_COLS, R.CONFIDENCE_COLS, R._CONT_COVARIATES, R._BINARY_COVARIATES,
         config.model.control_size) = saved


def _model_row(df: pd.DataFrame, cfg: Config, label: str, cols: tuple[str, ...],
               primary: bool, target: str, alpha_corr: float) -> dict:
    """One pre-registered model's per-scheme OOS deltas + the both-schemes verdict flags."""
    summ = _cv_for_model(df, cfg, cols, target=target)
    det = summ[summ["target"] == target] if not summ.empty else summ
    by_scheme = {r.group_scheme: r for r in det.itertuples()} if not det.empty else {}
    schemes = {
        s: {"n": int(by_scheme[s].n), "mae": round(float(by_scheme[s].mae), 4),
            "baseline_mae": round(float(by_scheme[s].baseline_mae), 4),
            "delta": round(float(by_scheme[s].delta), 4),
            "delta_lo": round(float(by_scheme[s].delta_lo), 4),
            "delta_hi": round(float(by_scheme[s].delta_hi), 4),
            "p_value": round(float(by_scheme[s].p_value), 4)}
        for s in by_scheme
    }
    both_pos = all(schemes[s]["delta"] > 0 for s in ("species", "location") if s in schemes)
    both_sig = all(s in schemes and schemes[s]["p_value"] < alpha_corr for s in ("species", "location"))
    return {"feature": label, "primary": primary, "target": target, "schemes": schemes,
            "both_schemes_positive": both_pos, "both_schemes_significant_bonferroni": both_sig}


def run(config: Config | None = None) -> Path:
    """Run the GLEE distance model-swap; write and return glee_experiment_summary.json."""
    cfg = config or Config()
    df = augment(cfg)
    alpha_corr = 0.05 / _BONFERRONI_M

    models: dict[str, list[dict]] = {"pDetA": [], "pAssA": []}
    for target in ("pDetA", "pAssA"):
        if pd.to_numeric(df.get(target), errors="coerce").notna().sum() == 0:
            continue
        for label, cols, primary in _PREREG:
            models[target].append(_model_row(df, cfg, label, cols, primary, target, alpha_corr))

    primary = next((m for m in models["pDetA"] if m["primary"]), None)
    validated = bool(primary and primary["both_schemes_significant_bonferroni"])
    verdict = (
        "GLEE distances VALIDATE OOS on pDetA (both schemes clear Bonferroni) — the null is SAM-3-specific"
        if validated else
        "GLEE distances do NOT predict pDetA OOS — the label-free null is TASK-GENERAL, not SAM-3-specific"
    )
    out = {
        "experiment": "glee_model_swap_distances",
        "framing": "same SA-FARI cells / GT / four distances / GLM+CV; only the frozen tracker changes "
                   "(SAM 3 -> GLEE). Tests SAM-3-specific vs task-general null.",
        "primary_target": "pDetA",
        "primary_feature": "taxonomic_distance",
        "n_pdeta_cells": int(pd.to_numeric(df.get("pDetA"), errors="coerce").notna().sum()),
        "n_passa_cells": int(pd.to_numeric(df.get("pAssA"), errors="coerce").notna().sum()),
        "bonferroni_m": _BONFERRONI_M,
        "alpha_corrected": round(alpha_corr, 4),
        "verdict": verdict,
        "pAssA_caveat": "association target near-degenerate on SA-FARI; any pAssA signal is a leakage flag",
        "models": models,
    }
    path = cfg.paths.outputs_root / "glee_experiment_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nGLEE experiment -> {path}\nVERDICT: {verdict}")
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML (point outputs_root at the GLEE run)")
    args = ap.parse_args()
    run(Config.load(args.config))


if __name__ == "__main__":
    main()
