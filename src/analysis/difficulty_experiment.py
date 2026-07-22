"""Difficulty experiment (POC, MVP-1) — does intrinsic label-free difficulty predict pDetA out of sample?

The "difficulty, not novelty" pivot: the size confound that dissolved ``visual_distance`` hints that SAM 3's
transfer is governed by the intrinsic DIFFICULTY of the target imagery, not by distance-from-training. The four
before-running distances were all *novelty* axes and all came up H0. Here we promote the EXISTING label-free
difficulty signals — low-light/IR (``achromatic_fraction``) and ``clutter``, already in ``features.parquet``,
computed per location from the frames with no SAM 3 run and no target label — from nuisance covariates to
**predictors of interest**, and validate them at the exact bar the distances failed.

Honesty: these are **before-running** (image statistics, no SAM 3 tracking run), **label-free** (per-location
frame properties; no species/place annotation), and **non-circular** (an image property, NOT a function of SAM
3's output — categorically unlike the excluded ATC confidence estimator). Motivation: the low-light↔pDetA
correlate is the strongest yet seen (location-level r≈-0.377). ``log_area`` and ``log(n_frames)`` are forced
into every fit (the size/support confounds). Detection (``pDetA``) only; no association claim.

This is MVP-1 (zero new features, zero inference). If it validates, the per-cell FG–BG *conspicuity* embedding
is a stronger follow-up; if it nulls, it hardens H0. Pre-registered family below; report whatever comes out.

Run: ``PYTHONPATH=. python -m src.analysis.difficulty_experiment [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis import regression as R
from src.analysis.confidence_experiment import _log_area  # torch-free log_area (reuse)
from src.analysis.cross_val import _SCHEME_COLUMN, _summarise, oos_predictions
from src.config import Config
from src.io import read_parquet, write_parquet
from src.splits import build_location_partition

_CELL_KEYS = ["category_id", "species", "location_id", "time"]
# Pre-registered difficulty models: (label, continuous predictors, binary predictors, is_primary).
_PREREG = [
    ("lowlight", ("achromatic_fraction",), (), True),          # the strongest existing correlate
    ("clutter", ("clutter",), (), False),
    ("night_ir", (), ("is_night_ir",), False),
    ("difficulty_combined", ("achromatic_fraction", "clutter"), ("is_night_ir",), False),  # exploratory
]
_BONFERRONI_M = 3  # the three single-signal tests
# Nested size decomposition: (label, continuous predictors, binary predictors, include log_area?).
# Isolates whether the apparent difficulty gain is really the size covariate.
_DECOMP = [
    ("log_support_only", (), (), False),
    ("log_area_only", (), (), True),
    ("lowlight_only_no_size", ("achromatic_fraction",), (), False),
    ("lowlight_plus_size", ("achromatic_fraction",), (), True),
]


def augment(config: Config) -> pd.DataFrame:
    """Add ``log_area`` (torch-free) to ``features.parquet`` → ``features_diff.parquet``."""
    outputs = config.paths.outputs_root
    features = read_parquet(outputs / "features.parquet")
    partition = build_location_partition(config)
    log_area = _log_area(config, partition)
    df = features.copy()
    for key in _CELL_KEYS:
        df[key] = df[key].astype(str)
    df["log_area"] = df["category_id"].map(log_area)
    write_parquet(df, outputs / "features_diff.parquet")
    n_pdeta = int(pd.to_numeric(df["pDetA"], errors="coerce").notna().sum())
    print(f"features_diff -> {outputs/'features_diff.parquet'} ({len(df)} cells, {n_pdeta} with pDetA) "
          f"| log_area nonnull={int(df['log_area'].notna().sum())}")
    return df


def _cv_for_model(
    df: pd.DataFrame, config: Config, cont: tuple[str, ...], binary: tuple[str, ...],
    target: str = "pDetA", control_size: bool = True
) -> pd.DataFrame:
    """OOS summary (per scheme) for the isolated model ``cont + binary [+ log_area] + log(support)``.

    Pins the estimation core's predictor tuples so exactly these features enter the design; ``control_size``
    toggles whether ``log_area`` is in the design (used by the nested size-decomposition). Restores globals.
    """
    saved = (R.DISTANCE_COLS, R.CONFIDENCE_COLS, R._CONT_COVARIATES, R._BINARY_COVARIATES,
             config.model.control_size)
    R.DISTANCE_COLS, R.CONFIDENCE_COLS, R._CONT_COVARIATES, R._BINARY_COVARIATES = cont, (), (), binary
    config.model.control_size = control_size
    try:
        frames = [
            preds
            for scheme, col in _SCHEME_COLUMN.items()
            if col in df.columns
            for preds in [oos_predictions(df, target, col, config)]
            if not preds.empty
        ]
        cv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return _summarise(cv, config)
    finally:
        (R.DISTANCE_COLS, R.CONFIDENCE_COLS, R._CONT_COVARIATES, R._BINARY_COVARIATES,
         config.model.control_size) = saved


def run(config: Config | None = None) -> Path:
    """Run the pre-registered difficulty experiment; write + return the summary JSON path."""
    cfg = config or Config()
    df = augment(cfg)
    alpha_corr = 0.05 / _BONFERRONI_M

    models = []
    for label, cont, binary, is_primary in _PREREG:
        summ = _cv_for_model(df, cfg, cont, binary, target="pDetA")
        det = summ[summ["target"] == "pDetA"] if not summ.empty else summ
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
        models.append({"feature": label, "primary": is_primary, "schemes": schemes,
                       "both_schemes_positive": both_pos,
                       "both_schemes_significant_bonferroni": both_sig})

    primary = next(m for m in models if m["primary"])

    # Nested decomposition: is the gain the SIZE covariate or the difficulty feature?
    decomposition = []
    for name, cont, binary, cs in _DECOMP:
        summ = _cv_for_model(df, cfg, cont, binary, control_size=cs)
        det = summ[summ["target"] == "pDetA"] if not summ.empty else summ
        schemes = {
            r.group_scheme: {"delta": round(float(r.delta), 4), "delta_lo": round(float(r.delta_lo), 4),
                             "delta_hi": round(float(r.delta_hi), 4), "p_value": round(float(r.p_value), 4),
                             "mae": round(float(r.mae), 4)}
            for r in det.itertuples()
        } if not det.empty else {}
        decomposition.append({"model": name, "schemes": schemes})
    dmap = {d["model"]: d["schemes"] for d in decomposition}

    def _val(name: str) -> bool:  # validates on both schemes at the uncorrected 5%
        s = dmap.get(name, {})
        return bool(s) and all(s.get(sc, {}).get("p_value", 1.0) < 0.05 for sc in ("species", "location"))

    size_carries = _val("log_area_only") and not _val("lowlight_only_no_size")
    decomposition_verdict = (
        "The apparent difficulty 'validation' is the SIZE effect: log_area alone carries the whole OOS gain, "
        "while low-light alone does NOT validate — low-light/clutter/night-IR add ~nothing over size."
        if size_carries else "Decomposition inconclusive — inspect the per-model deltas."
    )
    verdict = (
        "MISLEADING at face value — the difficulty models clear the bar, but the nested decomposition shows the "
        "gain is the SIZE covariate (log_area), NOT low-light/clutter/night-IR difficulty."
        if size_carries else
        ("VALIDATED (both schemes clear Bonferroni)" if primary["both_schemes_significant_bonferroni"]
         else "NOT validated at the full bar")
    )
    out = {
        "experiment": "intrinsic_difficulty_detection (POC MVP-1)",
        "framing": "before-running, label-free, non-circular difficulty (low-light/IR + clutter) vs pDetA; "
                   "the 'difficulty not novelty' pivot",
        "target": "pDetA", "n_pdeta_cells": int(pd.to_numeric(df["pDetA"], errors="coerce").notna().sum()),
        "bonferroni_m": _BONFERRONI_M, "alpha_corrected": round(alpha_corr, 4),
        "primary_feature": "achromatic_fraction (low-light/IR)",
        "verdict": verdict, "decomposition_verdict": decomposition_verdict,
        "models": models, "decomposition": decomposition,
    }
    path = cfg.paths.outputs_root / "difficulty_experiment_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\ndifficulty experiment -> {path}\nVERDICT: {verdict}")
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    run(Config.load(args.config))


if __name__ == "__main__":
    main()
