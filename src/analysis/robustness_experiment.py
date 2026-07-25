"""Design-robustness sweep (R6) — does the label-free null survive alternative design choices?

Re-runs the SAME leave-species-out + leave-location-out bar the primary distances faced, but with each
design knob swapped: the visual encoder (DINOv2 -> CLIP), the crop (mask -> whole-frame bounding box), the
prompt (species -> generic ``"animal"``), and the distance itself (nearest-prototype -> distributional
Frechet / MMD). Each variant's feature table is built on the GPU pod under its own config
(``scripts/run_robustness.sh``) and dropped here as ``outputs/features_<key>.parquet``; this driver only fits
the standing **4-distance GLM** through the unchanged ``cross_val`` core and reports, per variant x scheme,
the out-of-sample gain over the mean baseline.

Pre-registration (fixed before running): the null "survives" a variant when its 4-distance model does **not**
beat the mean on both schemes (``delta <= 0`` or not significant after Bonferroni over the variant family). A
variant that DOES clear the bar is a genuine positive, reported as such — nothing is swapped in post hoc. The
distance model matches the primary analysis exactly (``CONFIDENCE_COLS`` pinned empty, size not controlled).

Run: ``PYTHONPATH=. python -m src.analysis.robustness_experiment [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis import regression as R
from src.analysis.cross_val import _SCHEME_COLUMN, _summarise, oos_predictions
from src.config import Config
from src.io import read_parquet

# (key, human label, features filename, is_baseline). The distances GLM is identical for every row; only the
# underlying feature values (or, for the generic prompt, the target scores baked into the table) differ.
_VARIANTS = [
    ("baseline", "baseline (DINOv2, mask-crop, species prompt)", "features.parquet", True),
    ("clip", "CLIP encoder", "features_clip.parquet", False),
    ("wholeframe", "whole-frame (bbox + background)", "features_wholeframe.parquet", False),
    ("generic", 'generic "animal" prompt', "features_generic.parquet", False),
    ("frechet", "Frechet distributional distance", "features_frechet.parquet", False),
    ("mmd", "MMD distributional distance", "features_mmd.parquet", False),
]
_TARGET = "pDetA"  # detection; pAssA tracks it (near-degenerate) — CLAUDE.md §12


def _cv_summary(df: pd.DataFrame, config: Config, target: str = _TARGET) -> pd.DataFrame:
    """Standard 4-distance GLM OOS summary (both schemes) for one variant's feature table.

    Pins ``CONFIDENCE_COLS`` empty so the model is exactly the primary label-free distances (a variant
    table may or may not carry other columns); restores the module globals afterwards.
    """
    saved = (R.DISTANCE_COLS, R.CONFIDENCE_COLS)
    R.CONFIDENCE_COLS = ()
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
        R.DISTANCE_COLS, R.CONFIDENCE_COLS = saved


def _schemes_dict(summ: pd.DataFrame, target: str) -> dict:
    """Per-scheme ``{n, mae, baseline_mae, delta, delta_lo, delta_hi, p_value}`` for one variant."""
    det = summ[summ["target"] == target] if not summ.empty else summ
    by_scheme = {r.group_scheme: r for r in det.itertuples()} if not det.empty else {}
    return {
        s: {
            "n": int(getattr(by_scheme[s], "n", 0)),
            "mae": round(float(by_scheme[s].mae), 4),
            "baseline_mae": round(float(by_scheme[s].baseline_mae), 4),
            "delta": round(float(by_scheme[s].delta), 4),
            "delta_lo": round(float(by_scheme[s].delta_lo), 4),
            "delta_hi": round(float(by_scheme[s].delta_hi), 4),
            "p_value": round(float(by_scheme[s].p_value), 4),
        }
        for s in by_scheme
    }


def run(config: Config | None = None) -> Path:
    """Run the pre-registered robustness sweep over whichever variant tables are present; write the JSON."""
    cfg = config or Config()
    outputs = cfg.paths.outputs_root
    present = [(k, lbl, fn, base) for (k, lbl, fn, base) in _VARIANTS if (outputs / fn).exists()]
    tested = [v for v in present if not v[3]]
    alpha_corr = 0.05 / max(1, len(tested))

    models = []
    for key, label, fn, is_base in present:
        df = read_parquet(outputs / fn)
        for col in ("category_id", "location_id", "species", "time"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        schemes = _schemes_dict(_cv_summary(df, cfg), _TARGET)
        both_pos = bool(schemes) and all(
            schemes[s]["delta"] > 0 for s in ("species", "location") if s in schemes
        )
        both_sig = all(
            s in schemes and schemes[s]["p_value"] < alpha_corr for s in ("species", "location")
        )
        models.append({
            "variant": key,
            "label": label,
            "is_baseline": is_base,
            "n_cells": int(pd.to_numeric(df[_TARGET], errors="coerce").notna().sum()),
            "schemes": schemes,
            "both_schemes_positive": both_pos,
            "both_schemes_significant_bonferroni": bool(both_sig and not is_base),
        })

    validated = [m["label"] for m in models if m["both_schemes_significant_bonferroni"]]
    verdict = (
        f"NULL SURVIVES all {len(tested)} design variants (none beats the mean on both schemes)"
        if tested and not validated
        else f"A VARIANT VALIDATES: {', '.join(validated)} — report as a genuine positive"
        if validated
        else "no variant tables found — build them on the GPU pod first"
    )
    out = {
        "experiment": "design_robustness_sweep (R6)",
        "framing": "same leave-species/location-out bar as the primary null, with "
        "encoder/crop/prompt/distance swapped",
        "target": _TARGET,
        "variants_tested": [v[0] for v in tested],
        "variants_missing": [k for (k, _, fn, _) in _VARIANTS if not (outputs / fn).exists()],
        "bonferroni_m": len(tested),
        "alpha_corrected": round(alpha_corr, 4),
        "verdict": verdict,
        "models": models,
    }
    path = outputs / "robustness_experiment_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nrobustness sweep -> {path}\nVERDICT: {verdict}")
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    run(Config.load(args.config))


if __name__ == "__main__":
    main()
