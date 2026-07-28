"""MammAlps (R7) replication analysis — clean, separated model specs + confound diagnostics.

The pod pipeline fits ONE support-weighted logit GLM over every available predictor. On MammAlps (20 cells,
5 species) that kitchen-sink design (distances + confidence + covariates ≈ 10 params on 20 points)
quasi-separates (coefficients ~1e15), so the coefficient table is uninterpretable. This script refits the
SAME leakage-free leave-species-out / leave-camera-out machinery (``src.analysis.cross_val.oos_predictions``
+ ``_summarise``) on **cleanly separated** predictor sets, so each claim rests on a convergent model:

* ``distances (tax+vis+env)`` — the pre-registered label-free model (the headline null).
* ``visual_distance only`` / ``taxonomic_distance only`` — single-distance diagnostics.
* ``conf_mean_score [power probe]`` — a genuinely predictive per-cell feature (in-sample r≈+0.57); if even it
  fails out-of-sample, the n=20 CV is underpowered (the load-bearing, non-circular power argument).
* ``SIZE only`` / ``CONFIDENCE atc`` — the SA-FARI positive controls (structurally degenerate here: both are
  species-constant / regime-broken on MammAlps, so their failure is NOT a clean power readout).

It also prints the confound diagnostics for ``visual_distance`` (wrong sign, support confound, species-constant
tied values) that demote its lone OOS "significance" to an artifact. Numbers here were adversarially verified
against the raw parquet; see ``docs/mammalps_replication.md``.

Run: ``PYTHONPATH=. python scripts/mammalps_replication_analysis.py --features outputs_mammalps/features.parquet``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from scipy import stats

from src.analysis.cross_val import _SCHEME_COLUMN, _summarise, oos_predictions
from src.analysis.regression import TARGETS
from src.config import Config

warnings.filterwarnings("ignore")

# Columns the DesignBuilder may pick up as predictors; dropping the ones outside a spec isolates that spec.
_PRED_UNIVERSE = (
    "taxonomic_distance", "temporal_gap", "visual_distance", "environment_distance",
    "conf_atc_coverage", "conf_mean_score", "conf_median_score", "conf_frame_coverage",
    "clutter", "is_night_ir", "log_area", "achromatic_fraction", "familiarity_proxy",
)

# name -> (predictor columns kept, control_size flag)
_SPECS: dict[str, tuple[tuple[str, ...], bool]] = {
    "distances (tax+vis+env)":     (("taxonomic_distance", "visual_distance", "environment_distance"), False),
    "visual_distance only":        (("visual_distance",), False),
    "taxonomic_distance only":     (("taxonomic_distance",), False),
    "conf_mean_score [power probe]": (("conf_mean_score",), False),
    "SIZE only [pos. control]":    (("log_area",), True),
    "CONFIDENCE atc [pos. control]": (("conf_atc_coverage", "log_area"), True),
}


def _run_spec(fe: pd.DataFrame, cfg0: Config, keep: tuple[str, ...], control_size: bool) -> pd.DataFrame:
    """Leave-species-out + leave-camera-out ΔMAE vs a mean baseline, for one isolated predictor set."""
    cfg = cfg0.model_copy(deep=True)
    cfg.model.control_size = control_size
    drop = [c for c in _PRED_UNIVERSE if c not in keep and c in fe.columns]
    df = fe.drop(columns=drop)
    parts = [
        oos_predictions(df, target, col, cfg)
        for scheme, col in _SCHEME_COLUMN.items()
        if col in df.columns
        for target in TARGETS
    ]
    parts = [p for p in parts if len(p)]
    return _summarise(pd.concat(parts, ignore_index=True), cfg) if parts else pd.DataFrame()


def _diagnostics(fe: pd.DataFrame) -> None:
    """Print the confound evidence that demotes visual_distance's OOS significance to an artifact."""
    def corr(a: str, b: str) -> tuple[float, float]:
        m = fe[a].notna() & fe[b].notna()
        return stats.pearsonr(fe.loc[m, a], fe.loc[m, b])

    print("\n=== visual_distance confound diagnostics ===")
    r, p = corr("visual_distance", "pDetA")
    print(f"  corr(visual_distance, pDetA) = {r:+.3f} (p={p:.3f})  <- WRONG sign for novelty->failure")
    for c in ("n_frames", "n_masklets", "log_area"):
        if c in fe.columns:
            r, p = corr("visual_distance", c)
            print(f"  corr(visual_distance, {c:11s}) = {r:+.3f} (p={p:.3f})")
    keep = fe[~fe["species"].isin(["hare", "wolf"])]
    if len(keep) > 2:
        r, p = stats.pearsonr(keep["visual_distance"], keep["pDetA"])
        print(f"  drop hare+wolf (4/20 cells): corr(visual, pDetA) = {r:+.3f} (p={p:.3f})  <- sign flips, n.s.")
    print("  visual_distance is species-constant:")
    print(fe.groupby("species")["visual_distance"].mean().round(3).to_string().replace("\n", "\n    "))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="outputs_mammalps/features.parquet")
    ap.add_argument("--config", default="configs/mammalps.yaml")
    args = ap.parse_args()

    fe = pd.read_parquet(Path(args.features))
    cfg0 = Config.load(args.config)
    rows = []
    for name, (keep, cs) in _SPECS.items():
        summary = _run_spec(fe, cfg0, keep, cs)
        for _, r in summary.iterrows():
            rows.append({
                "spec": name, "scheme": r.group_scheme, "target": r.target, "n": int(r.n),
                "MAE": round(r.mae, 3), "baseline": round(r.baseline_mae, 3), "dMAE": round(r.delta, 4),
                "CI_lo": round(r.delta_lo, 3), "CI_hi": round(r.delta_hi, 3),
                "p": round(r.p_value, 3), "sig": bool(r.significant),
            })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for target in TARGETS:
        print(f"\n================ {target} (leave-species-out + leave-camera-out) ================")
        print(res[res.target == target].drop(columns="target").to_string(index=False))
    _diagnostics(fe)


if __name__ == "__main__":
    main()
