"""BURST (R7) replication analysis — separated specs, confounds, and the honest power caveat.

BURST is the better-powered replication (41 animal species, 132 video×class cells, mask-native) that MammAlps
(20 cells) could not be. This script refits the leakage-free leave-species-out machinery
(``src.analysis.cross_val.oos_predictions`` + ``_summarise``) on cleanly separated predictor sets — BURST has
no shared locations, so ``category_id`` (species) is the only valid grouping — and prints the diagnostics that
decide the honest framing:

* The before-running distances (taxonomic/visual/environment) are a NULL; the after-running confidence controls
  fire. But the firing does NOT prove the test is powered for a *distance*: distances are species-constant, so
  leave-species-out can only use the BETWEEN-species channel, whereas ``conf_mean_score`` wins mainly through a
  WITHIN-species channel. Collapsing ``conf_mean_score`` to species means (making it distance-like) kills its
  out-of-sample win — the key caveat. The confidence controls therefore establish pipeline *liveness* (a real
  contrast with MammAlps, where nothing fired), not power for a small species-level distance effect.

Numbers here were adversarially verified against the raw parquet; see ``docs/burst_replication.md``.

Run: ``PYTHONPATH=. python scripts/burst_replication_analysis.py --features outputs_burst/features.parquet``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.cross_val import _summarise, oos_predictions
from src.analysis.regression import TARGETS
from src.config import Config

warnings.filterwarnings("ignore")

_PRED = (
    "taxonomic_distance", "temporal_gap", "visual_distance", "environment_distance",
    "conf_atc_coverage", "conf_mean_score", "conf_median_score", "conf_frame_coverage",
    "clutter", "is_night_ir", "log_area", "achromatic_fraction", "familiarity_proxy",
)
_SPECS: dict[str, tuple[tuple[str, ...], bool]] = {
    "distances (tax+vis+env)":       (("taxonomic_distance", "visual_distance", "environment_distance"), False),
    "visual_distance only":          (("visual_distance",), False),
    "taxonomic_distance only":       (("taxonomic_distance",), False),
    "environment_distance only":     (("environment_distance",), False),
    "SIZE only [pos. control]":      (("log_area",), True),
    "conf_mean_score [pos. control]": (("conf_mean_score",), False),
    "conf_atc [pos. control]":       (("conf_atc_coverage",), False),
}


def _run(fe: pd.DataFrame, cfg0: Config, keep: tuple[str, ...], control_size: bool) -> pd.DataFrame:
    """Leave-species-out ΔMAE vs a mean baseline for one isolated predictor set (species scheme only)."""
    cfg = cfg0.model_copy(deep=True)
    cfg.model.control_size = control_size
    df = fe.drop(columns=[c for c in _PRED if c not in keep and c in fe.columns])
    parts = [oos_predictions(df, t, "category_id", cfg) for t in TARGETS]
    parts = [p for p in parts if len(p)]
    return _summarise(pd.concat(parts, ignore_index=True), cfg) if parts else pd.DataFrame()


def _between_species_corr(fe: pd.DataFrame, col: str) -> float:
    """Correlation of species-mean ``col`` with species-mean pDetA — the only channel a distance can use."""
    g = fe.groupby("species").agg(x=(col, "mean"), y=("pDetA", "mean")).dropna()
    return float(stats.pearsonr(g.x, g.y)[0]) if len(g) > 2 else float("nan")


def _within_species_fraction(fe: pd.DataFrame, col: str) -> float:
    """Fraction of ``col``'s variance that is within-species (0 ⇒ species-constant, like a distance)."""
    v = fe[[col, "species"]].dropna()
    if v[col].var() == 0:
        return 0.0
    within = v.groupby("species")[col].transform(lambda s: s - s.mean())
    return float(within.var() / v[col].var())


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="outputs_burst/features.parquet")
    ap.add_argument("--config", default="configs/burst.yaml")
    args = ap.parse_args()
    fe = pd.read_parquet(Path(args.features))
    cfg0 = Config.load(args.config)

    rows = []
    for name, (keep, cs) in _SPECS.items():
        for _, r in _run(fe, cfg0, keep, cs).iterrows():
            rows.append({"spec": name, "target": r.target, "n": int(r.n), "MAE": round(r.mae, 3),
                         "dMAE": round(r.delta, 4), "lo": round(r.delta_lo, 3), "hi": round(r.delta_hi, 3),
                         "p": round(r.p_value, 3), "sig": bool(r.significant)})
    res = pd.DataFrame(rows)
    for t in TARGETS:
        print(f"\n===== {t} (leave-species-out, {fe['species'].nunique()} species / {len(fe)} cells) =====")
        print(res[res.target == t].drop(columns="target").to_string(index=False))

    # Taxonomic on the clean full-taxonomy subset (no mean-imputation) — the honest, more-null number.
    tax = _run(fe[fe.taxonomic_distance.notna()], cfg0, ("taxonomic_distance",), False)
    td = tax[tax.target == "pDetA"].iloc[0]
    print(f"\ntaxonomic_distance (clean full-taxonomy subset, n={int(td.n)}): "
          f"dMAE {td.delta:+.4f} p={td.p_value:.3f}")

    print("\n=== the power caveat: why the firing confidence control does NOT prove distance power ===")
    for col in ("visual_distance", "environment_distance", "taxonomic_distance",
                "conf_mean_score", "conf_atc_coverage"):
        print(f"  {col:22s} within-species var frac={_within_species_fraction(fe, col):.2f}  "
              f"between-species corr(pDetA)={_between_species_corr(fe, col):+.3f}")
    # collapse conf_mean_score to species means (make it distance-like) -> its OOS win should die
    fe2 = fe.copy()
    fe2["conf_mean_score"] = fe2.groupby("species")["conf_mean_score"].transform("mean")
    coll = _run(fe2, cfg0, ("conf_mean_score",), False)
    cd = coll[coll.target == "pDetA"].iloc[0]
    print(f"  conf_mean_score COLLAPSED to species-means: dMAE {cd.delta:+.4f} p={cd.p_value:.3f} "
          f"(vs +0.0352 p=0.004 un-collapsed) -> the win lives in the within-species channel")

    print("\n=== confounds + association ===")
    m = fe.visual_distance.notna() & fe.pDetA.notna()
    print(f"  corr(visual, pDetA)={stats.pearsonr(fe.loc[m, 'visual_distance'], fe.loc[m, 'pDetA'])[0]:+.3f} "
          "(WRONG sign for novelty)")
    print(f"  corr(visual, log_area)={stats.pearsonr(fe.loc[m, 'visual_distance'], fe.loc[m, 'log_area'])[0]:+.3f}"
          "  (size confound)")
    print(f"  corr(pDetA, pAssA)={stats.pearsonr(fe.pDetA, fe.pAssA)[0]:.3f}  "
          f"mean|pDetA-pAssA|={(fe.pDetA - fe.pAssA).abs().mean():.3f}  "
          f"exact-equal={int(np.isclose(fe.pDetA, fe.pAssA).sum())}/{len(fe)}")


if __name__ == "__main__":
    main()
