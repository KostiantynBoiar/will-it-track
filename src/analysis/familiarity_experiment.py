"""Familiarity-proxy experiment (T2.5) — does SAM 3's own feature-space separability predict pDetA OOS?

The four before-running *novelty distances* were H0, and the one that neared significance (``visual_distance``)
was a wrong-direction size confound. This experiment adds one more before-running, label-free, non-circular
axis: **familiarity read from SAM 3's own vision-encoder embeddings** (T2.5), hedging the undisclosed-pretraining
problem. It measures separability three ways (``silhouette`` / ``nearest_prototype`` / ``mahalanobis``; see
:mod:`src.features.familiarity`) and validates each at the *exact* leave-species-out bar the distances faced,
then runs the honest head-to-head: **does familiarity add out-of-sample gain over ``visual_distance`` + size?**

Two stages: ``extract`` (GPU — embeds crops with SAM 3, computes the three metric columns into
``features_fam.parquet``; reuses the cached embeddings across metrics) and ``run`` (CPU — the CV, Bonferroni
over the three metrics, the size decomposition, and the correlations; writes ``familiarity_experiment_summary
.json``). Expected outcome, pre-registered: most likely **hardens H0** — separability confounds with the size
effect that already dissolved the visual signal.

Run: ``PYTHONPATH=. python -m src.analysis.familiarity_experiment --stage all [--origins test] [--config ...]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.confidence_experiment import _log_area  # torch-free log_area (reuse)
from src.analysis.difficulty_experiment import _CELL_KEYS, _cv_for_model  # reuse the CV harness
from src.config import Config
from src.io import read_parquet, write_parquet
from src.splits import build_species_partition

# familiarity_metric -> the column it is stored under in features_fam.parquet
_FAM_COLS = {
    "silhouette": "familiarity_silhouette",
    "nearest_prototype": "familiarity_nearest",
    "mahalanobis": "familiarity_mahalanobis",
}
_BONFERRONI_M = 3  # the three metric variants


def extract(config: Config, origins: tuple[str, ...] = ("train", "test")) -> pd.DataFrame:
    """GPU: embed crops with SAM 3, compute the three familiarity metrics → ``features_fam.parquet``.

    Builds the species hold-out (Split A) partition, computes each metric (the SAM 3 embeddings are cached, so
    the second and third metrics are free), maps the per-species series onto the base ``features.parquet``, and
    adds the torch-free ``log_area`` covariate.
    """
    from src.features.familiarity import FamiliarityProxy

    outputs = config.paths.outputs_root
    features = read_parquet(outputs / "features.parquet")
    partition = build_species_partition(config, origins=origins)
    proxy = FamiliarityProxy(config)

    df = features.copy()
    for key in _CELL_KEYS:
        df[key] = df[key].astype(str)
    for metric, col in _FAM_COLS.items():
        config.features.familiarity_metric = metric
        series = proxy.compute(partition)  # embeds once, then reuses the cached vectors
        df[col] = df["category_id"].map(series)
        print(f"  {col}: {int(df[col].notna().sum())}/{len(df)} cells non-null", flush=True)
    df["log_area"] = df["category_id"].map(_log_area(config, partition))
    path = write_parquet(df, outputs / "features_fam.parquet")
    print(f"features_fam -> {path} ({len(df)} cells)")
    return df


def _schemes(summary: pd.DataFrame, target: str = "pDetA") -> dict:
    """Per-scheme ``delta/CI/p/mae`` dict for one CV summary (species + location)."""
    det = summary[summary["target"] == target] if not summary.empty else summary
    return {
        r.group_scheme: {
            "delta": round(float(r.delta), 4), "delta_lo": round(float(r.delta_lo), 4),
            "delta_hi": round(float(r.delta_hi), 4), "p_value": round(float(r.p_value), 4),
            "mae": round(float(r.mae), 4), "baseline_mae": round(float(r.baseline_mae), 4),
            "n": int(r.n),
        }
        for r in det.itertuples()
    } if not det.empty else {}


def _corr(df: pd.DataFrame, a: str, b: str) -> float | None:
    """Pearson correlation between two numeric columns over their shared non-null cells."""
    x, y = pd.to_numeric(df.get(a), errors="coerce"), pd.to_numeric(df.get(b), errors="coerce")
    m = x.notna() & y.notna()
    return round(float(np.corrcoef(x[m], y[m])[0, 1]), 3) if m.sum() > 2 else None


def run(config: Config | None = None) -> Path:
    """CPU: read ``features_fam.parquet``, validate each metric + head-to-head vs visual; write the summary."""
    cfg = config or Config()
    outputs = cfg.paths.outputs_root
    fam_path = outputs / "features_fam.parquet"
    if not fam_path.exists():
        raise FileNotFoundError(f"{fam_path} missing — run the `extract` stage on the GPU first")
    df = read_parquet(fam_path)
    alpha_corr = 0.05 / _BONFERRONI_M

    # Pre-registered family: each familiarity metric, size-controlled, at the leave-species-out bar.
    models = []
    for metric, col in _FAM_COLS.items():
        if col not in df.columns:
            continue
        schemes = _schemes(_cv_for_model(df, cfg, (col,), (), target="pDetA", control_size=True))
        both_pos = all(schemes.get(s, {}).get("delta", -1) > 0 for s in ("species", "location") if s in schemes)
        both_sig = all(s in schemes and schemes[s]["p_value"] < alpha_corr for s in ("species", "location"))
        models.append({"metric": metric, "column": col, "schemes": schemes,
                       "both_schemes_positive": both_pos, "both_schemes_significant_bonferroni": both_sig,
                       "corr_visual": _corr(df, col, "visual_distance"),
                       "corr_log_area": _corr(df, col, "log_area")})

    # Head-to-head: does familiarity add OOS gain over visual_distance + size? (the decisive honest test)
    visual_plus_size = _schemes(_cv_for_model(df, cfg, ("visual_distance",), (), control_size=True))
    decomposition = [{"model": "visual_plus_size", "schemes": visual_plus_size}]
    for metric, col in _FAM_COLS.items():
        if col not in df.columns:
            continue
        decomposition += [
            {"model": f"{metric}_no_size",
             "schemes": _schemes(_cv_for_model(df, cfg, (col,), (), control_size=False))},
            {"model": f"{metric}_and_visual_plus_size",
             "schemes": _schemes(_cv_for_model(df, cfg, (col, "visual_distance"), (), control_size=True))},
        ]
    dmap = {d["model"]: d["schemes"] for d in decomposition}

    def _species_delta(name: str) -> float:
        return dmap.get(name, {}).get("species", {}).get("delta", float("nan"))

    adds_over_visual = any(
        _species_delta(f"{m}_and_visual_plus_size") > _species_delta("visual_plus_size") + 1e-4
        and dmap.get(f"{m}_and_visual_plus_size", {}).get("species", {}).get("p_value", 1.0) < 0.05
        for m in _FAM_COLS
    )
    any_validated = any(m["both_schemes_significant_bonferroni"] for m in models)
    verdict = (
        "VALIDATED — a familiarity metric clears the Bonferroni bar on both schemes AND adds over visual+size"
        if any_validated and adds_over_visual else
        "NULL — SAM 3 familiarity does not predict transfer beyond visual_distance + size (hardens H0)"
        if not any_validated else
        "AMBIGUOUS — clears the bar but does not add over visual+size (re-encodes DINOv2 visual + size)"
    )
    out = {
        "experiment": "sam3_familiarity_proxy_detection (T2.5)",
        "framing": "before-running, label-free, non-circular: separability of the species in SAM 3's own "
                   "vision-encoder feature space vs pDetA; hedges the undisclosed pretraining set",
        "target": "pDetA",
        "n_pdeta_cells": int(pd.to_numeric(df["pDetA"], errors="coerce").notna().sum()),
        "bonferroni_m": _BONFERRONI_M, "alpha_corrected": round(alpha_corr, 4),
        "metrics": list(_FAM_COLS), "adds_over_visual_plus_size": adds_over_visual,
        "verdict": verdict, "models": models, "decomposition": decomposition,
    }
    path = outputs / "familiarity_experiment_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nfamiliarity experiment -> {path}\nVERDICT: {verdict}")
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--stage", choices=("extract", "run", "all"), default="all")
    ap.add_argument("--origins", nargs="*", default=None, help="species-pool origins, e.g. --origins test")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    origins = tuple(args.origins) if args.origins else ("train", "test")
    if args.stage in ("extract", "all"):
        extract(cfg, origins=origins)
    if args.stage in ("run", "all"):
        run(cfg)


if __name__ == "__main__":
    main()
