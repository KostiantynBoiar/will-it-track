"""False-positive / hallucination analysis on hard negatives.

On a hard negative the queried species is absent, so a *correct* run returns no masklet. Here we ask the
detection-precision question the pDetA fit cannot --- pDetA conditions on the animal being present, so it
measures recall-given-present, not hallucination. Does SAM 3 return a masklet anyway, and does that
false-positive rate rise with the label-free novelty (taxonomic / visual distance) of the queried species?

We read the harness prediction JSONs directly (a non-empty prediction above a score threshold = a
hallucination), attach each hard-negative species' taxonomic and visual distance, and summarise at the
*species* level (so the significance test is over species, not pseudo-replicated probes). Writes
``outputs/false_positives.parquet`` + ``outputs/false_positives_summary.json``.

Run: ``PYTHONPATH=. python -m src.analysis.false_positives --split test [--threshold 0.5]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import SAFARI
from src.features.taxonomic import TaxonomicDistance
from src.features.visual import VisualDistance
from src.io import write_parquet
from src.splits import build_species_partition


def fp_table(split: str, config: Config, threshold: float) -> pd.DataFrame:
    """Per hard-negative probe: hallucination flag + max score + taxonomic/visual distance.

    A hard negative whose prediction JSON holds a masklet scoring ``>= threshold`` is a false positive.
    Distances are computed on a partition whose probe side is the hard-negative species (reference set
    unchanged), so taxonomic distance resolves for any species with full taxonomy; visual distance is
    ``NaN`` for species that never appear positively (no crops to embed).
    """
    records = [r for r in SAFARI(split, config).records() if r.is_hard_negative]
    pred_dir = (
        config.paths.outputs_root
        / config.inference.predictions_subdir
        / split
        / config.inference.prompt_mode
    )
    base = build_species_partition(config)
    partition = base.model_copy(update={"probe_species": sorted({r.category_id for r in records})})
    taxonomic = TaxonomicDistance(config).compute(partition)
    visual = VisualDistance(config).compute(partition)

    rows = []
    for record in records:
        path = pred_dir / f"{record.video_id}.json"
        if not path.exists():
            continue  # not inferred (e.g. a sampled subset) — omit rather than count as a rejection
        scores = [float(entry.get("score", 0.0)) for entry in json.loads(path.read_text())]
        max_score = max(scores) if scores else 0.0
        rows.append(
            {
                "video_id": record.video_id,
                "category_id": record.category_id,
                "species": record.species,
                "location_id": record.location_id,
                "n_pred": len(scores),
                "max_score": max_score,
                "fp": int(max_score >= threshold),
                "taxonomic_distance": float(taxonomic.get(record.category_id, float("nan"))),
                "visual_distance": float(visual.get(record.category_id, float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Support-weighted Pearson correlation (``NaN`` if either side is constant)."""
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx, sy = np.sqrt(np.average((x - mx) ** 2, weights=w)), np.sqrt(np.average((y - my) ** 2, weights=w))
    return float(cov / (sx * sy)) if sx > 0 and sy > 0 else float("nan")


def _corr_over_species(species: pd.DataFrame, dist_col: str, seed: int, n_boot: int) -> dict:
    """Support-weighted correlation of per-species FP rate vs a distance, with a species bootstrap CI."""
    data = species[["fp_rate", dist_col, "n"]].dropna()
    if len(data) < 3 or data[dist_col].std() == 0:
        return {"pearson_r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_species": int(len(data)), "significant": False}
    x = data[dist_col].to_numpy(float)
    y = data["fp_rate"].to_numpy(float)
    w = data["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    boots = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        r = _weighted_pearson(x[take], y[take], w[take])
        if np.isfinite(r):
            boots.append(r)
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5])) if boots else (float("nan"), float("nan"))
    return {"pearson_r": _weighted_pearson(x, y, w), "ci_lo": lo, "ci_hi": hi,
            "n_species": int(len(data)), "significant": bool(lo > 0 or hi < 0)}


def summarise(df: pd.DataFrame, config: Config, threshold: float) -> dict:
    """Overall FP rate, a taxonomic-tercile breakdown, and species-level FP-rate<->distance correlations."""
    species = (
        df.groupby("category_id")
        .agg(
            species=("species", "first"),
            fp_rate=("fp", "mean"),
            n=("fp", "size"),
            taxonomic_distance=("taxonomic_distance", "first"),
            visual_distance=("visual_distance", "first"),
        )
        .reset_index()
    )
    tercile = df.dropna(subset=["taxonomic_distance"]).copy()
    by_tercile = {}
    if len(tercile) >= 3 and tercile["taxonomic_distance"].nunique() >= 3:
        tercile["bin"] = pd.qcut(tercile["taxonomic_distance"], 3, labels=["near", "mid", "far"],
                                 duplicates="drop")
        by_tercile = {str(k): round(float(v), 4) for k, v in tercile.groupby("bin")["fp"].mean().items()}
    return {
        "split": config.experiment,
        "threshold": threshold,
        "n_probes": int(len(df)),
        "n_species": int(species["category_id"].nunique()),
        "overall_fp_rate": round(float(df["fp"].mean()), 4) if len(df) else float("nan"),
        "fp_rate_at_any_prediction": round(float((df["n_pred"] > 0).mean()), 4) if len(df) else float("nan"),
        "fp_rate_by_taxonomic_tercile": by_tercile,
        "fp_vs_taxonomic": _corr_over_species(species, "taxonomic_distance", config.seed, config.cv.n_bootstrap),
        "fp_vs_visual": _corr_over_species(species, "visual_distance", config.seed, config.cv.n_bootstrap),
    }


def analyse(split: str = "test", config: Config | None = None, threshold: float = 0.5) -> Path:
    """Build the FP table + summary; write both under ``outputs/``; return the parquet path."""
    cfg = config or Config()
    df = fp_table(split, cfg, threshold)
    path = write_parquet(df, cfg.paths.outputs_root / "false_positives.parquet")
    summary = summarise(df, cfg, threshold)
    (cfg.paths.outputs_root / "false_positives_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"false positives -> {path} ({len(df)} probes)")
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--threshold", type=float, default=0.5, help="min masklet score to count as a hallucination")
    args = ap.parse_args()
    analyse(args.split, Config.load(args.config), args.threshold)


if __name__ == "__main__":
    main()
