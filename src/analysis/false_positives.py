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
from src.features.size import SizeFeature
from src.features.taxonomic import TaxonomicDistance
from src.features.visual import VisualDistance
from src.inference.harness import probe_filename
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
    size = SizeFeature(config).compute(partition)  # log_area, to test the visual/size confound on FPs too

    rows = []
    for record in records:
        path = pred_dir / probe_filename(record.video_id, record.category_id)
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
                "log_area": float(size.get(record.category_id, float("nan"))),
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


def _partial_corr_over_species(
    species: pd.DataFrame, x_col: str, z_col: str, seed: int, n_boot: int
) -> dict:
    """Support-weighted partial correlation of per-species FP rate with ``x_col`` controlling ``z_col``.

    Uses the standard partial-correlation identity on the three weighted pairwise correlations, with a
    species bootstrap CI. If the visual FP effect is really a size artefact, controlling ``log_area`` here
    shrinks it toward zero (mirroring the size-controlled coefficient ablation).
    """
    data = species[["fp_rate", x_col, z_col, "n"]].dropna()
    if len(data) < 4 or data[x_col].std() == 0 or data[z_col].std() == 0:
        return {"partial_r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_species": int(len(data)), "significant": False}

    def partial(frame: pd.DataFrame) -> float:
        w = frame["n"].to_numpy(float)
        fp, x, z = (frame[c].to_numpy(float) for c in ("fp_rate", x_col, z_col))
        r_fx, r_fz, r_xz = (_weighted_pearson(a, b, w) for a, b in ((fp, x), (fp, z), (x, z)))
        denom = np.sqrt(max(0.0, (1 - r_fz**2) * (1 - r_xz**2)))
        return (r_fx - r_fz * r_xz) / denom if denom > 0 else float("nan")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    boots = [r for _ in range(n_boot) if np.isfinite(r := partial(data.iloc[rng.choice(idx, len(idx))]))]
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5])) if boots else (float("nan"), float("nan"))
    return {"partial_r": partial(data), "ci_lo": lo, "ci_hi": hi,
            "n_species": int(len(data)), "significant": bool(lo > 0 or hi < 0)}


def _wls_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least-squares line ``y ~ x`` -> (slope, intercept)."""
    if w.sum() <= 0:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    var = np.average((x - mx) ** 2, weights=w)
    slope = np.average((x - mx) * (y - my), weights=w) / var if var > 0 else 0.0
    return float(slope), float(my - slope * mx)


def fp_predictor_cv(species: pd.DataFrame, config: Config, predictor: str = "visual_distance") -> dict:
    """Leave-one-species-out validation: predict a held-out species' FP rate from ``predictor``.

    Each fold fits a support-weighted line on the other species and predicts the held-out one; the gain over
    a mean-predictor baseline is tested with a species bootstrap. This is what earns the ``predictor'' claim
    (per the decision rule) for the hallucination-risk signal, beyond the in-sample correlation.
    """
    data = species[["fp_rate", predictor, "n"]].dropna().reset_index(drop=True)
    n = len(data)
    nan = {"mae": float("nan"), "baseline_mae": float("nan"), "delta": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan"),
           "significant": False, "n_species": int(n)}
    if n < 5 or data[predictor].std() == 0:
        return nan
    x, y, w = (data[c].to_numpy(float) for c in (predictor, "fp_rate", "n"))
    pred, base = np.empty(n), np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        slope, intercept = _wls_line(x[tr], y[tr], w[tr])
        pred[i] = intercept + slope * x[i]
        base[i] = np.average(y[tr], weights=w[tr])  # leave-one-out mean baseline
    delta = np.abs(y - base) - np.abs(y - pred)  # >0 => the distance predictor beats the mean
    rng = np.random.default_rng(config.seed)
    boots = np.array([np.average(delta[s], weights=w[s])
                      for s in (rng.choice(n, n) for _ in range(config.cv.n_bootstrap))])
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return {"mae": float(np.average(np.abs(y - pred), weights=w)),
            "baseline_mae": float(np.average(np.abs(y - base), weights=w)),
            "delta": float(np.average(delta, weights=w)),
            "ci_lo": lo, "ci_hi": hi, "p_value": float((boots <= 0).mean()),
            "significant": bool(lo > 0), "n_species": int(n)}


def summarise(df: pd.DataFrame, config: Config, threshold: float) -> dict:
    """Overall FP rate, a taxonomic-tercile breakdown, and species-level FP-rate<->distance correlations."""
    agg = {
        "species": ("species", "first"),
        "fp_rate": ("fp", "mean"),
        "n": ("fp", "size"),
        "taxonomic_distance": ("taxonomic_distance", "first"),
        "visual_distance": ("visual_distance", "first"),
    }
    if "log_area" in df.columns:
        agg["log_area"] = ("log_area", "first")
    species = df.groupby("category_id").agg(**agg).reset_index()
    tercile = df.dropna(subset=["taxonomic_distance"]).copy()
    by_tercile = {}
    if len(tercile) >= 3 and tercile["taxonomic_distance"].nunique() >= 2:
        # Auto-label the surviving bins: the taxonomic distance is discrete (integer tree steps), so
        # qcut with duplicates="drop" may collapse to fewer than three bins -- fixed labels would then
        # mismatch the bin count. Interval labels adapt to however many terciles survive.
        tercile["bin"] = pd.qcut(tercile["taxonomic_distance"], 3, duplicates="drop")
        by_tercile = {
            str(k): round(float(v), 4)
            for k, v in tercile.groupby("bin", observed=True)["fp"].mean().items()
        }
    seed, n_boot = config.seed, config.cv.n_bootstrap
    out = {
        "split": config.experiment,
        "threshold": threshold,
        "n_probes": int(len(df)),
        "n_species": int(species["category_id"].nunique()),
        "overall_fp_rate": round(float(df["fp"].mean()), 4) if len(df) else float("nan"),
        "fp_rate_at_any_prediction": round(float((df["n_pred"] > 0).mean()), 4) if len(df) else float("nan"),
        "fp_rate_by_taxonomic_tercile": by_tercile,
        "fp_vs_taxonomic": _corr_over_species(species, "taxonomic_distance", seed, n_boot),
        "fp_vs_visual": _corr_over_species(species, "visual_distance", seed, n_boot),
        # leave-species-out validation: does visual distance predict a held-out species' FP rate?
        "fp_predictor_oos": fp_predictor_cv(species, config, "visual_distance"),
    }
    if "log_area" in species.columns:  # the size-confound checks on the visual FP effect
        out["fp_vs_size"] = _corr_over_species(species, "log_area", seed, n_boot)
        out["fp_vs_visual_size_controlled"] = _partial_corr_over_species(
            species, "visual_distance", "log_area", seed, n_boot
        )
    return out


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
