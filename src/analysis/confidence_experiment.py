"""After-running confidence experiment (T4.1c) — the pre-registered ATC detection estimator.

Augments the standing ``features.parquet`` with the ATC-style confidence block (``features/confidence.py``)
plus ``log_area`` (the mandatory size covariate, computed here torch-free), then validates — at the **exact**
bar the label-free distances had to clear — whether SAM 3's own detection confidence predicts ``pDetA`` out
of sample. Detection only; no ``pAssA`` claim (the association target barely varies — ``CLAUDE.md`` §12).

Pre-registration (fixed before fitting):
  * PRIMARY: ``conf_atc_coverage -> pDetA``, controlling ``log(n_frames)`` + ``log_area``. Claim a result
    only if **both** leave-species-out and leave-location-out give a positive OOS gain over the mean baseline
    with ``p < 0.05`` after Bonferroni over the ``{atc_coverage, mean_score, frame_coverage}`` family.
  * SECONDARY (Bonferroni): ``conf_mean_score``, ``conf_frame_coverage``.
  * The reframe (stated everywhere): this is *after running, before labelling* — a weaker, but realistic,
    claim than the before-running distances.

Each model is fit through the **unchanged** estimation core (``regression.DesignBuilder`` + ``cross_val``):
we scope it to a single confidence predictor by temporarily setting ``regression.DISTANCE_COLS`` /
``CONFIDENCE_COLS`` to the pre-registered set, so ``conf_atc_coverage + log_area + log(support)`` is the
isolated ATC model. Reports whatever comes out (positive or null); no feature is swapped in post hoc.

Run: ``PYTHONPATH=. python -m src.analysis.confidence_experiment [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import regression as R
from src.analysis.cross_val import _SCHEME_COLUMN, _summarise, oos_predictions
from src.config import Config
from src.dataset import SAFARI
from src.features.confidence import CONF_COLS, ConfidenceFeature
from src.io import read_parquet, write_parquet
from src.splits import Partition, build_location_partition, probe_records

_CELL_KEYS = ["category_id", "species", "location_id", "time"]
# Pre-registered confidence models: (label, tuple-of-confidence-cols, is_primary).
_PREREG = [
    ("atc_coverage", ("conf_atc_coverage",), True),
    ("mean_score", ("conf_mean_score",), False),
    ("frame_coverage", ("conf_frame_coverage",), False),
]
_BONFERRONI_M = len(_PREREG)  # 3-feature family


def _reference_pdeta(config: Config) -> float:
    """Mean ``pDetA`` over the reference (train) scoring — the ATC calibration anchor."""
    for name in ("scores_train.parquet", "scores.parquet"):
        path = config.paths.outputs_root / name
        if path.exists():
            df = read_parquet(path)
            if "pDetA" in df.columns:
                return float(pd.to_numeric(df["pDetA"], errors="coerce").mean())
    raise FileNotFoundError("no reference scoring found to calibrate the ATC threshold")


def _log_area(config: Config, partition: Partition) -> pd.Series:
    """``log_area`` per probe ``category_id`` (log1p mean GT mask pixels) — a torch-free SizeFeature.

    Replicates ``features.size.SizeFeature`` without importing the embedding stack (PIL / open_clip / torch),
    so it runs in a bare analysis environment. Ground-truth areas only; no prediction leaks in.
    """
    safari = {"train": SAFARI("train", config), "test": SAFARI("test", config)}
    cap = config.features.max_masklets_per_species
    frame_cap = max(1, config.features.n_frames_per_masklet)
    seen: Counter[str] = Counter()
    areas: dict[str, list[float]] = defaultdict(list)
    for record in probe_records(partition, config):
        cid = record.category_id
        if cap and seen[cid] >= cap:
            continue
        origin, _, raw_id = record.video_id.partition(":")
        reader = safari[origin]
        anns = [a for a in reader.annotations_by_video().get(raw_id, []) if str(a["category_id"]) == cid]
        for ann in anns:
            frames = [i for i, seg in enumerate(ann.get("segmentations") or []) if seg is not None]
            if not frames:
                continue
            step = max(1, len(frames) // frame_cap)
            for i in frames[::step][:frame_cap]:
                areas[cid].append(float(reader.mask_at(ann, i).sum()))
            seen[cid] += 1
            if cap and seen[cid] >= cap:
                break
    rows = {cid: float(np.log1p(np.mean(a))) for cid, a in areas.items() if a}
    return pd.Series(rows, name="log_area", dtype="float64")


def augment(config: Config) -> pd.DataFrame:
    """Add the ``conf_*`` block + ``log_area`` to ``features.parquet`` → ``features_conf.parquet``."""
    outputs = config.paths.outputs_root
    features = read_parquet(outputs / "features.parquet")
    partition = build_location_partition(config)

    ref_pdeta = _reference_pdeta(config)
    conf = ConfidenceFeature(config).compute(partition, reference_pdeta=ref_pdeta)
    log_area = _log_area(config, partition)

    df = features.copy()
    for key in _CELL_KEYS:
        df[key] = df[key].astype(str)
    df["log_area"] = df["category_id"].map(log_area)
    conf_df = conf.reset_index()
    for key in _CELL_KEYS:
        conf_df[key] = conf_df[key].astype(str)
    df = df.merge(conf_df, on=_CELL_KEYS, how="left")

    write_parquet(df, outputs / "features_conf.parquet")
    cov = {c: int(df[c].notna().sum()) for c in [*CONF_COLS, "log_area", "pDetA"]}
    print(f"features_conf -> {outputs/'features_conf.parquet'} ({len(df)} cells) coverage={cov} "
          f"| ATC ref-pDetA={ref_pdeta:.4f}")
    return df


def _cv_for_model(
    df: pd.DataFrame, config: Config, confidence_cols: tuple[str, ...], target: str = "pDetA"
) -> pd.DataFrame:
    """OOS summary (per scheme) for the isolated model ``confidence_cols + log_area + log(support)``.

    Scopes the design to exactly ``confidence_cols`` by pinning ``regression.DISTANCE_COLS = ()`` and
    ``CONFIDENCE_COLS = confidence_cols`` (the estimation core reads these module globals), with the
    ``log_area`` covariate forced on. Restores the globals afterwards.
    """
    saved = (R.DISTANCE_COLS, R.CONFIDENCE_COLS, config.model.control_size)
    R.DISTANCE_COLS, R.CONFIDENCE_COLS, config.model.control_size = (), confidence_cols, True
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
        R.DISTANCE_COLS, R.CONFIDENCE_COLS, config.model.control_size = saved


def run(config: Config | None = None) -> Path:
    """Run the pre-registered confidence experiment; write + return the summary JSON path."""
    cfg = config or Config()
    df = augment(cfg)
    alpha_corr = 0.05 / _BONFERRONI_M

    models = []
    for label, conf_cols, is_primary in _PREREG:
        summ = _cv_for_model(df, cfg, conf_cols, target="pDetA")
        det = summ[summ["target"] == "pDetA"] if not summ.empty else summ
        by_scheme = {r.group_scheme: r for r in det.itertuples()} if not det.empty else {}
        schemes = {
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
        both_pos = all(schemes[s]["delta"] > 0 for s in ("species", "location") if s in schemes)
        both_sig = all(
            s in schemes and schemes[s]["p_value"] < alpha_corr for s in ("species", "location")
        )
        models.append({
            "feature": label, "primary": is_primary, "schemes": schemes,
            "both_schemes_positive": both_pos,
            "both_schemes_significant_bonferroni": both_sig,
        })

    primary = next(m for m in models if m["primary"])
    verdict = (
        "VALIDATED (after-running detection estimator; both schemes clear Bonferroni)"
        if primary["both_schemes_significant_bonferroni"]
        else "NOT validated (does not clear the pre-registered bar on both schemes)"
    )
    out = {
        "experiment": "after_running_confidence_detection (T4.1c)",
        "reframe": "after running, before labelling — weaker than the before-running distances",
        "target": "pDetA", "n_pdeta_cells": int(pd.to_numeric(df["pDetA"], errors="coerce").notna().sum()),
        "bonferroni_m": _BONFERRONI_M, "alpha_corrected": round(alpha_corr, 4),
        "primary_feature": "conf_atc_coverage",
        "verdict": verdict,
        "models": models,
    }
    path = cfg.paths.outputs_root / "confidence_experiment_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nconfidence experiment -> {path}\nVERDICT: {verdict}")
    return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    run(Config.load(args.config))


if __name__ == "__main__":
    main()
