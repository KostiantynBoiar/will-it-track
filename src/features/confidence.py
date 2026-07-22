"""After-running confidence features — an ATC-style detection-reliability estimator.

The four label-free *distances* are *before-running* signals (decide trust before spending any compute).
These features are the complementary *after-running, before-labelling* signal: they read SAM 3's **own
outputs** on the target cell (masklet confidence scores + per-frame presence) — you have paid the inference
cost but not the annotation cost. This is a weaker claim than the before-running predictor, but the realistic
conservation operating point (inference is cheap; annotation is the bottleneck), and a confirmatory
application of the average-thresholded-confidence family (ATC; Garg et al. 2022 / DoC; Guillory et al. 2021)
to a promptable video *tracker*. It targets **detection** (``pDetA``) only — the association half leaves
almost no distinct variance to predict (see ``CLAUDE.md`` §12), so no ``pAssA`` claim is made from it.

Per positive cell we read the harness prediction JSONs (one per ``(video, species)`` probe, keyed by origin)
and aggregate:

- ``conf_mean_score`` / ``conf_median_score`` — over the cell's kept masklets. **Honest caveat:** the stored
  per-masklet ``score`` is the *max over frames* of the per-frame object score (``sam3_tracker.py`` collapses
  the trajectory at write time), so this is a *masklet-level* ATC, not the canonical per-frame ATC.
- ``conf_frame_coverage`` — fraction of the cell's frames on which at least one masklet is present (from the
  ``segmentations`` ``None``-pattern).
- ``conf_atc_coverage`` — fraction of the cell's masklets scoring ``>= t``, where the single threshold ``t``
  is calibrated **once** on the frozen reference (seen) split so its mean coverage matches the reference mean
  ``pDetA``, then frozen before any probe cell is scored (:func:`calibrate_atc_threshold`). This is the
  canonical ATC construction, ported to masklet confidences.

Keyed by ``(category_id, species, location_id, time)`` like :class:`~src.features.temporal.TemporalGap`, so it
merges onto the cell grid identically. Deliberately torch/PIL-free (reads JSON + parquet only), so it runs in
a bare analysis environment without the SAM 3 / embedding stack.
"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import _year
from src.splits import probe_records, reference_records

if TYPE_CHECKING:
    from pathlib import Path

    from src.dataset import VideoRecord
    from src.splits import Partition

_CELL_KEYS = ["category_id", "species", "location_id", "time"]
CONF_COLS = ["conf_atc_coverage", "conf_mean_score", "conf_median_score", "conf_frame_coverage"]


def _probe_prediction_path(record: VideoRecord, config: Config) -> Path:
    """On-disk prediction JSON for a probe, keyed by ``(video, species)`` under the record's origin.

    ``pooled_records`` namespaces ``video_id`` as ``"<origin>:<raw>"``; the harness wrote the file under the
    raw id, so strip the namespace. Mirrors ``harness.probe_filename`` / ``harness._out_dir`` without
    importing the (torch-bound) harness module.
    """
    raw_id = record.video_id.split(":", 1)[-1]
    return (
        config.paths.outputs_root
        / config.inference.predictions_subdir
        / record.origin
        / config.inference.prompt_mode
        / f"{raw_id}_{record.category_id}.json"
    )


def _masklet_scores_and_frames(path: Path) -> tuple[list[float], int, int]:
    """Read one probe JSON → (masklet max-scores, frames with ≥1 present masklet, total frames).

    A hard negative / total miss writes ``[]`` → ``([], 0, 0)``. ``segmentations`` is a per-frame list with
    ``None`` on absent frames; a frame is *covered* if any masklet is present on it.
    """
    entries = json.loads(path.read_text())
    scores = [float(entry.get("score", 0.0)) for entry in entries]
    n_frames = max((len(entry.get("segmentations") or []) for entry in entries), default=0)
    covered = 0
    for i in range(n_frames):
        if any((entry.get("segmentations") or [])[i] is not None for entry in entries):
            covered += 1
    return scores, covered, n_frames


def _reference_scores(config: Config, partition: Partition) -> list[float]:
    """Pool every masklet max-score over the partition's reference (seen) probes — the ATC calibration set."""
    scores: list[float] = []
    for record in reference_records(partition, config):
        path = _probe_prediction_path(record, config)
        if path.exists():
            scores.extend(_masklet_scores_and_frames(path)[0])
    return scores


def calibrate_atc_threshold(reference_scores: list[float], reference_pdeta: float) -> float:
    """The single ATC threshold ``t``: frozen so the reference thresholded-coverage matches its mean ``pDetA``.

    Chooses ``t`` such that the fraction of reference masklets scoring ``>= t`` equals ``reference_pdeta``
    (Average Thresholded Confidence; Garg et al. 2022). With ``p = reference_pdeta`` this is the
    ``(1 - p)`` quantile of the pooled reference scores. Calibrated on the seen set only and returned once,
    so no probe/held-out label ever informs it.
    """
    if not reference_scores:
        return 0.5
    p = float(min(1.0, max(0.0, reference_pdeta)))
    return float(np.quantile(np.asarray(reference_scores, dtype=float), 1.0 - p))


class ConfidenceFeature:
    """Per-cell ATC-style detection-confidence features from SAM 3's own prediction JSONs."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``paths.outputs_root``, ``inference.predictions_subdir`` / ``prompt_mode``).
        """
        self.config = config or Config()

    def compute(
        self,
        partition: Partition,
        *,
        threshold: float | None = None,
        reference_pdeta: float | None = None,
    ) -> pd.DataFrame:
        """Return the confidence-feature block per probe cell.

        Args:
            partition: The active split (its probe side supplies the cells; its reference side calibrates ``t``).
            threshold: A pre-frozen ATC threshold ``t``. When ``None`` it is calibrated on the reference
                (needs ``reference_pdeta``).
            reference_pdeta: Mean ``pDetA`` over the reference cells — the ATC calibration target. Required
                when ``threshold`` is ``None``.

        Returns:
            A frame with a ``(category_id, species, location_id, time)`` MultiIndex and the columns
            ``conf_atc_coverage``, ``conf_mean_score``, ``conf_median_score``, ``conf_frame_coverage``.
        """
        if threshold is None:
            if reference_pdeta is None:
                raise ValueError("calibrate the ATC threshold: pass threshold=… or reference_pdeta=…")
            threshold = calibrate_atc_threshold(_reference_scores(self.config, partition), reference_pdeta)

        scores: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        covered: dict[tuple[str, str, str, str], int] = defaultdict(int)
        total: dict[tuple[str, str, str, str], int] = defaultdict(int)
        seen: set[tuple[str, str, str, str]] = set()
        for record in probe_records(partition, self.config):
            path = _probe_prediction_path(record, self.config)
            if not path.exists():
                continue  # not inferred (sampled subset) — leave the cell to be NaN after the left merge
            key = (record.category_id, record.species, record.location_id, _year(record.creation_datetime))
            seen.add(key)
            s, cov, n = _masklet_scores_and_frames(path)
            scores[key].extend(s)
            covered[key] += cov
            total[key] += n

        rows = {
            key: {
                "conf_atc_coverage": (
                    float(np.mean([1.0 if v >= threshold else 0.0 for v in scores[key]])) if scores[key] else 0.0
                ),
                "conf_mean_score": float(np.mean(scores[key])) if scores[key] else 0.0,
                "conf_median_score": float(median(scores[key])) if scores[key] else 0.0,
                "conf_frame_coverage": float(covered[key] / total[key]) if total[key] else 0.0,
            }
            for key in seen
        }
        index = pd.MultiIndex.from_tuples(rows or [], names=_CELL_KEYS)
        return pd.DataFrame(list(rows.values()), index=index, columns=CONF_COLS).astype("float64")
