"""Score with the OFFICIAL VEval evaluator and aggregate to cells.

Produces the dependent variables — per-cell ``pDetA`` / ``pAssA`` / ``pHOTA``. Runs the vendored VEval
evaluator (the metric is never re-implemented) over the harness predictions and the split annotations,
then aggregates to one row per ``(category_id, species, location_id, time)`` cell in
``outputs/scores.parquet``, carrying support counts and the prompt condition.

Run: ``python -m src.eval.score --split test [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import SAFARI
from src.io import write_parquet

_METRICS = ("pDetA", "pAssA", "pHOTA")

# VEval reports HOTA-family metrics; the exact per-entry spelling (bare, ``phrase``-prefixed, or
# ``p``-prefixed) is confirmed once against the vendored toy output (runbook cell 7). Each of our
# columns is resolved from the first candidate field present, so a spelling drift never silently NaNs.
_METRIC_KEYS = {
    "pDetA": ("pDetA", "DetA", "phrase_DetA", "video_mask_all_phrase_DetA"),
    "pAssA": ("pAssA", "AssA", "phrase_AssA", "video_mask_all_phrase_AssA"),
    "pHOTA": ("pHOTA", "HOTA", "phrase_HOTA", "video_mask_all_phrase_HOTA"),
}


def _first_present(entry: dict, keys: tuple[str, ...]) -> float | None:
    """First non-null value among ``keys`` in ``entry`` as a float (``None`` if none present)."""
    for key in keys:
        if entry.get(key) is not None:
            return float(entry[key])
    return None


class Scorer:
    """Dispatch to the vendored VEval scorer and aggregate its per-probe metrics to cells."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize."""
        self.config = config or Config()

    def _support(self, split: str) -> dict[tuple[str, str], tuple[int, int]]:
        """Per probe ``(video_id, category_id) -> (n_annotated_frames, n_masklets)`` from the GT."""
        safari = SAFARI(split, self.config)
        by_video = safari.annotations_by_video()
        support: dict[tuple[str, str], tuple[int, int]] = {}
        for record in safari.records():
            anns = [
                a
                for a in by_video.get(record.video_id, [])
                if str(a["category_id"]) == record.category_id
            ]
            frames = {
                i
                for a in anns
                for i, seg in enumerate(a.get("segmentations") or [])
                if seg is not None
            }
            support[(record.video_id, record.category_id)] = (len(frames), len(anns))
        return support

    def aggregate(self, per_probe: dict[tuple[str, str], dict], split: str) -> pd.DataFrame:
        """Join per-probe VEval metrics + support onto the cell grid → one row per cell.

        Args:
            per_probe: ``{(video_id, category_id): {"pDetA": .., "pAssA": .., "pHOTA": ..}}``.
            split: The split whose probes/cells to aggregate.

        Returns:
            A DataFrame with a row per ``(category_id, species, location_id, time)`` cell.
        """
        safari = SAFARI(split, self.config)
        support = self._support(split)
        cells: dict[tuple, dict] = defaultdict(
            lambda: {m: [] for m in _METRICS} | {"n_frames": 0, "n_masklets": 0, "n_videos": 0}
        )
        for record in safari.records():
            cell = safari.cell_of(record)
            agg = cells[(cell.category_id, cell.species, cell.location_id, cell.time)]
            metrics = per_probe.get((record.video_id, record.category_id))
            if metrics:
                for m in _METRICS:
                    if metrics.get(m) is not None:
                        agg[m].append(float(metrics[m]))
            n_frames, n_masklets = support.get((record.video_id, record.category_id), (0, 0))
            agg["n_frames"] += n_frames
            agg["n_masklets"] += n_masklets
            agg["n_videos"] += 1

        rows = []
        for (category_id, species, location_id, time), agg in cells.items():
            rows.append(
                {
                    "category_id": category_id,
                    "species": species,
                    "location_id": location_id,
                    "time": time,
                    **{m: float(np.mean(agg[m])) if agg[m] else float("nan") for m in _METRICS},
                    "n_frames": agg["n_frames"],
                    "n_masklets": agg["n_masklets"],
                    "n_videos": agg["n_videos"],
                    "prompt_mode": self.config.inference.prompt_mode,
                }
            )
        return pd.DataFrame(rows)

    def _run_veval(self, pred_file: Path, gt_file: Path) -> dict:
        """Run the vendored VEval script on (predictions, GT) and return its result JSON.

        The exact CLI and result schema are finalised against the vendored ``saco_veval_eval.py`` on the
        GPU box; kept isolated here.
        """
        script = self.config.paths.third_party_root / self.config.eval.veval_script
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            result_path = Path(tmp.name)
        subprocess.run(
            [
                sys.executable,
                str(script),
                "one",
                "--gt_annot_file",
                str(gt_file),
                "--pred_file",
                str(pred_file),
                "--eval_res_file",
                str(result_path),
            ],
            check=True,
        )
        return json.loads(result_path.read_text())

    def _parse_veval(self, result: dict) -> dict[tuple[str, str], dict]:
        """Map the VEval result into ``{(video_id, category_id): {metric: value}}``.

        The evaluator emits ``{"dataset_results": {..aggregate..}, "video_np_results": [{video_id,
        category_id, **metrics}]}``; we key each per-probe entry by ``(video_id, category_id)`` (as
        strings, to match the harness records) and resolve each metric via :data:`_METRIC_KEYS`. Absent
        ``video_np_results`` yields an empty map (scores become NaN, support is still counted).
        """
        entries = (
            result.get("video_np_results")
            or result.get("per_video")
            or result.get("results")
            or []
        )
        per_probe: dict[tuple[str, str], dict] = {}
        for entry in entries:
            key = (str(entry["video_id"]), str(entry["category_id"]))
            per_probe[key] = {m: _first_present(entry, keys) for m, keys in _METRIC_KEYS.items()}
        return per_probe

    def score(self, split: str = "test") -> Path:
        """Score the harness predictions and write ``outputs/scores.parquet``; return its path."""
        inf = self.config.inference
        pred_file = (
            self.config.paths.outputs_root
            / inf.predictions_subdir
            / split
            / f"{inf.prompt_mode}.json"
        )
        result = self._run_veval(pred_file, SAFARI(split, self.config).ann_path)
        df = self.aggregate(self._parse_veval(result), split)
        return write_parquet(df, self.config.paths.outputs_root / "scores.parquet")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    args = ap.parse_args()
    print("scores ->", Scorer(Config.load(args.config)).score(args.split))


if __name__ == "__main__":
    main()
