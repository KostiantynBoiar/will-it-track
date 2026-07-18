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
import os
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

# VEval's per-probe ``video_np_results`` entries report HOTA-family metrics prefixed by annotation type
# (``mask_*`` / ``bbox_*``), confirmed against the vendored toy output (runbook cell 7). SA-FARI is
# mask-based, so we take the ``mask_*`` value first and fall back to ``bbox_*`` (then bare spellings).
_METRIC_KEYS = {
    "pDetA": ("mask_DetA", "bbox_DetA", "pDetA", "DetA"),
    "pAssA": ("mask_AssA", "bbox_AssA", "pAssA", "AssA"),
    "pHOTA": ("mask_HOTA", "bbox_HOTA", "pHOTA", "HOTA"),
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

        The scorer is run as a standalone script (not pip-installed), so its own package must be on
        ``PYTHONPATH`` — we point it at the vendored clone root so ``import sam3`` resolves regardless of
        the caller's environment.
        """
        script = self.config.paths.third_party_root / self.config.eval.veval_script
        sam3_root = self.config.paths.third_party_root / "sam3"  # clone dir → makes `import sam3` work
        env = {**os.environ}
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(sam3_root), env.get("PYTHONPATH", "")]))
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
            env=env,
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
        summary_path = self.config.paths.outputs_root / "veval_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result.get("dataset_results", result), indent=2))
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
