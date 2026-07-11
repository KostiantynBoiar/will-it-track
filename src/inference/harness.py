"""Frozen SAM 3 inference harness.

Runs frozen SAM 3 promptable tracking over every probe of a split and writes predicted masklets per
``(video, prompt)`` to ``outputs/predictions/`` in the evaluator's expected format. Species-specific
prompts are primary; a generic ``"animal"`` prompt is the robustness condition. Hard negatives are kept
(they should yield no masklet). Per-video JSONs make the run resumable across Colab session timeouts.

Run: ``python -m src.inference.harness --split test [--limit N] [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools import mask as coco_mask

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.features.frames import ensure_frames, load_frame
from src.inference.sam3_tracker import Masklet, Tracker


def _bbox(rle: dict) -> list[float]:
    """COCO ``[x, y, w, h]`` bounding box for one RLE mask."""
    packed = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return [float(v) for v in coco_mask.toBbox(packed).tolist()]


class InferenceHarness:
    """Run frozen SAM 3 promptable tracking over a split, one resumable video JSON at a time."""

    def __init__(self, config: Config | None = None, tracker: Tracker | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``inference.*``, ``paths.*``).
            tracker: The tracker to use (defaults to the real :class:`Sam3Tracker`; tests inject a fake).
        """
        self.config = config or Config()
        if tracker is None:
            from src.inference.sam3_tracker import Sam3Tracker

            tracker = Sam3Tracker(self.config)
        self.tracker = tracker

    def _prompt(self, record: VideoRecord) -> str:
        """The text prompt for a probe (species-specific, or the generic robustness prompt)."""
        if self.config.inference.prompt_mode == "generic":
            return "animal"
        return record.noun_phrase or record.species

    def _out_dir(self, split: str) -> Path:
        """Per-(split, prompt-mode) predictions directory."""
        return (
            self.config.paths.outputs_root
            / self.config.inference.predictions_subdir
            / split
            / self.config.inference.prompt_mode
        )

    def _frames(self, record: VideoRecord) -> list:
        """Load a video's frames, fetching only the ones not already on disk."""
        frames = [load_frame(fn, self.config) for fn in record.file_names]
        missing = [fn for fn, frame in zip(record.file_names, frames, strict=True) if frame is None]
        if missing:
            ensure_frames(missing, record.origin, self.config)
            frames = [
                frame if frame is not None else load_frame(fn, self.config)
                for fn, frame in zip(record.file_names, frames, strict=True)
            ]
        return frames

    def _predict_video(self, record: VideoRecord) -> dict:
        """Track one probe and build its prediction entry (empty ``masklets`` when nothing is found)."""
        frames = self._frames(record)
        if not frames or any(frame is None for frame in frames):
            masklets: list[Masklet] = []  # frames unavailable → no prediction (scored as a miss)
        else:
            masklets = self.tracker.track(frames, self._prompt(record))
        return {
            "video_id": record.video_id,
            "category_id": record.category_id,
            "noun_phrase": record.noun_phrase,
            "prompt": self._prompt(record),
            "n_frames": len(record.file_names),
            "masklets": [
                {
                    "segmentations": m.segmentations,
                    "bboxes": [_bbox(s) if s is not None else None for s in m.segmentations],
                    "score": m.score,
                }
                for m in masklets
            ],
        }

    def run(self, split: str = "test", limit: int | None = None) -> Path:
        """Predict masklets for every probe and write per-video JSONs; return the predictions dir.

        Args:
            split: ``"train"`` / ``"test"``.
            limit: Optional cap on the number of probes (for a subset smoke test).

        Returns:
            The per-(split, prompt-mode) predictions directory.
        """
        out_dir = self._out_dir(split)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = SAFARI(split, self.config).records()
        if limit is not None:
            records = records[:limit]

        for i, record in enumerate(records):
            path = out_dir / f"{record.video_id}.json"
            if path.exists():
                continue  # resumable: already predicted
            path.write_text(json.dumps(self._predict_video(record)))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(records)} probes predicted")

        combined = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
        combined_path = out_dir.with_suffix(".json")
        combined_path.write_text(json.dumps({"predictions": combined}))
        print(f"predictions -> {out_dir} ({len(combined)} videos) | combined -> {combined_path}")
        return out_dir


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--limit", type=int, default=None, help="cap probes for a subset smoke test")
    args = ap.parse_args()
    InferenceHarness(Config.load(args.config)).run(args.split, args.limit)


if __name__ == "__main__":
    main()
