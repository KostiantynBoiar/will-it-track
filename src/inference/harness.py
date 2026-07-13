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
from tqdm import tqdm

from src.config import Config
from src.dataset import SAFARI, VideoRecord
from src.features.frames import ensure_frames, load_frame
from src.inference.sam3_tracker import Masklet, Tracker


def _packed(rle: dict) -> dict:
    """A pycocotools-ready RLE (``counts`` as bytes) from our JSON-serialisable ``{size, counts}``."""
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def _bbox(rle: dict) -> list[float]:
    """COCO ``[x, y, w, h]`` bounding box for one RLE mask."""
    return [float(v) for v in coco_mask.toBbox(_packed(rle)).tolist()]


def _area(rle: dict) -> int:
    """Mask area (pixel count) for one RLE mask."""
    return int(coco_mask.area(_packed(rle)))


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

    def _predict_video(self, record: VideoRecord) -> list[dict]:
        """Track one probe → a flat list of per-masklet entries in the official VEval schema.

        Each entry is one masklet: integer ``video_id`` / ``category_id`` matching the split GT (VEval
        joins predictions to ground truth on that pair), a single ``score``, and equal-length per-frame
        ``segmentations`` (RLE), ``bboxes`` (COCO ``[x, y, w, h]``) and ``areas`` — ``None`` / ``0`` on
        frames where the object is absent. Hard negatives (and unavailable frames) contribute no entries.
        """
        frames = self._frames(record)
        if not frames or any(frame is None for frame in frames):
            masklets: list[Masklet] = []  # frames unavailable → no prediction (scored as a miss)
        else:
            masklets = self.tracker.track(frames, self._prompt(record))
        video_id, category_id = int(record.video_id), int(record.category_id)
        return [
            {
                "video_id": video_id,
                "category_id": category_id,
                "score": m.score,
                "segmentations": m.segmentations,
                "bboxes": [_bbox(s) if s is not None else None for s in m.segmentations],
                "areas": [_area(s) if s is not None else 0 for s in m.segmentations],
            }
            for m in masklets
        ]

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

        desc = f"SAM 3 {split}/{self.config.inference.prompt_mode}"
        for record in tqdm(records, desc=desc, unit="probe"):
            path = out_dir / f"{record.video_id}.json"
            if path.exists():
                continue  # resumable: already predicted
            path.write_text(json.dumps(self._predict_video(record)))

        masklets = [
            entry for p in sorted(out_dir.glob("*.json")) for entry in json.loads(p.read_text())
        ]
        combined_path = out_dir.with_suffix(".json")
        combined_path.write_text(json.dumps(masklets))  # flat list — the format VEval ingests
        print(f"predictions -> {out_dir} ({len(masklets)} masklets) | combined -> {combined_path}")
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
