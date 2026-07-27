"""MammAlps → SA-Co adapter (R7 independent replication; box-only Alpine camera-trap tracker).

MammAlps ships one JSON per video clip: an ``info`` block (site/cam/fps/file_path/resolution) and a ``frames``
list where each frame carries ``detections`` with a ``track_id``, an ``bbox`` (**xyxy** absolute pixels), and
``attributes.species`` (one of 5 Alpine mammals). There are **no masks** and the clips are 30 fps.

This converter emits the SA-Co ``_ext`` schema the pipeline already reads, with two box-specific choices
(validated against the loader/scorer contract): each GT box becomes a **filled-rectangle RLE** in
``segmentations`` (so the mask-based visual/size/environment features run as box features) **and** the native
box in ``bboxes`` (so ``eval.prefer_bbox`` scores box-HOTA). The 30 fps clips are subsampled to
``mammalps.target_fps`` and capped per ``(species, camera)`` cell so SAM 3 inference stays tractable given the
red-deer-dominated distribution. Location = ``site_cam`` (the environment axis MammAlps uniquely supports);
time is left empty (the 6-week window makes the temporal axis untestable).

All clips go to the probe (``test``) file for a pooled leave-species-out / leave-camera-out analysis; the
reference (``train``) file is emitted empty (Split A draws reference=probe from ``origins=("test",)``). Extra
per-video fields ``source_video`` and ``sample_step`` are carried for the frame-extraction script; the loader
ignores them.

Run: ``PYTHONPATH=. python -m src.adapters.mammalps --config configs/mammalps.yaml``
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from src.adapters.boxes import box_to_rle
from src.adapters.taxonomy import ClassTaxonomy, normalise_name
from src.config import Config
from src.dataset import _TAXONOMY_FIELDS

_PACKAGED_CSV = Path(__file__).parent / "mammalps_taxonomy.csv"


def _xywh(xyxy: list[float]) -> list[float]:
    """MammAlps boxes are ``[x_min, y_min, x_max, y_max]`` → COCO ``[x, y, w, h]``."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


class MammAlpsAdapter:
    """Emit ``mammalps_{train,test}_ext.json`` (test=all capped clips; train=empty) in the SA-Co schema."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (loads the 5-species taxonomy + a stable species→category_id map)."""
        self.config = config or Config()
        csv_path = (
            self.config.paths.data_root / self.config.mammalps.taxonomy_csv
            if self.config.mammalps.taxonomy_csv
            else _PACKAGED_CSV
        )
        self.taxonomy = ClassTaxonomy(csv_path)
        self._cat_id = {name: i + 1 for i, name in enumerate(sorted(self.taxonomy.names()))}

    def _location(self, info: dict) -> str:
        """Cell/location key: ``site_cam`` (S1_C1 …) or the coarser ``site``."""
        site, cam = str(info.get("site_id", "")), str(info.get("cam_id", ""))
        return site if self.config.mammalps.location_by == "site" else f"{site}_{cam}"

    def _selected_frames(self, info: dict, frames: list) -> tuple[list[int], int]:
        """Subsampled 0-indexed frame positions (step = clip_fps/target_fps), capped; + the step."""
        m = self.config.mammalps
        fps = float(info.get("fps", 30) or 30)
        step = max(1, round(fps / max(0.1, m.target_fps)))
        n = int(info.get("num_frames", len(frames)) or len(frames))
        selected = list(range(0, n, step))[: m.max_frames_per_clip]
        return selected, step

    def _clip_to_video(self, cid: int, info: dict, frames: list) -> tuple[dict, list[dict], set[int]] | None:
        """Convert one clip → (video, per-track annotations, present category_ids); None if unusable."""
        selected, step = self._selected_frames(info, frames)
        if len(selected) < self.config.mammalps.min_frames:
            return None
        w, h = (int(v) for v in str(info.get("resolution", "1920x1080")).lower().split("x"))
        file_id = Path(str(info.get("file_path", cid))).stem

        # frame_id -> {track_id: (bbox_xyxy, species)}
        by_frame: dict[int, dict[int, tuple]] = {}
        track_species: dict[int, str] = {}
        for fr in frames:
            fid = int(fr.get("frame_id", -1))
            for det in fr.get("detections") or []:
                sp = normalise_name((det.get("attributes") or {}).get("species", ""))
                if sp not in self.taxonomy.names():
                    continue
                tid = int(det["track_id"])
                by_frame.setdefault(fid, {})[tid] = (det.get("bbox"), sp)
                track_species.setdefault(tid, sp)
        if not track_species:
            return None

        video = {
            "id": cid,
            "video_name": file_id,
            "file_names": [f"{file_id}/{i:06d}.jpg" for i in range(len(selected))],
            "height": h,
            "width": w,
            "length": len(selected),
            "video_num_frames": int(info.get("num_frames", len(frames)) or len(frames)),
            "video_fps": float(self.config.mammalps.target_fps),
            "video_creation_datetime": "",  # 6-week window → temporal axis untestable
            "location_id": self._location(info),
            "source_video": str(info.get("file_path", "")),  # for the frame-extraction script
            "sample_step": step,
        }
        annotations = []
        for tid, sp in track_species.items():
            segs: list[dict | None] = []
            bboxes: list[list[float] | None] = []
            areas: list[float] = []
            has_box = False
            for fid in selected:
                box = (by_frame.get(fid, {}).get(tid) or (None, None))[0]
                rle = box_to_rle(*_xywh(box), h, w) if box else None
                if rle is not None:
                    segs.append(rle)
                    xywh = _xywh(box)
                    bboxes.append(xywh)
                    areas.append(float(xywh[2] * xywh[3]))
                    has_box = True
                else:
                    segs.append(None)
                    bboxes.append(None)
                    areas.append(0.0)
            if not has_box:
                continue
            annotations.append({
                "id": -1, "video_id": cid, "category_id": self._cat_id[sp],
                "height": h, "width": w, "iscrowd": 0, "noun_phrase": sp,
                "segmentations": segs, "bboxes": bboxes, "areas": areas,
            })
        present = {a["category_id"] for a in annotations}
        return (video, annotations, present) if present else None

    def convert(self) -> dict:
        """Parse + cap + subsample the dense-annotation clips into one SA-Co ``_ext`` dict (the probe set)."""
        raw = self.config.paths.data_root / self.config.mammalps.raw_dir
        clip_files = sorted(glob.glob(str(raw / "**" / "*.json"), recursive=True))
        cell_count: Counter = Counter()
        videos, annotations, pairs = [], [], []
        for path in clip_files:
            try:
                data = json.loads(Path(path).read_text())
            except (ValueError, OSError):
                continue
            info, frames = data.get("info", {}), data.get("frames", [])
            # primary species for the cap (clips are near-single-species)
            sp_counts = Counter(
                normalise_name((det.get("attributes") or {}).get("species", ""))
                for fr in frames for det in (fr.get("detections") or [])
            )
            primary = next((s for s, _ in sp_counts.most_common() if s in self.taxonomy.names()), None)
            if primary is None:
                continue
            cell = (primary, self._location(info))
            if cell_count[cell] >= self.config.mammalps.max_clips_per_cell:
                continue
            built = self._clip_to_video(len(videos), info, frames)
            if built is None:
                continue
            video, anns, present = built
            cell_count[cell] += 1
            videos.append(video)
            for a in anns:
                a["id"] = len(annotations)
                annotations.append(a)
            for cat in present:
                pairs.append({"id": len(pairs), "video_id": video["id"], "category_id": cat,
                              "noun_phrase": next(k for k, v in self._cat_id.items() if v == cat),
                              "num_masklets": sum(1 for a in anns if a["category_id"] == cat)})
        return {
            "info": {"description": "MammAlps -> SA-Co (probe; box GT as rectangle RLE + bboxes)"},
            "videos": videos, "annotations": annotations,
            "categories": self._categories({a["category_id"] for a in annotations}),
            "video_np_pairs": pairs,
        }

    def _categories(self, ids: set[int]) -> list[dict]:
        """SA-Co category entries for the present species (id, name, 7 capitalized taxonomy levels)."""
        id_name = {v: k for k, v in self._cat_id.items()}
        out = []
        for cid in sorted(ids):
            name = id_name[cid]
            tax = self.taxonomy.taxonomy_of(name)
            entry: dict = {"id": cid, "name": name}
            entry.update({fld: tax.get(fld) for fld in _TAXONOMY_FIELDS})
            out.append(entry)
        return out

    def run(self) -> tuple[Path, Path]:
        """Write ``data.test_ann`` (all clips) + an empty ``data.train_ann``; return the two paths."""
        out_dir = self.config.paths.data_root / self.config.data.annotations_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = self.convert()
        test_path = out_dir / self.config.data.test_ann
        test_path.write_text(json.dumps(probe))
        empty = {"info": {"description": "MammAlps reference (empty; Split A uses origins=test)"},
                 "videos": [], "annotations": [], "categories": probe["categories"], "video_np_pairs": []}
        train_path = out_dir / self.config.data.train_ann
        train_path.write_text(json.dumps(empty))
        print(f"probe: {len(probe['videos'])} clips, {len(probe['annotations'])} tracks, "
              f"{len(probe['categories'])} species, {len(probe['video_np_pairs'])} probes -> {test_path}")
        return train_path, test_path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config (e.g. configs/mammalps.yaml)")
    args = ap.parse_args()
    MammAlpsAdapter(Config.load(args.config)).run()


if __name__ == "__main__":
    main()
