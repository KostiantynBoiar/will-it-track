"""BURST → SA-Co adapter (R7 independent replication).

Converts the downloaded BURST split JSONs (``sequences`` of per-frame RLE tracks over the LVIS vocabulary)
into the SA-Co ``_ext`` schema the pipeline already reads (``videos`` / ``annotations`` / ``categories`` /
``video_np_pairs``), keeping only the **animal** categories that resolve to a taxonomy. Because the
abstraction boundary is the JSON schema, the loader, the vendored VEval scorer, the splits, and every distance
run unchanged on the output --- point ``data.train_ann`` / ``data.test_ann`` at the emitted files.

Mapping: BURST ``sequence`` → ``video``; each track (``track_category_ids`` + per-frame ``segmentations``) →
one masklet ``annotation`` with per-frame COCO-RLE ``{size, counts}``; present animal categories + the
sequence's ``neg_category_ids`` → ``video_np_pairs`` (the latter as ``num_masklets=0`` hard negatives). BURST
has no location or capture time, so ``location_id`` / ``video_creation_datetime`` are emitted empty (the
environment and temporal distances degrade to ``NaN`` --- reported as untested, not as a tested null).

Run: ``PYTHONPATH=. python -m src.adapters.burst [--config configs/burst.yaml]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools import mask as coco_mask

from src.adapters.burst_taxonomy import BurstTaxonomy
from src.config import Config
from src.dataset import _TAXONOMY_FIELDS


def _rle_area(rle_counts: str, height: int, width: int) -> int:
    """Pixel area of one RLE mask (from the counts string; no full decode)."""
    return int(coco_mask.area({"size": [height, width], "counts": rle_counts.encode("ascii")}))


class BurstAdapter:
    """Emit ``{train,test}`` SA-Co ``_ext`` JSONs from the BURST split JSONs (animal categories only)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (loads the class → taxonomy map)."""
        self.config = config or Config()
        self.taxonomy = BurstTaxonomy(self.config)

    def _animal_categories(self, data: dict) -> dict[int, str]:
        """``category_id → name`` for the BURST categories that resolve to a taxonomy (animals)."""
        return {
            int(c["id"]): str(c["name"])
            for c in data.get("categories", [])
            if self.taxonomy.taxonomy_of(str(c["name"]))
        }

    def _track_annotations(
        self, seq: dict, vid: int, height: int, width: int, keep: dict[int, str]
    ) -> tuple[list[dict], dict[int, int]]:
        """Per-track masklet annotations for one sequence + a ``category_id → #masklets`` count."""
        frames = seq.get("annotated_image_paths") or []
        segs = seq.get("segmentations") or []  # per annotated frame: {track_id: {rle, bbox, ...}}
        track_cat = {str(t): int(c) for t, c in (seq.get("track_category_ids") or {}).items()}
        anns: list[dict] = []
        counts: dict[int, int] = {}
        for track_id, cid in track_cat.items():
            if cid not in keep:
                continue
            segmentations: list[dict | None] = []
            bboxes: list[list[float] | None] = []
            areas: list[float] = []
            has_mask = False
            for fi in range(len(frames)):
                obj = (segs[fi] if fi < len(segs) else {}).get(track_id)
                if obj and obj.get("rle"):
                    segmentations.append({"size": [height, width], "counts": obj["rle"]})
                    bboxes.append([float(x) for x in obj["bbox"]] if obj.get("bbox") else None)
                    areas.append(float(_rle_area(obj["rle"], height, width)))
                    has_mask = True
                else:
                    segmentations.append(None)
                    bboxes.append(None)
                    areas.append(0.0)
            if not has_mask:
                continue
            anns.append({
                "id": -1,  # renumbered globally in convert_split
                "video_id": vid,
                "category_id": cid,
                "height": height,
                "width": width,
                "iscrowd": 0,
                "noun_phrase": keep[cid],
                "segmentations": segmentations,
                "bboxes": bboxes,
                "areas": areas,
            })
            counts[cid] = counts.get(cid, 0) + 1
        return anns, counts

    def convert_split(self, burst_json: Path, origin: str) -> dict:
        """Convert one BURST split JSON into a SA-Co ``_ext`` dict (animal categories only)."""
        data = json.loads(Path(burst_json).read_text())
        keep = self._animal_categories(data)

        videos: list[dict] = []
        annotations: list[dict] = []
        pairs: list[dict] = []
        for seq in data.get("sequences", []):
            frames = seq.get("annotated_image_paths") or []
            if len(frames) < self.config.burst.min_frames:
                continue
            vid = int(seq["id"])
            height, width = int(seq["height"]), int(seq["width"])
            videos.append({
                "id": vid,
                "video_name": str(seq.get("seq_name", vid)),
                "file_names": list(frames),
                "height": height,
                "width": width,
                "length": len(frames),
                "video_num_frames": len(seq.get("all_image_paths") or frames),
                "video_fps": float(seq.get("fps", 0) or 0),
                "video_creation_datetime": "",
                "location_id": "",
            })
            anns, present = self._track_annotations(seq, vid, height, width, keep)
            for ann in anns:
                ann["id"] = len(annotations)
                annotations.append(ann)
            for cid, n in present.items():
                pairs.append({"id": len(pairs), "video_id": vid, "category_id": cid,
                              "noun_phrase": keep[cid], "num_masklets": n})
            if self.config.burst.keep_hard_negatives:
                for raw in seq.get("neg_category_ids") or []:
                    cid = int(raw)
                    if cid in keep and cid not in present:
                        pairs.append({"id": len(pairs), "video_id": vid, "category_id": cid,
                                      "noun_phrase": keep[cid], "num_masklets": 0})

        return {
            "info": {"description": f"BURST -> SA-Co ({origin}); animal subset only"},
            "videos": videos,
            "annotations": annotations,
            "categories": self._categories(keep),
            "video_np_pairs": pairs,
        }

    def _categories(self, keep: dict[int, str]) -> list[dict]:
        """SA-Co category entries (id, name, + the 7 capitalized taxonomy levels; ``None`` where coarse)."""
        out = []
        for cid, name in sorted(keep.items()):
            tax = self.taxonomy.taxonomy_of(name)
            entry: dict = {"id": cid, "name": name}
            entry.update({f: tax.get(f) for f in _TAXONOMY_FIELDS})
            out.append(entry)
        return out

    def run(self) -> tuple[Path, Path]:
        """Convert both BURST splits → ``data.train_ann`` / ``data.test_ann``; return the two paths."""
        b = self.config.burst
        raw = self.config.paths.data_root / b.raw_dir
        out_dir = self.config.paths.data_root / self.config.data.annotations_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for origin, src, dst in (
            ("train", b.train_json, self.config.data.train_ann),
            ("test", b.test_json, self.config.data.test_ann),
        ):
            converted = self.convert_split(raw / src, origin)
            path = out_dir / dst
            path.write_text(json.dumps(converted))
            print(
                f"{origin}: {len(converted['videos'])} videos, {len(converted['annotations'])} masklets, "
                f"{len(converted['categories'])} animal categories, "
                f"{len(converted['video_np_pairs'])} probes -> {path}"
            )
            written.append(path)
        return written[0], written[1]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config (e.g. configs/burst.yaml)")
    args = ap.parse_args()
    BurstAdapter(Config.load(args.config)).run()


if __name__ == "__main__":
    main()
