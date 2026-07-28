"""BURST → SA-Co adapter (R7 many-species replication; mask-native LVIS-class tracker).

BURST ships one ``all_classes.json`` per split: a list of ``sequences``, each with a source ``dataset`` +
``seq_name`` (which locate the TAO frames), ``annotated_image_paths``, per-annotated-frame COCO-RLE
``segmentations`` keyed by track id, and ``track_category_ids`` (track → LVIS category). Masks are native, so
unlike the box-only MammAlps adapter this emits the RLE directly and scores mask-HOTA (``eval.prefer_bbox`` off).

The adapter keeps only the **animal** categories — those present in the packaged ``burst_taxonomy.csv``, a
WordNet-derived whitelist of the 41 animal LVIS classes in BURST val — prompts SAM 3 with each class name, and
caps sequences per category so inference stays tractable (``dog``/``person``-style dominance). BURST carries no
site or timestamp metadata, so the cell/location key is the per-video ``seq_name`` and time is empty; only
leave-species-out is meaningful (set ``cv.group_schemes = ("species",)``), which is exactly the many-species
detection↔novelty test MammAlps was too small to power. Extra per-video fields ``source_dataset`` and
``source_seq`` are carried for the frame-extraction script; the loader ignores them.

All probes go to the ``test`` file; the reference (``train``) file is emitted empty (Split A draws
reference=probe from ``origins=("test",)``).

Run: ``PYTHONPATH=. python -m src.adapters.burst --config configs/burst.yaml``
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pycocotools import mask as coco_mask

from src.adapters.taxonomy import ClassTaxonomy, normalise_name
from src.config import Config
from src.dataset import _TAXONOMY_FIELDS

_PACKAGED_CSV = Path(__file__).parent / "burst_taxonomy.csv"


class BurstAdapter:
    """Emit ``burst_{train,test}_ext.json`` (test=capped animal probes; train=empty) in the SA-Co schema."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (loads the animal-class taxonomy whitelist)."""
        self.config = config or Config()
        csv_path = (
            self.config.paths.data_root / self.config.burst.taxonomy_csv
            if self.config.burst.taxonomy_csv
            else _PACKAGED_CSV
        )
        self.taxonomy = ClassTaxonomy(csv_path)
        self._animals = self.taxonomy.names()  # normalised animal class names

    def _load(self) -> dict:
        """Read the BURST ``all_classes.json`` (list-of-sequences dict)."""
        path = self.config.paths.data_root / self.config.data.annotations_subdir / self.config.burst.ann_file
        return json.loads(path.read_text())

    def _kept_frame_indices(self, seq: dict) -> list[int]:
        """Evenly-spaced annotated-frame indices, capped to ``max_frames_per_video``."""
        n = len(seq["annotated_image_paths"])
        cap = self.config.burst.max_frames_per_video
        if n <= cap:
            return list(range(n))
        return sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})

    def _rle(self, entry: dict, h: int, w: int) -> dict | None:
        """A BURST segmentation entry → a COCO-RLE ``{size, counts}`` dict (``None`` if empty)."""
        counts = entry.get("rle") if isinstance(entry, dict) else None
        if not counts:
            return None
        return {"size": [int(h), int(w)], "counts": counts}

    def _tracks_for_category(
        self, seq: dict, cat_id: int, kept: list[int], h: int, w: int
    ) -> list[dict]:
        """Per-track annotations (RLE + mask-derived bbox/area) for one category in one sequence."""
        track_ids = [str(t) for t, c in seq["track_category_ids"].items() if int(c) == cat_id]
        segs_by_frame = seq["segmentations"]
        annotations = []
        for tid in track_ids:
            segs: list[dict | None] = []
            bboxes: list[list[float] | None] = []
            areas: list[float] = []
            has_mask = False
            for idx in kept:
                rle = self._rle((segs_by_frame[idx] or {}).get(tid), h, w) if idx < len(segs_by_frame) else None
                if rle is not None:
                    segs.append(rle)
                    bboxes.append([float(v) for v in coco_mask.toBbox(rle).tolist()])
                    areas.append(float(coco_mask.area(rle)))
                    has_mask = True
                else:
                    segs.append(None)
                    bboxes.append(None)
                    areas.append(0.0)
            if has_mask:
                annotations.append({
                    "id": -1, "video_id": -1, "category_id": cat_id,
                    "height": h, "width": w, "iscrowd": 0,
                    "segmentations": segs, "bboxes": bboxes, "areas": areas,
                })
        return annotations

    def convert(self) -> dict:
        """Parse + cap the BURST animal sequences into one SA-Co ``_ext`` dict (the probe set)."""
        data = self._load()
        id_to_name = {int(c["id"]): c["name"] for c in data["categories"]}
        videos, annotations, pairs = [], [], []
        per_cat: Counter = Counter()  # sequences emitted per animal category (the cap)
        cap = self.config.burst.max_videos_per_category

        for seq in data["sequences"]:
            h, w = int(seq["height"]), int(seq["width"])
            present = {int(c) for c in seq["track_category_ids"].values()}
            animal_cats = [
                cid for cid in present
                if normalise_name(id_to_name.get(cid, "")) in self._animals and per_cat[cid] < cap
            ]
            if not animal_cats:
                continue
            kept = self._kept_frame_indices(seq)
            if len(kept) < self.config.burst.min_frames:
                continue

            video_name = f"{seq['dataset']}__{seq['seq_name']}"
            file_names = [f"{video_name}/{seq['annotated_image_paths'][i]}" for i in kept]
            emitted_here = []
            for cid in animal_cats:
                anns = self._tracks_for_category(seq, cid, kept, h, w)
                if not anns:
                    continue
                emitted_here.append((cid, anns))
            if not emitted_here:
                continue

            vid = len(videos)
            videos.append({
                "id": vid, "video_name": video_name, "file_names": file_names,
                "height": h, "width": w, "length": len(file_names),
                "video_num_frames": len(seq.get("all_image_paths", file_names)),
                "video_fps": float(seq.get("fps", 1) or 1),
                "video_creation_datetime": "",  # TAO videos carry no reliable date → temporal untestable
                "location_id": seq["seq_name"],  # per-video cell (BURST has no site metadata)
                "source_dataset": seq["dataset"], "source_seq": seq["seq_name"],  # for frame extraction
            })
            for cid, anns in emitted_here:
                per_cat[cid] += 1
                for a in anns:
                    a["id"] = len(annotations)
                    a["video_id"] = vid
                    a["noun_phrase"] = id_to_name[cid]
                    annotations.append(a)
                pairs.append({
                    "id": len(pairs), "video_id": vid, "category_id": cid,
                    "noun_phrase": id_to_name[cid], "num_masklets": len(anns),
                })
        return {
            "info": {"description": "BURST val (animal subset) -> SA-Co (mask-native)"},
            "videos": videos, "annotations": annotations,
            "categories": self._categories({p["category_id"] for p in pairs}, id_to_name),
            "video_np_pairs": pairs,
        }

    def _categories(self, ids: set[int], id_to_name: dict[int, str]) -> list[dict]:
        """SA-Co category entries for the present animal classes (id, name, resolvable taxonomy levels)."""
        out = []
        for cid in sorted(ids):
            name = id_to_name[cid]
            tax = self.taxonomy.taxonomy_of(name)
            entry: dict = {"id": cid, "name": name}
            entry.update({fld: tax.get(fld) for fld in _TAXONOMY_FIELDS})
            out.append(entry)
        return out

    def run(self) -> tuple[Path, Path]:
        """Write ``data.test_ann`` (animal probes) + an empty ``data.train_ann``; return the two paths."""
        out_dir = self.config.paths.data_root / self.config.data.annotations_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = self.convert()
        test_path = out_dir / self.config.data.test_ann
        test_path.write_text(json.dumps(probe))
        empty = {"info": {"description": "BURST reference (empty; Split A uses origins=test)"},
                 "videos": [], "annotations": [], "categories": probe["categories"], "video_np_pairs": []}
        train_path = out_dir / self.config.data.train_ann
        train_path.write_text(json.dumps(empty))
        n_species = len(probe["categories"])
        print(f"probe: {len(probe['videos'])} videos, {len(probe['annotations'])} tracks, "
              f"{n_species} animal species, {len(probe['video_np_pairs'])} probes -> {test_path}")
        return train_path, test_path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config (e.g. configs/burst.yaml)")
    args = ap.parse_args()
    BurstAdapter(Config.load(args.config)).run()


if __name__ == "__main__":
    main()
