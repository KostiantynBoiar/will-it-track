"""BURST → SA-Co adapter — hermetic (synthetic ``all_classes.json``, no download/GPU).

Verifies the mask-native converter emits a schema the SA-FARI loader + RLE decoding accept: BURST COCO-RLE
masks round-trip, only animal categories are kept (``person`` dropped), species map to a taxonomy, sequences
are capped per category, and the per-video ``seq_name`` becomes the location key.
"""

from __future__ import annotations

import json

import numpy as np
from pycocotools import mask as coco_mask

from src.adapters.burst import BurstAdapter
from src.config import BurstConfig, Config, DataConfig, PathsConfig
from src.dataset import SAFARI


def _rle(h: int, w: int, box) -> str:
    """A real COCO-RLE counts string for a filled rectangle in an ``h×w`` mask."""
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return coco_mask.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _seq(seq_id: int, name: str, dataset: str, cat_id: int, h: int, w: int, box, n: int = 4) -> dict:
    """A synthetic BURST sequence: one track of ``cat_id`` masked on every annotated frame."""
    frames = [f"frame{i:04d}.jpg" for i in range(n)]
    segs = [{"1": {"rle": _rle(h, w, box)}} for _ in range(n)]
    return {
        "id": seq_id, "seq_name": name, "dataset": dataset, "width": w, "height": h, "fps": 30,
        "all_image_paths": frames, "annotated_image_paths": frames,
        "neg_category_ids": [], "not_exhaustive_category_ids": [],
        "segmentations": segs, "track_category_ids": {"1": cat_id},
    }


def _burst_json() -> dict:
    """Synthetic BURST val: 3 zebra + 1 dog + 1 person (non-animal) sequences."""
    cats = [{"id": 100, "name": "zebra"}, {"id": 200, "name": "dog"}, {"id": 805, "name": "person"}]
    seqs = [
        _seq(0, "zeb_a", "LaSOT", 100, 480, 640, [50, 60, 300, 400]),
        _seq(1, "zeb_b", "YFCC100M", 100, 480, 640, [10, 10, 200, 300]),
        _seq(2, "zeb_c", "LaSOT", 100, 480, 640, [20, 20, 120, 220]),
        _seq(3, "dog_a", "HACS", 200, 360, 640, [30, 40, 90, 140]),
        _seq(4, "per_a", "AVA", 805, 480, 640, [5, 5, 55, 105]),
    ]
    return {"sequences": seqs, "categories": cats, "split": "val"}


def _config(tmp_path) -> Config:
    """BURST config rooted at tmp_path with a synthetic all_classes.json and a low per-category cap."""
    ann = tmp_path / "annotations"
    ann.mkdir(parents=True)
    (ann / "val_all_classes.json").write_text(json.dumps(_burst_json()))
    return Config(
        paths=PathsConfig(data_root=tmp_path),
        data=DataConfig(train_ann="burst_train_ext.json", test_ann="burst_test_ext.json"),
        burst=BurstConfig(ann_file="val_all_classes.json", max_videos_per_category=2,
                          max_frames_per_video=20, min_frames=2),
    )


def test_masks_roundtrip_and_animal_filter(tmp_path) -> None:
    out = BurstAdapter(_config(tmp_path)).convert()
    assert {"videos", "annotations", "categories", "video_np_pairs"} <= set(out)
    names = {c["name"] for c in out["categories"]}
    assert names == {"zebra", "dog"}                          # person (non-animal) dropped
    ann = out["annotations"][0]
    seg = next(s for s in ann["segmentations"] if s)
    m = coco_mask.decode(seg)
    assert m.sum() > 0                                        # BURST RLE decodes
    bb = next(b for b in ann["bboxes"] if b)
    assert len(bb) == 4 and bb[2] > 0 and bb[3] > 0           # mask-derived box carried


def test_cap_taxonomy_and_per_video_location(tmp_path) -> None:
    out = BurstAdapter(_config(tmp_path)).convert()
    zebra = next(c for c in out["categories"] if c["name"] == "zebra")
    assert zebra["Species"] == "Equus quagga" and zebra["Order"] == "Perissodactyla"
    # cap=2 on zebra: only 2 of the 3 zebra sequences kept
    zebra_vids = [p["video_id"] for p in out["video_np_pairs"] if p["category_id"] == zebra["id"]]
    assert len(zebra_vids) == 2
    # location = per-video seq_name
    locs = {v["location_id"] for v in out["videos"]}
    assert {"zeb_a", "zeb_b", "dog_a"} <= locs and "zeb_c" not in locs


def test_output_loads_through_the_safari_loader(tmp_path) -> None:
    cfg = _config(tmp_path)
    BurstAdapter(cfg).run()
    safari = SAFARI("test", cfg)
    records = safari.records()
    assert len(records) >= 3
    tax = safari.taxonomy()
    assert any(len(v) == 7 for v in tax.values())             # zebra/dog resolve to full taxonomy
    ann = next(a for a in safari.annotations_for(str(records[0].video_id)))
    seg_idx = next(i for i, s in enumerate(ann["segmentations"]) if s)
    assert safari.mask_at(ann, seg_idx).sum() > 0
