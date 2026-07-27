"""BURST → SA-Co adapter — hermetic (synthetic BURST JSON, no download, no GPU).

Verifies the converter emits a schema the existing SA-FARI loader + RLE decoding accept: animal categories
kept (non-animals dropped), tracks → masklets with decodable per-frame RLE, coarse taxonomy honestly partial,
and BURST ``neg_category_ids`` surfaced as ``num_masklets=0`` hard negatives.
"""

from __future__ import annotations

import json

import numpy as np
from pycocotools import mask as coco_mask

from src.adapters.burst import BurstAdapter
from src.config import BurstConfig, Config, DataConfig, PathsConfig
from src.dataset import SAFARI


def _rle(h: int, w: int, box: tuple[int, int, int, int]) -> str:
    """A valid COCO-RLE counts string for a rectangular mask (BURST stores the counts as a string)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    y, x, bh, bw = box
    mask[y : y + bh, x : x + bw] = 1
    enc = coco_mask.encode(np.asfortranarray(mask))
    return enc["counts"].decode("ascii")


def _burst_json() -> dict:
    """A tiny BURST split: seqA (dog track + car track + a 'cat' hard negative), seqB (a bird track)."""
    return {
        "split": "train",
        "categories": [
            {"id": 62, "name": "dog", "synset": "dog.n.01"},
            {"id": 3, "name": "car", "synset": "car.n.01"},   # non-animal → dropped
            {"id": 17, "name": "cat", "synset": "cat.n.01"},
            {"id": 55, "name": "bird", "synset": "bird.n.01"},  # coarse taxonomy
        ],
        "sequences": [
            {
                "id": 0, "seq_name": "seqA", "dataset": "TAO", "fps": 1, "height": 40, "width": 50,
                "all_image_paths": ["seqA/0.jpg", "seqA/1.jpg"],
                "annotated_image_paths": ["seqA/0.jpg", "seqA/1.jpg"],
                "neg_category_ids": [17],  # cat known-absent → hard negative
                "track_category_ids": {"1": 62, "2": 3},
                "segmentations": [
                    {"1": {"rle": _rle(40, 50, (2, 3, 8, 9)), "bbox": [3, 2, 9, 8]},
                     "2": {"rle": _rle(40, 50, (0, 0, 5, 5)), "bbox": [0, 0, 5, 5]}},
                    {"1": {"rle": _rle(40, 50, (4, 5, 8, 9)), "bbox": [5, 4, 9, 8]}},
                ],
            },
            {
                "id": 1, "seq_name": "seqB", "dataset": "TAO", "fps": 1, "height": 30, "width": 30,
                "all_image_paths": ["seqB/0.jpg", "seqB/1.jpg", "seqB/2.jpg"],
                "annotated_image_paths": ["seqB/0.jpg", "seqB/1.jpg", "seqB/2.jpg"],
                "neg_category_ids": [],
                "track_category_ids": {"5": 55},
                "segmentations": [
                    {"5": {"rle": _rle(30, 30, (1, 1, 6, 6)), "bbox": [1, 1, 6, 6]}},
                    {},  # bird absent this frame → None segmentation
                    {"5": {"rle": _rle(30, 30, (2, 2, 6, 6)), "bbox": [2, 2, 6, 6]}},
                ],
            },
        ],
    }


def _config(tmp_path) -> Config:
    """A BURST config rooted at tmp_path, CSV-only taxonomy (no SA-FARI seed), one synthetic split file."""
    (tmp_path / "burst" / "raw").mkdir(parents=True)
    (tmp_path / "burst" / "raw" / "split.json").write_text(json.dumps(_burst_json()))
    (tmp_path / "burst" / "class_taxonomy.csv").write_text(
        "name,Kingdom,Phylum,Class,Order,Family,Genus,Species\n"
        "dog,Animalia,Chordata,Mammalia,Carnivora,Canidae,Canis,Canis lupus\n"
        "cat,Animalia,Chordata,Mammalia,Carnivora,Felidae,Felis,Felis catus\n"
        "bird,Animalia,Chordata,Aves,,,,\n"  # coarse → 3 real levels only
    )
    return Config(
        paths=PathsConfig(data_root=tmp_path),
        data=DataConfig(train_ann="burst_train_ext.json", test_ann="burst_test_ext.json"),
        burst=BurstConfig(
            seed_taxonomy_from_safari=False, taxonomy_csv="burst/class_taxonomy.csv",
            raw_dir="burst/raw", train_json="split.json", test_json="split.json", min_frames=2,
        ),
    )


def test_convert_keeps_animals_drops_nonanimals_and_maps_tracks(tmp_path) -> None:
    cfg = _config(tmp_path)
    out = BurstAdapter(cfg).convert_split(tmp_path / "burst" / "raw" / "split.json", "train")

    assert {"info", "videos", "annotations", "categories", "video_np_pairs"} <= set(out)
    names = {c["name"] for c in out["categories"]}
    assert names == {"dog", "cat", "bird"}  # car dropped (no taxonomy)
    # two masklets: the dog track and the bird track (car track dropped)
    assert len(out["annotations"]) == 2
    assert {a["category_id"] for a in out["annotations"]} == {62, 55}
    # capitalized taxonomy on categories; bird coarse (Order.. are None)
    dog = next(c for c in out["categories"] if c["name"] == "dog")
    bird = next(c for c in out["categories"] if c["name"] == "bird")
    assert dog["Kingdom"] == "Animalia" and dog["Species"] == "Canis lupus"
    assert bird["Class"] == "Aves" and bird["Order"] is None


def test_hard_negatives_and_present_probes(tmp_path) -> None:
    cfg = _config(tmp_path)
    out = BurstAdapter(cfg).convert_split(tmp_path / "burst" / "raw" / "split.json", "train")
    pairs = {(p["video_id"], p["category_id"]): p["num_masklets"] for p in out["video_np_pairs"]}
    assert pairs[(0, 62)] == 1        # dog present in seqA
    assert pairs[(0, 17)] == 0        # cat: neg_category_ids → hard negative
    assert pairs[(1, 55)] == 1        # bird present in seqB
    assert (0, 3) not in pairs        # car never a probe (non-animal)


def test_output_loads_through_the_safari_loader_and_rle_decodes(tmp_path) -> None:
    cfg = _config(tmp_path)
    BurstAdapter(cfg).run()  # writes data/annotations/burst_{train,test}_ext.json

    safari = SAFARI("train", cfg)
    records = safari.records()
    assert {r.category_id for r in records} == {"62", "17", "55"}
    assert any(r.is_hard_negative for r in records)  # the cat probe
    tax = safari.taxonomy()
    assert len(tax["62"]) == 7 and len(tax["55"]) == 3  # dog full, bird coarse
    # the dog masklet's first-frame RLE decodes to the expected mask
    dog_ann = next(a for a in safari.annotations_for("0") if str(a["category_id"]) == "62")
    mask = safari.mask_at(dog_ann, 0)
    assert mask.shape == (40, 50) and mask.sum() == 8 * 9
