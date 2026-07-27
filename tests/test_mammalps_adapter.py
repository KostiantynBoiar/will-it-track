"""MammAlps → SA-Co adapter — hermetic (synthetic clip JSONs, no download/ffmpeg/GPU).

Verifies the box-only converter emits a schema the SA-FARI loader + RLE decoding accept: boxes become
decodable filled-rectangle masks AND carry native ``bboxes``, species map to a taxonomy, the 30fps clip is
subsampled, and cells are capped per (species, camera).
"""

from __future__ import annotations

import json

from pycocotools import mask as coco_mask

from src.adapters.mammalps import MammAlpsAdapter
from src.config import Config, DataConfig, MammAlpsConfig, PathsConfig
from src.dataset import SAFARI


def _clip(file_path: str, species: str, box_xyxy, n_frames: int = 30) -> dict:
    """A synthetic MammAlps clip JSON: one track of ``species`` present on the first few frames."""
    frames = []
    for fid in range(n_frames):
        det = [{"track_id": 1, "bbox": box_xyxy, "attributes": {"species": species}}] if fid < 12 else []
        frames.append({"frame_id": fid, "detections": det})
    site, cam = file_path.split("/")[:2]
    return {
        "info": {"site_id": site, "cam_id": cam, "fps": 30.0, "num_frames": n_frames,
                 "resolution": "1920x1080", "file_path": file_path},
        "frames": frames,
    }


def _config(tmp_path) -> Config:
    """MammAlps config rooted at tmp_path with synthetic clips and a low per-cell cap."""
    dense = tmp_path / "dense"
    dense.mkdir(parents=True)
    # red_deer at S1_C1 x3 (cap=2 → keep 2); roe_deer at S1_C1 x1; fox at S2_C1 x1
    (dense / "a.json").write_text(json.dumps(_clip("S1/C1/S1_C1_E1_V1.mp4", "red_deer", [100, 200, 500, 900])))
    (dense / "b.json").write_text(json.dumps(_clip("S1/C1/S1_C1_E2_V2.mp4", "red_deer", [50, 60, 300, 400])))
    (dense / "c.json").write_text(json.dumps(_clip("S1/C1/S1_C1_E3_V3.mp4", "red_deer", [10, 10, 200, 300])))
    (dense / "d.json").write_text(json.dumps(_clip("S1/C1/S1_C1_E4_V4.mp4", "roe_deer", [20, 20, 120, 220])))
    (dense / "e.json").write_text(json.dumps(_clip("S2/C1/S2_C1_E5_V5.mp4", "fox", [30, 40, 90, 140])))
    return Config(
        paths=PathsConfig(data_root=tmp_path),
        data=DataConfig(train_ann="mammalps_train_ext.json", test_ann="mammalps_test_ext.json"),
        mammalps=MammAlpsConfig(raw_dir="dense", target_fps=3.0, max_frames_per_clip=120,
                                max_clips_per_cell=2, min_frames=2, location_by="site_cam"),
    )


def test_boxes_become_decodable_masks_with_native_bboxes(tmp_path) -> None:
    out = MammAlpsAdapter(_config(tmp_path)).convert()
    assert {"videos", "annotations", "categories", "video_np_pairs"} <= set(out)
    ann = out["annotations"][0]
    seg = next(s for s in ann["segmentations"] if s)
    m = coco_mask.decode(seg)
    assert m.shape == (1080, 1920) and m.sum() > 0            # rectangle RLE decodes
    bb = next(b for b in ann["bboxes"] if b)
    assert len(bb) == 4 and bb[2] > 0 and bb[3] > 0           # native box [x,y,w,h] carried
    # 30fps subsampled to 3fps (step 10) → detections on frames 0..11 → positions 0,10 kept
    assert sum(1 for s in ann["segmentations"] if s) >= 1


def test_cap_and_taxonomy_and_locations(tmp_path) -> None:
    out = MammAlpsAdapter(_config(tmp_path)).convert()
    names = {c["name"] for c in out["categories"]}
    assert names == {"red deer", "roe deer", "fox"}          # normalised species names
    reddeer = next(c for c in out["categories"] if c["name"] == "red deer")
    assert reddeer["Species"] == "Cervus elaphus" and reddeer["Order"] == "Artiodactyla"
    # cap=2 on (red_deer, S1_C1): only 2 of the 3 red_deer clips kept
    locs = {v["location_id"] for v in out["videos"]}
    assert "S1_C1" in locs and "S2_C1" in locs
    reddeer_clips = [v for v in out["videos"]
                     if any(a["video_id"] == v["id"] and a["category_id"] == reddeer["id"]
                            for a in out["annotations"])]
    assert len(reddeer_clips) == 2                            # capped


def test_output_loads_through_the_safari_loader(tmp_path) -> None:
    cfg = _config(tmp_path)
    MammAlpsAdapter(cfg).run()
    safari = SAFARI("test", cfg)
    records = safari.records()
    assert len(records) >= 3 and all(r.location_id.startswith("S") for r in records)
    tax = safari.taxonomy()
    assert any(len(v) == 7 for v in tax.values())            # deer/fox resolve to full taxonomy
    # a GT box decodes via mask_at
    ann = next(a for a in safari.annotations_for(str(records[0].video_id)))
    seg_idx = next(i for i, s in enumerate(ann["segmentations"]) if s)
    assert safari.mask_at(ann, seg_idx).sum() > 0
