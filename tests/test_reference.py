"""Reference tests — location-disjoint Split B, honest species-overlap report, and manifest.

Skips until the annotations are fetched (`python -m src.acquire --annotations`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.dataset import SAFARI, _year
from src.reference import Reference
from src.splits import build_location_partition, probe_records

_CFG = Config()
_ANN_PRESENT = SAFARI("train", _CFG).ann_path.exists() and SAFARI("test", _CFG).ann_path.exists()
pytestmark = pytest.mark.skipif(not _ANN_PRESENT, reason="SA-FARI annotations not fetched")


def test_location_holdout_disjoint_by_location_not_species() -> None:
    """Split B holds out locations (disjoint) while species are shared (reported, not asserted)."""
    report = Reference(_CFG, build_location_partition(_CFG)).check_axis()
    assert report["held_axis"] == "location"
    assert report["location_overlap"] == 0  # test locations are unseen
    assert report["species_overlap"] > 0  # species recur across the split (expected)


def test_manifest_covers_every_probe_cell(tmp_path: Path) -> None:
    """The frozen Split B manifest lists exactly the distinct probe cells, each with its metadata."""
    cfg = Config()
    cfg.paths.reference_root = tmp_path  # don't write into the repo's data/reference
    part = build_location_partition(cfg)
    ref = Reference(cfg, part)
    ref.freeze()

    manifest = ref.load()["manifest"]
    assert manifest
    assert all(
        {"category_id", "species", "location_id", "time", "taxonomy", "n_videos"} <= set(e)
        for e in manifest
    )

    manifest_cells = {(e["category_id"], e["location_id"], e["time"]) for e in manifest}
    expected = {
        (r.category_id, r.location_id, _year(r.creation_datetime)) for r in probe_records(part, cfg)
    }
    assert manifest_cells == expected
