"""Seen/unseen reference tests (T0.2) — species+location disjointness + manifest coverage.

Skips until the annotations are fetched (`python -m src.acquire --annotations`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.dataset import SAFARI
from src.reference import Reference
from src.types import Cell

_CFG = Config()
_ANN_PRESENT = SAFARI("train", _CFG).ann_path.exists() and SAFARI("test", _CFG).ann_path.exists()
pytestmark = pytest.mark.skipif(not _ANN_PRESENT, reason="SA-FARI annotations not fetched (T0.2)")


def test_test_split_disjoint_from_train() -> None:
    """No test species or location also appears in the train (seen) split."""
    Reference(_CFG).assert_disjoint()  # raises AssertionError on any overlap


def test_manifest_covers_every_test_cell(tmp_path: Path) -> None:
    """The frozen manifest lists exactly the distinct test cells, each with its metadata."""
    cfg = Config()
    cfg.paths.reference_root = tmp_path  # don't write into the repo's data/reference
    ref = Reference(cfg)
    ref.freeze()

    manifest = ref.load()["manifest"]
    assert manifest
    assert all(
        {"species", "location_id", "time", "taxonomy", "n_videos"} <= set(e) for e in manifest
    )

    manifest_cells = {
        Cell(species=e["species"], location_id=e["location_id"], time=e["time"]) for e in manifest
    }
    test = SAFARI("test", cfg)
    expected_cells = {test.cell_of(r) for r in test.records()}
    assert manifest_cells == expected_cells
