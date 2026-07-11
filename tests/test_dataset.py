"""SA-FARI loader tests — records parse and RLE masks decode via pycocotools.

Skips until the minimal slice is fetched (`python -m src.acquire --annotations`).
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.dataset import SAFARI

_CFG = Config()
_ANN_PRESENT = SAFARI("test", _CFG).ann_path.exists()
pytestmark = pytest.mark.skipif(not _ANN_PRESENT, reason="SA-FARI annotations not fetched")


def test_records_parse() -> None:
    """Every record has an id, frame paths, a category id, and a consistent hard-negative flag."""
    records = SAFARI("test", _CFG).records()
    assert len(records) > 0
    assert all(r.file_names for r in records)
    assert all(r.category_id for r in records)
    assert all(isinstance(r.num_masklets, int) for r in records)
    assert all(r.is_hard_negative == (r.num_masklets == 0) for r in records)


def test_present_species_from_positive_pairs() -> None:
    """The present set (positive probes) is resolved by category id and is non-trivial."""
    ds = SAFARI("test", _CFG)
    present = ds.present_category_ids()
    assert len(present) >= 80  # ~83 present species in the test split
    # every present category resolves to a canonical species name
    assert all(ds._species_name(cid) for cid in present)


def test_rle_decodes_to_2d_bool_mask() -> None:
    """A positive query's first annotated frame decodes to a 2-D boolean mask (pycocotools)."""
    ds = SAFARI("test", _CFG)
    record = next(r for r in ds.records() if not r.is_hard_negative)
    anns = ds.annotations_for(record.video_id)
    assert anns, "a positive record must have annotations"
    mask = ds.mask_at(anns[0], 0)
    assert mask.ndim == 2
    assert mask.dtype == bool
    assert mask.shape[0] > 0 and mask.shape[1] > 0
