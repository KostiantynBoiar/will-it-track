"""SA-FARI loader tests (T0.1) — records parse + RLE masks decode via pycocotools.

Skips until the minimal slice is fetched (`python -m src.acquire --annotations`).
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.dataset import SAFARI

_CFG = Config()
_ANN_PRESENT = SAFARI("test", _CFG).ann_path.exists()
pytestmark = pytest.mark.skipif(not _ANN_PRESENT, reason="SA-FARI annotations not fetched (T0.1)")


def test_records_parse() -> None:
    """Every record has an id + frame paths; hard-negative flags are booleans."""
    records = SAFARI("test", _CFG).records()
    assert len(records) > 0
    assert all(r.file_names for r in records)
    assert all(isinstance(r.is_hard_negative, bool) for r in records)


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
