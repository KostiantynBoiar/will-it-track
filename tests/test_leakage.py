"""Feature-leakage tests — the reference is frozen and a species never anchors on itself."""

from __future__ import annotations

import pytest

from src.config import Config
from src.dataset import SAFARI
from src.features.taxonomic import TaxonomicDistance
from src.splits import build_location_partition, build_species_partition, reference_records

_CFG = Config()
_ANN_PRESENT = SAFARI("train", _CFG).ann_path.exists() and SAFARI("test", _CFG).ann_path.exists()
_needs_ann = pytest.mark.skipif(not _ANN_PRESENT, reason="SA-FARI annotations not fetched")


@_needs_ann
def test_reference_records_by_split() -> None:
    """Split A draws reference from both origins (all present); Split B from train only."""
    a = build_species_partition(_CFG)
    ref_a = reference_records(a, _CFG)
    assert {r.origin for r in ref_a} == {"train", "test"}
    assert {r.category_id for r in ref_a} == set(a.reference_species)

    b = build_location_partition(_CFG)
    ref_b = reference_records(b, _CFG)
    assert {r.origin for r in ref_b} == {"train"}
    assert {r.location_id for r in ref_b if r.location_id != "nan"}.isdisjoint(b.probe_locations)


@_needs_ann
def test_loso_excludes_self() -> None:
    """Leave-one-species-out never lets a species be its own reference (else every distance is 0)."""
    part = build_species_partition(_CFG)
    loso = TaxonomicDistance(_CFG).compute(part).dropna()
    with_self = TaxonomicDistance(_CFG).compute(part.model_copy(update={"loso": False})).dropna()

    assert (with_self == 0).all()  # self-inclusion collapses every distance to 0
    assert (loso > 0).sum() > 0  # LOSO recovers the real (mostly non-zero) distances
    assert not loso.equals(with_self)


@_needs_ann
def test_swapping_reference_changes_distances() -> None:
    """Shrinking the reference set changes the distances (label-free, no probe leak)."""
    part = build_species_partition(_CFG)
    full = TaxonomicDistance(_CFG).compute(part).dropna()
    shrunk_part = part.model_copy(update={"reference_species": part.reference_species[::2]})
    shrunk = TaxonomicDistance(_CFG).compute(shrunk_part).dropna()

    common = full.index.intersection(shrunk.index)
    assert not full.loc[common].equals(shrunk.loc[common])


@pytest.mark.skip(reason="needs the assembled feature table")
def test_feature_table_has_no_unexplained_nulls() -> None:
    """outputs/features.parquet has the four distances + proxy + support with no stray nulls."""
    ...
