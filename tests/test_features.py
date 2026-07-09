"""Label-free distance feature tests (T2.1 taxonomic, T2.4 temporal).

The pure LCA-distance logic is tested unconditionally; the end-to-end ``compute(partition)`` tests skip
until the annotations are fetched (`python -m src.acquire --annotations`).
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.dataset import SAFARI
from src.features.taxonomic import TaxonomicDistance, tree_distance
from src.features.temporal import TemporalGap
from src.splits import build_species_partition

_CFG = Config()
_ANN_PRESENT = SAFARI("train", _CFG).ann_path.exists() and SAFARI("test", _CFG).ann_path.exists()
_needs_ann = pytest.mark.skipif(not _ANN_PRESENT, reason="SA-FARI annotations not fetched (T0.2)")

_LEO = ["animalia", "chordata", "mammalia", "carnivora", "felidae", "panthera", "leo"]


def test_tree_distance() -> None:
    """LCA distance counts the tree steps to the shared ancestor."""
    same_genus = ["animalia", "chordata", "mammalia", "carnivora", "felidae", "panthera", "tigris"]
    same_family = ["animalia", "chordata", "mammalia", "carnivora", "felidae", "felis", "catus"]
    same_order = ["animalia", "chordata", "mammalia", "carnivora", "canidae", "canis", "lupus"]
    disjoint = ["plantae", "u", "v", "w", "x", "y", "z"]

    assert tree_distance(_LEO, _LEO) == 0
    assert tree_distance(_LEO, same_genus) == 1
    assert tree_distance(_LEO, same_family) == 2
    assert tree_distance(_LEO, same_order) == 3
    assert tree_distance(_LEO, disjoint) == len(_LEO)


def test_tree_distance_stops_at_empty_level() -> None:
    """An empty (missing) level counts as a mismatch and halts the shared prefix."""
    assert tree_distance(["a", "", "c"], ["a", "b", "c"]) == 2


@_needs_ann
def test_taxonomic_distances_split_a() -> None:
    """Leave-one-species-out distances: one row per present species, full-taxonomy ones in [0, n]."""
    n_levels = len(_CFG.features.taxonomic_levels)
    part = build_species_partition(_CFG)
    series = TaxonomicDistance(_CFG).compute(part)

    assert set(series.index) == set(part.probe_species)  # one row per present species
    valid = series.dropna()
    assert valid.between(0, n_levels).all()
    assert valid.size >= 70  # ~72 full-taxonomy species
    assert (valid == 0).sum() >= 1  # the pig/wild-boar taxonomy twin(s)
    assert (valid >= 3).sum() >= 10  # a real spread of clearly-novel species


@_needs_ann
def test_temporal_gaps_split_a() -> None:
    """Every probe cell gets a non-negative, non-null year gap, keyed by cell."""
    part = build_species_partition(_CFG)
    series = TemporalGap(_CFG).compute(part)

    assert not series.empty
    assert series.notna().all()
    assert (series >= 0).all()
    assert list(series.index.names) == ["category_id", "species", "location_id", "time"]
