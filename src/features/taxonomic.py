"""Taxonomic distance (T2.1).

Goal: how far a probe's species sits from the nearest *reference* species on the tree of life.
Input: an analysis :class:`~src.splits.Partition` + per-``category_id`` taxonomy.
Output: ``taxonomic_distance`` keyed by ``category_id`` (probe species).
Method: LCA tree steps to the nearest reference species (shares Genus = 1, Family = 2, Order = 3, …).
    On Split A (``loso=True``) each probe species excludes itself from the reference. Species without a
    full 7-level taxonomy get ``NaN`` (they carry only visual/environment novelty).
Done when: distances vary (on SA-FARI's LOSO, 0–4 — the 0 is the ``pig``/``wild boar`` taxonomy twin).
Depends on: T0.3 (the split defines reference vs probe species).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.config import Config
from src.dataset import _TAXONOMY_FIELDS, SAFARI

if TYPE_CHECKING:
    from src.splits import Partition

_N_LEVELS = len(_TAXONOMY_FIELDS)


def tree_distance(a: list[str], b: list[str]) -> int:
    """Lowest-common-ancestor tree-step distance between two taxonomy paths.

    The distance is ``len(path) - shared`` where ``shared`` is the number of leading levels that are
    equal and non-empty (the shared root-to-LCA prefix); the walk stops at the first mismatch or empty
    level. For 7-level paths: same genus -> 1, same family -> 2, same order -> 3, fully disjoint -> 7.

    Args:
        a: Ordered taxonomy values (kingdom -> species) for one species.
        b: Ordered taxonomy values for the other species (same level order).

    Returns:
        The number of tree steps separating the two species.
    """
    shared = 0
    for x, y in zip(a, b, strict=False):
        if x and x == y:
            shared += 1
        else:
            break
    return len(a) - shared


class TaxonomicDistance:
    """LCA tree-step distance from each probe species to the nearest reference species."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.taxonomic_levels`` order is fixed by the schema).
        """
        self.config = config or Config()

    def _full_paths(self) -> dict[str, list[str]]:
        """``category_id`` → kingdom→species path, for full-7-level-taxonomy categories only."""
        taxonomy = SAFARI("test", self.config).taxonomy()  # identical vocab across splits
        return {
            cid: [tax[field.lower()] for field in _TAXONOMY_FIELDS]
            for cid, tax in taxonomy.items()
            if len(tax) == _N_LEVELS
        }

    def compute(self, partition: Partition) -> pd.Series:
        """Return ``taxonomic_distance`` per probe species (``category_id``).

        Args:
            partition: The active split (its ``loso`` flag drives self-exclusion).

        Returns:
            A Series indexed by probe ``category_id``; ``NaN`` for species without full taxonomy.
        """
        paths = self._full_paths()
        reference = [cid for cid in partition.reference_species if cid in paths]
        rows: dict[str, float] = {}
        for cid in partition.probe_species:
            if cid not in paths:
                rows[cid] = float("nan")
                continue
            others = [paths[c] for c in reference if not (partition.loso and c == cid)]
            rows[cid] = (
                float(min(tree_distance(paths[cid], p) for p in others)) if others else float("nan")
            )
        return pd.Series(rows, name="taxonomic_distance", dtype="float64")
