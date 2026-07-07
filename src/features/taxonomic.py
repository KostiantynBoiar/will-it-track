"""Taxonomic distance (T2.1).

Goal: how far a test species sits from the nearest training species on the tree of life.
Input: the seen (train) split's taxonomy + the test species taxonomy (both label-free metadata).
Output: ``taxonomic_distance`` keyed by species.
Method: distance = tree steps to the nearest training species via lowest common ancestor
    (shares Genus = 1, Family = 2, Order = 3, ...). Minimum is >= genus-level since splits are disjoint.
Done when: every test species has a finite distance; spot-checks match intuition (a felid closer to
    another felid than to a bird).
Depends on: T0.2 (the train split is the frozen reference).
"""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.dataset import SAFARI


def tree_distance(a: list[str], b: list[str]) -> int:
    """Lowest-common-ancestor tree-step distance between two taxonomy paths.

    The distance is ``len(path) - shared`` where ``shared`` is the number of leading levels that
    are equal and non-empty (the shared root-to-LCA prefix); the walk stops at the first mismatch
    or empty level. For 7-level paths: same genus -> 1, same family -> 2, same order -> 3, fully
    disjoint -> 7.

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
    """LCA tree-step distance from each test species to the nearest seen species."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.taxonomic_levels``).
        """
        self.config = config or Config()

    def _path(self, taxonomy: dict[str, str]) -> list[str]:
        """Order a species' taxonomy dict into a level-aligned, lowercased path."""
        return [taxonomy.get(level, "").lower() for level in self.config.features.taxonomic_levels]

    def compute(self) -> pd.Series:
        """Return ``taxonomic_distance`` per test species.

        Returns:
            A Series indexed by test species; each value is the LCA tree-step distance to the
            nearest seen (train) species.
        """
        train_paths = [self._path(t) for t in SAFARI("train", self.config).taxonomy().values()]
        test_taxonomy = SAFARI("test", self.config).taxonomy()
        distances = {
            species: min(tree_distance(self._path(taxonomy), tp) for tp in train_paths)
            for species, taxonomy in test_taxonomy.items()
        }
        return pd.Series(distances, name="taxonomic_distance", dtype="int64")
