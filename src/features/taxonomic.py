"""Taxonomic distance (T2.1).

Goal: how far a test species sits from the nearest training species on the tree of life.
Input: the frozen reference manifest (T0.2), test species taxonomy.
Output: ``taxonomic_distance`` keyed by species.
Method: distance = tree steps to the nearest training species via lowest common ancestor
    (shares Genus = 1, Family = 2, Order = 3, ...). Minimum is >= genus-level since splits are disjoint.
Done when: every test species has a finite distance; spot-checks match intuition (a felid closer to
    another felid than to a bird).
Depends on: T0.2.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


class TaxonomicDistance:
    """LCA tree-step distance from each test species to the nearest seen species."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.taxonomic_levels``, ``reference`` manifest).
        """
        self.config = config or Config()

    def compute(self) -> pd.Series:
        """Return ``taxonomic_distance`` per test species.

        Returns:
            A Series indexed by species.
        """
        raise NotImplementedError("T2.1: LCA tree-step distance to nearest seen species")
