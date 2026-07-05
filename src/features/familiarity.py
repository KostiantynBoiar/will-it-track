"""SAM 3 familiarity proxy (T2.5).

Goal: hedge against the unknown true pretraining set by measuring representational familiarity
    directly (SAM 3's true corpus is unknown; distances are scoped to the SA-FARI train split).
Input: SAM 3 features on test-species frames.
Output: ``familiarity_proxy`` per test species.
Method: measure how separable/typical the test species is in SAM 3's own feature space (e.g. distance
    to the nearest seen-species cluster in SAM 3 embeddings).
Done when: computed for all test species and correlates sensibly (loosely) with taxonomic/visual
    distance.
Depends on: T0.2.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


class FamiliarityProxy:
    """Separability of each test species in SAM 3's own feature space."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (SAM 3 features).
        """
        self.config = config or Config()

    def compute(self) -> pd.Series:
        """Return ``familiarity_proxy`` per test species.

        Returns:
            A Series indexed by species.
        """
        raise NotImplementedError("T2.5: separability in SAM 3 feature space")
