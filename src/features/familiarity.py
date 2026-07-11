"""SAM 3 familiarity proxy.

Hedges against the unknown true pretraining set by measuring representational familiarity directly:
how separable/typical each test species is in SAM 3's own feature space (e.g. distance to the nearest
seen-species cluster in SAM 3 embeddings), computed per test species.
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
        raise NotImplementedError("separability in SAM 3 feature space")
