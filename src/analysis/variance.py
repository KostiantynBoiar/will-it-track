"""Variance partitioning & decomposition test.

Attributes variance to each factor and tests the headline claim (detection <- species novelty;
association <- environment) via dominance / commonality (Shapley) analysis. Reports VIF for
collinearity and contrasts standardised coefficients across the two targets.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


class VariancePartition:
    """Dominance/commonality analysis + VIF + the standardised-coefficient contrast."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config.
        """
        self.config = config or Config()

    def partition(self, target: str) -> pd.DataFrame:
        """Return the unique + shared R^2 per factor for one target.

        Args:
            target: ``"pDetA"`` or ``"pAssA"``.

        Returns:
            The variance-partition table.
        """
        raise NotImplementedError("dominance/Shapley partition + VIF")
