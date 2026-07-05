"""Variance partitioning & decomposition test (T3.2).

Goal: attribute variance to each factor and test the headline (detection <- species novelty;
    association <- environment).
Input: fitted models + design matrix.
Output: variance-partition table (unique + shared R^2 per factor, per target); VIF report;
    standardised-coefficient contrast figure.
Method: dominance / commonality (Shapley) analysis; VIF for collinearity; compare standardised
    coefficients across the two targets and quantify the contrast.
Done when: the decomposition claim is quantified (supported or not) with the collinearity caveat
    addressed.
Depends on: T3.1.
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
        raise NotImplementedError("T3.2: dominance/Shapley partition + VIF")
