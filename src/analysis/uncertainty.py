"""Uncertainty & the predictive-line figure.

Produces bootstrap confidence intervals (for scores, coefficients, and out-of-sample error) by
resampling over cells/groups, and plots the held-out predicted-vs-actual predictive-line figures
to ``outputs/figures/predictive_line_{det,assoc}.png``.
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config


class Uncertainty:
    """Bootstrap confidence intervals + the predicted-vs-actual predictive-line figures."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``cv.n_bootstrap``, ``paths.outputs_root``).
        """
        self.config = config or Config()

    def predictive_line(self, target: str) -> Path:
        """Write ``outputs/figures/predictive_line_<target>.png`` with bootstrap error bars.

        Args:
            target: ``"det"`` or ``"assoc"``.

        Returns:
            Path to the written figure.
        """
        raise NotImplementedError("bootstrap CIs + predictive-line plot")
