"""Uncertainty & the predictive-line figure (T4.2).

Goal: honest error bars and the headline visual.
Input: ``cv_results.parquet``, per-cell scores.
Output: bootstrap CIs (scores, coefficients, OOS error); ``outputs/figures/predictive_line_{det,assoc}.png``
    (predicted vs actual with error bars).
Method: bootstrap over cells/groups; plot held-out predictions.
Done when: figures reproduce from script; CIs reported everywhere a point estimate appears.
Depends on: T4.1.
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
        raise NotImplementedError("T4.2: bootstrap CIs + predictive-line plot")
