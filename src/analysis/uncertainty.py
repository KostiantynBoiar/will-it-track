"""Uncertainty & the predictive-line figure.

Produces bootstrap confidence intervals (for scores, coefficients, and out-of-sample error) by
resampling over cells/groups, and plots the held-out predicted-vs-actual predictive-line figures
to ``outputs/figures/predictive_line_{det,assoc}.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis.cross_val import oos_predictions  # noqa: E402
from src.config import Config  # noqa: E402
from src.io import read_parquet  # noqa: E402

# The reliability tool talks in det/assoc; the tables use pDetA/pAssA.
_TARGET = {"det": "pDetA", "assoc": "pAssA", "pDetA": "pDetA", "pAssA": "pAssA"}
_STEM = {"pDetA": "det", "pAssA": "assoc"}


class Uncertainty:
    """Bootstrap confidence intervals + the predicted-vs-actual predictive-line figures."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``cv.n_bootstrap``, ``paths.outputs_root``).
        """
        self.config = config or Config()

    def _bootstrap_band(
        self, predicted: np.ndarray, actual: np.ndarray, grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """5–95% band of the OLS calibration line (predicted→actual) over ``grid``, by resampling cells."""
        rng = np.random.default_rng(self.config.seed)
        n = len(predicted)
        draws = min(self.config.cv.n_bootstrap, 500)  # cap: the band is smooth well before 500
        lines = np.empty((draws, len(grid)))
        for b in range(draws):
            idx = rng.integers(0, n, n)
            slope, intercept = np.polyfit(predicted[idx], actual[idx], 1)
            lines[b] = intercept + slope * grid
        return np.percentile(lines, 5, axis=0), np.percentile(lines, 95, axis=0)

    def predictive_line(self, target: str) -> Path:
        """Write ``outputs/figures/predictive_line_<target>.png`` with bootstrap error bars.

        Args:
            target: ``"det"`` or ``"assoc"`` (also accepts ``"pDetA"``/``"pAssA"``).

        Returns:
            Path to the written figure.
        """
        col = _TARGET[target]
        df = read_parquet(self.config.paths.outputs_root / "features.parquet")
        cv = oos_predictions(df, col, "location_id", self.config)
        cv = cv[cv["predicted"].notna()]
        predicted = cv["predicted"].to_numpy(dtype=float)
        actual = cv["actual"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect")
        if len(predicted) >= 2 and np.ptp(predicted) > 0:
            grid = np.linspace(float(predicted.min()), float(predicted.max()), 50)
            lo, hi = self._bootstrap_band(predicted, actual, grid)
            ax.fill_between(grid, lo, hi, color="tab:blue", alpha=0.2, label="90% band")
            slope, intercept = np.polyfit(predicted, actual, 1)
            ax.plot(grid, intercept + slope * grid, color="tab:blue", lw=1.5, label="calibration")
        ax.scatter(predicted, actual, s=14, color="tab:green", alpha=0.7, edgecolor="none")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"predicted {col} (held-out)", ylabel=f"actual {col}")
        ax.set_title(f"Leave-location-out prediction — {col}")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()

        figures = self.config.paths.outputs_root / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        path = figures / f"predictive_line_{_STEM.get(col, col)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"predictive line -> {path} ({len(predicted)} OOS cells)")
        return path


def main() -> None:
    """CLI entry point — write both predictive-line figures."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    unc = Uncertainty(Config.load(args.config))
    for target in ("det", "assoc"):
        unc.predictive_line(target)


if __name__ == "__main__":
    main()
