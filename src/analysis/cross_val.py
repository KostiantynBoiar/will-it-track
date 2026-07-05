"""Grouped cross-validation (T4.1).

Goal: turn a fitted curve into a PREDICTOR — predict held-out species/places from distance alone.
Input: modelling table, fitting code from T3.1.
Output: ``outputs/validation/cv_results.parquet`` (predicted vs actual per held-out group; MAE/RMSE;
    calibration).
Method: leave-one-species-out and leave-one-location-out (WHOLE groups held out); predict
    ``pDetA``/``pAssA`` from distances only.
Done when: out-of-sample errors reported for both schemes; leakage-free grouping asserted.
    (Phase-4 decision gate — if no signal, invoke H0 and pivot to representational probing.)
Depends on: T3.1.

Run: ``PYTHONPATH=. .venv/bin/python -m src.analysis.cross_val [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config


class GroupedCV:
    """Leave-one-group-out CV (whole species / whole locations) for both targets."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``cv.group_schemes``).
        """
        self.config = config or Config()

    def run(self, table_path: Path) -> Path:
        """Run grouped CV and write ``outputs/validation/cv_results.parquet``.

        Args:
            table_path: Merged scores-x-features parquet.

        Returns:
            Path to the CV results parquet.
        """
        raise NotImplementedError("T4.1: leave-species-out + leave-location-out; assert no leakage")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    GroupedCV(cfg).run(cfg.paths.outputs_root / "features.parquet")


if __name__ == "__main__":
    main()
