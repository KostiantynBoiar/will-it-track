"""Grouped cross-validation.

Turns a fitted curve into a predictor: predicts held-out species/places from distance alone
using leave-one-species-out and leave-one-location-out schemes (whole groups held out). Reports
out-of-sample error and calibration to ``outputs/validation/cv_results.parquet`` and asserts
leakage-free grouping.

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
        raise NotImplementedError("leave-species-out + leave-location-out; assert no leakage")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    GroupedCV(cfg).run(cfg.paths.outputs_root / "features.parquet")


if __name__ == "__main__":
    main()
