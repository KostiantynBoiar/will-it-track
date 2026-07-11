"""Assemble the feature table.

Joins the four distances + familiarity proxy + support counts on the cell key into a single modelling
table written to ``outputs/features.parquet`` — one row per cell — re-verifying that no test label
leaks into any column.

Run: ``PYTHONPATH=. .venv/bin/python -m src.features.assemble [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config


class FeatureAssembler:
    """Join the four distances + proxy + support into one per-cell modelling table."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``paths.outputs_root``, ``features.*``).
        """
        self.config = config or Config()

    def assemble(self) -> Path:
        """Build and write ``outputs/features.parquet``.

        Returns:
            Path to the written ``features.parquet``.
        """
        raise NotImplementedError("join distances + proxy + support; re-verify no leakage")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    FeatureAssembler(Config.load(args.config)).assemble()


if __name__ == "__main__":
    main()
