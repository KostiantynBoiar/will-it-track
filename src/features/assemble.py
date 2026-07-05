"""Assemble the feature table (T2.6).

Goal: a single modelling table.
Input: T2.1-T2.5 outputs, support counts.
Output: ``outputs/features.parquet`` — one row per cell with the four distances + proxy + support.
Method: join on cell key; re-verify no test label leaks into any column (unit test: swapping the
    seen set changes distances as expected).
Done when: the leakage unit test passes; the table has no unexplained nulls.
Depends on: T2.1-T2.5.

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
        raise NotImplementedError("T2.6: join distances + proxy + support; re-verify no leakage")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    FeatureAssembler(Config.load(args.config)).assemble()


if __name__ == "__main__":
    main()
