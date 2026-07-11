"""Score and decompose with the OFFICIAL VEval evaluator.

Produces the dependent variables — per-cell ``pDetA``, ``pAssA``, ``pHOTA``. Runs the official
evaluator (pHOTA is never re-implemented) over ``outputs/predictions/`` and the test annotations,
then aggregates to one row per ``(species, location_id, time)`` cell (and per video) in
``outputs/scores.parquet``, carrying support counts, taxonomy, location, datetime, and prompt
condition.

Run: ``PYTHONPATH=. .venv/bin/python -m src.eval.score [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config


class Scorer:
    """Dispatch to the vendored official VEval scorer and aggregate to cells."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``eval.veval_module``, ``paths.*``).
        """
        self.config = config or Config()

    def score(self, predictions_dir: Path, annotations: Path) -> Path:
        """Score predictions and write ``outputs/scores.parquet``.

        Args:
            predictions_dir: ``outputs/predictions/`` from the inference harness.
            annotations: Test-split ``_ext`` annotation JSON.

        Returns:
            Path to the written ``scores.parquet``.
        """
        raise NotImplementedError("dispatch to official VEval; aggregate to cells")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    Scorer(cfg).score(cfg.paths.outputs_root / "predictions", cfg.paths.data_root / "annotations")


if __name__ == "__main__":
    main()
