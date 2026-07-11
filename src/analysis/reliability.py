"""Reliability estimator — the practical "will it work here?" tool.

The deployment output: takes four distances (+ proxy) for a new ``(species, place)`` and returns
predicted ``pDetA``/``pAssA`` with confidence intervals, by wrapping the fitted models with the
feature pipeline and a bootstrap interval. Exposes a minimal CLI/notebook demo.

Run: ``PYTHONPATH=. .venv/bin/python -m src.analysis.reliability --species ... --location ...``
"""

from __future__ import annotations

import argparse

from src.config import Config


class ReliabilityEstimator:
    """Predict pDetA/pAssA + CI for a new (species, place) from label-free distances."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (loads the fitted models).

        Args:
            config: Project config (``paths.outputs_root`` for the fitted models).
        """
        self.config = config or Config()

    def predict(self, distances: dict[str, float]) -> dict[str, tuple[float, float, float]]:
        """Predict both targets with confidence intervals.

        Args:
            distances: The four distances (+ proxy) for the query cell.

        Returns:
            ``{"pDetA": (point, lo, hi), "pAssA": (point, lo, hi)}``.
        """
        raise NotImplementedError("features -> fitted models -> prediction + bootstrap CI")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--species", required=True)
    ap.add_argument("--location", required=True)
    args = ap.parse_args()
    ReliabilityEstimator(Config.load(args.config)).predict({})


if __name__ == "__main__":
    main()
