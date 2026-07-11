"""Frozen SAM 3 inference harness.

Runs frozen (inference-only) SAM 3 promptable tracking over every test probe and writes predicted
masklets per ``(video, prompt)`` to ``outputs/predictions/`` in the evaluator's expected format.
Runs both a species-specific prompt (primary) and a generic ("animal") prompt, keeping
hard-negative queries.

Run: ``PYTHONPATH=. .venv/bin/python -m src.inference.harness [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config


class InferenceHarness:
    """Run frozen SAM 3 promptable tracking over the test split."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (loads frozen SAM 3 weights).

        Args:
            config: Project config (``inference.*``, ``paths.*``).
        """
        self.config = config or Config()

    def run(self) -> Path:
        """Predict masklets for every test ``(video, prompt)`` and write ``outputs/predictions/``.

        Returns:
            The predictions directory.
        """
        raise NotImplementedError(
            "frozen SAM 3 promptable tracking (species + generic prompts)"
        )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    InferenceHarness(Config.load(args.config)).run()


if __name__ == "__main__":
    main()
