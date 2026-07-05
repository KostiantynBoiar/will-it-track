"""Frozen SAM 3 inference harness (T1.1).

Goal: produce SAM 3's zero-shot predictions on every test probe.
Input: test annotations (video-noun-phrase pairs, incl. hard negatives), ``data/frames/``, SAM 3
    checkpoints.
Output: ``outputs/predictions/`` — predicted masklets per ``(video, prompt)`` in the evaluator's
    expected format.
Method: for each test video + query, run SAM 3 promptable tracking. Primary condition =
    species-specific prompt; also run generic ("animal"). Batch frames; checkpoint partial results;
    keep hard-negative queries. SAM 3 is FROZEN (inference-only, no gradients).
Done when: predictions exist for all test (video, prompt) pairs in both conditions; hard negatives
    are represented.

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
            "T1.1: frozen SAM 3 promptable tracking (species + generic prompts)"
        )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    InferenceHarness(Config.load(args.config)).run()


if __name__ == "__main__":
    main()
