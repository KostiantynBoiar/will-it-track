"""Per-target regression (T3.1).

Goal: model ``pDetA`` and ``pAssA`` SEPARATELY on the four distances.
Input: ``scores.parquet`` join ``features.parquet``.
Output: ``outputs/models/{pdeta,passa}_beta.pkl``, coefficient tables.
Method: beta regression / logit-link GLM (scores in [0,1]); WEIGHT by support; include
    ``log(n_frames)`` covariate so rare != far.
Done when: models converge; residuals inspected; coefficients + CIs reported for both targets.
Depends on: T1.2, T2.6.

Run: ``PYTHONPATH=. .venv/bin/python -m src.analysis.regression [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config


class TargetRegression:
    """Fit a support-weighted beta/logit-GLM for one bounded target (pDetA or pAssA)."""

    def __init__(self, target: str, config: Config | None = None) -> None:
        """Initialize.

        Args:
            target: ``"pDetA"`` or ``"pAssA"``.
            config: Project config (``model.*``).
        """
        self.target = target
        self.config = config or Config()

    def fit(self, table_path: Path) -> Path:
        """Fit the model and write ``outputs/models/<target>_beta.pkl``.

        Args:
            table_path: Merged scores-x-features parquet.

        Returns:
            Path to the pickled fitted model.
        """
        raise NotImplementedError(
            "T3.1: support-weighted beta/logit GLM with log(n_frames) covariate"
        )


def main() -> None:
    """CLI entry point — fit both targets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    for target in ("pDetA", "pAssA"):
        TargetRegression(target, cfg).fit(cfg.paths.outputs_root / "features.parquet")


if __name__ == "__main__":
    main()
