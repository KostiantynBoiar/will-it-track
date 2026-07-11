"""Ablations & robustness.

Factor ablations show what each distance contributes by refitting and re-validating with one
factor dropped at a time (R^2 / OOS-error change), written to
``outputs/ablations/factor_ablation.csv``. The robustness sweep confirms findings are not
artifacts of a modelling choice by re-running under alternative designs: DINOv2 vs CLIP,
mask-cropped vs whole-frame, cell-level vs per-species aggregation, species-specific vs generic
prompts, alternative distance definitions, and with/without the support covariate.
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config


class Ablations:
    """Factor-drop ablation + design/confound robustness sweep."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (the robustness variants are config toggles).
        """
        self.config = config or Config()

    def factor_ablation(self) -> Path:
        """Refit/re-validate dropping each factor; write ``outputs/ablations/factor_ablation.csv``.

        Returns:
            Path to the ablation table.
        """
        raise NotImplementedError("drop-one-factor refit + re-validate")

    def robustness_sweep(self) -> Path:
        """Re-run under alternative design choices (encoder, crop, aggregation, prompt, ...).

        Returns:
            Path to the robustness tables.
        """
        raise NotImplementedError("design/confound robustness variants")
