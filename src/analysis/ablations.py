"""Ablations & robustness (T5.1, T5.2).

T5.1 — factor ablations. Goal: show what each distance contributes. Method: refit and re-validate
    leaving out one factor at a time. Output: ``outputs/ablations/factor_ablation.csv`` (R^2 /
    OOS-error change when each factor is dropped).
T5.2 — design & confound robustness. Goal: confirm findings aren't artifacts of a modelling choice.
    Method: DINOv2 vs CLIP; mask-cropped vs whole-frame; cell-level vs per-species aggregation;
    species-specific vs generic prompts; alternative distance definitions; with/without support
    covariate. Output: robustness section + tables.
Done when: the detection/association contrast holds (or its limits are documented) across variants.
Depends on: T3.2, T4.1.
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config


class Ablations:
    """Factor-drop ablation (T5.1) + design/confound robustness sweep (T5.2)."""

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
        raise NotImplementedError("T5.1: drop-one-factor refit + re-validate")

    def robustness_sweep(self) -> Path:
        """Re-run under alternative design choices (encoder, crop, aggregation, prompt, ...).

        Returns:
            Path to the robustness tables.
        """
        raise NotImplementedError("T5.2: design/confound robustness variants")
