"""Ablations & robustness.

Factor ablations show what each distance contributes by refitting and re-validating with one factor
dropped at a time (pseudo-R^2 + out-of-sample gain), plus the ``+size`` and ``-support`` design variants —
written to ``outputs/ablations/factor_ablation.csv``. The variance partition (LMG/Shapley unique R^2 + VIF,
from :mod:`src.analysis.variance`) is written alongside. Together they show the null is not an artifact of a
single factor or covariate: no distance carries unique variance, and only animal size moves the score.

The heavier design robustness (DINOv2 vs CLIP, mask-crop vs whole-frame, species vs generic prompt) needs
fresh GPU inference/embedding passes and is run separately (:meth:`robustness_sweep`).

Run: ``PYTHONPATH=. python -m src.analysis.ablations [--config configs/default.yaml]``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import regression as R
from src.analysis.cross_val import _SCHEME_COLUMN, _summarise, oos_predictions
from src.analysis.regression import DISTANCE_COLS, TARGETS, DesignBuilder, _num, _pseudo_r2, fit_glm
from src.analysis.variance import VariancePartition
from src.config import Config
from src.io import read_parquet


def _fit_validate(
    df: pd.DataFrame, config: Config, distance_cols: tuple[str, ...], control_size: bool,
    log_support: bool, target: str,
) -> tuple[float, dict[str, tuple[float, float]]]:
    """Pseudo-R^2 + per-scheme OOS gain for an isolated design (predictor set pinned via the module globals)."""
    saved = (R.DISTANCE_COLS, R.CONFIDENCE_COLS, config.model.control_size,
             config.model.log_support_covariate)
    R.DISTANCE_COLS, R.CONFIDENCE_COLS = distance_cols, ()
    config.model.control_size, config.model.log_support_covariate = control_size, log_support
    try:
        rows = df[_num(df[target]).notna()]
        builder = DesignBuilder(config).fit(rows)
        r2 = _pseudo_r2(fit_glm(rows, target, builder, config))
        frames = [p for s, c in _SCHEME_COLUMN.items() if c in df.columns
                  for p in [oos_predictions(df, target, c, config)] if not p.empty]
        summ = _summarise(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), config)
        det = summ[summ["target"] == target] if not summ.empty else summ
        gains = {r.group_scheme: (round(float(r.delta), 4), round(float(r.p_value), 3))
                 for r in det.itertuples()}
        return round(r2, 4), gains
    finally:
        (R.DISTANCE_COLS, R.CONFIDENCE_COLS, config.model.control_size,
         config.model.log_support_covariate) = saved


class Ablations:
    """Factor-drop ablation + variance partition (+ the deferred design/confound robustness sweep)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (the robustness variants are config toggles).
        """
        self.config = config or Config()
        self._outdir = self.config.paths.outputs_root / "ablations"

    def _table(self) -> Path:
        outputs = self.config.paths.outputs_root
        p = outputs / "features_diff.parquet"  # has log_area for the +size design
        return p if p.exists() else outputs / "features.parquet"

    def factor_ablation(self, target: str = "pDetA") -> Path:
        """Refit/re-validate dropping each factor (+ ``+size`` / ``-support``); write the ablation CSV.

        Returns:
            Path to ``outputs/ablations/factor_ablation.csv``.
        """
        df = read_parquet(self._table())
        designs = [
            ("full (4 distances)", DISTANCE_COLS, False, True),
            *[(f"drop {c.split('_')[0]}", tuple(x for x in DISTANCE_COLS if x != c), False, True)
              for c in DISTANCE_COLS],
            ("$+$ animal size", DISTANCE_COLS, True, True),
            ("$-$ support covariate", DISTANCE_COLS, False, False),
        ]
        rows = []
        for label, dcols, csize, support in designs:
            r2, gains = _fit_validate(df, self.config, dcols, csize, support, target)
            rows.append({
                "design": label, "target": target, "pseudo_r2": r2,
                "oos_species_delta": gains.get("species", (float("nan"), float("nan")))[0],
                "oos_species_p": gains.get("species", (float("nan"), float("nan")))[1],
                "oos_location_delta": gains.get("location", (float("nan"), float("nan")))[0],
                "oos_location_p": gains.get("location", (float("nan"), float("nan")))[1],
            })
        self._outdir.mkdir(parents=True, exist_ok=True)
        path = self._outdir / "factor_ablation.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"factor ablation ({target}) -> {path}")
        return path

    def variance_partition(self) -> Path:
        """LMG/Shapley unique R^2 + VIF per distance, for both targets; write the CSV."""
        vp = VariancePartition(self.config)
        frames = []
        for target in TARGETS:
            t = vp.partition(target)
            t.insert(0, "target", target)
            frames.append(t)
        self._outdir.mkdir(parents=True, exist_ok=True)
        path = self._outdir / "variance_partition.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"variance partition -> {path}")
        return path

    def robustness_sweep(self) -> Path:
        """Re-run under alternative design choices (encoder, crop, prompt) — needs fresh GPU passes."""
        raise NotImplementedError("design/confound robustness variants (encoder/crop/prompt) — run on GPU")


def main() -> None:
    """CLI entry point — factor ablation (both targets) + variance partition."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    abl = Ablations(Config.load(args.config))
    abl.factor_ablation("pDetA")  # detection is the focus; pAssA tracks it (near-degenerate)
    abl.variance_partition()


if __name__ == "__main__":
    main()
