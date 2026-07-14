"""Per-target regression.

Models ``pDetA`` and ``pAssA`` separately on the four distances using a support-weighted
logit-link GLM on bounded scores (the "beta / logit-link GLM, weighted by support" of CLAUDE.md:
a fractional-logit ``Binomial`` GLM whose endog is a proportion in ``[0, 1]`` and whose
``var_weights`` are the per-cell support). A ``log(n_frames)`` covariate is added so rare is not
mistaken for far. Writes the fitted model + a standardised coefficient table to ``outputs/models/``.

The :class:`DesignBuilder` (fit-on-train, transform-any) and :func:`fit_glm` here are the shared
estimation core reused by :mod:`src.analysis.cross_val`, :mod:`src.analysis.variance`, and
:mod:`src.analysis.uncertainty` so every stage standardises features identically and leakage-free.

Run: ``PYTHONPATH=. .venv/bin/python -m src.analysis.regression [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import Config
from src.io import read_parquet

# The four label-free distances (the factors of interest) + the scene covariates.
DISTANCE_COLS = ("taxonomic_distance", "temporal_gap", "visual_distance", "environment_distance")
_CONT_COVARIATES = ("clutter",)
_BINARY_COVARIATES = ("is_night_ir",)
TARGETS = ("pDetA", "pAssA")


def _num(series: pd.Series) -> pd.Series:
    """Coerce to float, non-numeric → NaN."""
    return pd.to_numeric(series, errors="coerce")


class DesignBuilder:
    """Standardising design-matrix builder: ``fit`` on a training frame, ``transform`` any frame.

    Continuous predictors (distances, clutter, ``log(support)``) are z-scored using means/SDs learned
    at ``fit`` time only — so a cross-validation fold never sees held-out statistics. Missing values are
    mean-imputed (to the training mean). Binary covariates pass through as ``0/1``. An intercept is added.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize."""
        self.config = config or Config()
        self.cont_: list[str] = []
        self.binary_: list[str] = []
        self.stats_: dict[str, tuple[float, float]] = {}
        self.support_col_: str | None = None

    def fit(self, df: pd.DataFrame) -> DesignBuilder:
        """Learn standardisation statistics from ``df`` (a training frame)."""
        continuous = [c for c in (*DISTANCE_COLS, *_CONT_COVARIATES) if c in df.columns]
        self.cont_ = []
        self.stats_ = {}
        for col in continuous:
            values = _num(df[col])
            if values.notna().any():
                self.stats_[col] = (float(values.mean()), float(values.std(ddof=0)) or 1.0)
                self.cont_.append(col)
        model = self.config.model
        self.support_col_ = (
            model.support_col
            if model.log_support_covariate and model.support_col in df.columns
            else None
        )
        if self.support_col_:
            values = np.log1p(_num(df[self.support_col_]))
            self.stats_["log_support"] = (float(values.mean()), float(values.std(ddof=0)) or 1.0)
        self.binary_ = [c for c in _BINARY_COVARIATES if c in df.columns]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the (standardised, intercepted) design matrix for ``df``."""
        design = pd.DataFrame(index=df.index)
        for col in self.cont_:
            mean, sd = self.stats_[col]
            design[col] = (_num(df[col]).fillna(mean) - mean) / sd
        if self.support_col_:
            mean, sd = self.stats_["log_support"]
            design["log_support"] = (np.log1p(_num(df[self.support_col_])).fillna(mean) - mean) / sd
        for col in self.binary_:
            design[col] = _num(df[col]).fillna(0.0).astype(float)
        return sm.add_constant(design, has_constant="add")

    @property
    def feature_names(self) -> list[str]:
        """Predictor names in design order (excluding the intercept)."""
        support = ["log_support"] if self.support_col_ else []
        return [*self.cont_, *support, *self.binary_]


def _weights(df: pd.DataFrame, config: Config) -> np.ndarray | None:
    """Per-cell support weights (``var_weights``), or ``None`` when weighting is off."""
    if not config.model.support_weight or config.model.support_col not in df.columns:
        return None
    return _num(df[config.model.support_col]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)


def fit_glm(df: pd.DataFrame, target: str, builder: DesignBuilder, config: Config):  # noqa: ANN201
    """Fit the support-weighted fractional-logit GLM for ``target`` on ``builder``'s design.

    ``builder`` must already be ``fit`` (on the training rows). Rows with a missing target are dropped.
    Returns the fitted statsmodels ``GLMResults``.
    """
    rows = df[_num(df[target]).notna()]
    design = builder.transform(rows)
    endog = _num(rows[target]).clip(0.0, 1.0)
    model = sm.GLM(endog, design, family=sm.families.Binomial(), var_weights=_weights(rows, config))
    return model.fit()


def _pseudo_r2(result) -> float:  # noqa: ANN001
    """McFadden-style pseudo-R^2 (``1 - deviance/null_deviance``); ``0`` if the null is degenerate."""
    null = float(result.null_deviance)
    return 1.0 - float(result.deviance) / null if null > 0 else 0.0


class TargetRegression:
    """Fit a support-weighted logit-link GLM for one bounded target (pDetA or pAssA)."""

    def __init__(self, target: str, config: Config | None = None) -> None:
        """Initialize.

        Args:
            target: ``"pDetA"`` or ``"pAssA"``.
            config: Project config (``model.*``).
        """
        self.target = target
        self.config = config or Config()

    def fit(self, table_path: Path) -> Path:
        """Fit the model and write ``outputs/models/<target>_beta.pkl`` (+ a coefficient CSV).

        Args:
            table_path: Merged scores-x-features parquet.

        Returns:
            Path to the pickled fitted model.
        """
        df = read_parquet(table_path)
        rows = df[_num(df[self.target]).notna()].copy()
        builder = DesignBuilder(self.config).fit(rows)
        result = fit_glm(rows, self.target, builder, self.config)

        models_dir = self.config.paths.outputs_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        pkl = models_dir / f"{self.target}_beta.pkl"
        result.save(str(pkl))

        ci = result.conf_int()
        coef = pd.DataFrame(
            {
                "coef": result.params,
                "std_err": result.bse,
                "z": result.tvalues,
                "p_value": result.pvalues,
                "ci_lo": ci[0],
                "ci_hi": ci[1],
            }
        )
        coef.to_csv(models_dir / f"{self.target}_coef.csv")
        print(f"{self.target}: n={int(result.nobs)}  pseudo-R2={_pseudo_r2(result):.3f}  ->  {pkl}")
        return pkl


def main() -> None:
    """CLI entry point — fit both targets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    for target in TARGETS:
        TargetRegression(target, cfg).fit(cfg.paths.outputs_root / "features.parquet")


if __name__ == "__main__":
    main()
