"""Variance partitioning & decomposition test.

Attributes variance to each factor and tests the headline claim (detection <- species novelty;
association <- environment) via dominance / commonality (Shapley) analysis. Reports VIF for
collinearity and contrasts standardised coefficients across the two targets.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.analysis.regression import DISTANCE_COLS, TARGETS, DesignBuilder, _num, fit_glm
from src.config import Config
from src.io import read_parquet


def _standardised_factors(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Mean-imputed, z-scored distance predictors + the clipped target (rows with a target only)."""
    rows = df[_num(df[target]).notna()]
    cols = [c for c in DISTANCE_COLS if c in rows.columns and _num(rows[c]).notna().any()]
    x = pd.DataFrame(index=rows.index)
    for col in cols:
        values = _num(rows[col])
        values = values.fillna(values.mean())
        x[col] = (values - values.mean()) / (values.std(ddof=0) or 1.0)
    return x, _num(rows[target]).clip(0.0, 1.0), cols


def _ols_r2(y: pd.Series, x: pd.DataFrame) -> float:
    """OLS R^2 of ``y`` on ``x`` (0 for the empty predictor set)."""
    if x.shape[1] == 0:
        return 0.0
    return float(sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy(), has_constant="add")).fit().rsquared)


def _shapley_r2(y: pd.Series, x: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """LMG / Shapley decomposition: each factor's average marginal R^2 over all predictor subsets."""
    cache = {
        combo: _ols_r2(y, x[list(combo)])
        for k in range(len(cols) + 1)
        for combo in combinations(cols, k)
    }
    p = len(cols)
    contrib = {c: 0.0 for c in cols}
    for col in cols:
        others = [c for c in cols if c != col]
        for k in range(len(others) + 1):
            weight = factorial(k) * factorial(p - k - 1) / factorial(p)
            for combo in combinations(others, k):
                with_col = tuple(c for c in cols if c in {*combo, col})
                contrib[col] += weight * (cache[with_col] - cache[combo])
    return contrib


class VariancePartition:
    """Dominance/commonality analysis + VIF + the standardised-coefficient contrast."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config.
        """
        self.config = config or Config()
        self._table_path = self.config.paths.outputs_root / "features.parquet"

    def partition(self, target: str) -> pd.DataFrame:
        """Return the unique + shared R^2 per factor for one target.

        Args:
            target: ``"pDetA"`` or ``"pAssA"``.

        Returns:
            One row per factor: ``lmg_r2`` (Shapley share of R^2), ``share`` (fraction of the model
            R^2), and ``vif`` (collinearity). Sorted by ``lmg_r2`` descending.
        """
        df = read_parquet(self._table_path)
        x, y, cols = _standardised_factors(df, target)
        lmg = _shapley_r2(y, x, cols)
        total = sum(lmg.values()) or 1.0
        design = sm.add_constant(x.to_numpy(), has_constant="add")
        vif = {col: variance_inflation_factor(design, i + 1) for i, col in enumerate(cols)}
        table = pd.DataFrame(
            {
                "factor": cols,
                "lmg_r2": [lmg[c] for c in cols],
                "share": [lmg[c] / total for c in cols],
                "vif": [vif[c] for c in cols],
            }
        )
        return table.sort_values("lmg_r2", ascending=False).reset_index(drop=True)

    def contrast(self) -> pd.DataFrame:
        """Standardised GLM coefficients for both targets side by side (the H1-vs-H2 headline)."""
        df = read_parquet(self._table_path)
        coefs = {}
        for target in TARGETS:
            rows = df[_num(df[target]).notna()]
            builder = DesignBuilder(self.config).fit(rows)
            coefs[target] = fit_glm(rows, target, builder, self.config).params
        return pd.DataFrame(coefs)


def main() -> None:
    """CLI entry point — print the partition + contrast for both targets."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    vp = VariancePartition(Config.load(args.config))
    for target in TARGETS:
        print(f"\n== {target} ==")
        print(vp.partition(target).to_string(index=False))
    print("\n== standardised-coefficient contrast ==")
    print(vp.contrast().to_string())


if __name__ == "__main__":
    main()
