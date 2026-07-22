"""Generate the dissertation's result figures from the committed outputs (reproducible).

Writes PDF/PNG figures to ``report/dissertation/figures/`` — the honest-null visuals: a coefficient
forest plot (every distance's CI crossing zero), the out-of-sample predicted-vs-actual cloud (no
predictive power), and the size-confound scatter (the one real correlate). The ``features_in_action``
images are generated separately (they need a real frame + GT mask); see ``figures_features.py``.

Run: ``PYTHONPATH=. python -m src.analysis.figures [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.regression import TARGETS  # noqa: E402
from src.config import Config  # noqa: E402
from src.io import read_parquet  # noqa: E402

NAVY, BURNT, FOREST, GREY = "#1e407c", "#d6681a", "#22783c", "#9aa0a6"
_ROWS = ("taxonomic_distance", "visual_distance", "environment_distance", "temporal_gap",
         "clutter", "is_night_ir", "log_support")
_LABEL = {"taxonomic_distance": "Taxonomic dist.", "visual_distance": "Visual dist.",
          "environment_distance": "Environment dist.", "temporal_gap": "Temporal gap",
          "clutter": "Clutter", "is_night_ir": "Night / IR", "log_support": "log(support)"}
plt.rcParams.update({"font.size": 9})


def _figdir(cfg: Config) -> Path:
    d = cfg.paths.outputs_root.parent / "report" / "dissertation" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def forest_plot(cfg: Config) -> Path:
    """Every distance's standardised coefficient + 95% CI, for both targets; CIs crossing zero are greyed."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, target in zip(axes, TARGETS, strict=False):
        d = pd.read_csv(cfg.paths.outputs_root / "models" / f"{target}_coef.csv", index_col=0)
        present = [r for r in _ROWS if r in d.index]
        for i, r in enumerate(present):
            lo, hi, c = float(d.loc[r, "ci_lo"]), float(d.loc[r, "ci_hi"]), float(d.loc[r, "coef"])
            crosses = lo <= 0 <= hi
            color = GREY if crosses else BURNT
            ax.plot([lo, hi], [i, i], color=color, lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot([c], [i], "o", color=color, ms=5, zorder=3)
        ax.axvline(0, color="k", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([_LABEL.get(r, r) for r in present])
        ax.set_title(target, color=NAVY, fontweight="bold")
        ax.set_xlabel("standardised coef. [95% CI]")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = _figdir(cfg) / "res_forest.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def predicted_vs_actual(cfg: Config, target: str = "pDetA") -> Path:
    """Out-of-sample predicted vs actual (detection): a flat cloud around the mean, not the diagonal."""
    cv = read_parquet(cfg.paths.outputs_root / "validation" / "cv_results.parquet")
    d = cv[(cv["target"] == target) & cv["predicted"].notna()]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for scheme, col in (("species", NAVY), ("location", FOREST)):
        s = d[d["group_scheme"] == scheme]
        ax.scatter(s["actual"], s["predicted"], s=12, alpha=0.45, color=col,
                   edgecolors="none", label=f"leave-{scheme}-out")
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.9, ls=(0, (4, 3)), label="perfect (y=x)")
    ax.axhline(float(d["actual"].mean()), color=BURNT, lw=1.3, label="mean baseline")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"actual {target}"); ax.set_ylabel(f"predicted {target} (out-of-sample)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = _figdir(cfg) / "res_pred_vs_actual.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def size_confound(cfg: Config, target: str = "pDetA") -> Path:
    """The one real correlate: detection score vs animal size (log GT mask area)."""
    src = cfg.paths.outputs_root / "features_diff.parquet"
    f = read_parquet(src if src.exists() else cfg.paths.outputs_root / "features.parquet")
    d = f[pd.to_numeric(f[target], errors="coerce").notna() & pd.to_numeric(f["log_area"], errors="coerce").notna()]
    x = pd.to_numeric(d["log_area"], errors="coerce").to_numpy()
    y = pd.to_numeric(d[target], errors="coerce").to_numpy()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(x, y, s=12, alpha=0.4, color=NAVY, edgecolors="none")
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, a + b * xs, color=BURNT, lw=1.8, label=f"trend (r={np.corrcoef(x, y)[0, 1]:+.2f})")
    ax.set_xlabel("animal size: log GT mask area"); ax.set_ylabel(f"{target} (detection score)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = _figdir(cfg) / "res_size_confound.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def main() -> None:
    """Generate all result figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = Config.load(ap.parse_args().config)
    for fn in (forest_plot, predicted_vs_actual, size_confound):
        print("figure ->", fn(cfg))


if __name__ == "__main__":
    main()
