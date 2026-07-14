"""Assemble the feature table.

Joins the four distances (+ a deferred familiarity-proxy placeholder) onto the per-cell scores into a
single modelling table written to ``outputs/features.parquet`` — one row per scored cell, keyed by
``(category_id, species, location_id, time)`` — re-verifying that no test label leaks into any column.

The distances live at different granularities and are broadcast onto the cell grid:
taxonomic/visual are per ``category_id``; the temporal gap is already per cell; environment (+ its
night/IR and clutter covariates) is per ``location_id``. The scores' own ``pDetA``/``pAssA``/support
columns ride along, so the output is the ready-to-fit table the regression consumes directly.

Run: ``PYTHONPATH=. .venv/bin/python -m src.features.assemble [--config configs/default.yaml]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.features.environment import EnvironmentDistance
from src.features.taxonomic import TaxonomicDistance
from src.features.temporal import TemporalGap
from src.features.visual import VisualDistance
from src.io import read_parquet, write_parquet
from src.splits import Partition, build_location_partition

_CELL_KEYS = ["category_id", "species", "location_id", "time"]
_ENV_COLS = ["environment_distance", "is_night_ir", "clutter", "achromatic_fraction"]


def merge_features(
    scores: pd.DataFrame,
    *,
    taxonomic: pd.Series,
    visual: pd.Series,
    temporal: pd.Series,
    environment: pd.DataFrame,
    familiarity: pd.Series | None = None,
) -> pd.DataFrame:
    """Broadcast the distance objects onto the per-cell ``scores`` grid → the merged modelling table.

    Args:
        scores: The per-cell scores (``category_id, species, location_id, time`` + ``pDetA``/... ).
        taxonomic: ``taxonomic_distance`` indexed by ``category_id``.
        visual: ``visual_distance`` indexed by ``category_id``.
        temporal: ``temporal_gap`` with a ``(category_id, species, location_id, time)`` MultiIndex.
        environment: Per-``location_id`` frame (``environment_distance`` + covariates).
        familiarity: Optional ``familiarity_proxy`` per ``category_id`` (``NaN`` column when omitted).

    Returns:
        One row per input cell, with the four distance columns + covariates joined on.
    """
    df = scores.copy()
    for key in _CELL_KEYS:
        df[key] = df[key].astype(str)

    df["taxonomic_distance"] = df["category_id"].map(taxonomic)
    df["visual_distance"] = df["category_id"].map(visual)

    temporal_df = temporal.reset_index()
    for key in _CELL_KEYS:
        temporal_df[key] = temporal_df[key].astype(str)
    df = df.merge(temporal_df, on=_CELL_KEYS, how="left")

    env_df = environment.reset_index()
    env_df["location_id"] = env_df["location_id"].astype(str)
    keep = ["location_id", *[c for c in _ENV_COLS if c in env_df.columns]]
    df = df.merge(env_df[keep], on="location_id", how="left")

    df["familiarity_proxy"] = (
        df["category_id"].map(familiarity) if familiarity is not None else np.nan
    )
    return df


class FeatureAssembler:
    """Join the four distances + proxy + support into one per-cell modelling table."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``paths.outputs_root``, ``features.*``).
        """
        self.config = config or Config()

    @staticmethod
    def _assert_no_leakage(partition: Partition) -> None:
        """Firewall: the held axis must be disjoint between reference and probe."""
        if partition.held_axis == "location":
            overlap = set(partition.probe_locations) & set(partition.reference_locations)
            if overlap:
                raise ValueError(
                    f"location leakage: {len(overlap)} probe locations also in the reference set"
                )
        elif partition.held_axis == "species" and not partition.loso:
            overlap = set(partition.probe_species) & set(partition.reference_species)
            if overlap:
                raise ValueError(f"species leakage: {len(overlap)} probe species in the reference")

    def assemble(self) -> Path:
        """Build and write ``outputs/features.parquet``.

        Returns:
            Path to the written ``features.parquet``.
        """
        scores = read_parquet(self.config.paths.outputs_root / "scores.parquet")
        partition = build_location_partition(self.config)
        self._assert_no_leakage(partition)

        merged = merge_features(
            scores,
            taxonomic=TaxonomicDistance(self.config).compute(partition),
            visual=VisualDistance(self.config).compute(partition),
            temporal=TemporalGap(self.config).compute(partition),
            environment=EnvironmentDistance(self.config).compute(partition),
        )
        path = write_parquet(merged, self.config.paths.outputs_root / "features.parquet")
        covered = int(merged["environment_distance"].notna().sum())
        print(f"features -> {path} ({len(merged)} cells, {covered} with an environment distance)")
        return path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    args = ap.parse_args()
    FeatureAssembler(Config.load(args.config)).assemble()


if __name__ == "__main__":
    main()
