"""Temporal gap (T2.4).

Goal: capture drift over years.
Input: ``video_creation_datetime`` (test + train).
Output: ``temporal_gap`` per cell.
Method: min |Delta year| between a test cell's footage and the nearest training footage (locations
    are disjoint, so "nearest relevant training footage" reduces to the nearest train footage year).
Done when: gaps are non-negative and non-null for every cell whose year parses.
Depends on: T0.2 (the train split is the frozen reference).
"""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.dataset import SAFARI, _year


class TemporalGap:
    """Years from each test cell to the nearest training footage."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config.
        """
        self.config = config or Config()

    def compute(self) -> pd.Series:
        """Return ``temporal_gap`` per cell.

        Returns:
            A Series indexed by a ``(species, location_id, time)`` MultiIndex; each value is the
            smallest year gap from the cell's footage to the nearest seen (train) footage. Cells
            whose year does not parse are omitted.
        """
        train = SAFARI("train", self.config)
        train_years = {int(y) for r in train.records() if (y := _year(r.creation_datetime))}

        test = SAFARI("test", self.config)
        cells = {test.cell_of(r) for r in test.records()}
        gaps = {
            (cell.species, cell.location_id, cell.time): min(
                abs(int(cell.time) - year) for year in train_years
            )
            for cell in cells
            if cell.time and train_years
        }
        index = pd.MultiIndex.from_tuples(gaps, names=["species", "location_id", "time"])
        return pd.Series(list(gaps.values()), index=index, name="temporal_gap", dtype="int64")
