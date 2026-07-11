"""Temporal gap.

Captures drift over years: the minimum |Δyear| between a probe cell's footage and the nearest
reference footage year, computed per probe cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.config import Config
from src.dataset import _year
from src.splits import probe_records

if TYPE_CHECKING:
    from src.splits import Partition


class TemporalGap:
    """Years from each probe cell to the nearest reference footage."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config.
        """
        self.config = config or Config()

    def compute(self, partition: Partition) -> pd.Series:
        """Return ``temporal_gap`` per probe cell.

        Args:
            partition: The active split.

        Returns:
            A Series with a ``(category_id, species, location_id, time)`` MultiIndex; each value is the
            smallest year gap from the cell's footage to the nearest reference footage year.
        """
        reference_years = {int(y) for y in partition.reference_years if y.isdigit()}
        gaps: dict[tuple[str, str, str, str], int] = {}
        for record in probe_records(partition, self.config):
            year = _year(record.creation_datetime)
            if not year or not reference_years:
                continue
            key = (record.category_id, record.species, record.location_id, year)
            gaps[key] = min(abs(int(year) - ref) for ref in reference_years)
        index = pd.MultiIndex.from_tuples(
            gaps, names=["category_id", "species", "location_id", "time"]
        )
        return pd.Series(list(gaps.values()), index=index, name="temporal_gap", dtype="int64")
