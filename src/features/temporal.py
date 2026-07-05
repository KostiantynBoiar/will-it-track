"""Temporal gap (T2.4).

Goal: capture drift over years.
Input: ``video_creation_datetime`` (test + train).
Output: ``temporal_gap`` per cell.
Method: years between a test cell's footage and the nearest relevant training footage.
Done when: values are within the 2014-2024 range and non-null for all cells.
Depends on: T0.2.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


class TemporalGap:
    """Years from each test cell to the nearest relevant training footage."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config.
        """
        self.config = config or Config()

    def compute(self) -> pd.Series:
        """Return ``temporal_gap`` per cell.

        Returns:
            A Series indexed by cell.
        """
        raise NotImplementedError("T2.4: year gap to nearest training footage")
