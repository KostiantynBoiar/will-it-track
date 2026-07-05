"""Small IO helpers — parquet tables (scores/features/CV) and JSON (reference).

Thin wrappers so every module reads/writes the shared artifacts the same way. Implemented alongside
the first task that needs them (T1.2 for scores).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a dataframe to parquet (creating parent dirs).

    Args:
        df: The table.
        path: Destination ``.parquet`` path.

    Returns:
        ``path``.
    """
    raise NotImplementedError("T1.2: parquet write with parent mkdir")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet table.

    Args:
        path: A ``.parquet`` path.

    Returns:
        The table.
    """
    raise NotImplementedError("T1.2: parquet read")
