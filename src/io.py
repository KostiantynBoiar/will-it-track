"""Small IO helpers — parquet tables (scores/features/CV) and JSON (reference).

Thin wrappers so every module reads/writes the shared artifacts the same way.
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet table.

    Args:
        path: A ``.parquet`` path.

    Returns:
        The table.
    """
    return pd.read_parquet(path)
