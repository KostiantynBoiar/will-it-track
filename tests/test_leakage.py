"""Feature-leakage test (T2.6) — no test information enters a distance. Needs the feature table."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="T2.6: implement src.features (needs assembled feature table)")


def test_swapping_seen_set_changes_distances() -> None:
    """Swapping the frozen seen set changes the distances as expected (label-free, no test leak)."""
    ...


def test_feature_table_has_no_unexplained_nulls() -> None:
    """outputs/features.parquet has the four distances + proxy + support with no stray nulls."""
    ...
