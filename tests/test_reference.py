"""Seen/unseen reference tests (T0.2) — species+location disjointness. Needs the _ext JSONs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="T0.2: implement src.reference (needs SA-FARI annotations)")


def test_test_split_disjoint_from_train() -> None:
    """No test species or location also appears in the train (seen) split."""
    ...


def test_manifest_covers_every_test_cell() -> None:
    """The frozen manifest lists every test cell with taxonomy + location_id + timestamp."""
    ...
