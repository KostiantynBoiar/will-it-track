"""Freeze the seen/unseen reference (T0.2).

Goal: fix the "distance from training" reference once, to prevent leakage — every distance is later
computed against this frozen train-split set only.
Input: ``sa_fari_train_ext.json``, ``sa_fari_test_ext.json``.
Output: ``data/reference/{seen_species,seen_locations}.json`` + a per-test-cell manifest (taxonomy +
``location_id`` + timestamps).
Method: enumerate train-split species/locations; record taxonomy path + metadata; verify test
species/locations are disjoint from train.
Done when: disjointness is asserted in a test; the manifest lists every test cell with its metadata.
"""

from __future__ import annotations

from src.config import Config


class Reference:
    """Build + load the frozen seen-set reference and the test-cell manifest."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``paths.reference_root``, ``reference.*``, ``data.*``).
        """
        self.config = config or Config()

    def freeze(self) -> None:
        """Write the frozen seen species/locations + the test-cell manifest to ``reference_root``."""
        raise NotImplementedError("T0.2: enumerate train species/locations + taxonomy manifest")

    def assert_disjoint(self) -> None:
        """Assert the test split is disjoint from train by species AND location.

        Raises:
            AssertionError: If any test species or location also appears in train.
        """
        raise NotImplementedError("T0.2: species/location disjointness check")
