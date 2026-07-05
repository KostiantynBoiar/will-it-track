"""Environment distance (T2.3).

Goal: scene/appearance gap between the test location and training locations.
Input: ``data/frames/`` (background regions), ``location_id``, colour statistics.
Output: ``environment_distance`` + interpretable covariates (``is_night_ir``, clutter/motion proxy)
    per location.
Method: embed the scene (whole frame or animal-masked-out background); distance = embedding distance
    to the nearest training location; derive day/night-IR from colour stats.
Done when: distances separate obviously different habitats; the night/IR flag is validated on samples.
Depends on: T0.2.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


class EnvironmentDistance:
    """Scene-embedding distance per test location + interpretable covariates."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.scene_encoder``, ``night_ir_from_color``).
        """
        self.config = config or Config()

    def compute(self) -> pd.DataFrame:
        """Return per-location ``environment_distance`` + ``is_night_ir`` + clutter/motion.

        Returns:
            A DataFrame indexed by ``location_id``.
        """
        raise NotImplementedError("T2.3: background scene embedding -> nearest-location distance")
