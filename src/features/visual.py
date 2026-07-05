"""Visual distance (T2.2).

Goal: how unfamiliar the animal looks to a frozen vision encoder.
Input: ``data/frames/``, masks (to crop), the reference species set.
Output: ``visual_distance`` per test species.
Method: embed MASK-CROPPED animal frames with DINOv2 (§9 DO); represent each species by its
    embedding distribution; distance = cosine distance to the nearest training-species prototype
    (plus a distributional variant — Frechet/MMD — for robustness, ``features.distance_variant``).
Done when: distances are stable under resampling; background is demonstrably not the dominant signal
    (cropped vs uncropped sanity check).
Depends on: T0.2, T1.2 (masks).
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


class VisualDistance:
    """DINOv2 mask-crop embedding distance from each test species to the nearest seen prototype."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.visual_encoder``, ``mask_crop``, ``distance_variant``).
        """
        self.config = config or Config()

    def compute(self) -> pd.Series:
        """Return ``visual_distance`` per test species.

        Returns:
            A Series indexed by species.
        """
        raise NotImplementedError("T2.2: DINOv2 mask-crop embeddings -> nearest-prototype distance")
