"""Animal size (ground-truth mask area) per species --- a control for the visual-distance confound.

Visually-distinctive species tend to be large, high-contrast animals that are *easy* to segment, so
``visual_distance`` can pick up ease rather than novelty. This feature gives the mean ground-truth mask
area per species (log pixels), to add as a nuisance covariate and test whether the (wrong-signed) visual
effect is really a size artefact. Keyed by ``category_id`` (probe species), mirroring the distances; the
same per-species sampling caps as the visual/environment features bound its cost.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import Config
from src.features.pipeline import record_annotations, safari_by_origin
from src.splits import probe_records

if TYPE_CHECKING:
    from src.splits import Partition


class SizeFeature:
    """Mean log ground-truth mask-area per probe species (``log_area``)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``features.max_masklets_per_species`` / ``n_frames_per_masklet``
                cap the sampling, exactly as the visual feature does).
        """
        self.config = config or Config()

    def compute(self, partition: Partition) -> pd.Series:
        """Return ``log_area`` per probe species (``category_id``).

        Args:
            partition: The active split (its probe side supplies the species to size).

        Returns:
            A Series indexed by probe ``category_id`` (``log1p`` of the mean GT mask pixel-count);
            ``NaN`` for species with no usable ground-truth mask.
        """
        safari = safari_by_origin(self.config)
        masklet_cap = self.config.features.max_masklets_per_species
        frame_cap = max(1, self.config.features.n_frames_per_masklet)
        seen: Counter[str] = Counter()
        areas: dict[str, list[float]] = defaultdict(list)
        for record in probe_records(partition, self.config):
            cid = record.category_id
            if masklet_cap and seen[cid] >= masklet_cap:
                continue
            reader = safari[record.origin]
            for ann in record_annotations(record, safari, cid):
                frames = [i for i, seg in enumerate(ann.get("segmentations") or []) if seg is not None]
                if not frames:
                    continue
                step = max(1, len(frames) // frame_cap)
                for i in frames[::step][:frame_cap]:
                    areas[cid].append(float(reader.mask_at(ann, i).sum()))
                seen[cid] += 1
                if masklet_cap and seen[cid] >= masklet_cap:
                    break
        rows = {cid: float(np.log1p(np.mean(a))) for cid, a in areas.items() if a}
        for cid in partition.probe_species:
            rows.setdefault(cid, float("nan"))
        return pd.Series(rows, name="log_area", dtype="float64")
