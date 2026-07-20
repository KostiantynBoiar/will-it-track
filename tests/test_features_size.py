"""SizeFeature per-species area aggregation — hermetic (monkeypatched loaders, no gated data)."""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.dataset import VideoRecord
from src.features import size as S
from src.features.size import SizeFeature
from src.splits import Partition


def _rec(video_id: str, category_id: str) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        file_names=[f"{video_id}/0.jpg"],
        category_id=category_id,
        species=f"sp{category_id}",
        noun_phrase=f"sp{category_id}",
        location_id="L",
        creation_datetime="2020",
        origin="test",
        num_masklets=1,
        is_hard_negative=False,
    )


class _FakeSafari:
    """mask_at returns a square mask whose pixel count is the annotation's planted ``_area``."""

    def mask_at(self, annotation: dict, frame_index: int) -> np.ndarray:
        side = int(round(annotation["_area"] ** 0.5))
        return np.ones((side, side), dtype=bool)


def _partition(species: list[str]) -> Partition:
    return Partition(
        name="t",
        held_axis="species",
        loso=True,
        reference_species=species,
        probe_species=species,
        reference_locations=[],
        probe_locations=[],
        reference_years=[],
        probe_origins=["test"],
    )


def test_size_feature_per_species(monkeypatch) -> None:
    """log_area is log1p(mean GT mask area), computed per species from the sampled GT masks."""
    recs = [_rec("a", "1"), _rec("b", "2")]
    areas = {"1": 100, "2": 400}
    monkeypatch.setattr(S, "probe_records", lambda partition, config: recs)
    monkeypatch.setattr(
        S,
        "record_annotations",
        lambda record, safari, cid: [{"segmentations": [{"rle": 1}], "_area": areas[cid]}],
    )
    monkeypatch.setattr(S, "safari_by_origin", lambda config: {"test": _FakeSafari()})

    series = SizeFeature(Config()).compute(_partition(["1", "2"]))

    assert series.name == "log_area"
    assert series["1"] == np.log1p(100.0)
    assert series["2"] == np.log1p(400.0)


def test_size_feature_nan_for_species_without_masks(monkeypatch) -> None:
    """A probe species with no ground-truth masks gets NaN (not 0)."""
    monkeypatch.setattr(S, "probe_records", lambda partition, config: [])
    monkeypatch.setattr(S, "record_annotations", lambda record, safari, cid: [])
    monkeypatch.setattr(S, "safari_by_origin", lambda config: {"test": _FakeSafari()})

    series = SizeFeature(Config()).compute(_partition(["9"]))
    assert np.isnan(series["9"])
