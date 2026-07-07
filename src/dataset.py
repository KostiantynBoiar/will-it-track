"""SA-FARI loader — the ``_ext`` annotation schema + RLE masks + cell grouping.

Reads the YTVIS-style ``sa_fari_{train,test}_ext.json`` (``videos`` with ``file_names`` + per-video
``location_id`` / ``video_creation_datetime``, ``annotations`` with per-frame RLE ``segmentations`` +
``noun_phrase``, and ``video_np_pairs`` = the (video, query) probes incl. hard negatives). Decodes
per-frame RLE masks with ``pycocotools``; hard negatives are kept (§9). The exact ``_ext`` field
names are validated at load and treated as verify-at-runtime — the loader fails loudly if the schema
differs. ``records()`` + ``mask_at()`` implemented at T0.1; ``cell_of()`` at T1.2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
from pydantic import BaseModel

from src.config import Config
from src.types import Cell

# Capitalized taxonomy fields as they appear on SA-FARI `categories` (HF dataset card).
_TAXONOMY_FIELDS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")


def _year(dt: str) -> str:
    """Return the first 4-digit year in a datetime string, or ``""`` if none."""
    match = re.search(r"\d{4}", dt or "")
    return match.group(0) if match else ""


def _taxonomy_of(category: dict) -> dict[str, str]:
    """Lowercased 7-level taxonomy from one raw ``categories`` entry (absent levels omitted)."""
    return {
        field.lower(): str(category[field]) for field in _TAXONOMY_FIELDS if category.get(field)
    }


class VideoRecord(BaseModel):
    """One SA-FARI (video, query) probe.

    Attributes:
        video_id: Identifier.
        file_names: Ordered 6 fps frame paths (bucket-relative).
        species: Queried noun phrase (species); may be ``None`` if absent.
        location_id: Camera/location identifier.
        creation_datetime: Per-video timestamp.
        is_hard_negative: True when the query has no positive masklet (species absent).
    """

    video_id: str
    file_names: list[str]
    species: str | None
    location_id: str
    creation_datetime: str
    is_hard_negative: bool


class SAFARI:
    """Iterate SA-FARI (video, query) records and decode their masks."""

    def __init__(self, split: str, config: Config | None = None) -> None:
        """Initialize.

        Args:
            split: ``"train"`` (seen/reference) or ``"test"`` (transfer probes).
            config: Project config (``paths.data_root``, ``data.*``).
        """
        self.split = split
        self.config = config or Config()
        self._data: dict | None = None

    @property
    def ann_path(self) -> Path:
        """Path to this split's ``_ext`` annotation JSON."""
        d = self.config.data
        name = d.train_ann if self.split == "train" else d.test_ann
        return self.config.paths.data_root / d.annotations_subdir / name

    def _load(self) -> dict:
        """Load + cache the annotation JSON, validating the core YTVIS keys.

        Raises:
            FileNotFoundError: If the annotations have not been fetched.
            KeyError: If the JSON lacks the expected top-level keys.
        """
        if self._data is None:
            path = self.ann_path
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run `python -m src.acquire --annotations` (needs HF login)"
                )
            with open(path) as f:
                self._data = json.load(f)
            missing = {"videos", "annotations"} - set(self._data)
            if missing:
                raise KeyError(f"unexpected SA-FARI schema in {path}: missing {sorted(missing)}")
        return self._data

    def records(self) -> list[VideoRecord]:
        """Return every ``(video, query)`` record in the split (hard negatives included by default)."""
        data = self._load()
        videos = {v["id"]: v for v in data["videos"]}
        positive = {(a["video_id"], a.get("noun_phrase")) for a in data["annotations"]}

        records: list[VideoRecord] = []
        pairs = data.get("video_np_pairs")
        if pairs:
            for pair in pairs:
                video = videos.get(pair.get("video_id"))
                if video is None:
                    continue
                noun = pair.get("noun_phrase")
                records.append(
                    self._record(video, noun, (pair.get("video_id"), noun) not in positive)
                )
        else:
            # fall back to the distinct (video, noun_phrase) pairs seen in annotations
            for key in dict.fromkeys(
                (a["video_id"], a.get("noun_phrase")) for a in data["annotations"]
            ):
                video = videos.get(key[0])
                if video is not None:
                    records.append(self._record(video, key[1], is_negative=False))

        if not self.config.data.keep_hard_negatives:
            records = [r for r in records if not r.is_hard_negative]
        return records

    def _record(self, video: dict, noun: str | None, is_negative: bool) -> VideoRecord:
        """Build a :class:`VideoRecord` from a video object + query."""
        return VideoRecord(
            video_id=str(video["id"]),
            file_names=list(video.get("file_names", [])),
            species=noun,
            location_id=str(video.get("location_id", "")),
            creation_datetime=str(video.get("video_creation_datetime", "")),
            is_hard_negative=is_negative,
        )

    def annotations_for(self, video_id: str) -> list[dict]:
        """Return the raw annotation dicts for one video."""
        data = self._load()
        return [a for a in data["annotations"] if str(a["video_id"]) == str(video_id)]

    def categories(self) -> list[dict]:
        """Return the raw ``categories`` list (species + their 7-level taxonomy)."""
        return self._load().get("categories", [])

    def taxonomy(self) -> dict[str, dict[str, str]]:
        """Map each species name in this split to its lowercased 7-level taxonomy.

        Returns:
            ``{species_name: {"kingdom": ..., ..., "species": ...}}`` keyed by the category
            ``name`` (fallback ``Species``); one entry per category, absent levels omitted.
        """
        index: dict[str, dict[str, str]] = {}
        for category in self.categories():
            name = category.get("name") or category.get("Species")
            if name:
                index[str(name)] = _taxonomy_of(category)
        return index

    def mask_at(self, annotation: dict, frame_index: int) -> np.ndarray:
        """Decode one annotation's per-frame RLE mask to a dense ``bool`` array.

        Args:
            annotation: A raw annotation dict (with ``segmentations`` per frame).
            frame_index: Frame position.

        Returns:
            A ``(H, W)`` boolean mask (all-``False`` where the frame is unlabelled).
        """
        segs = annotation.get("segmentations") or []
        seg = segs[frame_index] if 0 <= frame_index < len(segs) else None
        if seg is None:
            h, w = annotation.get("height"), annotation.get("width")
            return np.zeros((h, w), dtype=bool) if h and w else np.zeros((0, 0), dtype=bool)
        return coco_mask.decode(seg).astype(bool)

    def cell_of(self, record: VideoRecord) -> Cell:
        """Map a video record to its ``(species, location_id, time)`` cell.

        Args:
            record: A video record.

        Returns:
            The cell it belongs to.
        """
        return Cell(
            species=record.species or "",
            location_id=record.location_id,
            time=_year(record.creation_datetime),
        )
