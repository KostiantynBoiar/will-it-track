"""SA-FARI loader — the ``_ext`` annotation schema + RLE masks + cell grouping.

Reads the YTVIS-style ``sa_fari_{train,test}_ext.json``. Species and their 7-level taxonomy live on
``categories`` keyed by ``category_id`` (a large shared noun-phrase vocabulary; only animal concepts
carry real taxonomy, the rest are ``NaN``). ``video_np_pairs`` are the (video, prompt) probes and
carry ``num_masklets`` (``0`` ⇒ hard negative); ``annotations`` hold the per-frame RLE masks. Per-video
``location_id`` / ``video_creation_datetime`` live on ``videos``. Hard negatives are kept (§9). The
species identity is the ``category_id``; ``species`` is the canonical category ``name`` (display only).
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
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


def _is_real(value: object) -> bool:
    """True if a value is a real string — not ``None``, float ``NaN``, or ``"nan"``/``"none"``/``""``."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in ("", "nan", "none")


def _taxonomy_of(category: dict) -> dict[str, str]:
    """Lowercased 7-level taxonomy from one ``categories`` entry (``NaN``/absent levels omitted)."""
    return {f.lower(): str(category[f]) for f in _TAXONOMY_FIELDS if _is_real(category.get(f))}


@lru_cache(maxsize=4)
def _read_annotation_json(path_str: str) -> dict:
    """Read + validate an annotation JSON once, cached by path (the files are large and immutable)."""
    with open(path_str) as f:
        data = json.load(f)
    missing = {"videos", "annotations"} - set(data)
    if missing:
        raise KeyError(f"unexpected SA-FARI schema in {path_str}: missing {sorted(missing)}")
    return data


class VideoRecord(BaseModel):
    """One SA-FARI (video, prompt) probe.

    Attributes:
        video_id: Identifier (namespaced by origin in the pooled view).
        file_names: Ordered 6 fps frame paths (bucket-relative).
        category_id: Stable species identity — the grouping / cross-validation key.
        species: Canonical species label (category ``name``); ``""`` if unresolved.
        noun_phrase: Raw prompt text (equals ``species`` in practice; kept for prompt-mode robustness).
        location_id: Camera/location identifier.
        creation_datetime: Per-video timestamp.
        origin: Source split file (``"train"`` / ``"test"``).
        num_masklets: Ground-truth masklets for this probe (``0`` ⇒ hard negative).
        is_hard_negative: ``num_masklets == 0`` (queried species absent from the video).
    """

    video_id: str
    file_names: list[str]
    category_id: str
    species: str
    noun_phrase: str | None
    location_id: str
    creation_datetime: str
    origin: str
    num_masklets: int
    is_hard_negative: bool


class SAFARI:
    """Iterate SA-FARI (video, prompt) records and decode their masks."""

    def __init__(self, split: str, config: Config | None = None) -> None:
        """Initialize.

        Args:
            split: ``"train"`` (seen/reference) or ``"test"`` (transfer probes).
            config: Project config (``paths.data_root``, ``data.*``).
        """
        self.split = split
        self.config = config or Config()
        self._data: dict | None = None
        self._cat_by_id: dict[str, dict] | None = None
        self._anns_by_video: dict[str, list[dict]] | None = None

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
            self._data = _read_annotation_json(str(path))
        return self._data

    def categories(self) -> list[dict]:
        """Return the raw ``categories`` list (the shared noun-phrase vocabulary + taxonomy)."""
        return self._load().get("categories", [])

    def categories_by_id(self) -> dict[str, dict]:
        """Map ``category_id`` (as ``str``) to its raw ``categories`` entry (cached)."""
        if self._cat_by_id is None:
            self._cat_by_id = {str(c["id"]): c for c in self.categories()}
        return self._cat_by_id

    def _species_name(self, category_id: str) -> str:
        """Canonical species name for a ``category_id`` (category ``name``, fallback ``Species``)."""
        cat = self.categories_by_id().get(str(category_id), {})
        name = cat.get("name") or cat.get("Species")
        return str(name) if _is_real(name) else ""

    def records(self) -> list[VideoRecord]:
        """Return every ``(video, prompt)`` probe in the split (hard negatives included by default)."""
        data = self._load()
        videos = {v["id"]: v for v in data["videos"]}

        records: list[VideoRecord] = []
        pairs = data.get("video_np_pairs")
        if pairs:
            for pair in pairs:
                video = videos.get(pair.get("video_id"))
                if video is not None:
                    records.append(self._record(video, pair))
        else:
            # fall back to distinct (video, category) pairs from annotations (num_masklets unknown → 1)
            seen: dict[tuple, dict] = {}
            for a in data["annotations"]:
                seen.setdefault((a["video_id"], a.get("category_id")), a)
            for (vid, cid), a in seen.items():
                video = videos.get(vid)
                if video is not None:
                    pair = {
                        "category_id": cid,
                        "noun_phrase": a.get("noun_phrase"),
                        "num_masklets": 1,
                    }
                    records.append(self._record(video, pair))

        if not self.config.data.keep_hard_negatives:
            records = [r for r in records if not r.is_hard_negative]
        return records

    def _record(self, video: dict, pair: dict) -> VideoRecord:
        """Build a :class:`VideoRecord` from a video object + its ``video_np_pairs`` probe."""
        category_id = str(pair.get("category_id", ""))
        n_masklets = int(pair.get("num_masklets", 0) or 0)
        return VideoRecord(
            video_id=str(video["id"]),
            file_names=list(video.get("file_names", [])),
            category_id=category_id,
            species=self._species_name(category_id),
            noun_phrase=pair.get("noun_phrase"),
            location_id=str(video.get("location_id", "")),
            creation_datetime=str(video.get("video_creation_datetime", "")),
            origin=self.split,
            num_masklets=n_masklets,
            is_hard_negative=(n_masklets == 0),
        )

    def present_category_ids(self) -> set[str]:
        """Category ids with at least one positive (``num_masklets > 0``) probe in this split."""
        pairs = self._load().get("video_np_pairs") or []
        return {str(p["category_id"]) for p in pairs if int(p.get("num_masklets", 0) or 0) > 0}

    def annotations_for(self, video_id: str) -> list[dict]:
        """Return the raw annotation dicts for one video."""
        return self.annotations_by_video().get(str(video_id), [])

    def annotations_by_video(self) -> dict[str, list[dict]]:
        """Map ``video_id`` (as ``str``) to its annotation dicts (cached; avoids O(N) rescans)."""
        if self._anns_by_video is None:
            index: dict[str, list[dict]] = {}
            for a in self._load()["annotations"]:
                index.setdefault(str(a["video_id"]), []).append(a)
            self._anns_by_video = index
        return self._anns_by_video

    def taxonomy(self) -> dict[str, dict[str, str]]:
        """Map ``category_id`` → lowercased real taxonomy, for taxonomy-bearing categories only.

        Categories with no real taxonomy (the generic vocabulary) are simply absent from the map, so
        ``category_id not in taxonomy()`` means "no taxonomy" (distinct from an empty taxonomy).
        """
        out: dict[str, dict[str, str]] = {}
        for category in self.categories():
            tax = _taxonomy_of(category)
            if tax:
                out[str(category["id"])] = tax
        return out

    def taxonomy_by_name(self) -> dict[str, dict[str, str]]:
        """Display view: canonical species ``name`` → real taxonomy (taxonomy-bearing categories)."""
        out: dict[str, dict[str, str]] = {}
        for category in self.categories():
            tax = _taxonomy_of(category)
            name = category.get("name") or category.get("Species")
            if tax and _is_real(name):
                out.setdefault(str(name), tax)
        return out

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
        """Map a probe record to its ``(category_id, species, location_id, time)`` cell.

        Args:
            record: A video record.

        Returns:
            The cell it belongs to (keyed by ``category_id``).
        """
        return Cell(
            category_id=record.category_id,
            species=record.species,
            location_id=record.location_id,
            time=_year(record.creation_datetime),
        )


def pooled_records(config: Config | None = None) -> list[VideoRecord]:
    """Every probe from both splits, ``origin``-tagged with ``video_id`` namespaced by origin.

    The species identity (``category_id``) is shared across splits (the ``categories`` vocabulary is
    identical), so records for the same species from train and test map to one group.
    """
    cfg = config or Config()
    out: list[VideoRecord] = []
    for split in ("train", "test"):
        for record in SAFARI(split, cfg).records():
            out.append(record.model_copy(update={"video_id": f"{split}:{record.video_id}"}))
    return out
