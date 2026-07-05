"""SA-FARI loader — the ``_ext`` annotation schema + RLE masks + cell grouping.

Reads the YTVIS-style ``sa_fari_{train,test}_ext.json`` (with ``video_num_frames``, ``video_fps``,
``video_creation_datetime``, ``location_id``, and the 7-level taxonomy), decodes per-frame RLE masks
with ``pycocotools``, and groups videos into ``(species, location_id, time)`` cells. Hard negatives
(queries with 0 masklets) are kept (§9). Shared by inference, eval, and the feature builders.
Implemented at T0.1/T1.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config
from src.types import Cell


@dataclass
class VideoRecord:
    """One SA-FARI video + its query.

    Attributes:
        video_id: Identifier.
        file_names: Ordered 6 fps frame paths.
        species: Queried species (noun phrase); ``None``/absent for a hard negative.
        location_id: Camera/location identifier.
        creation_datetime: Per-video timestamp.
        is_hard_negative: True when the query returns 0 masklets (species absent).
    """

    video_id: str
    file_names: list[str]
    species: str | None
    location_id: str
    creation_datetime: str
    is_hard_negative: bool


class SAFARI:
    """Iterate SA-FARI video records and map them to cells."""

    def __init__(self, split: str, config: Config | None = None) -> None:
        """Initialize.

        Args:
            split: ``"train"`` (seen/reference) or ``"test"`` (transfer probes).
            config: Project config (``paths.data_root``, ``data.*``).
        """
        self.split = split
        self.config = config or Config()

    def records(self) -> list[VideoRecord]:
        """Return all video records in the split (hard negatives included)."""
        raise NotImplementedError("T0.1: parse the _ext JSON (pycocotools RLE)")

    def cell_of(self, record: VideoRecord) -> Cell:
        """Map a video record to its ``(species, location_id, time)`` cell.

        Args:
            record: A video record.

        Returns:
            The cell it belongs to.
        """
        raise NotImplementedError("T1.2: bucket species/location_id/time into a Cell")
