"""Shared record types.

The **cell** — ``(species, location_id, time)`` — is the unit at which scores are recorded and the
law is fit; **support** is how much data backs a cell (used for weighting, controlled for in the
models). Both are pure data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Cell(BaseModel):
    """The analysis unit: a ``(species, location_id, time)`` group.

    Frozen so it is hashable (used as a set/dict key when de-duplicating cells).

    Attributes:
        species: SA-FARI species label.
        location_id: Anonymised camera/location identifier.
        time: Coarse time bucket (e.g. year) from ``video_creation_datetime``.
    """

    model_config = ConfigDict(frozen=True)

    species: str
    location_id: str
    time: str


class Support(BaseModel):
    """How much data backs a cell's score.

    Attributes:
        n_frames: Annotated frames.
        n_masklets: Distinct masklet identities.
        n_videos: Videos contributing to the cell.
    """

    n_frames: int
    n_masklets: int
    n_videos: int
