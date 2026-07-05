"""Shared record types.

The **cell** — ``(species, location_id, time)`` — is the unit at which scores are recorded and the
law is fit; **support** is how much data backs a cell (used for weighting, controlled for in the
models). Both are pure data (no ``NotImplementedError``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    """The analysis unit: a ``(species, location_id, time)`` group.

    Attributes:
        species: SA-FARI species label.
        location_id: Anonymised camera/location identifier.
        time: Coarse time bucket (e.g. year) from ``video_creation_datetime``.
    """

    species: str
    location_id: str
    time: str


@dataclass
class Support:
    """How much data backs a cell's score.

    Attributes:
        n_frames: Annotated frames.
        n_masklets: Distinct masklet identities.
        n_videos: Videos contributing to the cell.
    """

    n_frames: int
    n_masklets: int
    n_videos: int
