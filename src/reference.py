"""Freeze the reference for a split (T0.2 — the leakage firewall).

For a given :class:`~src.splits.Partition`, freeze the reference species/locations and a probe-cell
manifest once, so every distance is later computed against a fixed anchor. Disjointness is asserted on
the split's *held axis* (location for Split B; leave-one-species-out for Split A) and the other-axis
overlap is *reported* honestly — the SA-FARI split shares species, so species overlap is expected, not
an error.

Run: ``PYTHONPATH=. .venv/bin/python -m src.reference --freeze`` / ``--check``.
"""

from __future__ import annotations

import argparse
import json

from pydantic import BaseModel

from src.config import Config
from src.dataset import SAFARI, _year
from src.splits import (
    Partition,
    build_location_partition,
    build_species_partition,
    probe_records,
    save,
)


class ManifestCell(BaseModel):
    """One probe cell in a frozen manifest.

    Attributes:
        category_id: Species identity.
        species: Canonical species label.
        location_id: Camera/location identifier.
        time: Coarse time bucket (year).
        taxonomy: Lowercased taxonomy (``{}`` if the category has none).
        n_videos: Number of probe videos backing this cell.
    """

    category_id: str
    species: str
    location_id: str
    time: str
    taxonomy: dict[str, str]
    n_videos: int


class Reference:
    """Build, check, and load the frozen reference for one partition."""

    def __init__(self, config: Config | None = None, partition: Partition | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``paths.reference_root``, ``reference.*``).
            partition: The split whose reference is frozen (required for freeze/load/check).
        """
        self.config = config or Config()
        self.partition = partition

    @property
    def _root(self):
        """Per-partition reference directory."""
        assert self.partition is not None, "Reference needs a partition"
        return self.config.paths.reference_root / self.partition.name

    def check_axis(self) -> dict:
        """Assert the split's held axis is disjoint; report the other-axis overlap.

        Returns:
            ``{"held_axis", "location_overlap", "species_overlap"}``.

        Raises:
            AssertionError: If Split B's probe locations intersect the reference, or Split A is not LOSO.
        """
        p = self.partition
        assert p is not None
        location_overlap = sorted(set(p.reference_locations) & set(p.probe_locations))
        species_overlap = sorted(set(p.reference_species) & set(p.probe_species))
        if p.held_axis == "location":
            assert not location_overlap, f"location overlap on Split B: {location_overlap[:10]}"
        elif p.held_axis == "species":
            assert p.loso, "the species split must use leave-one-species-out"
        return {
            "held_axis": p.held_axis,
            "location_overlap": len(location_overlap),
            "species_overlap": len(species_overlap),
        }

    def freeze(self) -> None:
        """Write reference species/locations + the probe-cell manifest to ``reference_root/<split>/``."""
        assert self.partition is not None
        root = self._root
        root.mkdir(parents=True, exist_ok=True)
        ref = self.config.reference
        (root / ref.reference_species_file).write_text(
            json.dumps(self.partition.reference_species, indent=2)
        )
        (root / ref.reference_locations_file).write_text(
            json.dumps(self.partition.reference_locations, indent=2)
        )

        taxonomy = SAFARI("test", self.config).taxonomy()
        cells: dict[tuple[str, str, str], ManifestCell] = {}
        for record in probe_records(self.partition, self.config):
            time = _year(record.creation_datetime)
            key = (record.category_id, record.location_id, time)
            if key in cells:
                cells[key].n_videos += 1
                continue
            cells[key] = ManifestCell(
                category_id=record.category_id,
                species=record.species,
                location_id=record.location_id,
                time=time,
                taxonomy=taxonomy.get(record.category_id, {}),
                n_videos=1,
            )
        manifest = [c.model_dump() for c in cells.values()]
        (root / ref.manifest_file).write_text(json.dumps(manifest, indent=2))

        report = self.check_axis()
        print(
            f"froze reference -> {root}: {len(self.partition.reference_species)} reference species, "
            f"{len(self.partition.reference_locations)} reference locations, {len(manifest)} probe cells "
            f"(species overlap {report['species_overlap']}, location overlap {report['location_overlap']})"
        )

    def load(self) -> dict:
        """Read the three frozen files back for the partition."""
        ref = self.config.reference
        return {
            "reference_species": json.loads((self._root / ref.reference_species_file).read_text()),
            "reference_locations": json.loads(
                (self._root / ref.reference_locations_file).read_text()
            ),
            "manifest": json.loads((self._root / ref.manifest_file).read_text()),
        }


def main() -> None:
    """CLI: build both splits, ``--check`` disjointness and/or ``--freeze`` their references."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument(
        "--check", action="store_true", help="assert held-axis disjointness + report overlap"
    )
    ap.add_argument("--freeze", action="store_true", help="persist the splits + frozen references")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    partitions = [build_location_partition(cfg), build_species_partition(cfg)]
    if args.check:
        for partition in partitions:
            print(partition.name, Reference(cfg, partition).check_axis())
    if args.freeze:
        for partition in partitions:
            save(partition, cfg)
            Reference(cfg, partition).freeze()


if __name__ == "__main__":
    main()
