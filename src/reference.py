"""Freeze the seen/unseen reference (T0.2).

Goal: fix the "distance from training" reference once, to prevent leakage — every distance is later
computed against this frozen train-split set only.
Input: ``sa_fari_train_ext.json``, ``sa_fari_test_ext.json``.
Output: ``data/reference/{seen_species,seen_locations}.json`` + a per-test-cell manifest (taxonomy +
``location_id`` + timestamps).
Method: enumerate train-split species/locations; record taxonomy path + metadata; verify test
species/locations are disjoint from train.
Done when: disjointness is asserted in a test; the manifest lists every test cell with its metadata.

Run: ``PYTHONPATH=. .venv/bin/python -m src.reference --freeze`` / ``--check``.
"""

from __future__ import annotations

import argparse
import json

from pydantic import BaseModel

from src.config import Config
from src.dataset import SAFARI
from src.types import Cell

# Capitalized taxonomy fields as they appear on SA-FARI `categories` (HF dataset card).
_TAXONOMY_FIELDS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")


class ManifestCell(BaseModel):
    """One test cell in the frozen manifest.

    Attributes:
        species: Queried species (noun phrase / category name).
        location_id: Camera/location identifier.
        time: Coarse time bucket (year).
        taxonomy: Lowercased 7-level taxonomy (``{}`` if the species was unmatched).
        n_videos: Number of test videos backing this cell.
    """

    species: str
    location_id: str
    time: str
    taxonomy: dict[str, str]
    n_videos: int


class Reference:
    """Build + load the frozen seen-set reference and the test-cell manifest."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (loaders are lazy — no file access until a method is called).

        Args:
            config: Project config (``paths.reference_root``, ``reference.*``, ``data.*``).
        """
        self.config = config or Config()
        self.train = SAFARI("train", self.config)
        self.test = SAFARI("test", self.config)

    @staticmethod
    def _species(safari: SAFARI) -> list[str]:
        """Canonical species names for a split (category ``name``, fallback ``Species``)."""
        names = []
        for category in safari.categories():
            name = category.get("name") or category.get("Species")
            if name:
                names.append(str(name))
        return names

    @staticmethod
    def _locations(safari: SAFARI) -> set[str]:
        """Distinct ``location_id``s appearing in a split."""
        return {r.location_id for r in safari.records() if r.location_id}

    def _taxonomy_index(self) -> dict[str, dict[str, str]]:
        """Map ``species_key.lower()`` -> lowercased taxonomy, from train + test categories."""
        index: dict[str, dict[str, str]] = {}
        for safari in (self.train, self.test):
            for category in safari.categories():
                taxonomy = {
                    field.lower(): str(category[field])
                    for field in _TAXONOMY_FIELDS
                    if category.get(field)
                }
                for key in (category.get("name"), category.get("Species")):
                    if key:
                        index.setdefault(str(key).lower(), taxonomy)
        return index

    def freeze(self) -> None:
        """Write the frozen seen species/locations + the test-cell manifest to ``reference_root``."""
        root = self.config.paths.reference_root
        root.mkdir(parents=True, exist_ok=True)
        ref = self.config.reference

        seen_species = sorted(set(self._species(self.train)))
        seen_locations = sorted(self._locations(self.train))
        (root / ref.seen_species_file).write_text(json.dumps(seen_species, indent=2))
        (root / ref.seen_locations_file).write_text(json.dumps(seen_locations, indent=2))

        taxonomy_index = self._taxonomy_index()
        cells: dict[Cell, ManifestCell] = {}
        unmatched: set[str] = set()
        for record in self.test.records():
            cell = self.test.cell_of(record)
            if cell in cells:
                cells[cell].n_videos += 1
                continue
            taxonomy = taxonomy_index.get(cell.species.lower(), {})
            if cell.species and not taxonomy:
                unmatched.add(cell.species)
            cells[cell] = ManifestCell(
                species=cell.species,
                location_id=cell.location_id,
                time=cell.time,
                taxonomy=taxonomy,
                n_videos=1,
            )
        manifest = [c.model_dump() for c in cells.values()]
        (root / ref.manifest_file).write_text(json.dumps(manifest, indent=2))

        note = f" ({len(unmatched)} species without taxonomy)" if unmatched else ""
        print(
            f"froze reference -> {root}: {len(seen_species)} seen species, "
            f"{len(seen_locations)} seen locations, {len(manifest)} test cells{note}"
        )

    def assert_disjoint(self) -> None:
        """Assert the test split is disjoint from train by species AND location.

        Raises:
            AssertionError: If any test species or location also appears in train.
        """
        train_species = {s.lower() for s in self._species(self.train)}
        test_species = {s.lower() for s in self._species(self.test)}
        species_overlap = sorted(train_species & test_species)
        location_overlap = sorted(self._locations(self.train) & self._locations(self.test))
        assert not species_overlap, f"species overlap train∩test: {species_overlap[:10]}"
        assert not location_overlap, f"location overlap train∩test: {location_overlap[:10]}"

    def load(self) -> dict:
        """Read the three frozen files back (for downstream distance features, T2.x).

        Returns:
            ``{"seen_species": [...], "seen_locations": [...], "manifest": [...]}``.
        """
        root = self.config.paths.reference_root
        ref = self.config.reference
        return {
            "seen_species": json.loads((root / ref.seen_species_file).read_text()),
            "seen_locations": json.loads((root / ref.seen_locations_file).read_text()),
            "manifest": json.loads((root / ref.manifest_file).read_text()),
        }


def main() -> None:
    """CLI: ``--check`` (assert disjointness) and/or ``--freeze`` (write the reference)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--check", action="store_true", help="assert train/test disjointness")
    ap.add_argument("--freeze", action="store_true", help="write the frozen reference + manifest")
    args = ap.parse_args()
    ref = Reference(Config.load(args.config))
    if args.check:
        ref.assert_disjoint()
        print("disjointness OK: test species & locations are disjoint from train")
    if args.freeze:
        ref.freeze()


if __name__ == "__main__":
    main()
