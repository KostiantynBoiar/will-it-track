"""BURST animal-class → biological taxonomy, for the R7 replication adapter.

Maps BURST/LVIS class names to a (capitalized) 7-level taxonomy so the existing taxonomic distance works
unchanged. Two sources, the CSV taking precedence: a curated ``data/burst/class_taxonomy.csv`` and
(optionally) the SA-FARI ``categories`` themselves --- the same shared LVIS vocabulary the main study used ---
so BURST animals inherit the study's own taxonomy. A name with no resolvable taxonomy is simply absent, which
is exactly how the adapter decides a class is *not* an animal to keep. Coarse classes (``bird`` → only
``Aves``) resolve to a partial path, which the loader's full-7-level filter turns into an honest ``NaN``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.config import Config
from src.dataset import _TAXONOMY_FIELDS, SAFARI, _is_real

_PACKAGED_CSV = Path(__file__).parent / "class_taxonomy.csv"  # the curated base map (tracked with the code)


class BurstTaxonomy:
    """Resolve a BURST class name to its capitalized ``level → value`` taxonomy path."""

    def __init__(self, config: Config | None = None) -> None:
        """Load the taxonomy map (SA-FARI seed first, CSV overrides on top)."""
        self.config = config or Config()
        self._by_name: dict[str, dict[str, str]] = {}
        if self.config.burst.seed_taxonomy_from_safari:
            self._seed_from_safari()
        self._load_csv()

    def _seed_from_safari(self) -> None:
        """Fill ``name → taxonomy`` from the SA-FARI categories (skipped if the annotations are absent)."""
        try:
            categories = SAFARI("test", self.config).categories()
        except FileNotFoundError:
            return
        for category in categories:
            name = str(category.get("name", "")).strip().lower()
            tax = {f: str(category[f]) for f in _TAXONOMY_FIELDS if _is_real(category.get(f))}
            if name and tax and name not in self._by_name:
                self._by_name[name] = tax

    def _load_csv(self) -> None:
        """Overlay the curated CSV (takes precedence over the SA-FARI seed).

        Uses the packaged base map by default; a non-empty ``burst.taxonomy_csv`` overrides it with a
        ``data_root``-relative file (e.g. a fuller BURST animal list added at acquisition, or a test fixture).
        """
        csv_cfg = self.config.burst.taxonomy_csv
        path = (self.config.paths.data_root / csv_cfg) if csv_cfg else _PACKAGED_CSV
        if not path.exists():
            return
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name", "")).strip().lower()
                tax = {f: str(row[f]).strip() for f in _TAXONOMY_FIELDS if _is_real(row.get(f))}
                if name:
                    self._by_name[name] = tax  # CSV overrides the SA-FARI seed

    def taxonomy_of(self, name: str) -> dict[str, str]:
        """Capitalized ``level → value`` for a class name (real levels only; empty dict if unknown)."""
        return dict(self._by_name.get(str(name).strip().lower(), {}))

    def animal_names(self) -> set[str]:
        """Lowercased class names with any resolvable taxonomy --- the animal subset the adapter keeps."""
        return set(self._by_name)
