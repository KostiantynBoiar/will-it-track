"""Class-name → 7-level biological taxonomy, shared by dataset adapters.

Loads a CSV (name + the 7 capitalized levels Kingdom…Species) into a normalised lookup so the existing
taxonomic distance works unchanged. Names are normalised (LVIS-style _(disambiguation) suffixes dropped,
underscores→spaces) so dataset-specific spellings match. A name with no resolvable taxonomy is simply absent,
which is also how an adapter decides a class is not an animal to keep.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from src.dataset import _TAXONOMY_FIELDS, _is_real


def normalise_name(name: str) -> str:
    """Lookup key: drop a trailing _(...) and turn underscores into spaces (red_deer → red deer)."""
    n = str(name).strip().lower()
    n = re.sub(r"_\([^)]*\)$", "", n)
    return n.replace("_", " ").strip()


class ClassTaxonomy:
    """Resolve a class name to its capitalized level → value taxonomy path from a CSV."""

    def __init__(self, csv_path: str | Path) -> None:
        """Load csv_path (name,Kingdom,…,Species) into a normalised name → taxonomy map."""
        self._by_name: dict[str, dict[str, str]] = {}
        path = Path(csv_path)
        if not path.exists():
            return
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = normalise_name(row.get("name", ""))
                tax = {fld: str(row[fld]).strip() for fld in _TAXONOMY_FIELDS if _is_real(row.get(fld))}
                if name:
                    self._by_name[name] = tax

    def taxonomy_of(self, name: str) -> dict[str, str]:
        """Capitalized level → value for a class name (real levels only; empty dict if unknown)."""
        return dict(self._by_name.get(normalise_name(name), {}))

    def names(self) -> set[str]:
        """Normalised class names with a resolvable taxonomy."""
        return set(self._by_name)
