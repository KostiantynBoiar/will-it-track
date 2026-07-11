"""Analysis splits — the two experiments over the pooled SA-FARI probes.

A :class:`Partition` is a frozen reference/probe assignment on one *held axis*. Species identity is the
``category_id`` throughout.

- **Split A — species hold-out (primary, H1):** ``held_axis="species"``, ``loso=True``. Reference and
  probe are both the pooled present set; each probe species' distance excludes itself (leave-one-species
  -out). Environment stays ~familiar.
- **Split B — location hold-out (secondary, H2):** ``held_axis="location"``, ``loso=False``. Reference =
  the official train split, probe = test; species distance ≈ 0, environment/temporal vary.

Because SAM 3 is frozen and zero-shot on all of SA-FARI, a partition is only our reference anchor for
distances, so re-partitioning is sound.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel

from src.config import Config
from src.dataset import _TAXONOMY_FIELDS, SAFARI, VideoRecord, _is_real, _year, pooled_records
from src.features.taxonomic import tree_distance

_N_LEVELS = len(_TAXONOMY_FIELDS)


def _full_taxonomy(taxonomy: dict[str, str] | None) -> bool:
    """True if a category has all 7 real taxonomy levels (usable for a clean tree distance)."""
    return bool(taxonomy) and len(taxonomy) == _N_LEVELS


def taxonomy_path(taxonomy: dict[str, str]) -> list[str]:
    """Order a (lowercased-key) taxonomy dict into a kingdom→species path for :func:`tree_distance`."""
    return [taxonomy.get(field.lower(), "") for field in _TAXONOMY_FIELDS]


class Partition(BaseModel):
    """A frozen reference/probe assignment for one experiment (species identity = ``category_id``).

    Attributes:
        name: File-stem identifier (e.g. ``"species_holdout"``).
        held_axis: The disjoint axis — ``"species"`` (Split A) or ``"location"`` (Split B).
        loso: Leave-one-species-out — each probe species excludes itself from the reference (Split A).
        reference_species: Reference ``category_id``s.
        probe_species: Probe ``category_id``s.
        reference_locations: Reference ``location_id``s.
        probe_locations: Probe ``location_id``s.
        reference_years: Distinct reference footage years (for the temporal gap).
        probe_origins: Which source files the probe records come from (``["test"]`` / both).
    """

    name: str
    held_axis: str
    loso: bool
    reference_species: list[str]
    probe_species: list[str]
    reference_locations: list[str]
    probe_locations: list[str]
    reference_years: list[str]
    probe_origins: list[str]


def _present(records: list[VideoRecord]) -> list[VideoRecord]:
    """Records with at least one ground-truth masklet (positive probes)."""
    return [r for r in records if r.num_masklets > 0]


def _years(records: list[VideoRecord]) -> list[str]:
    """Sorted distinct footage years present in a record list."""
    return sorted({_year(r.creation_datetime) for r in records if _year(r.creation_datetime)})


def _locations(records: list[VideoRecord]) -> list[str]:
    """Sorted distinct real ``location_id``s (drops missing/``nan``)."""
    return sorted({r.location_id for r in records if _is_real(r.location_id)})


def build_species_partition(config: Config | None = None) -> Partition:
    """Split A — leave-one-species-out over the pooled present set."""
    cfg = config or Config()
    records = _present(pooled_records(cfg))
    species = sorted({r.category_id for r in records})
    locations = _locations(records)
    return Partition(
        name="species_holdout",
        held_axis="species",
        loso=True,
        reference_species=species,
        probe_species=species,
        reference_locations=locations,
        probe_locations=locations,
        reference_years=_years(records),
        probe_origins=["train", "test"],
    )


def build_location_partition(config: Config | None = None) -> Partition:
    """Split B — the official train (reference) vs test (probe) location hold-out."""
    cfg = config or Config()
    ref = _present(SAFARI("train", cfg).records())
    probe = _present(SAFARI("test", cfg).records())
    return Partition(
        name="location_holdout",
        held_axis="location",
        loso=False,
        reference_species=sorted({r.category_id for r in ref}),
        probe_species=sorted({r.category_id for r in probe}),
        reference_locations=_locations(ref),
        probe_locations=_locations(probe),
        reference_years=_years(ref),
        probe_origins=["test"],
    )


def save(partition: Partition, config: Config | None = None) -> Path:
    """Persist a partition to ``paths.splits_root/<name>.json``."""
    cfg = config or Config()
    cfg.paths.splits_root.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.splits_root / f"{partition.name}.json"
    path.write_text(partition.model_dump_json(indent=2))
    return path


def load(name: str, config: Config | None = None) -> Partition:
    """Load a persisted partition by name."""
    cfg = config or Config()
    return Partition.model_validate_json((cfg.paths.splits_root / f"{name}.json").read_text())


def probe_records(partition: Partition, config: Config | None = None) -> list[VideoRecord]:
    """Pooled records assigned to the partition's probe side (present species only)."""
    cfg = config or Config()
    probe_species = set(partition.probe_species)
    origins = set(partition.probe_origins)
    return [
        r
        for r in _present(pooled_records(cfg))
        if r.category_id in probe_species and r.origin in origins
    ]


def reference_records(partition: Partition, config: Config | None = None) -> list[VideoRecord]:
    """Pooled records on the partition's reference side (present species only).

    Split A (``held_axis="species"``) draws from both origins — all present species (leave-one-species-out
    is applied later at prototype selection, not here). Split B (``held_axis="location"``) draws only from
    ``train`` (the seen locations).
    """
    cfg = config or Config()
    ref_species = set(partition.reference_species)
    origins = {"train", "test"} if partition.held_axis == "species" else {"train"}
    return [
        r
        for r in _present(pooled_records(cfg))
        if r.category_id in ref_species and r.origin in origins
    ]


def structural_report(config: Config | None = None) -> dict:
    """Sanity report for Split A: present counts, LOSO taxonomic spread, location decoupling."""
    cfg = config or Config()
    taxonomy = SAFARI("test", cfg).taxonomy()  # category_id → taxonomy (identical across splits)
    records = _present(pooled_records(cfg))
    present = sorted({r.category_id for r in records})
    full = [cid for cid in present if _full_taxonomy(taxonomy.get(cid))]
    paths = {cid: taxonomy_path(taxonomy[cid]) for cid in full}

    dist: Counter[int] = Counter()
    for cid in full:
        others = [p for other, p in paths.items() if other != cid]
        dist[min(tree_distance(paths[cid], p) for p in others)] += 1

    species_locations: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if _is_real(r.location_id):
            species_locations[r.category_id].add(r.location_id)
    location_species: dict[str, set[str]] = defaultdict(set)
    for cid, locs in species_locations.items():
        for loc in locs:
            location_species[loc].add(cid)
    shared_fracs = [
        sum(1 for loc in locs if location_species[loc] - {cid}) / len(locs)
        for cid, locs in species_locations.items()
        if locs
    ]

    return {
        "present_species": len(present),
        "full_taxonomy_species": len(full),
        "loso_taxonomic_distribution": dict(sorted(dist.items())),
        "n_locations": len(_locations(records)),
        "location_decoupling": round(statistics.mean(shared_fracs), 3) if shared_fracs else 0.0,
    }
