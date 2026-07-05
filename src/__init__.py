"""SAM-3 zero-shot transfer prediction — an empirical study, not a tracking method.

Predicts a frozen promptable tracker's (SAM 3) zero-shot performance on unseen species/places —
per-cell ``pDetA`` (detection) and ``pAssA`` (association) — from four label-free distances
(taxonomic, visual, environment, temporal) plus a SAM-3 familiarity proxy, and validates the law
out-of-sample. See ``.claude/CLAUDE.md`` for the spec, ``.claude/IMPLEMENTATION_PLAN.md`` for the
task breakdown, and ``ROADMAP.md`` for the phase summary.
"""

__all__ = ["__version__"]

__version__ = "0.0.0"
