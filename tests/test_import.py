"""Scaffold sanity — the package imports cleanly and the config parses.

Passes at scaffold time (stubs raise ``NotImplementedError`` only when *called*, not on import).
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.types import Cell, Support


def test_import_package() -> None:
    """The top-level package imports and exposes a version."""
    import src

    assert src.__version__


def test_import_all_modules() -> None:
    """Every module imports without error (no import-time side effects in the stubs)."""
    import src.analysis.ablations  # noqa: F401
    import src.analysis.cross_val  # noqa: F401
    import src.analysis.regression  # noqa: F401
    import src.analysis.reliability  # noqa: F401
    import src.analysis.uncertainty  # noqa: F401
    import src.analysis.variance  # noqa: F401
    import src.dataset  # noqa: F401
    import src.eval.score  # noqa: F401
    import src.features.assemble  # noqa: F401
    import src.features.environment  # noqa: F401
    import src.features.familiarity  # noqa: F401
    import src.features.taxonomic  # noqa: F401
    import src.features.temporal  # noqa: F401
    import src.features.visual  # noqa: F401
    import src.inference.harness  # noqa: F401
    import src.io  # noqa: F401
    import src.reference  # noqa: F401


def test_types() -> None:
    """The core record types construct."""
    cell = Cell(species="impala", location_id="loc0", time="2020")
    assert cell.species == "impala"
    assert Support(n_frames=10, n_masklets=2, n_videos=1).n_frames == 10


def test_default_config() -> None:
    """Dataclass defaults are sane."""
    cfg = Config()
    assert cfg.data.fps == 6
    assert cfg.model.family == "beta"
    assert cfg.data.keep_hard_negatives is True


def test_config_yaml_overlay() -> None:
    """configs/default.yaml overlays onto the config without error."""
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = Config.load(path)
    assert cfg.features.mask_crop is True
    assert cfg.model.log_support_covariate is True
    assert list(cfg.cv.group_schemes) == ["species", "location"]
