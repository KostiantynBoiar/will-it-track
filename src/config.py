"""Central configuration.

Single source of truth for every constant (paths, data, reference, inference, eval, features,
model, cross-validation). Defaults are overridable via a YAML file (``configs/*.yaml``) or
``SAFARI_*`` environment variables. Every §9 hard constraint from ``.claude/CLAUDE.md`` is a config
toggle here (``keep_hard_negatives``, ``mask_crop``, ``support_weight``, ``log_support_covariate``,
``group_schemes``) so ablations (T5) flip config, not code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# repo root (this file is src/config.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(env_var: str, default: Path) -> Path:
    """Return ``$env_var`` as an expanded path if set, else ``default``."""
    return Path(os.path.expanduser(os.environ.get(env_var, str(default))))


@dataclass
class PathsConfig:
    """Filesystem locations.

    Attributes:
        data_root: SA-FARI media + annotations (``$SAFARI_DATA_ROOT``).
        outputs_root: Derived artifacts — predictions, parquet, models, figures (``$SAFARI_OUTPUTS_ROOT``).
        reference_root: The frozen seen-set (train) reference (``$SAFARI_REFERENCE_ROOT``).
        third_party_root: Vendored SAM 3 + official VEval scorer (``$SAFARI_THIRD_PARTY_ROOT``).
    """

    data_root: Path = field(
        default_factory=lambda: _env_path("SAFARI_DATA_ROOT", _REPO_ROOT / "data")
    )
    outputs_root: Path = field(
        default_factory=lambda: _env_path("SAFARI_OUTPUTS_ROOT", _REPO_ROOT / "outputs")
    )
    reference_root: Path = field(
        default_factory=lambda: _env_path(
            "SAFARI_REFERENCE_ROOT", _REPO_ROOT / "data" / "reference"
        )
    )
    third_party_root: Path = field(
        default_factory=lambda: _env_path("SAFARI_THIRD_PARTY_ROOT", _REPO_ROOT / "third_party")
    )


@dataclass
class DataConfig:
    """SA-FARI dataset facts (see .claude/CLAUDE.md §4).

    Attributes:
        fps: Frame rate the annotations align to (6 fps downsampled).
        annotations_subdir: Subdirectory of ``data_root`` holding the annotation JSONs.
        frames_subdir: Subdirectory of ``data_root`` holding the 6 fps frames.
        train_ann: Train-split ``_ext`` annotation JSON (the seen set / reference).
        test_ann: Test-split ``_ext`` annotation JSON (the transfer probes).
        keep_hard_negatives: Keep queries that return 0 (§9 DON'T filter them).
        hf_repo: Gated Hugging Face dataset id for the annotations.
        gcs_bucket: Public GCS bucket holding the 6 fps frames (anonymous access).
    """

    fps: int = 6
    annotations_subdir: str = "annotations"
    frames_subdir: str = "frames"
    train_ann: str = "sa_fari_train_ext.json"
    test_ann: str = "sa_fari_test_ext.json"
    keep_hard_negatives: bool = True
    hf_repo: str = "facebook/SA-FARI"
    gcs_bucket: str = "cxl-public-camera-trap"


@dataclass
class ReferenceConfig:
    """The frozen seen/unseen reference (T0.2). Filenames resolved under ``paths.reference_root``.

    Attributes:
        seen_species_file: Frozen train-split species set.
        seen_locations_file: Frozen train-split location set.
        manifest_file: Per-test-cell manifest (taxonomy + location_id + timestamps).
    """

    seen_species_file: str = "seen_species.json"
    seen_locations_file: str = "seen_locations.json"
    manifest_file: str = "cell_manifest.json"


@dataclass
class InferenceConfig:
    """Frozen SAM 3 promptable inference (T1.1).

    Attributes:
        prompt_mode: ``"species"`` (primary) or ``"generic"`` ("animal", robustness).
        sam3_weights: Path/id of the frozen SAM 3 checkpoint.
        device: Torch device for inference.
        batch_frames: Frames per inference batch.
    """

    prompt_mode: str = "species"
    sam3_weights: str = ""
    device: str = "cuda"
    batch_frames: int = 16


@dataclass
class EvalConfig:
    """Scoring with the OFFICIAL VEval evaluator (T1.2).

    Attributes:
        metrics: Reported metrics (computed by veval; the metric is never re-implemented).
        veval_module: Import path into the vendored VEval scorer.
    """

    metrics: tuple[str, ...] = ("pHOTA", "pDetA", "pAssA")
    veval_module: str = "veval"


@dataclass
class FeaturesConfig:
    """Label-free distance features (T2.x).

    Attributes:
        visual_encoder: ``"dinov2"`` (primary) or ``"clip"`` (robustness, T5.2).
        mask_crop: Crop visual embeddings to the animal mask (§9 DO); T5.2 toggles it off.
        scene_encoder: Encoder for the environment/background embedding.
        taxonomic_levels: 7-level taxonomy (Kingdom -> Species) for LCA tree distance.
        distance_variant: ``"nearest_prototype"`` (primary), or ``"frechet"`` / ``"mmd"`` (robustness).
        night_ir_from_color: Derive the day/night-IR flag from colour statistics.
    """

    visual_encoder: str = "dinov2"
    mask_crop: bool = True
    scene_encoder: str = "dinov2"
    taxonomic_levels: tuple[str, ...] = (
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    )
    distance_variant: str = "nearest_prototype"
    night_ir_from_color: bool = True


@dataclass
class ModelConfig:
    """Per-target regression on the distances (T3.1).

    Attributes:
        family: ``"beta"`` / logit-link GLM on bounded scores.
        support_weight: Weight observations by support (§9).
        support_col: Support column used for weighting and the covariate.
        log_support_covariate: Add ``log(n_frames)`` so "rare" is never mistaken for "far" (§9).
    """

    family: str = "beta"
    support_weight: bool = True
    support_col: str = "n_frames"
    log_support_covariate: bool = True


@dataclass
class CVConfig:
    """Group-aware cross-validation + bootstrap (T4).

    Attributes:
        group_schemes: Whole-group hold-out schemes (leave-species-out, leave-location-out) (§9).
        n_bootstrap: Bootstrap resamples for confidence intervals.
    """

    group_schemes: tuple[str, ...] = ("species", "location")
    n_bootstrap: int = 1000


@dataclass
class Config:
    """Top-level configuration aggregating every section.

    Attributes:
        paths: Filesystem locations.
        data: SA-FARI dataset facts.
        reference: Frozen seen/unseen reference.
        inference: SAM 3 inference options.
        eval: VEval scoring options.
        features: Distance-feature options.
        model: Regression options.
        cv: Cross-validation + bootstrap options.
        seed: Random seed.
        raw: Raw parsed YAML, for any extra keys.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    seed: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Config:
        """Build a config from defaults, then overlay a YAML file if given.

        Args:
            path: Optional YAML config path; defaults-only when ``None``.

        Returns:
            The resolved :class:`Config`.
        """
        cfg = cls()
        if path is not None:
            with open(path) as f:
                cfg.raw = yaml.safe_load(f) or {}
            cfg._apply(cfg.raw)
        return cfg

    def _apply(self, raw: dict[str, Any]) -> None:
        """Overlay parsed YAML onto the nested sections in place.

        Args:
            raw: Parsed YAML mapping; recognised sections mirror the dataclass field names, plus a
                top-level ``seed``.
        """
        sections = (
            "paths",
            "data",
            "reference",
            "inference",
            "eval",
            "features",
            "model",
            "cv",
        )
        for section in sections:
            values = raw.get(section)
            if not values:
                continue
            sub = getattr(self, section)
            for key, value in values.items():
                if not hasattr(sub, key):
                    continue
                if section == "paths":
                    value = Path(os.path.expanduser(str(value)))
                setattr(sub, key, value)
        if "seed" in raw:
            self.seed = int(raw["seed"])
