"""Central configuration.

Single source of truth for every constant (paths, data, reference, inference, eval, features,
model, cross-validation). Defaults are overridable via a YAML file (``configs/*.yaml``) or
``SAFARI_*`` environment variables. Every §9 hard constraint from ``.claude/CLAUDE.md`` is a config
toggle here (``keep_hard_negatives``, ``mask_crop``, ``support_weight``, ``log_support_covariate``,
``group_schemes``) so ablations (T5) flip config, not code.

Pydantic v2 + pydantic-settings: any field is overridable via a ``SAFARI_*`` env var, nested with
``__`` (e.g. ``SAFARI_DATA__FPS``, ``SAFARI_PATHS__DATA_ROOT``, ``SAFARI_SEED``). ``Config.load(path)``
reads a YAML file over the defaults; env vars fill anything the YAML omits.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root (this file is src/config.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]


class PathsConfig(BaseModel):
    """Filesystem locations.

    Attributes:
        data_root: SA-FARI media + annotations.
        outputs_root: Derived artifacts — predictions, parquet, models, figures.
        reference_root: The frozen reference artifacts (per split).
        splits_root: The persisted analysis splits (species / location partitions).
        third_party_root: Vendored SAM 3 + official VEval scorer.
    """

    data_root: Path = _REPO_ROOT / "data"
    outputs_root: Path = _REPO_ROOT / "outputs"
    reference_root: Path = _REPO_ROOT / "data" / "reference"
    splits_root: Path = _REPO_ROOT / "data" / "splits"
    third_party_root: Path = _REPO_ROOT / "third_party"


class DataConfig(BaseModel):
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


class ReferenceConfig(BaseModel):
    """The frozen reference (T0.2). Files resolved under ``paths.reference_root/<split-name>/``.

    Attributes:
        reference_species_file: Frozen reference species set (``category_id``s) for the split.
        reference_locations_file: Frozen reference location set for the split.
        manifest_file: Per-probe-cell manifest (category_id + taxonomy + location_id + timestamps).
    """

    reference_species_file: str = "reference_species.json"
    reference_locations_file: str = "reference_locations.json"
    manifest_file: str = "cell_manifest.json"


class SplitsConfig(BaseModel):
    """Analysis-split construction (T0.3).

    Attributes:
        seed: Seed for any stochastic split choices (the LOSO protocol is deterministic).
        min_present_videos: Minimum positive videos for a species to enter the present set.
    """

    seed: int = 0
    min_present_videos: int = 1


class InferenceConfig(BaseModel):
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


class EvalConfig(BaseModel):
    """Scoring with the OFFICIAL VEval evaluator (T1.2).

    Attributes:
        metrics: Reported metrics (computed by veval; the metric is never re-implemented).
        veval_module: Import path into the vendored VEval scorer.
    """

    metrics: tuple[str, ...] = ("pHOTA", "pDetA", "pAssA")
    veval_module: str = "veval"


class FeaturesConfig(BaseModel):
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


class ModelConfig(BaseModel):
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


class CVConfig(BaseModel):
    """Group-aware cross-validation + bootstrap (T4).

    Attributes:
        group_schemes: Whole-group hold-out schemes (leave-species-out, leave-location-out) (§9).
        n_bootstrap: Bootstrap resamples for confidence intervals.
    """

    group_schemes: tuple[str, ...] = ("species", "location")
    n_bootstrap: int = 1000


class Config(BaseSettings):
    """Top-level configuration aggregating every section.

    Any field is overridable via a ``SAFARI_*`` env var (nested with ``__``), e.g.
    ``SAFARI_DATA__FPS=8`` or ``SAFARI_PATHS__DATA_ROOT=/mnt/safari``.

    Attributes:
        paths: Filesystem locations.
        data: SA-FARI dataset facts.
        reference: Frozen reference artifacts (per split).
        splits: Analysis-split construction.
        inference: SAM 3 inference options.
        eval: VEval scoring options.
        features: Distance-feature options.
        model: Regression options.
        cv: Cross-validation + bootstrap options.
        seed: Random seed.
    """

    model_config = SettingsConfigDict(
        env_prefix="SAFARI_", env_nested_delimiter="__", extra="ignore"
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Build a config from defaults + ``SAFARI_*`` env vars, overlaying a YAML file if given.

        Args:
            path: Optional YAML config path; defaults + env only when ``None``.

        Returns:
            The resolved :class:`Config` (YAML sections coerced into the sub-models; extras ignored).
        """
        if path is None:
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
