"""Central configuration.

Single source of truth for every constant (paths, data, reference, inference, eval, features,
model, cross-validation). Defaults are overridable via a YAML file (``configs/*.yaml``) or
``SAFARI_*`` environment variables. Every hard constraint is a config toggle here
(``keep_hard_negatives``, ``mask_crop``, ``support_weight``, ``log_support_covariate``,
``group_schemes``) so ablations flip config, not code.

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
    """SA-FARI dataset facts.

    Attributes:
        fps: Frame rate the annotations align to (6 fps downsampled).
        annotations_subdir: Subdirectory of ``data_root`` holding the annotation JSONs.
        frames_subdir: Subdirectory of ``data_root`` holding the 6 fps frames.
        train_ann: Train-split ``_ext`` annotation JSON (the seen set / reference).
        test_ann: Test-split ``_ext`` annotation JSON (the transfer probes).
        keep_hard_negatives: Keep queries that return 0 (do not filter them).
        hf_repo: Gated Hugging Face dataset id for the annotations.
        gcs_bucket: Public GCS bucket holding the 6 fps frames (anonymous access).
        frames_gcs_dir: Bucket-relative frame root (``{split}``-templated); ``file_names`` append to it.
    """

    fps: int = 6
    annotations_subdir: str = "annotations"
    frames_subdir: str = "frames"
    train_ann: str = "sa_fari_train_ext.json"
    test_ann: str = "sa_fari_test_ext.json"
    keep_hard_negatives: bool = True
    hf_repo: str = "facebook/SA-FARI"
    gcs_bucket: str = "cxl-public-camera-trap"
    frames_gcs_dir: str = "sa_fari/sa_fari_{split}/JPEGImages_6fps"
    download_workers: int = 16  # parallel GCS frame downloads (I/O-bound; 1 = serial)


class ReferenceConfig(BaseModel):
    """The frozen reference. Files resolved under ``paths.reference_root/<split-name>/``.

    Attributes:
        reference_species_file: Frozen reference species set (``category_id``s) for the split.
        reference_locations_file: Frozen reference location set for the split.
        manifest_file: Per-probe-cell manifest (category_id + taxonomy + location_id + timestamps).
    """

    reference_species_file: str = "reference_species.json"
    reference_locations_file: str = "reference_locations.json"
    manifest_file: str = "cell_manifest.json"


class SplitsConfig(BaseModel):
    """Analysis-split construction.

    Attributes:
        seed: Seed for any stochastic split choices (the LOSO protocol is deterministic).
        min_present_videos: Minimum positive videos for a species to enter the present set.
    """

    seed: int = 0
    min_present_videos: int = 1


class InferenceConfig(BaseModel):
    """Frozen SAM 3 promptable inference.

    Attributes:
        sam3_model: Hugging Face id (or local path) of the frozen SAM 3 checkpoint.
        prompt_mode: ``"species"`` (primary) or ``"generic"`` ("animal", robustness).
        device: Torch device for inference.
        precision: ``"bf16"`` / ``"fp16"`` / ``"fp32"`` (tune to the GPU tier).
        batch_frames: Frames per inference batch.
        score_threshold: Minimum confidence to keep a predicted masklet; ``0.0`` writes raw scores so
            VEval (not the harness) owns the operating point (its own ``prob_thresh`` decides HOTA).
        predictions_subdir: Where per-video prediction JSONs are written, under ``paths.outputs_root``.
        max_videos_per_species: Cap on present (positive) videos scored per species — a stratified
            sample so the huge train split (~31k probes) yields the species hold-out at feasible cost;
            ``None`` runs every probe. Hard negatives are skipped when capping (positives-only H1 fit).
    """

    sam3_model: str = "facebook/sam3"
    prompt_mode: str = "species"
    device: str = "cuda"
    precision: str = "bf16"
    batch_frames: int = 16
    score_threshold: float = 0.0
    predictions_subdir: str = "predictions"
    max_videos_per_species: int | None = None


class EvalConfig(BaseModel):
    """Scoring with the OFFICIAL VEval evaluator (never re-implemented).

    Attributes:
        metrics: Reported metrics (computed by the vendored VEval scorer).
        veval_script: Path to the vendored evaluator, under ``paths.third_party_root``.
        prefer_bbox: Read VEval's ``bbox_*`` HOTA metrics before ``mask_*``. On for a box-only dataset
            (e.g. MammAlps), whose GT is emitted as filled-rectangle masks: a tight predicted mask vs a
            rectangle GT would deflate mask-IoU, so box-IoU HOTA (``bbox_*``) is the honest score there.
    """

    metrics: tuple[str, ...] = ("pHOTA", "pDetA", "pAssA")
    veval_script: str = "sam3/sam3/eval/saco_veval_eval.py"  # <clone>/sam3/eval/... (repo dir + package)
    prefer_bbox: bool = False


class FeaturesConfig(BaseModel):
    """Label-free distance features.

    Attributes:
        visual_encoder: ``"dinov2"`` (primary) or ``"clip"`` (robustness).
        mask_crop: Crop visual embeddings to the animal mask; robustness ablation toggles it off.
        scene_encoder: Encoder for the environment/background embedding.
        taxonomic_levels: 7-level taxonomy (Kingdom -> Species) for LCA tree distance.
        distance_variant: ``"nearest_prototype"`` (primary), or ``"frechet"`` / ``"mmd"`` (robustness).
        night_ir_from_color: Derive the day/night-IR flag from colour statistics.
        n_frames_per_masklet: Annotated frames sampled per masklet for embedding.
        max_masklets_per_species: Cap on masklets embedded per species (bounds the prototype cost).
        max_frames_per_location: Cap on masked-background frames embedded per location.
        embed_batch: Images per forward pass.
        embed_crop_chunk: Crops loaded + embedded per chunk — bounds peak RAM (only this many decoded
            frames are ever alive at once), independent of the sampling caps.
        embed_load_workers: Threads that load + crop frames per chunk in parallel (the embedding path
            is I/O-bound on frame reads, not the GPU); ``1`` = serial.
        embed_device: Torch device for embedding (``"mps"``/``"cuda"``/``"cpu"``; falls back to CPU).
        min_mask_pixels: Smallest mask (in pixels) that yields a usable animal crop.
        background_fill: How to neutralise the animal in the scene embedding (``"mean"``/``"zero"``).
        night_ir_threshold: Mean channel-spread (0-255) below which a frame counts as night/IR.
        embeddings_subdir: Cache dir for embeddings, under ``paths.outputs_root``.
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
    n_frames_per_masklet: int = 5
    max_masklets_per_species: int = 40
    max_frames_per_location: int = 200
    embed_batch: int = 32
    embed_crop_chunk: int = 256
    embed_load_workers: int = 16
    embed_device: str = "mps"
    min_mask_pixels: int = 64
    background_fill: str = "mean"
    night_ir_threshold: float = 12.0
    embeddings_subdir: str = "embeddings"


class ModelConfig(BaseModel):
    """Per-target regression on the distances.

    Attributes:
        family: ``"beta"`` / logit-link GLM on bounded scores.
        support_weight: Weight observations by support.
        support_col: Support column used for weighting and the covariate.
        log_support_covariate: Add ``log(n_frames)`` so "rare" is never mistaken for "far".
        cluster_ci: Report group-cluster-bootstrap coefficient CIs (refit per resample) instead of the
            naive model CIs. The naive CIs are anti-conservative here — ``var_weights`` inflates the
            effective N and the predictors are constant within species/location (pseudo-replication) —
            so the honest interval resamples whole groups. Naive CIs are still written alongside.
        cluster_cols: Grouping columns bootstrapped over; the reported CI is the conservative envelope
            across them (widest), since novelty predictors vary per species and scene ones per location.
        control_size: Add ``log_area`` (mean GT mask-area per species) as a nuisance covariate. Off by
            default so committed fits are unchanged; turned on for the confound ablation that tests whether
            the (wrong-signed) visual-distance effect is really an animal-size artefact.
    """

    family: str = "beta"
    support_weight: bool = True
    support_col: str = "n_frames"
    log_support_covariate: bool = True
    cluster_ci: bool = True
    cluster_cols: tuple[str, ...] = ("category_id", "location_id")
    control_size: bool = False


class CVConfig(BaseModel):
    """Group-aware cross-validation + bootstrap.

    Attributes:
        group_schemes: Whole-group hold-out schemes (leave-species-out, leave-location-out).
        n_bootstrap: Bootstrap resamples for confidence intervals.
    """

    group_schemes: tuple[str, ...] = ("species", "location")
    n_bootstrap: int = 1000


class MammAlpsConfig(BaseModel):
    """MammAlps → SA-Co adapter (R7 independent replication; box-only Alpine camera-trap tracker).

    MammAlps ships per-clip JSONs of bounding-box tracks (5 species, 9 cameras / 3 sites, 30 fps MP4s, no
    masks). The adapter emits each box as a filled-rectangle RLE (segmentations) plus the native box (bboxes),
    subsamples the 30 fps clips to ``target_fps`` and caps per ``(species, camera)`` cell so the SAM 3
    inference stays tractable; scoring uses ``eval.prefer_bbox`` (box-HOTA).

    Attributes:
        raw_dir: Directory of the extracted dense-annotation JSONs, relative to ``data_root``.
        taxonomy_csv: Optional override for the packaged ``src/adapters/mammalps_taxonomy.csv``.
        target_fps: Subsample the 30 fps clips to this rate (step = round(clip_fps / target_fps)).
        max_frames_per_clip: Cap on kept (subsampled) frames per clip.
        max_clips_per_cell: Cap on clips per ``(species, camera)`` cell (bounds red-deer dominance).
        min_frames: Skip clips with fewer than this many kept frames.
        location_by: ``"site_cam"`` (S1_C1 …, the cell/location key) or ``"site"`` (S1/S2/S3, coarser).
    """

    raw_dir: str = "dense"
    taxonomy_csv: str = ""  # empty ⇒ packaged src/adapters/mammalps_taxonomy.csv
    target_fps: float = 3.0
    max_frames_per_clip: int = 120
    max_clips_per_cell: int = 12
    min_frames: int = 4
    location_by: str = "site_cam"


class BurstConfig(BaseModel):
    """BURST → SA-Co adapter (R7 many-species replication; mask-native LVIS-class tracker).

    BURST ships one ``all_classes.json`` per split: a list of ``sequences``, each carrying a source
    ``dataset`` + ``seq_name`` (locating the TAO frames), ``annotated_image_paths``, per-frame COCO-RLE
    ``segmentations`` keyed by track id, and ``track_category_ids`` (track → LVIS category). The adapter
    keeps only the animal categories (those in the packaged taxonomy CSV — a WordNet-derived whitelist),
    emits the RLE masks directly (mask-native, so ``eval.prefer_bbox`` stays off), and caps videos per
    category so SAM 3 inference stays tractable. Location is the per-video ``seq_name`` (no site metadata),
    time is empty, so only leave-species-out is meaningful (``cv.group_schemes = ("species",)``).

    Attributes:
        ann_file: The extracted BURST annotation JSON (relative to ``annotations_subdir``).
        taxonomy_csv: Optional override for the packaged ``src/adapters/burst_taxonomy.csv``.
        max_videos_per_category: Cap on sequences per animal category (bounds dog/person dominance).
        max_frames_per_video: Cap on kept annotated frames per sequence.
        min_frames: Skip sequences with fewer than this many annotated frames of the target category.
    """

    ann_file: str = "val_all_classes.json"
    taxonomy_csv: str = ""  # empty ⇒ packaged src/adapters/burst_taxonomy.csv
    max_videos_per_category: int = 8
    max_frames_per_video: int = 20
    min_frames: int = 2


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
        experiment: Experiment tag for the exporter's output filenames (``"location"`` keeps the
            unsuffixed names; e.g. ``"species"`` writes ``coefficients_species.tex``), so two experiments'
            artefacts sit side by side without clobbering. Overridable via ``SAFARI_EXPERIMENT``.
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
    mammalps: MammAlpsConfig = Field(default_factory=MammAlpsConfig)
    burst: BurstConfig = Field(default_factory=BurstConfig)
    seed: int = 0
    experiment: str = "location"

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
