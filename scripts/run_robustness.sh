#!/usr/bin/env bash
# R6 design-robustness sweep — build each variant's feature table on the GPU pod, then run the driver.
#
# Each variant swaps ONE design choice and writes outputs/features_<key>.parquet (the driver reads those):
#   clip        — DINOv2 -> CLIP visual+scene encoder            (re-embed only, reuse baseline scores)
#   wholeframe  — mask-crop -> whole-frame bbox (background kept) (re-embed only, reuse baseline scores)
#   generic     — species prompt -> generic "animal"             (fresh SAM 3 inference: new Y)
#   frechet/mmd — nearest-prototype -> distributional distance    (re-embed only, reuse baseline scores)
#
# Prereqs (see docs/runpod_runbook.md): env set up, third_party/sam3 cloned, annotations acquired, and a
# baseline outputs/scores.parquet present (species-prompt scoring). Assumes SAFARI_PATHS__DATA_ROOT and a
# base SAFARI_PATHS__OUTPUTS_ROOT (=$OUT_BASE) are exported, plus SAFARI_FEATURES__EMBED_DEVICE=cuda.
set -euo pipefail

OUT_BASE="${SAFARI_PATHS__OUTPUTS_ROOT:-/workspace/outputs}"
BASE_SCORES="$OUT_BASE/scores.parquet"
[[ -f "$BASE_SCORES" ]] || { echo "error: baseline $BASE_SCORES missing — run the species-prompt inference+score first" >&2; exit 1; }

# Build one variant: run assemble under an isolated outputs_root, then copy its features to the driver's name.
# Args: <key> <extra SAFARI_* env assignments...>
build_variant() {
    local key="$1"; shift
    local vout="$OUT_BASE/out_$key"
    echo "==> building variant '$key' -> $vout"
    mkdir -p "$vout/embeddings"
    cp "$BASE_SCORES" "$vout/scores.parquet"                      # reuse the baseline target Y
    # share the baseline DINOv2 embedding caches where the encoder/crop match (skips redundant embedding)
    cp -n "$OUT_BASE"/embeddings/*.npz "$vout/embeddings/" 2>/dev/null || true
    env "$@" SAFARI_PATHS__OUTPUTS_ROOT="$vout" SAFARI_FEATURES__EMBED_DEVICE=cuda \
        python -m src.features.assemble --partition location
    cp "$vout/features.parquet" "$OUT_BASE/features_$key.parquet"
}

# --- variants that reuse the baseline scores (re-embed only) ---------------------------------------------
build_variant clip       SAFARI_FEATURES__VISUAL_ENCODER=clip SAFARI_FEATURES__SCENE_ENCODER=clip
build_variant wholeframe SAFARI_FEATURES__MASK_CROP=false
build_variant frechet    SAFARI_FEATURES__DISTANCE_VARIANT=frechet
build_variant mmd        SAFARI_FEATURES__DISTANCE_VARIANT=mmd

# --- generic prompt: needs a FRESH SAM 3 inference + scoring (new Y), then assemble ----------------------
GOUT="$OUT_BASE/out_generic"
echo "==> building variant 'generic' (fresh inference) -> $GOUT"
mkdir -p "$GOUT/embeddings"
cp -n "$OUT_BASE"/embeddings/*.npz "$GOUT/embeddings/" 2>/dev/null || true
env SAFARI_PATHS__OUTPUTS_ROOT="$GOUT" SAFARI_INFERENCE__PROMPT_MODE=generic \
    SAFARI_INFERENCE__PRECISION="${SAFARI_INFERENCE__PRECISION:-bf16}" \
    python -m src.inference.harness --split test
env SAFARI_PATHS__OUTPUTS_ROOT="$GOUT" SAFARI_INFERENCE__PROMPT_MODE=generic \
    python -m src.eval.score --split test
env SAFARI_PATHS__OUTPUTS_ROOT="$GOUT" SAFARI_FEATURES__EMBED_DEVICE=cuda \
    python -m src.features.assemble --partition location
cp "$GOUT/features.parquet" "$OUT_BASE/features_generic.parquet"

# --- fit the standing 4-distance GLM per variant + write the comparison JSON -----------------------------
echo "==> robustness driver"
python -m src.analysis.robustness_experiment
echo "done -> $OUT_BASE/robustness_experiment_summary.json"
