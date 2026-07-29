#!/usr/bin/env bash
# T2.5 SAM 3 familiarity proxy (pod, GPU): embed crops with SAM 3's OWN vision encoder → separability metrics
# (silhouette / nearest-prototype / mahalanobis) → leave-species-out CV + head-to-head vs the DINOv2 visual
# distance. Two datasets: `burst` (cheap smoke — already staged, shakes out Sam3Embedder) then `safari` (the
# meaningful target — needs SA-FARI frames/annotations + a baseline features.parquet staged).
#
# Usage: scripts/run_familiarity.sh burst        # smoke first
#        scripts/run_familiarity.sh safari        # the real run
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SAFARI_FEATURES__EMBED_DEVICE=cuda
export SAFARI_INFERENCE__DEVICE=cuda
export SAFARI_INFERENCE__PRECISION="${SAFARI_INFERENCE__PRECISION:-bf16}"
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")"
export HF_HUB_DISABLE_XET=1

DATASET="${1:-burst}"
case "$DATASET" in
  burst)
    export SAFARI_PATHS__DATA_ROOT=/workspace/data_burst
    export SAFARI_PATHS__OUTPUTS_ROOT=/workspace/outputs_burst
    CFG=configs/burst.yaml
    ORIGINS="test"                       # BURST Split A pools test-only
    ;;
  safari)
    export SAFARI_PATHS__DATA_ROOT=/workspace/data
    export SAFARI_PATHS__OUTPUTS_ROOT=/workspace/outputs
    CFG=""                               # default config
    ORIGINS="train test"                 # SA-FARI pools train+test species (Split A)
    ;;
  *) echo "usage: $0 {burst|safari}" >&2; exit 2 ;;
esac
CFG_ARG=(); [ -n "$CFG" ] && CFG_ARG=(--config "$CFG")
BASE="$SAFARI_PATHS__OUTPUTS_ROOT/features.parquet"

echo "== [1/3] prereq check ($DATASET) =="
if [ ! -f "$BASE" ]; then
  echo "ERROR: baseline $BASE missing — build it first (assemble --partition species) so the familiarity" >&2
  echo "       columns can be added onto the existing 4-distance features + scores." >&2
  exit 1
fi
python3 -c "import transformers, torch; print('transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

echo "== [2/3] extract SAM 3 embeddings + 3 familiarity metrics, then CV (species + location) =="
# --stage all: GPU extract (embeds once, caches sam3_crop*.npz, reuses across the 3 metrics) → features_fam.parquet
#              then the CPU CV (Bonferroni over the 3 metrics + head-to-head vs visual_distance + size control).
python3 -m src.analysis.familiarity_experiment --stage all --origins $ORIGINS "${CFG_ARG[@]}"

echo "== [3/3] report table =="
python3 -m src.analysis.report "${CFG_ARG[@]}"

echo "== FAMILIARITY_DONE ($DATASET) =="
