#!/usr/bin/env bash
# R7 MammAlps end-to-end (pod): extract frames → SAM 3 inference → box-HOTA score → distances → GLM/CV → report.
# Split A (leave-species-out) on the pooled probe set; leave-camera-out is the environment scheme.
# Prereqs: mammalps_v1.zip downloaded, dense annotations converted (src.adapters.mammalps), SA-FARI env + the
# VEval scorer present, HF token set (SAM 3 checkpoint).
set -euo pipefail

export SAFARI_PATHS__DATA_ROOT=/workspace/data_mammalps
export SAFARI_PATHS__OUTPUTS_ROOT=/workspace/outputs_mammalps
export SAFARI_FEATURES__EMBED_DEVICE=cuda
export SAFARI_INFERENCE__PRECISION="${SAFARI_INFERENCE__PRECISION:-bf16}"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_HUB_DISABLE_XET=1
CFG=configs/mammalps.yaml
ZIP=/workspace/data_mammalps/mammalps_v1.zip

echo "== [1/7] extract frames from the video zip =="
python3 scripts/extract_mammalps_frames.py --config "$CFG" --zip "$ZIP"

echo "== [2/7] SAM 3 promptable inference over the probe clips =="
python3 -m src.inference.harness --split test --config "$CFG"

echo "== [3/7] score (box-HOTA via eval.prefer_bbox) =="
python3 -m src.eval.score --split test --config "$CFG"

echo "== [4/7] assemble distances (Split A: leave-species-out, reference=probe) =="
python3 -m src.features.assemble --partition species --origins test --config "$CFG"

echo "== [5/7] per-target regression + coefficients =="
python3 -m src.analysis.regression --config "$CFG"

echo "== [6/7] grouped cross-validation (leave-species-out + leave-camera-out) =="
python3 -m src.analysis.cross_val --config "$CFG"

echo "== [7/7] report tables =="
python3 -m src.analysis.report --config "$CFG"

echo "== MAMMALPS_DONE =="
