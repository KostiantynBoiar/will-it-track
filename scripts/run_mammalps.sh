#!/usr/bin/env bash
# R7 MammAlps end-to-end (pod): extract frames → SAM 3 inference → box-HOTA score → distances → GLM/CV → report.
# Split A (leave-species-out) on the pooled probe set; leave-camera-out is the environment scheme.
# Frames are streamed from the Zenodo zip via HTTP range (remotezip) — no 82 GiB local download needed.
# Prereqs: dense annotations converted (src.adapters.mammalps), remotezip + SA-FARI env + the VEval scorer,
# HF token set (SAM 3 checkpoint).
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"  # repo root, so `python3 scripts/...` can import `src`
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SAFARI_PATHS__DATA_ROOT=/workspace/data_mammalps
export SAFARI_PATHS__OUTPUTS_ROOT=/workspace/outputs_mammalps
export SAFARI_FEATURES__EMBED_DEVICE=cuda
export SAFARI_INFERENCE__PRECISION="${SAFARI_INFERENCE__PRECISION:-bf16}"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_HUB_DISABLE_XET=1
CFG=configs/mammalps.yaml
ZURL=https://zenodo.org/records/15588220/files/mammalps_v1.zip

echo "== [1/7] stream + extract frames from the Zenodo zip (HTTP range) =="
python3 scripts/extract_mammalps_frames.py --config "$CFG" --remote-url "$ZURL"

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
