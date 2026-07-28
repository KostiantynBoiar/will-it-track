#!/usr/bin/env bash
# R7 BURST end-to-end (pod): acquire masks → convert → stream frames → SAM 3 → mask-HOTA → distances → CV → report.
# Many-species leave-species-out on the animal subset of BURST val. Masks are native → mask-HOTA.
# Masks stream from RWTH omnomnom (HTTP range); frames stream from the gated HF TAO-Amodal zips (range + token).
# Prereqs: remotezip + SA-FARI env + the VEval scorer, HF token set (frames + SAM 3 checkpoint).
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"  # repo root, so `python3 scripts/...` can import `src`
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SAFARI_PATHS__DATA_ROOT=/workspace/data_burst
export SAFARI_PATHS__OUTPUTS_ROOT=/workspace/outputs_burst
export SAFARI_FEATURES__EMBED_DEVICE=cuda
export SAFARI_INFERENCE__PRECISION="${SAFARI_INFERENCE__PRECISION:-bf16}"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_HUB_DISABLE_XET=1
CFG=configs/burst.yaml
ANN=/workspace/data_burst/annotations/val_all_classes.json
BURST_ZIP=https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip

echo "== [0/8] acquire BURST val masks (HTTP range) if missing =="
mkdir -p /workspace/data_burst/annotations
if [ ! -f "$ANN" ]; then
  python3 - "$BURST_ZIP" "$ANN" <<'PY'
import sys
from remotezip import RemoteZip
with RemoteZip(sys.argv[1]) as zf, open(sys.argv[2], "wb") as out:
    out.write(zf.open("val/all_classes.json").read())
print("acquired", sys.argv[2])
PY
fi

echo "== [1/8] convert BURST (animal subset) → SA-Co schema =="
python3 -m src.adapters.burst --config "$CFG"

echo "== [2/8] stream + extract annotated frames from the HF TAO-Amodal zips (HTTP range) =="
python3 scripts/extract_burst_frames.py --config "$CFG" --split val

echo "== [3/8] SAM 3 promptable inference over the probe videos =="
python3 -m src.inference.harness --split test --config "$CFG"

echo "== [4/8] score (mask-HOTA) =="
python3 -m src.eval.score --split test --config "$CFG"

echo "== [5/8] assemble distances (Split A: leave-species-out, reference=probe) =="
python3 -m src.features.assemble --partition species --origins test --config "$CFG"

echo "== [6/8] per-target regression + coefficients =="
python3 -m src.analysis.regression --config "$CFG"

echo "== [7/8] grouped cross-validation (leave-species-out) =="
python3 -m src.analysis.cross_val --config "$CFG"

echo "== [8/8] report tables =="
python3 -m src.analysis.report --config "$CFG"

echo "== BURST_DONE =="
