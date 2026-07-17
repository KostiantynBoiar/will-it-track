#!/usr/bin/env bash
# One-command environment setup for a RunPod (or any CUDA) box.
#
#   bash scripts/setup_pod.sh            # CUDA 12.8 driver (RunPod default) — venv on /workspace
#   CU=cu124 bash scripts/setup_pod.sh   # a different driver's CUDA tag
#   VENV=/root/venv bash scripts/setup_pod.sh   # venv elsewhere (default: /workspace/venv, persists on the volume)
#
# Creates a persistent venv, installs a driver-matched torch, then everything in requirements.txt.
# After this, a fresh session only needs:  source "$VENV/bin/activate"
set -euo pipefail

CU="${CU:-cu128}"                 # match your GPU driver's CUDA (RunPod pods here report 12.8)
VENV="${VENV:-/workspace/venv}"   # on the network volume → survives pod restarts
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> venv: $VENV   torch: $CU"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

# torch first, from the CUDA-matched index (see requirements.txt header for why it's separate)
pip install --no-cache-dir torch torchvision --index-url "https://download.pytorch.org/whl/$CU"
pip install -r "$here/requirements.txt"

python -c "import torch, transformers; print('torch', torch.__version__, torch.version.cuda,
      '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
echo "==> done. New sessions: source $VENV/bin/activate"
