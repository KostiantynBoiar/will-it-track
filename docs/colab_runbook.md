# Colab runbook — SAM 3 inference → `scores.parquet`

Run frozen SAM 3 over SA-FARI on a **Colab Pro GPU** and score with VEval. This is *inference* (the model
is frozen), so it's a few GPU-hours, and it's **resumable** — a session timeout never loses progress.

## Before you start (one-time)
- **Runtime → Change runtime type → GPU** (prefer A100; L4/T4 work with `precision=fp16` + a smaller batch).
- Add **Colab Secrets** (🔑): `HF_TOKEN` (your granted token) and `GITHUB_PAT` (to clone the private repo).
- Confirm gated access: [`facebook/sam3`](https://huggingface.co/facebook/sam3) and
  [`facebook/SA-FARI`](https://huggingface.co/datasets/facebook/SA-FARI) both show the file browser.

## Cells

```python
# 1 — GPU + persist outputs/annotations to Drive (so they survive teardown + resume across sessions)
!nvidia-smi -L
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ["SAFARI_PATHS__DATA_ROOT"]    = "/content/drive/MyDrive/mambr/data"
os.environ["SAFARI_PATHS__OUTPUTS_ROOT"] = "/content/drive/MyDrive/mambr/outputs"
```

```python
# 2 — clone the private repo (GITHUB_PAT secret)
from google.colab import userdata
pat = userdata.get('GITHUB_PAT')
!git clone https://{pat}@github.com/KostiantynBoiar/mambr.git
%cd mambr
```

```python
# 3 — deps: our analysis stack + transformers (ships the SAM 3 modeling code)
!pip -q install -r requirements-local.txt
!pip -q install -U "transformers>=4.57" accelerate
```

```python
# 4 — vendor SAM 3 + the official VEval scorer (single clone provides both)
!git clone https://github.com/facebookresearch/sam3 third_party/sam3
!pip -q install -e third_party/sam3          # needs torch>=2.7 / CUDA 12.6 (Colab has CUDA)
```

```python
# 5 — HF login (HF_TOKEN secret) → downloads the gated checkpoints on first use
from huggingface_hub import login; login(userdata.get('HF_TOKEN'))
```

```python
# 6 — fetch the gated annotations (frames are pulled on demand during inference)
!python -m src.acquire --annotations
```

```python
# 7 — FINALISE THE VEval SCHEMA (do this once): inspect the vendored toy prediction + eval script,
#     then adjust src/inference/harness.py (_predict_video) and src/eval/score.py (_parse_veval / _run_veval)
#     if the field names / CLI differ.
!find third_party/sam3 -path '*veval*' \( -name '*.json' -o -name '*.py' -o -name 'README*' \) | head
```

```python
# 8 — smoke test: 3 probes end-to-end
!python -m src.inference.harness --split test --limit 3
!python -m src.eval.score --split test          # -> outputs/scores.parquet
import pandas as pd; pd.read_parquet(os.environ["SAFARI_PATHS__OUTPUTS_ROOT"] + "/scores.parquet").head()
```

```python
# 9 — full test split (resumable: re-run this cell after any disconnect to continue)
!python -m src.inference.harness --split test
!python -m src.eval.score --split test
```

## Notes
- **Resumable:** predictions are one JSON per video under `outputs/predictions/…`; a re-run skips finished
  videos. Because `outputs/` is on Drive (cell 1), progress survives session timeouts.
- **GPU tiers:** on L4/T4 set `os.environ["SAFARI_INFERENCE__PRECISION"]="fp16"` and lower
  `SAFARI_INFERENCE__BATCH_FRAMES`.
- **Robustness prompt:** re-run cells 8–9 with `os.environ["SAFARI_INFERENCE__PROMPT_MODE"]="generic"` for
  the generic-prompt condition.
- **Gate 1:** check the aggregate `pDetA`/`pAssA` land in the SA-FARI paper's SAM 3 ballpark before moving on.
- **Species split later:** `--split train` runs SAM 3 on the train videos too (a bigger batch), needed for
  the species-hold-out experiment.
