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
!git clone https://{pat}@github.com/KostiantynBoiar/will-it-track.git
%cd will-it-track
```

```python
# 3 — deps: our analysis stack + transformers (ships the SAM 3 modeling code)
!pip -q install -r requirements-local.txt
!pip -q install -U "transformers>=5.0" accelerate   # v5.0.0 first ships Sam3VideoModel; 4.x has none
# If the import in cell 4b fails, the classes may be main-only:
#   !pip -q install -U "git+https://github.com/huggingface/transformers"
```

```python
# 4 — vendor the official VEval SCORER only (the frozen SAM 3 model comes from transformers, not here)
!git clone https://github.com/facebookresearch/sam3 third_party/sam3
!pip -q install iopath   # lightweight scorer dep (add regex/ftfy too if cell 7 asks for them)
# Do NOT `pip install -e third_party/sam3`: its deps pin numpy<2 (and its README reinstalls torch),
# clobbering the stack transformers just validated. Run the scorer as a standalone script (cell 7).
# Only if a scorer import demands the package:  !pip -q install -e third_party/sam3 --no-deps
```

```python
# 4b — GUARD: assert the stack BEFORE any GPU work (fail here, not 30 min into a run)
import numpy, torch, transformers, platform
print("python", platform.python_version(), "| numpy", numpy.__version__,
      "| torch", torch.__version__, torch.version.cuda, "| transformers", transformers.__version__)
from transformers import Sam3VideoModel, Sam3VideoProcessor  # ImportError here => fix cell 3
print("SAM 3 video classes import OK")
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
# 7 — CONFIRM THE VEval SCHEMA (once): run the scorer on the vendored toy files and read the REAL keys.
#     The harness/scorer are already coded to the documented schema; this only confirms the metric
#     spelling inside `video_np_results` (add any new spelling to _METRIC_KEYS in src/eval/score.py).
!python third_party/sam3/sam3/eval/saco_veval_eval.py one \
    --gt_annot_file  third_party/sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_gt.json \
    --pred_file      third_party/sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_pred.json \
    --eval_res_file  /tmp/toy_res.json
import json; r = json.load(open("/tmp/toy_res.json"))
print("top-level keys:", list(r.keys()))
print("dataset_results keys:", list(r.get("dataset_results", {}).keys())[:12])
print("one per-probe entry:", (r.get("video_np_results") or [{}])[0])
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
- **GPU tiers:** `bf16` (the config default) needs an Ampere+ GPU (A100/L4). **T4 has no bf16** — you
  *must* set `os.environ["SAFARI_INFERENCE__PRECISION"]="fp16"` (and lower `SAFARI_INFERENCE__BATCH_FRAMES`).
  Only `accelerate` is needed for bf16; `bitsandbytes` is **not** (that is for 8/4-bit quant).
- **Operating point:** the harness writes **raw scores** (`score_threshold=0.0`) so VEval owns the
  threshold; it applies its own `prob_thresh` (~0.5) for HOTA and the full score range for AP-style metrics.
- **Robustness prompt:** re-run cells 8–9 with `os.environ["SAFARI_INFERENCE__PROMPT_MODE"]="generic"` for
  the generic-prompt condition.
- **Gate 1:** check the aggregate `pDetA`/`pAssA` land in the SA-FARI paper's SAM 3 ballpark before moving on.
- **Species split later:** `--split train` runs SAM 3 on the train videos too (a bigger batch), needed for
  the species-hold-out experiment.
