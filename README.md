# Predicting SAM 3's zero-shot transfer on SA-FARI

**An empirical-science ML dissertation — we predict a promptable video tracker's reliability, we do
not try to make it score higher.**

Can we predict **SAM 3's** zero-shot tracking performance on an *unseen* species and place *before
running it*, from properties measurable in advance — and which factors govern that transfer,
separately for **finding** the animal (`pDetA`) and **following** it (`pAssA`)?

- **H1 (detection):** `pDetA` falls with **species novelty** (taxonomic + visual distance).
- **H2 (association):** `pAssA` falls with **environment difficulty** (scene distance, day/night/IR,
  clutter, camera motion), largely regardless of species.
- **H0 (null, still a result):** distance explains little → pivot to representational probing.

**Success = a *validated, out-of-sample* predictor of `pDetA`/`pAssA` from label-free distances,
with honest confidence intervals.**

## Dataset — SA-FARI
The largest open multi-animal wild-animal tracking dataset (Meta × Conservation X Labs, 2025): 99
species with a full 7-level taxonomy, 741 locations across 4 continents, 2014–2024. Train/test are
**disjoint by species AND location**, so each test video is a controlled probe of transfer along a
measurable distance. Hard negatives are kept. Details in `.claude/CLAUDE.md §4`.

## Pipeline
```
data/  →  src/inference (frozen SAM 3)  →  src/eval (OFFICIAL VEval)  →  outputs/scores.parquet
                                                                              │
       src/features (taxonomic · visual · environment · temporal · familiarity)  →  outputs/features.parquet
                                                                              │
       src/analysis (beta regression · variance partition · grouped CV · bootstrap · reliability)
                                                     →  outputs/{models,validation,figures}/
```

## Layout
```
src/            config.py types.py dataset.py reference.py io.py
                inference/harness.py  eval/score.py
                features/{taxonomic,visual,environment,temporal,familiarity,assemble}.py
                analysis/{regression,variance,cross_val,uncertainty,ablations,reliability}.py
configs/        default.yaml
data/           annotations/ frames/ raw_videos/ reference/     (gitignored)
outputs/        predictions/ scores.parquet features.parquet models/ validation/ ablations/ figures/  (gitignored)
notebooks/      exploration + final figures
tests/  report/build.sh
```

## Run
```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-local.txt          # analysis + CPU (no GPU, no SAM 3)
.venv/bin/python -m pytest -q                            # import/config pass; data/SAM3/VEval tests skip
PYTHONPATH=. .venv/bin/python -m src.inference.harness    # GPU box: also needs requirements-gpu.txt
report/build.sh                                          # dissertation LaTeX
```

## Data access (T0.1)
Annotations are **gated** on Hugging Face; frames are **public** on GCS; SAM 3 checkpoints are
**gated** and take ~24–48 h to approve — so request them **now**.

1. Accept the license at **huggingface.co/datasets/facebook/SA-FARI**.
2. `.venv/bin/huggingface-cli login` (or `export HF_TOKEN=hf_…`) with a read token.
3. **Request SAM 3 checkpoint access** at **huggingface.co/facebook/sam3** (starts the clock; needed
   for inference at T1.1/T1.2, not for the data layer).
4. Fetch the minimal slice + verify RLE decode:
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.acquire --list             # public GCS layout probe (no auth)
   PYTHONPATH=. .venv/bin/python -m src.acquire --annotations      # gated HF snapshot (~1.86 GB JSON)
   PYTHONPATH=. .venv/bin/python -m src.acquire --frames --n-clips 3
   PYTHONPATH=. .venv/bin/python -m pytest tests/test_dataset.py   # records parse + RLE masks decode
   ```

SAM 3 itself needs a **separate Python 3.12 + CUDA** env (`requirements-gpu.txt`), not this `.venv`.

## Constraints (see `.claude/CLAUDE.md §9`)
Freeze the seen set **first**; compute every distance against the **train split only**; use the
**official evaluator** (never re-implement `pHOTA`); model bounded scores with **beta/logit GLM
weighted by support** with a `log(n_frames)` covariate; **group-aware CV** (hold out whole species +
locations); **no test information leaks** into a feature; **keep hard negatives**.

Full spec: `.claude/CLAUDE.md`. Task breakdown: `.claude/IMPLEMENTATION_PLAN.md`. Phase summary +
gates: `ROADMAP.md`.
