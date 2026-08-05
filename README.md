<div align="center">

# Will it track?

### Predicting a promptable video tracker's zero-shot reliability — *before you run it*

*An MSc dissertation in empirical ML. We predict the tracker's reliability; we do **not** try to make it score higher.*

</div>

---

> **TL;DR** — Modern trackers like **SAM 3** can follow *any* named animal in a video with zero training on that
> species. But nobody can tell, in advance, whether to trust the result. We asked whether four **label-free
> distances** — how novel the species is, how different it looks, its environment, its era — can forecast that
> reliability before running the model.
>
> **The answer is a clean, honest _no._** Nothing predicts transfer beyond a trivial animal-size effect — and that
> null holds across **three datasets** and **three different trackers**. The contribution is the rigorous negative result.

---

## The question

Conservation teams point promptable trackers at new species and places every day, with no way to know beforehand
whether the output can be trusted. So:

> **Can we predict a tracker's zero-shot accuracy on an unseen species and place *before running it* —
> and what governs that transfer, separately for _finding_ the animal and _following_ it?**

We split the tracking score two ways and study each on its own:

| | what it measures | symbol |
|---|---|---|
| **Detection** | did it *find* the animals? | `pDetA` |
| **Association** | did it *keep each identity* over time? | `pAssA` |

…and try to predict them from four things you can measure **without any label of the target**:

**taxonomic** distance · **visual** distance · **environment** distance · **temporal** distance

This combination — forecasting a **video tracker**, **before running it**, **split into detection vs
association** — has never been done. That is the novelty.

## The answer

**A robust, decomposed null.** Under a deliberately tough out-of-sample test (hold out whole species *and* whole
locations, honest bootstrap confidence intervals):

- **Detection → null.** No distance beats simply guessing the average. The one thing that *looks* predictive is
  really just **animal size**, and a second, independent attempt (image difficulty — low light, clutter, night/IR)
  collapses to the *same* size confound.
- **Association → null, by near-degeneracy.** Only ~13% of clips have multiple animals, and there `pAssA ≈ pDetA`
  — so there's almost no separate association signal to predict in the first place.
- **The test isn't broken — it fires when signal exists.** The model's *own confidence* (measured **after**
  running) predicts detection cleanly. We keep it only as a **positive control** — it's after-the-fact and
  circular, so it's excluded from the headline. It proves the null is a real absence of signal, not weak stats.
- **Not a fluke of one setup.** The null repeats across **SA-FARI, MammAlps, and BURST**, and — in a controlled
  swap that changes *only the tracker* — across **SAM 3, GLEE, and Florence-2 + SAM 2**. It's **task-general**.

> **The contribution is the null itself:** the first out-of-sample evidence that label-free, before-running signals
> do *not* predict a promptable video tracker's zero-shot transfer — hardened against its confounds and shown to
> hold across datasets and models.

## How it works

Everything hangs off **one frozen inference**. SAM 3 is never retrained — it runs once over the test videos to
produce per-clip scores. Every experiment then overlays a different label-free predictor on those *same* scores and
fits one tiny model. The only thing ever "trained" is that little regression.

```
                    ┌──────────────────────────────────────────────┐
   videos  ───────▶ │  SAM 3  (frozen, zero-shot — never retrained) │
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                       official evaluator (HOTA)  ──▶  scores:  pDetA / pAssA   (what we predict, Y)
                                       │
     four label-free distances  ──────────────────▶  features:  taxonomic · visual · environment · temporal   (X)
                                       │
                                       ▼
              small support-weighted regression  +  group-aware cross-validation  +  bootstrap CIs
                                       │
                                       ▼
                           a validated (or, here, honestly null) predictor
```

The **cross-model swap** keeps the clips, ground truth, distances, and regression fixed and changes *only the
tracker* — so any difference is the model's doing, not the setup's.

## The dataset

**SA-FARI** (Meta × Conservation X Labs, 2025) — the largest open wild-animal tracking dataset: **99 species**
with full taxonomy, **741 locations** on 4 continents, **11,609 videos**, 2014–2024. We add **MammAlps** and
**BURST** as independent replications. *(Full detail, including the split correction, is in the dissertation.)*

## Repository

```
src/
  inference/     frozen SAM 3 harness  ─┐
  eval/          official HOTA scoring  ─┴─▶  outputs/scores.parquet     (pDetA / pAssA)
  features/      taxonomic · visual · environment · temporal · size · … ▶ outputs/features.parquet
  analysis/      regression · variance · grouped CV · bootstrap · hallucination · reliability
  adapters/      burst · mammalps       (cross-dataset replication)
configs/         default · burst · mammalps
docs/            label_free_prediction_null.md   ← the consolidated write-up of the result
report/          dissertation/ (LaTeX)  ·  build.sh
tests/  notebooks/                       data/ & outputs/ are gitignored
```

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-local.txt     # analysis + CPU (no GPU / no SAM 3 needed)
.venv/bin/python -m pytest -q                        # sanity: imports + config
report/build.sh report/dissertation                  # build the dissertation PDF
```

Running SAM 3 itself needs a **separate Python 3.12 + CUDA** environment (`requirements-gpu.txt`) and gated access
to the [dataset](https://huggingface.co/datasets/facebook/SA-FARI) and the
[checkpoint](https://huggingface.co/facebook/sam3) — request both early (~24–48 h to approve).

---

<div align="center">

**The result, in one page:** [`docs/label_free_prediction_null.md`](docs/label_free_prediction_null.md)  ·
**Full write-up:** the dissertation in [`report/`](report/)

</div>
