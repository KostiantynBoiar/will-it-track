# How the pipeline works — plain English

A from-scratch, no-jargon tour of what this project actually does: how SA-FARI feeds
SAM 3, how we guarantee the run is genuinely *zero-shot* and really happened on the real
data, and how the experiments are structured. Every claim below points at the real code so
it can be checked.

---

## The big picture (one analogy)

Think of SAM 3 as a **contractor you hire for one day**. You hand them a video and say one
word — **"impala."** They point at every impala and follow it across the clip. You never
send them to school, never correct them, never let them practice on your footage. You just
watch what they do and grade it against the real answer sheet afterward.

- **SA-FARI** — the pile of videos + the answer sheet (where every animal actually is).
- **SAM 3** — the contractor.
- **This project** — grade the contractor, then ask: *could we have predicted the grade in
  advance, without watching?*

The honest answer we found is **no** (from label-free "distance" signals) — which is a real,
publishable result. This document is only about *how the machinery works*.

---

## Q1 — How does SA-FARI work with SAM 3?

SA-FARI is **data**; SAM 3 is a **model**. Three files connect them.

**1. The data — [`src/dataset.py`](../src/dataset.py).** Reads SA-FARI's
`sa_fari_test_ext.json`. The unit is a **"probe" = (one video, one text prompt)**. Each
probe carries:
- the video's frames,
- a **prompt** — a species name like `"impala"` (SA-FARI calls it the *noun phrase*),
- the **ground-truth masks** — the real answer: exactly which pixels are the animal, per
  frame,
- taxonomy, location, timestamp.

Some probes are **hard negatives** (`num_masklets == 0`): the prompt asks for an animal that
*isn't in the video* — a deliberate trap to see whether SAM 3 hallucinates one.

**2. The model — [`src/inference/sam3_tracker.py`](../src/inference/sam3_tracker.py).** Loads
SAM 3 from its public checkpoint and exposes exactly one method:

```python
track(frames, prompt) -> list of masklets
```

A **masklet** is one tracked object: a mask on each frame + a confidence score. You give it
pictures and a word; it gives back "here's what I found, and how sure I am."

**3. The glue — [`src/inference/harness.py`](../src/inference/harness.py).** Loops over every
probe: load the real frames → `track(frames, prompt)` → save the predicted masks to
`{video}_{species}.json`. SA-FARI supplies videos + prompts; SAM 3 answers; the harness
records the answers.

---

## Q2 — How do we *guarantee* it's zero-shot, and that it really ran on the real data?

This is what makes the whole dissertation trustworthy, so these are **checkable** facts, not
promises.

### It is zero-shot because the code *cannot* train the model

1. **No learning step exists.** Grep all of `src/`: there is no optimizer, no `.backward()`,
   no `.step()`, no loss function, no `.train()`. The only calls are `from_pretrained()`
   (download the finished model), `.eval()` (freeze it), and `no_grad()` (don't even compute
   gradients). There is *no mechanism* in the code to change SAM 3's weights.
2. **The answers never reach the model.** `track()` takes only `(frames, prompt)` — pictures
   and a word. Ground-truth masks and species labels are used **only afterward, by the
   grader**. The answer physically cannot leak into the prediction.
3. **The dataset is held-out.** SA-FARI is not in SAM 3's training data (verified at the
   dataset level from both papers). SAM 3 meets these species and places for the first time.
   We can verify non-membership at the *dataset* level, not frame-by-frame (SAM 3's full
   pretraining corpus is undisclosed) — so distances are always measured against the SA-FARI
   *train reference*, never against SAM 3's true unknown pretraining data.

### It really ran on the real data because

- The harness reads **actual frames from disk**, fetching them from the public Google bucket
  if missing. If a frame is genuinely unavailable, that probe is scored as a **miss** — never
  faked.
- Every prediction is written to an **inspectable JSON** in `outputs/predictions/`. You can
  open one and see the real masks. The run is **resumable** (it skips probes already done),
  which is why per-video files exist.
- **Grading uses Meta's own official scorer** ([`src/eval/score.py`](../src/eval/score.py)
  shells out to the vendored `VEval` script) — we never re-implement the metric.
- **The sanity check ("Gate 1") passed:** our dataset-level score (~0.65 mask pHOTA)
  **matches the SA-FARI paper's number** for SAM 3. Run it wrong, or on the wrong data, and
  that number wouldn't line up. It does — independent confirmation the pipeline is real.

> **On "editing" data:** we never edit SA-FARI's videos or labels. The one thing we
> "construct" is a re-labelling of *which species count as the reference set* (for the
> species split). Because SAM 3 is frozen, re-grouping the reference changes nothing about
> what SAM 3 did — it only changes what we measure *distance to*. The videos and answers are
> untouched.

---

## Q3 — How do the experiments actually work?

Two layers. Layer 1 runs once; Layer 2 is where every "experiment" lives.

### Layer 1 — MEASURE (run SAM 3 once, get the grades)

Run the harness + scorer over the test videos → for each **cell** (a species × place × time
group) you get two grades:

- **pDetA** — *did it find the animal?* (detection)
- **pAssA** — *did it keep each individual's identity over time?* (association)

These grades are the **target, `Y`**. This happens **exactly once** — SAM 3 is never re-run
per experiment.

### Layer 2 — PREDICT (the research question)

For each cell we compute four **label-free "distances" (`X`)** — *without running SAM 3 and
without the target's labels*:

- **taxonomic** — how far on the tree of life from known species,
- **visual** — how different the animal *looks* from known ones,
- **environment** — how different the *scene* looks,
- **temporal** — how far apart in time.

Then we fit a **tiny logistic regression `X → Y`**. The honest test: **hide whole species (or
whole locations), train on the rest, predict the hidden ones.** If "far from training →
unreliable" were true, this would beat simply guessing the average grade.

**The result:** it *doesn't* beat the average. The one signal that looked real (visual)
turned out to be **animal size** in disguise. So the honest answer is a **null** — you cannot
forecast SAM 3's transfer from these distances.

**The positive control:** the same test *does* catch the size effect. That proves the test
works and isn't simply too weak to find anything — a flat result for the distances is
genuinely "no signal," not "failed to look."

---

## One-line summary

> SA-FARI feeds videos + prompts to a **frozen** SAM 3; an **official** scorer grades the
> results into per-cell **pDetA/pAssA**; and every "experiment" is a small model trying — and
> honestly failing — to predict those grades in advance from **label-free distances**.

---

## Where to look in the code

| Question | File |
|---|---|
| How a video/prompt/answer is loaded | [`src/dataset.py`](../src/dataset.py) |
| How SAM 3 is loaded (frozen) and run | [`src/inference/sam3_tracker.py`](../src/inference/sam3_tracker.py) |
| How the whole test set is swept, resumably | [`src/inference/harness.py`](../src/inference/harness.py) |
| How the official metric grades it into cells | [`src/eval/score.py`](../src/eval/score.py) |
| The distances (`X`) | `src/features/{taxonomic,visual,environment,temporal,size}.py` |
| The GLM fit + leave-group-out validation | `src/analysis/{regression,cross_val}.py` |
