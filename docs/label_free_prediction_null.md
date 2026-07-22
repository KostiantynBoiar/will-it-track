# Label-free, before-running prediction of SAM 3 transfer — the consolidated (robust) null

A single, evidence-backed reference for the dissertation's central result. It pulls together every
before-running, label-free attempt into one honest conclusion, with the numbers that back it. Intended as raw
material for the Results/Discussion prose (which stays the student's).

## The question
Can we predict SAM 3's zero-shot promptable-tracking accuracy on an *unseen* (species, place) — split into
**detection** (`pDetA`) and **association** (`pAssA`) — from properties measurable **before running the model**
and **without any label** of the target? Validated out-of-sample by holding out whole species and whole
locations, at a strict bar (support-weighted logit GLM, leave-species-out AND leave-location-out vs a
mean-predictor baseline, paired group-bootstrap significance, size + support controlled).

## The answer: a robust null
Across **two independent families** of well-motivated label-free signals, **nothing predicts transfer beyond
a trivial, confounding size effect**. This is not one failed feature — it is a consistent convergence.

### 1. Novelty distances (the original H1/H2) — NULL
Four before-running label-free distances: **taxonomic** (tree distance to nearest reference species),
**visual** (DINOv2 mask-crop cosine to nearest species prototype), **environment** (DINOv2 masked-background
to nearest location), **temporal** (year gap). On both the constructed species hold-out and the official
location hold-out: every distance's group-cluster-bootstrap coefficient CI **spans zero**, and grouped CV does
**not** beat the mean baseline. The one coefficient that reached significance — visual distance, with the
*wrong* (positive) sign — **collapsed under a size control** (`log_area`): it was measuring "big,
high-contrast animals are easy to segment," not novelty.

### 2. The intrinsic-difficulty pivot — ALSO NULL (it reduces to size)
The size confound above motivated a pivot: maybe transfer is driven by intrinsic **difficulty**, not novelty.
We promoted the existing label-free difficulty signals — **low-light/IR** (`achromatic_fraction`),
**clutter**, **night/IR** — to predictors of interest, size-controlled. They *appear* to validate (each clears
Bonferroni on both schemes, ΔMAE ≈ +0.017), **but a nested decomposition shows the gain is entirely the size
covariate, not difficulty**:

| model (leave-species / leave-location OOS gain over mean baseline) | ΔMAE species | ΔMAE location |
|---|---|---|
| `log(n_frames)` only | +0.0005 (p 0.43) | +0.0018 (p 0.29) |
| **`log_area` only (size)** | **+0.0152 (p 0.014)** | **+0.0172 (p 0.001)** |
| low-light only (no size) | +0.0031 (p **0.23**) | +0.0044 (p **0.12**) |
| low-light + size | +0.0174 | +0.0202 |

Low-light **alone does not validate**; adding it on top of size buys ~+0.002 (marginal). Clutter and night/IR
behave identically. And the correlate that *motivated* the pivot — a location-level night/IR–pDetA
**r = −0.377** — is an **aggregation artifact** (Robinson 1950): at the cell level it is only **−0.13**, and
clutter's cell-level correlation with pDetA is **0.00**.

### 3. The one thing that survives: animal size (modest, and it IS the confound)
`log_area` (mean GT mask area per species) predicts `pDetA` out-of-sample (+0.017, both schemes, cell-level
r **+0.28**) — bigger animals detect better. It is honest (before-running, label-free, non-circular) but
**small**, **low-novelty** (the well-known small-object-detection difficulty), and it is precisely the
**confound** that dissolved every apparent effect. The coherent one-line story: *the only label-free property
that predicts detection is animal size — the same axis that confounded the visual distance.*

## Why the null is credible (a positive control, not a weak test)
The obvious objection to any null is "your test was too weak to find anything." It was not. The **exact same
pipeline, on the exact same 346 cells**, detects a strong signal when one exists: SAM 3's **own detection
confidence** (an ATC-style after-running feature) beats the baseline decisively (ΔMAE **+0.143**, p ≈ 0, both
schemes; pseudo-R² 0.76). So the test fires when there is signal to find. The distances and difficulty features
coming out flat is a genuine **absence of before-running label-free signal**, not an underpowered test.
(That confidence result is **after-running and mechanistically circular**, so it is deliberately excluded from
the dissertation's contribution — see `docs/after_running_confidence_estimator.md`.)

## The decomposition into detection vs association
`pAssA` barely varies from `pDetA` (only 13% of cells are multi-object; within them corr(pDetA, pAssA) = 0.94,
51% exact ties, association-specific residual std ≈ 0.10 over ~188 thinly-grouped cells). So **association is a
near-degenerate target** — the analysis is honestly **detection-only**, and the association null is reported
with these divergence numbers as its explanation, not as a separate failed search.

## The precision (false-positive) side — a marginal correlate, not a predictor
On the 1486 hard negatives, the hallucination rate is ≈9.8% and falls with **visual distinctiveness** (r ≈
−0.33, size-independent) — but its leave-species-out validation **just misses** (ΔMAE +0.008, p = 0.053). A
real correlate, not a validated predictor. It does not change the headline.

## What it means (framing for the write-up)
- The contribution is a **rigorous, decomposed, honest negative result** for a promptable video tracker — the
  first time this before-running/label-free question has been asked for a tracker and split into detection vs
  association. A robust null from a demonstrably-powerful test is a real, publishable finding.
- It is **hardened**, not just "we used lazy distances": two independent well-motivated families
  (novelty-distance *and* intrinsic-difficulty) both fail, and both failures route through the **same size
  confound** — a clean mechanistic explanation, not a bare "no effect."
- Scope caveat to state every time: distances are measured against the SA-FARI train split, **not** SAM 3's
  true (unknown, web-scale) pretraining corpus; the claim is scoped accordingly.

## Reproducibility (code)
- Distances + GLM + grouped-CV + cluster-bootstrap: `src/features/{taxonomic,visual,environment,temporal,
  size}.py`, `src/analysis/{regression,cross_val,variance,report}.py`.
- Difficulty experiment + the size decomposition: `src/analysis/difficulty_experiment.py`,
  `outputs/difficulty_experiment_summary.json` (the `decomposition_the_honest_verdict` field).
- Positive control (excluded): `src/features/confidence.py`, `src/analysis/confidence_experiment.py`,
  `outputs/confidence_experiment_summary.json`.
- False positives: `src/analysis/false_positives.py`.
