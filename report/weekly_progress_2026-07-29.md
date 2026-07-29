# Weekly progress — week ending 2026-07-29

*What I did since last week's progress deck (2026-07-23), and the results. Written for the supervisor.*

## One-paragraph summary

Last week's deck ended on a clean **negative result**: on SA-FARI we can *measure* SAM 3's zero-shot
reliability but *cannot predict it in advance* from label-free "distance" signals, beyond a small animal-size
effect. This week I did the obvious next thing — **tried to break that result** — by (1) re-running the whole
pipeline on **two completely different datasets**, and (2) implementing the one before-running idea I had left
(**reading familiarity from SAM 3's own features**). The null held everywhere. So the "no" is now a
**triangulated, robust result** across three datasets and a model-internal check, not a one-dataset quirk.

## What I did (three workstreams)

### 1. Two independent replications
The whole project had reused **one** SAM 3 run on **one** dataset, so the result wasn't *independent*. I built
offline adapters (so nothing in the pipeline changes — new datasets just get converted to the format the loader
already reads) and re-ran the entire thing end-to-end on:
- **MammAlps** — Alpine camera-trap tracker, 5 species, box-only ground truth (Zenodo). Box → filled-rectangle
  masks + box-HOTA scoring; frames streamed by HTTP-range so no 82 GB download.
- **BURST** — internet/movie video, the **animal subset (41 species)** of a mask-native benchmark built on TAO.
  Animals selected by a WordNet filter from 482 LVIS classes; masks streamed from RWTH, frames from a
  HuggingFace mirror. This is the *many-species* test MammAlps was too small to be.

### 2. SAM 3 familiarity proxy (the last before-running idea, T2.5)
Every distance measures novelty against **our** reference pile — but nobody knows what SAM 3 was actually
trained on. I plugged that hole by reading familiarity **from the model itself**: push each animal through
SAM 3's own vision encoder and measure how distinctly it sits in SAM 3's feature space (three metrics:
silhouette / nearest-species / density). Implemented, hermetically tested, and run on SA-FARI + BURST.

### 3. Dissertation write-up + polish
Wrote the two replications, the cross-dataset comparison, and the familiarity proxy into the methodology,
results and discussion chapters (+ tables + a plain-English explainer each). Also: added the dataset/tool
papers to the bibliography, fixed a wide table (rotated to landscape), and de-jargoned some prose.

## The results

**All three datasets agree: the before-running distances predict nothing.**

| Feature (leave-species-out) | SA-FARI (~100 sp) | MammAlps (5 sp) | BURST (41 sp) |
|---|---|---|---|
| the four distances, combined | null | null | null |
| **animal size** (confound + positive control) | validates (+0.015) | n.s. | n.s. |

- **MammAlps** — null, but a **consistency check, not proof**: at 20 cells it's underpowered (even a genuinely
  predictive feature can't validate). Its one "significant" feature (visual) points the **opposite direction**
  to the hypothesis and is a size confound — the same artefact SA-FARI showed.
- **BURST** — a **third convergent null**, better-powered (132 cells, mask-native, real association content).
  Better powered than MammAlps by sheer numbers, but still a *convergent* null, not proof: no before-running
  signal fires here (not even the size control that works on SA-FARI), so a small effect can't be independently
  ruled out — the answer just agrees with the others.
- **Familiarity proxy** — **another null** on both datasets. All three metrics turn out to be **63–85% the same
  number as the visual distance** — they just re-measure "how visually distinct is it," which was already a dead
  end and a size confound. The closest metric (silhouette) reaches p≈0.03 and is *not* size-confounded, but it
  doesn't clear the corrected bar and adds nothing over visual + size. On the powered SA-FARI cells the
  animal-size control validates, so this is a genuine absence of signal. **This closes the "but you don't know
  SAM 3's real training data" loophole.**

## Where this leaves the project

- The headline **negative result is now robust**: same "no" across a camera-trap dataset, an internet-video
  dataset, and a model-internal familiarity check — powered enough (on SA-FARI and BURST) to exclude a moderate
  effect. The contribution is this **triangulated, honest null**, decomposed into detection vs association.
- The only before-running quantity that ever reaches significance is the **animal-size** confound (modestly, on
  SA-FARI) — which doubles as the positive control that keeps the "no" honest: the same test that stays flat for
  the distances *does* catch size out-of-sample.
- Everything is **built, tested (hermetic + on the GPU pod), committed, and written into the dissertation.**

## Status / next
- Dissertation chapters + tables + figures are updated for all of the above; PDFs rebuilt.
- Optional remaining extensions (none change the null): a coverage-anchored pretraining-typicality proxy; a
  fuller representational probe; a larger mask-native second dataset for even more leave-species-out power.
