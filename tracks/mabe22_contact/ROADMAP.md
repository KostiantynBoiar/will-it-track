# MABe22 contact-geometry — PRIMARY track roadmap (M0–M9)

> The headline contribution: does **inter-animal mask-contact geometry** (mask IoU/overlap,
> contact-boundary length) beat the **keypoint-hull geometry** the field uses (SimBA/MARS) for
> social/contact behaviour (Chase, Huddle, Oral Contact, Oral–Genital Contact)?
> See `../../.claude/CLAUDE.md` for the full re-scope. Same philosophy as the MammAlps roadmap:
> thin slices, a reviewable artifact + go/no-go gate per stage. **M2 is the make-or-break and
> must finish in week 1.** Pivot rule = §9 of CLAUDE.md.

**Verified:** MABe22 mouse-triplet video exists — 2,614 clips, 1 min @ 30 Hz, 512×512 grayscale,
12 keypoints/mouse. So the open risk is identity-through-contact (M2), not "does video exist."

**Reuse from the MammAlps track** (`tracks/mammalps_baseline/mammalps_b1/`, or via `common/`):
the nested `Config` pattern, frozen encoder wrappers + `get_device`, the summation-fusion head +
multi-task loss, and the eval-canary + remotezip-fetch patterns.

```
M0 acquire+confirm ─► M1 LOCK label/eval contract ─► M2 MASKING PILOT (★ week-1 go/no-go)
   ─► M3 scoped mask cache (Kaggle) ─► M4 contact-geom + hull baseline
   ─► M5 head overfit ─► M6 mask-geom vs hull-geom HEAD-TO-HEAD ─► M7 per-behaviour
   ─► M8 multi-seed CIs ─► M9 fuse-on-VideoMAE redundancy + lock
```

---

## M0 — Acquire MABe22 & confirm the video subset  ✅ DONE
> Submission-set labels fetched + inspected (`mabe22/scripts/check_data.py`): **1830 labelled
> sequences × 1800 frames**; the 4 behaviours (`chases`, `huddles`, `oral_oral_contact`,
> `oral_genital_contact`) confirmed framewise; flat MABe format
> (`vocabulary`/`label_array[13, 3.294M]`/`frame_number_map`). Keypoints (949 MB) + video deferred
> to M4/M2. Gate passed.

- **Start from:** the Caltech / AIcrowd release.
- **Do:** acquire the mouse-triplet data; identify the **video-available** clips; confirm the 12
  keypoints/mouse are present and frame-aligned; confirm resolution (≈512×512) is segmentable.
- **Output:** a manifest of video-available clips (frame count, fps, resolution, which target
  behaviours each contains).
- **Gate:** video-available count known; keypoints present + frame-aligned. (Video existence is
  already confirmed; this is the concrete obtain-and-check.)

## M1 — Establish & LOCK the label/eval contract (the MABe22 "nail the split")  ✅ DONE
> `EVAL_CONTRACT.md` written; `eval.py` `MABeScorer` (per-behaviour F1 + AP, macro); leakage-safe
> split by sequence (`data/split.py`); format canary `scripts/emit_dummy.py` ran clean
> (macro F1≈0.04, AP≈0.02 on random preds); `tests/test_eval_canary.py` asserts split disjointness.
> **Contract locked.**

- **Do:** determine per-frame/window **label availability** for the four contact behaviours.
  Decide the eval design — **recommended:** supervised classification on labelled contact windows
  with a **leakage-safe split** (split by sequence/animal, never random window), metric per-class
  AP + macro mAP (match MammAlps reporting); or the MABe probe protocol if cleaner. Write a
  one-line eval contract + a **dummy-prediction scorer** that runs end-to-end (mirror the MammAlps
  format canary).
- **Output:** written eval contract; stub scorer that accepts a dummy file; printed metric on dummy preds.
- **Gate:** scorer runs on dummy preds; target-behaviour labels confirmed present; split leakage-safe
  and documented. **Lock before any modelling.**

## M2 — THE MASKING PILOT  ★ critical-path, week 1, go/no-go ★
- **Do:** pull 5–10 clips with **Huddle + Chase** (deepest contact). Prompt SAM 2 from the **given
  keypoints** (derive per-mouse boxes/points — free, no manual annotation). Propagate across the
  clip. Assess, **on the contact frames**, whether the **three identities survive**.
- **"Survived" criteria (tune at M1/M2):** mask count stays 3 across sustained contact (no long
  collapse to <3); no identity swaps (each mask stays consistent with its prompting keypoint set —
  mask↔own-keypoints IoU above threshold); per-mouse mask spatially stable frame-to-frame.
- **Output:** an overlay montage of contact frames + a short quantitative report (per clip:
  % frames with 3 stable masks, swap events, merge events).
- **Gate / DECISION RULE → §9 of CLAUDE.md** (PASS / PARTIAL=merged-mask-as-signal / FAIL=pivot to RQ4).
  This single gate de-risks the whole contribution.

## M3 — Scoped mask cache · PART A (Kaggle GPU, run once)
- **Do:** for the **behaviour-windowed, frame-subsampled (~5–6 Hz)** clips containing target
  behaviours, run SAM 2 propagation; cache **derived geometry** (+ enough mask info to recompute
  features). Deterministic per-clip seed. Respect Kaggle's ~30 h/week quota.
- **Output:** `cache/<clip_id>.npz` with per-(frame/window) mask geometry; a manifest (clip_id,
  split, behaviours present, feat_path). Mask-ready layout consistent with the MammAlps cache.
- **Gate:** every scoped clip cached; spot-check mask quality; no NaNs; compute within quota.

## M4 — Contact-geometry features + the (free) hull baseline
- **Do, per frame, per mouse-pair (3 pairs):** mask **IoU/overlap**, **contact-boundary length**
  (adjacent-boundary pixel count between two masks). **Mutual occlusion DEFERRED.** Per-mouse
  **shape descriptors** (area, aspect ratio, solidity, Hu moments) for the posture side. **Also
  the keypoint-hull baseline** (SimBA-style `polygon_pct_overlap` / `difference_area` from the
  given keypoints) — the comparison condition, essentially free.
- **Output:** per-clip feature series for `mask_contact`, `mask_shape`, `kp_hull`.
- **Gate:** sanity — mask overlap + contact-boundary length **rise during known contact**
  (Huddle/Chase), fall otherwise; hull baseline computed on the same clips.

## M5 — Head + overfit sanity
- **Do:** a **tiny temporal encoder** over the feature series + a small head; modality-toggle
  dataloader. Overfit ~50 clips.
- **Gate:** head overfits (loss → ~0) — wiring + label encoding + multiclass/multilabel correct.

## M6 — HEAD-TO-HEAD: mask-geometry vs hull-geometry  ★ key result ★
- **Do:** **identical-capacity** head, on the contact behaviours: (a) `kp_hull` only, (b)
  `mask_contact` only, (c) optionally (a)+(b); if integrating appearance, video vs video+`mask_contact`.
  Class-balanced sampling; fixed seed.
- **Output:** the central result table (per-task macro mAP + per-class AP) per condition.
- **Gate:** non-degenerate result; the mask-vs-hull comparison is clean and interpretable (the RQ3
  answer, positive *or* negative).

## M7 — Per-behaviour analysis
- **Do:** break down by Chase / Huddle / Oral Contact / Oral–Genital Contact. Where does mask
  geometry win/lose vs hulls? Tie to the mechanism (does it help most where keypoints occlude/swap?).
- **Gate:** per-behaviour table with CIs + a short interpretation.

## M8 — Multi-seed report
- **Do:** **≥5 seeds** per condition; aggregate to mean ± **95% CI** (the pre-registered design).
- **Gate:** a +0.02-type gain clears the noise band, or is reported honestly as within-noise.

## M9 — Fuse, lock, redundancy test
- **Do:** optionally fuse `mask_contact` **on top of frozen VideoMAE** to answer RQ1 directly —
  *does geometry add over what VideoMAE already sees?* Lock primary results; decide what carries to
  the secondary MammAlps track.
- **Gate:** documented answer to RQ3 (+ redundancy); reproducible from cache, one command per condition.

---

## Doc steps (done one-by-one, after the framing is settled)
- **Write the MABe22 proposal** ✅ → `report/proposal/proposal.{tex,pdf}` (contact geometry
  primary; MammAlps secondary; honest novelty framing; pre-registered ablations + success threshold;
  precise keypoint source + geometry formulas).
- **Write the MABe22 presentation** ✅ → `report/presentation/presentation.{tex,pdf}` (12-slide
  Beamer deck; worked TikZ examples of hull-overlap vs mask-IoU + contact-boundary; the re-scoped
  story for Marwa). Build both with `report/build.sh` (Tectonic + gs-normalize for GitHub).
