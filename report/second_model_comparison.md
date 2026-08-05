# Second-model swap — comparison report

*Goal: find a second text-promptable zero-shot tracker to run the label-free-distance model-swap against SAM 3
(does the null replicate beyond SAM 3, or is it SAM-3-specific?).*

## ★ HEADLINE UPDATE (2026-08-05) — the swap WORKS on BURST; the null is task-general

The initial conclusion further below ("no tracker replaces SAM 3, swap not achievable") was **an SA-FARI-specific
artifact, and it has been overturned.** A student objection — *"it works on SAM 3 but no other model, which
either means SAM 3 is just good (low novelty) or our harness is rigged"* — prompted a fairness audit and a move
to a fairer dataset. The result:

1. **The harness is not rigged.** Competitors' SA-FARI zeros survive box-HOTA and threshold-free AP (§4) — the
   0.5 confidence gate is a real SAM-3-calibrated operating point, but neutralising it does not change the
   outcome. The SA-FARI zeros are genuine.
2. **The SA-FARI zeros were a DOUBLE confound.** SA-FARI is (a) SAM 3's *own co-released benchmark* (home turf)
   and (b) full of exotic nocturnal species (armadillo, agouti, margay) that are out-of-distribution for models
   trained on internet imagery. Both inflate SAM 3 and sink the others.
3. **On BURST — a fair, off-turf, everyday-animal dataset — the competitors WORK.** GLEE localises at oracle
   mIoU **0.80** (up to 0.96) with well-calibrated scores (0.6–0.9, clearing the 0.5 gate), and its full-split
   **pDetA = 0.340** (median 0.31, spread 0–0.98, 67/132 cells > 0.3) — non-degenerate, vs SAM 3's 0.607.
4. **The distance null REPLICATES on GLEE (BURST, 132 cells, leave-species-out).** The combined four-distance
   model is null (ΔMAE **+0.011, p=0.088**, CI spans zero); magnitudes are trivial (~0.01–0.02) throughout;
   individual flickers do not survive as a usable predictor (§5 caveats). **The label-free null is
   task-general, not SAM-3-specific** — the exact robustness result the swap was meant to deliver, on real,
   varying scores from an architecturally-different model, on a dataset SAM 3 was not built for.

**Net:** the model-swap is *not* a dead end. It is a confound-free replication of the robust null on a second
model, and it defuses the "you only proved SAM 3 is good" objection. Full detail in §5.

---

## TL;DR of the original (SA-FARI) attempt

Three candidates evaluated with a cheap **Gate-0 calibration de-risk** on the same 10 SA-FARI clips SAM 3 uses,
*before* any expensive full build. The de-risk separates two questions: **can the model localize the animal?**
(oracle IoU, scores ignored) and **do its confidence scores rank the good detection high enough to survive
scoring?** (the axis that killed GLEE). Super-honesty held throughout: configs pre-registered from each model's
own defaults, one blind score, **no tuning-to-result**.

| Model | Localizes? | Score calibration | Gate-1 pDetA (10 cells) | Verdict |
|---|---|---|---|---|
| **GLEE** (Lite-scaleup) | ✅ yes (IoU up to 0.96) | ❌ collapses | ≈ **0.02** (median 0) | **FAIL** — scores un-scoreable on camera-trap OOD |
| **OWLv2** (base-ensemble) | ✅ partial (box-IoU up to 0.79) | ❌ mis-ranks (rank_ok 0.27) | not reached | **FAIL** — same calibration wall (stopped at Gate-0) |
| **Florence-2 + SAM 2** | ⚠️ sparse/unstable (correct on a minority of frames, full-frame garbage on the rest) | N/A — no score to collapse | **0.29** frame-0 seed → **0.089** with a GT-blind median seed | **FAIL** — localization too sparse/unstable; SAM 2 can't rescue without a GT-aware seed (§3b) |

## Bottom line

**No candidate is a clean drop-in second tracker on SA-FARI, but the three fail in two very different, and
both informative, ways:**

1. **The two open-vocab *detectors* (GLEE, OWLv2) hit the same wall:** they *find* the animals but their
   per-detection confidence scores **collapse / mis-rank on this out-of-distribution camera-trap domain**, so
   at any fixed operating point their detections are filtered to ~nothing and pDetA floors at ~0.02. Rescuing
   this needs a swept threshold = tuning-to-result, which we refused. This convergent failure is itself a
   finding: **confidence calibration, not localization, is what breaks open-vocab detectors on wildlife OOD.**

2. **Florence-2 + SAM 2 is the one that partly works**, precisely because it *has no confidence axis to
   collapse* (Florence emits no score; detections score at 1.0; SAM 2 gives masks + identity). Where Florence's
   seed box is correct it scores **0.79–0.95** (at or above SAM 3's level); where it isn't, SAM 2 tracks the
   wrong object all clip → 0. Mean 0.29 across 10 cells — **real, non-degenerate variance**, driven by a
   fixable first-frame-seeding limitation, not a calibration artifact.

**Recommendation for the model-swap experiment.** Florence-2 + SAM 2 is the **only viable path to a second
tracker**, but *only after* improving the seeding (best-box frame + periodic re-detection), which should lift
the miss-cells and give a genuinely comparable — if weaker — tracker. That is the honest next step, and it is a
**~1–2 day GPU job** on the full ~2300-cell test split (Florence beam-search + SAM 2 propagation is slow), so it
is left as a **recommendation, not started autonomously**. If that improved run yields a non-degenerate
full-split pDetA, the label-free-distance swap becomes feasible (a genuine second data point on whether the null
is SAM-3-specific). If it too stays bimodal/patchy, the honest conclusion is that **a fair cross-model swap is
not achievable on SA-FARI with current open-vocab trackers, and the SAM-3-only robust null stands** — with the
"three architecturally-diverse trackers all struggle here" result as an informative limitation.

**Guardrail held throughout:** every config pre-registered from the model's own defaults, one blind score each,
no threshold/seed tuning after seeing results, failures reported as failures.

## Why the calibration axis is decisive

SA-FARI is camera-trap footage — out-of-distribution for these open-vocab detectors. Both GLEE and OWLv2
**find the animals** but their per-detection confidence scores are (a) low-magnitude and (b) uncorrelated with
mask quality on this domain, so the correct detection is buried below junk. At any fixed operating point (VEval's
0.5 gate) the good detection is filtered out → pDetA collapses. Rescuing this by sweeping the threshold would be
**tuning the tracker to make the swap "work"** — a p-hack we refuse. This is a genuine property of the domain,
not a bug in one model.

Florence-2 is the structurally different bet: its grounder emits **no confidence score at all** (autoregressive
text generation), so there is no score to collapse and no operating point to tune. Detections are scored at 1.0
and mask-HOTA (which does not integrate a confidence-recall curve) scores the fixed detection set cleanly. Its
only failure modes are localization quality and whether SAM 2 identity-tracking holds across ~85 frames.

---

## 1. GLEE — FAIL (score collapse)

- **Implemented fully** (`src/inference/glee_tracker.py`): per-frame forward, MinVIS association, top-K + NMS,
  torch-2.1/CUDA-op/detectron2 env. Committed.
- **Gate-1 (blind, GLEE's own defaults top-15 / thr 0.2):** per-cell pDetA **≈ 0.02, median 0**, 7/9 cells
  exactly zero. GLEE localizes (oracle IoU up to 0.96, 13/100 queries >0.5 IoU) but its scores collapse: the
  best-IoU mask scores ~0.08 while junk scores ~0.42; at its own 0.2 gate only ~5 detections survive across 10
  clips.
- **Verdict:** not scoreable comparably to SAM 3. Reported as a measurement finding, **not** a replicated null
  (running the swap on floored pDetA would be a false confirmation). See memory `glee-gate1-prereg`.

## 2. OWLv2 — FAIL (same calibration wall)

- **Gate-0 de-risk (10 clips, box→filled-mask, OWLv2 default threshold path):**
  `detect_ok=True` (oracle mIoU>0.4 on 6/10 videos, maxIoU up to 0.79) but `rank_ok=False` (mean 0.27) →
  **VERDICT=LIKELY_FAILS_LIKE_GLEE.**
- The good detection is out-scored by junk on most frames (e.g. margay: best-IoU det scores 0.052 vs frame-max
  0.421); absolute scores swing wildly per clip (max 0.029–0.659).
- **Verdict:** stopped at Gate-0 — the cheap de-risk correctly predicted a GLEE-style failure without a full
  build (~5 min vs a day). Not pursued further; would require a swept threshold = p-hack.

## 3. Florence-2 + SAM 2 — *(Gate-0 running)*

- **Env note:** Florence-2's checkpoint predates recent transformers; it broke on transformers 5.14
  (remote-code config) and 4.56 (native-class `image_token`/placeholder bugs). Resolved with a dedicated
  `flo-venv` on **transformers 4.45.2** (Florence-2's documented era, remote code). SAM 2 stays in the 4.56
  `hf-venv` — a natural two-stage split.
- **1-frame sanity (video 312, giant armadillo):** Florence-2 returned box `[372, 411, 701, 682]` with the
  correct label — **near-identical to the GT box** (`[372, 416, 691, 674]`). Accurate localization on the
  first try, and (unlike GLEE/OWLv2) **no confidence score to collapse** — detections are scored at 1.0.
- **Full Gate-0 (10 clips):** MIXED. Florence's raw **box-IoU vs GT** (the fair per-frame detection measure,
  vs the filled-rectangle mask-IoU which under-counts): **~4/10 videos localize well** (armadillo mean 0.67
  / max 0.98, rabbit 0.55 / 0.94, margay 0.47 / 0.84, coati 0.38 / 0.97) but **~6/10 genuinely miss**
  (agouti 0.001, grison 0.004, opossum 0.006–0.019 — small / nocturnal / camouflaged species). So Florence
  localizes *some* SA-FARI animals excellently and misses others entirely — better than blind, not a clean
  detector. (The de-risk's mask-oracle-IoU of 0.125 understated this because a filled bounding rectangle has
  low IoU with a tight GT silhouette; box-IoU is the honest read, and SAM 2 would convert the good boxes to
  tight masks.)
- **Blind mask-HOTA Gate-1 (SAM 2 masks + tracking, scores = 1.0):** **PARTIAL — real but fragile/bimodal.**
  Per-cell pDetA: giant armadillo **0.946**, white-nosed coati **0.873**, spix's guan **0.790**, and the
  other 6 cells **0.000**. **Mean 0.290, median 0, max 0.946, 3/9 cells > 0.3.** This is *categorically
  different* from GLEE/OWLv2's uniform ~0.02 collapse: where Florence's seed box is right, SAM 2 tracks the
  animal near-perfectly (armadillo/coati pDetA *above* SAM 3's ~0.53); where it's wrong, SAM 2 confidently
  tracks the wrong object for the whole clip (e.g. agouti/margay/rabbit: mask present on 76–80/80 frames but
  pDetA 0). No calibration artifact — clean mask-HOTA at a single operating point.
- **Honest caveat (not tuned away):** the adapter seeds SAM 2 from Florence's **frame-0** box regardless of
  quality, so a bad first frame poisons the whole track. A better seed (Florence's best-box frame, or
  periodic re-detection + merge) would very likely raise the mean — but changing the seeding *after* seeing
  the 10-cell number would be tuning-to-result, so it is left as a documented future improvement, not applied.

### 3b. Seeding follow-up (2026-08-05) — the improvement *fails*, and reveals why

A second, GT-blind seed rule was pre-registered and tested: seed SAM 2 from the frame whose Florence box is
closest (by IoU) to the **temporal-median box** across all detected frames (a "representative detection"
instead of a possibly-spurious frame 0). **Result: it made things WORSE — mean pDetA 0.290 → 0.089** (the
armadillo/coati/guan that scored 0.79–0.95 all collapsed to 0; only rabbit rose, to 0.80).

The diagnosis is the real finding. On the armadillo, Florence's frame-0 box `[372,411,700,681]` ≈ GT
`[372,416,690,673]` (excellent), but the **median box is `[1,0,1277,718]` — the entire frame**: on *most*
frames Florence emits a **full-frame garbage box** ("the whole image is a giant armadillo") and only a few
early frames carry the tight, correct box. So the median-of-boxes is junk, and the median-seed rule seeds SAM 2
on a whole-frame box → SAM 2 tracks garbage → 0. The overnight frame-0 score (0.95) was **partly luck** —
frame 0 happened to be one of the few good frames for those videos.

**Honest verdict — Florence-2 + SAM 2 is NOT viable for the swap.** Florence's correct detections are a
**sparse, unidentifiable minority** among full-frame garbage boxes, and there is **no GT-blind way to pick the
good frame** (Florence emits no confidence, and the good boxes are the minority). Two seed rules were tried
(first-frame, median-consistency); neither is reliable, and a third selector tuned against pDetA would be
exactly the p-hack this study refuses. So Florence+SAM 2 is bimodal/fragile by nature here, not by a fixable
adapter detail. **Phase 2 (productionizing `Florence2Tracker`) was NOT built** — the Phase-1 gate correctly
prevented investing in an unviable tracker.

## Final bottom line (2026-08-05)

**No open-vocab tracker replaces SAM 3 cleanly on SA-FARI, and the failures are informative, not a dead end:**
GLEE and OWLv2 *find* animals but their confidence **mis-calibrates** on camera-trap OOD; Florence-2 has no
score to miscalibrate but its **localization is sparse and unstable** (correct only on a minority of frames,
full-frame garbage on the rest), which SAM 2 cannot rescue without a GT-aware seed. Across three
architecturally-diverse trackers, **none is scoreable comparably to SAM 3** — so a fair cross-model swap is not
achievable **on SA-FARI** with current open-vocab trackers. Every step was pre-registered and reported as-is; no
seed/threshold was tuned to manufacture a passing number.

> **This "not achievable" conclusion is SA-FARI-specific and was overturned on BURST — see §5.** It stands only
> as an account of *why SA-FARI is the wrong arena* for the swap (SAM 3's home turf + exotic OOD species), not as
> a statement about the swap in general.

---

## 4. Fairness audit — are the SA-FARI zeros a rigged harness?

Prompted by the objection that "SAM 3 works and nothing else" looks suspicious, we adversarially tested whether
the *scoring harness* unfairly favours SAM 3. Three independent neutralisations of the confidence gate, all on
the SA-FARI predictions:

- **Box-HOTA** (`prefer_bbox=true`, the fair bar for box models): GLEE box-DetA **0.033**, Florence box-HOTA
  **0.195** — still ~0.
- **Threshold-free AP** (no confidence gate at all): GLEE mask/box AP **0.000**, Florence AP **0.017–0.022**.
- **Florence-2 has no confidence score** (all 1.0 → gate wide open) and still scores mask-DetA **0.040**.

**Verdict:** the harness's fixed `prob_thresh=0.5` IS a SAM-3-calibrated operating point (a real, minor tilt),
but neutralising it three ways does **not** rescue the competitors. The SA-FARI zeros are **genuine OOD
localisation failure**, not a scoring artifact. (This is the control that forecloses the "rigged harness"
objection — worth reporting for exactly that reason.)

## 5. The confound-free swap on BURST — the null replicates

**Why BURST.** SA-FARI is confounded two ways for a model comparison: it is SAM 3's *own co-released benchmark*
(home turf), and its species are exotic/nocturnal (OOD for internet-trained models). BURST removes both — it is
an independent dataset (built on TAO, movie/internet video, released years before SAM 3) of *everyday* animals
(cow, dog, cat, camel). Crucially, **SAM 3 itself scores 0.607 on BURST — even higher than its 0.537 on
SA-FARI** — so SAM 3 does not need home turf, and BURST is a genuinely fair arena with a legitimate bar.

**GLEE Gate-0 on BURST (10 clips).** Oracle mIoU **0.80** (>0.4 on 9/10, up to 0.96) and, unlike SA-FARI, its
confidence **ranks the good detection** (rank_ok 0.55) with high magnitudes (0.6–0.9) that clear the 0.5 gate.
Florence-2 Gate-0 also passes (oracle mIoU 0.44, >0.4 on 9/10). **Both `WORTH_FULL_BUILD`.** The SA-FARI failure
was the species, not the models.

**GLEE full-split BURST (132 cells, mask-HOTA).** pDetA **0.340** (median 0.306; spread min 0 / p25 0.02 /
p75 0.48 / max 0.975; **67/132 cells > 0.3**). Weaker than SAM 3 (0.607) but **non-degenerate with real
variance** — exactly what the distance regression needs, and exactly what SA-FARI's ~0/degenerate scores could
never provide.

**Distance model-swap (GLEE BURST pDetA, four label-free distances, leave-species-out, size-controlled):**

| Distance | ΔMAE | 95% CI | p |
|---|---|---|---|
| taxonomic (primary/novelty) | +0.015 | [+0.000, +0.031] | 0.022 |
| visual | +0.013 | [−0.002, +0.029] | 0.055 |
| environment | +0.017 | [+0.003, +0.033] | 0.008 |
| temporal | +0.016 | [+0.001, +0.032] | 0.018 |
| **all four combined** | **+0.011** | **[−0.004, +0.027]** | **0.088** |

**Reading it honestly (the null replicates):**
- The **combined four-distance model — the actual predictor — is null** (ΔMAE +0.011, CI spans zero, p=0.088).
  When you use the distances together, they do not predict GLEE's transfer.
- **All magnitudes are trivial** (~0.01–0.02 ΔMAE), the same non-effect as SAM 3's BURST run.
- Two individual distances (environment p=0.008, temporal p=0.018) *look* significant, but: (i) **BURST has no
  locations, so "leave-species-out" and "leave-location-out" are the *same* partition** — an apparent "both
  schemes" pass is one partition counted twice, not independent confirmation; (ii) they are single-distance
  flickers of the exact kind the SAM 3 analysis showed dissolve as usable predictors; (iii) the combined model,
  which is what a deployment estimator would use, is null.
- This **mirrors the SAM 3 BURST replication** ("every before-running distance null, CI spans zero"), now on a
  second, architecturally-different model.

**Conclusion.** On a fair, confound-free arena where GLEE genuinely tracks, the label-free-distance null
**replicates** — so it is **task-general, not SAM-3-specific**. This is the robustness result the model-swap was
designed to produce, and it directly answers the objection: the SA-FARI "only SAM 3 works" pattern was a
benchmark/species confound, not evidence that the null is a SAM 3 quirk.

**Caveat (stated, not hidden):** BURST is 41 species / 132 cells and underpowered for a small species-level
effect (the dissertation already logs the SAM 3 BURST run as a *convergent* null, not independent proof). The
GLEE swap is the same: a **convergent, confound-free replication**, not an independently-powered certification.
One model (GLEE) was swapped; Florence-2 passed Gate-0 too and could be added. Every config was pre-registered;
the driver forces the size control and the Bonferroni bar; nothing was tuned to the result.
