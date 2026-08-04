# Second-model swap — overnight comparison report

*Autonomous run, night of 2026-08-04. Goal: find a second text-promptable zero-shot tracker to run the
label-free-distance model-swap against SAM 3 (does the null replicate beyond SAM 3, or is it SAM-3-specific?).*

## TL;DR (fill final verdict once Florence-2 resolves)

Three candidates evaluated with a cheap **Gate-0 calibration de-risk** on the same 10 SA-FARI clips SAM 3 uses,
*before* any expensive full build. The de-risk separates two questions: **can the model localize the animal?**
(oracle IoU, scores ignored) and **do its confidence scores rank the good detection high enough to survive
scoring?** (the axis that killed GLEE). Super-honesty held throughout: configs pre-registered from each model's
own defaults, one blind score, **no tuning-to-result**.

| Model | Localizes? | Score calibration | Gate-1 pDetA (10 cells) | Verdict |
|---|---|---|---|---|
| **GLEE** (Lite-scaleup) | ✅ yes (IoU up to 0.96) | ❌ collapses | ≈ **0.02** (median 0) | **FAIL** — scores un-scoreable on camera-trap OOD |
| **OWLv2** (base-ensemble) | ✅ partial (box-IoU up to 0.79) | ❌ mis-ranks (rank_ok 0.27) | not reached | **FAIL** — same calibration wall (stopped at Gate-0) |
| **Florence-2 + SAM 2** | ⚠️ mixed (4/10 well, box-IoU up to 0.98) | N/A — no score to collapse | **0.29** mean (max 0.946; 3/9 cells > 0.3) | **PARTIAL** — genuinely works on a subset; not uniformly comparable |

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
