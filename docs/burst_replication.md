# BURST (R7) — a better-powered, mask-native replication of the before-running null

A self-contained record of the second independent replication, its result, and the honest interpretation.
Raw material for the Results/Discussion prose (which stays the student's). Every number was adversarially
re-derived against the raw parquet by three independent skeptics; reproduce with
`scripts/burst_replication_analysis.py`.

## Why BURST

MammAlps ([`docs/mammalps_replication.md`](mammalps_replication.md)) was consistent with the null but
**underpowered** — only 5 species / 20 cells, so even a genuinely predictive feature could not validate. BURST
(mask-native, LVIS-class multi-object tracker built on TAO) supplies the missing scale: the **animal subset of
BURST val is 41 species / 132 video×class cells**, mask-native, with a genuinely non-degenerate association
target. It is the many-species test MammAlps could not be.

## What was built (this session)

Same abstraction boundary as MammAlps: an offline adapter emits the SA-Co `_ext` schema; the loader, VEval
scorer, splits, features and modelling are untouched.

- **`src/adapters/burst.py`** — parses BURST `all_classes.json` (sequences of COCO-RLE masks keyed by track +
  `track_category_ids`), keeps only the **animal** categories (the packaged `burst_taxonomy.csv`, a
  WordNet-derived whitelist of the 41 animal LVIS classes in val), and emits the RLE masks directly.
  **Mask-native**, so `eval.prefer_bbox` stays off and scoring is mask-HOTA.
- **`src/adapters/burst_taxonomy.csv`** — 41 animals; **25 with full 7-level biological taxonomy** (tree
  distances verified: zebra↔horse=1, dog↔cat=3, zebra↔chicken=5), the rest partial → `NaN` (honest).
- **Frame acquisition (the old blocker, now solved):** BURST masks stream from RWTH `omnomnom` and the TAO
  frames from the (now token-authorized) HF `chengyenhsieh/TAO-Amodal` per-source zips — both reachable and
  range-enabled, so `scripts/extract_burst_frames.py` pulls only the ~2,300 needed frames, never the full
  archive. The unreachable `motchallenge.net` is not used.
- Ran end-to-end on a RunPod RTX A4000. **115 videos, 41 species, 132 probes.**

Cell = (species, video); BURST has no site/time metadata, so location = per-video `seq_name`, time is empty,
and **leave-species-out is the only valid CV** (`cv.group_schemes = ("species",)`).

## Gate 1 — measurement (sanity)

Per-cell support-weighted **pDetA 0.607 / pAssA 0.688 / pHOTA 0.635** (mask-HOTA; dataset mask mAP 0.489, bbox
mAP 0.508). Non-degenerate — SAM 3 tracks the LVIS animals well zero-shot. Unlike MammAlps/SA-FARI the
**association target is real**: `corr(pDetA,pAssA)=0.755`, mean|pDetA−pAssA|=0.135, 71/132 multi-object cells,
**0/71** multi-object cells with pAssA exactly equal to pDetA (SA-FARI had 51%). So the pAssA null is meaningful.

## Result — a third convergent null (but honestly caveated)

The pod's kitchen-sink GLM gives leave-species-out ΔMAE **+0.048, p=0.003** — but that is **entirely the
confidence features** (`conf_mean_score` coef +4.10 [2.15, 7.28]; every distance coef CI spans zero). The clean
separated specs (leave-species-out, `scripts/burst_replication_analysis.py`):

| spec (pDetA) | ΔMAE | 95% CI | p | |
|---|---|---|---|---|
| **distances (tax+vis+env)** | +0.0016 | [−0.007, +0.011] | 0.374 | **null** |
| visual only | +0.0030 | [−0.005, +0.013] | 0.257 | null |
| taxonomic only (clean n=82) | −0.0058 | — | 0.952 | null |
| environment only | −0.0014 | [−0.005, +0.003] | 0.747 | null |
| size only [control] | +0.0050 | [−0.003, +0.014] | 0.143 | n.s. |
| `conf_mean_score` [after-running control] | +0.0352 | [+0.011, +0.063] | 0.004 | **fires** |
| `conf_atc` [after-running control] | +0.0287 | [+0.004, +0.055] | 0.012 | **fires** |

**1. The before-running distances are a genuine, leakage-free null.** Every distance's paired group-bootstrap
CI spans zero, on both pDetA and pAssA. Robust to dropping the 19 single-cell species (+0.0002), to a size
control, and to the clean full-taxonomy subset (which is *more* null). This is a **better-powered replication
than MammAlps** (132 cells vs 20), in a new mask-native many-species regime.

**2. The confidence controls fire — but this proves pipeline *liveness*, NOT power for a distance.** This is
the key honesty point (and where an initial over-reading was corrected). `conf_mean_score`/`conf_atc` validate
(p=0.004/0.012), a real contrast with MammAlps where *nothing* fired — so the test is not dead or misconfigured
at n=132. **But** the confidence win runs mainly through a **within-species** channel: `conf_mean_score` is 73%
within-species variance, and **collapsing it to species-means (making it distance-like) kills the win: +0.0044,
p=0.212.** Distances are 100% species-constant, so leave-species-out can only use the **between-species**
channel, where the correlations are visual **+0.37 (opposite direction)**, taxonomic −0.02, environment +0.09, versus
`conf_atc` +0.50. A power simulation put the 80%-power minimum detectable between-species |r| at ≈0.65–0.70. So
the design detects a *large* between-species predictor (conf_atc) but is **underpowered for a small one**.

**3. The one near-signal is a confound, not a finding.** `visual_distance` correlates with pDetA at +0.170 —
the **opposite direction** to the novelty hypothesis and a **size confound** (`corr(visual,log_area)=+0.40`; partial-r
given log_area = +0.082, p=0.348). It does not even reach OOS significance.

## Honest headline

BURST is a **third convergent null** on before-running label-free distances (after SA-FARI and MammAlps): under
leakage-free leave-species-out, no distance predicts detection *or* association transfer, in a better-powered,
mask-native, many-species, non-degenerate-association regime. It shows **no moderate-or-large before-running
distance signal**, consistent with SA-FARI — but the 41-group design is **underpowered to exclude a small
species-constant effect**, and the only feature that validates remains SAM 3's own after-running detection
confidence, a mechanistically near-circular, non-novel signal. It is convergent evidence for the null, **not
positive proof** of it.

## Scope caveats

- Underpowered for a *small* species-level effect (between-species channel only; MDE |r|≈0.65–0.70 at 80%).
- Only 3 of 4 distances testable (temporal all-NaN; taxonomic 25/41 species, and *more* null on the clean
  subset).
- 19/41 species are single-cell (null survives dropping them).
- Confidence controls are near-circular after-running self-assessments — used only as a liveness/consistency
  check, never as a novel predictor.
- Mask-native, no site/time metadata → leave-species-out is the only valid CV.

## Reproduce

```
# on the pod (GPU): full pipeline
bash scripts/run_burst.sh
# anywhere (CPU): separated specs + confounds + the species-mean-collapse power caveat
PYTHONPATH=. python scripts/burst_replication_analysis.py --features outputs_burst/features.parquet
```
