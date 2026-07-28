# MammAlps (R7) — an independent replication of the label-free null

A self-contained record of the second-dataset replication, its result, and the honest interpretation.
Raw material for the Results/Discussion prose (which stays the student's). Every number below was
adversarially re-derived against the raw parquet by three independent skeptics; reproduce with
`scripts/mammalps_replication_analysis.py`.

## Why MammAlps

Every SA-FARI experiment shares **one** frozen SAM 3 inference on **one** dataset, so the two splits × two
targets are near-orthogonal but not *independent* replications (a stated Discussion limitation). MammAlps
(Alpine camera-trap multi-animal tracker; Zenodo 15588220) is the honest independent-replication route: a
different continent, different species pool, different pipeline entry, and — uniquely — real per-camera
environment metadata. It is small by design, so it tests **consistency**, not power.

## What was built (this session)

The pipeline's abstraction boundary is the SA-Co `_ext` annotation schema, not the loader — so MammAlps is
added by an **offline adapter**, leaving the loader, VEval scorer, splits, features, and modelling untouched.

- **`src/adapters/mammalps.py`** — parses the per-clip dense JSONs, subsamples 30→3 fps, caps per
  `(species, camera)` cell, and emits the `_ext` schema. Box-only GT: each box becomes a filled-rectangle
  COCO-RLE (`src/adapters/boxes.py`) **and** a native bbox.
- **`eval.prefer_bbox`** — scores **bbox-HOTA** instead of mask-HOTA. Essential here: against filled-rectangle
  GT a tight predicted mask has near-zero IoU (dataset mask mAP ≈ 0.002), so mask-HOTA would falsely read
  "SAM 3 fails"; box-IoU is the honest metric (dataset bbox DetA 0.427).
- **`scripts/extract_mammalps_frames.py`** — streams only the ~135 needed clips from the 82 GiB Zenodo zip via
  HTTP range (`remotezip`) → ffmpeg, so no full download (peak disk ~5 GB).
- Ran end-to-end on a RunPod RTX A4000: extract → SAM 3 promptable inference → bbox-HOTA score → distances →
  leave-species-out + leave-camera-out CV.

**Scale:** 20 cells, 5 species (red deer 9, roe deer 6, hare 2, wolf 2, **fox 1**), 9 cameras.

## Gate 1 — measurement (sanity)

Per-cell support-weighted mean **pDetA ≈ 0.44 / pAssA ≈ 0.44 / pHOTA ≈ 0.44** (bbox-HOTA), non-degenerate:
SAM 3 genuinely finds Alpine mammals zero-shot. `corr(pDetA, pAssA) = 0.999` with 13/20 cells exactly equal —
so, as on SA-FARI, MammAlps adds nothing to the **association** question; everything below is detection.

## Result — the null replicates (directionally), but MammAlps is underpowered

The pod's single all-predictor GLM quasi-separates on 20 points (coefficients ~1e15), so it is uninterpretable.
The numbers below come from **cleanly separated** models fit with the identical leakage-free machinery
(`scripts/mammalps_replication_analysis.py`).

| spec (pDetA) | camera ΔMAE (p) | species ΔMAE (p) | reading |
|---|---|---|---|
| **distances (tax+vis+env)** | **+0.067 [−0.043,+0.144] (0.099)** | **+0.014 [−0.218,+0.162] (0.410)** | **NULL** (both CIs span 0) |
| visual_distance only | +0.078 (0.011) | +0.080 (0.010) | *spurious* — see below |
| taxonomic_distance only | +0.031 (0.239) | −0.083 (0.935) | n.s. |
| conf_mean_score [power probe] | −0.029 (0.920) | −0.054 (0.956) | fails despite in-sample r=+0.57 |
| SIZE only [pos. control] | +0.013 (0.244) | −0.015 (0.653) | fails (species-constant here) |
| CONFIDENCE atc [pos. control] | −0.036 (0.804) | −0.072 (0.852) | fails (regime-broken here) |

**1. The pre-registered distance model is a genuine, leakage-free null.** Both CIs span zero on both schemes
(camera p=0.099, species p=0.410). The CV is leakage-free (per-fold standardisation, group-disjoint firewall,
label-free DINOv2/GT-mask feature, conservative grand-mean baseline). This does **not overturn** the SA-FARI
null and is directionally consistent with it.

**2. It is a consistency check, not independent confirmation — the CV is underpowered.** The load-bearing,
non-circular evidence: `conf_mean_score` is a genuinely predictive per-cell feature (in-sample r=+0.57,
p=0.009) yet **still fails out-of-sample** (ΔMAE −0.03/−0.05, p>0.9). A real signal that cannot validate at
n=20 means the test lacks power. (Do **not** argue power from the `log_area`/`conf_atc` positive-control
failures: both are species-constant or regime-broken on MammAlps, so their failure is structural, not a clean
power readout.)

**3. The one "significant" feature is a confound, not a finding.** `visual_distance`-alone clears the bar
(ΔMAE +0.078/+0.080, p≈0.01) but must never be reported as a predictor:
- **Opposite direction.** `corr(visual_distance, pDetA) = +0.636` (p=0.003) — higher visual novelty ⇒ *better*
  detection, the opposite of the H1 hypothesis.
- **Support confound.** `corr(visual, n_frames) = +0.60`, `corr(visual, n_masklets) = +0.56`.
- **Species-constant, carried by 4 cells.** Values tie within species (red=roe=0.401, hare=wolf=0.154,
  fox=0.339); the effect lives entirely on the two rare, low-support species (hare, wolf) that share the
  lowest value and both fail. Drop those 4/20 cells → `corr = −0.325` (n.s.) and the OOS gain vanishes on both
  schemes. Its leave-species-out "validity" is the species-constant-feature pathology the double-CV bar exists
  to reject — a twin species (roe for red, wolf for hare) sits at the identical training coordinate.

## Scope caveats (state plainly)

- **Only 3 of 4 distances exist here:** `temporal_gap` and `familiarity_proxy` are all-NaN (6-week window,
  proxy deferred); `taxonomic_distance` is near-constant (2.0 everywhere except hare=4.0).
- **Underpowered by construction:** 20 cells, 5 species (one a single fox cell), 9 cameras.
- **Regime differs from SA-FARI:** box-only GT / bbox-HOTA, 3 fps, Alpine daytime video.

## Honest headline

MammAlps **reinforces, and does not contradict, the SA-FARI null**: the pre-registered label-free distance
model does not predict detection transfer out-of-sample, the only feature that clears the bar is an opposite-direction
support confound, and even a genuinely predictive per-cell feature cannot validate at this sample size. It is
an underpowered, directionally-consistent check — not independent confirmation of any positive result.

## Reproduce

```
# on the pod (GPU): full pipeline
bash scripts/run_mammalps.sh
# anywhere (CPU): the separated-spec analysis + confound diagnostics on the produced parquet
PYTHONPATH=. python scripts/mammalps_replication_analysis.py --features outputs_mammalps/features.parquet
```
