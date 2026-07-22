# After-running confidence estimator — method, novelty issue, mechanism, results

**Status:** deliberately **NOT in the dissertation.** It is a real, validated result, but it undercuts the
dissertation's novelty claim (see §2), so it lives here as a standalone record. The dissertation stays scoped
to the *before-running, label-free distance* story (a rigorous, decomposed H0) plus the false-positive
correlate.

Run once on the pod (`2026-07-20`), artefacts: `outputs/confidence_experiment_summary.json`,
`outputs/features_conf.parquet`. Code: `src/features/confidence.py`, `CONFIDENCE_COLS` in
`src/analysis/regression.py`, driver `src/analysis/confidence_experiment.py`, tests `tests/test_confidence.py`.

---

## 1. The method we used

An **ATC-style (Average Thresholded Confidence) detection-reliability estimator**, ported from image
classifiers to SAM 3's promptable *video tracking*. Instead of predicting `pDetA` from *external* label-free
distances (taxonomic / visual / environment / temporal), we predict it from **SAM 3's own outputs on the
target cell** — the confidence and coverage of the masklets it returned.

Per positive cell (species present) we read the prediction JSONs and compute four features:

- **`conf_atc_coverage`** — fraction of the cell's masklets scoring `≥ t`. The single threshold `t` is
  calibrated **once** on the frozen *train* reference so that the reference mean coverage equals the reference
  mean `pDetA`, then frozen before any test cell is scored (canonical ATC, Garg et al. 2022).
- **`conf_mean_score` / `conf_median_score`** — over the cell's masklet scores (each score is the
  *max-over-frames* per-masklet confidence; the tracker collapses the trajectory at write time — a
  masklet-level, not per-frame, ATC).
- **`conf_frame_coverage`** — fraction of the cell's frames on which ≥1 masklet is present.

**Validation — exactly the bar the distances had to clear.** Support-weighted fractional-logit GLM
(`var_weights = n_frames`), mandatory covariates `log(n_frames)` **and** `log_area` (the size confound that
killed the visual effect). Two honest bars:

1. **Grouped CV** — leave-one-**species**-out *and* leave-one-**location**-out, whole groups held out, refit
   per fold, vs a mean-predictor baseline; significance by a paired group-bootstrap (one-sided `p`).
2. **Group-cluster-bootstrap coefficient CIs** — resample whole species/locations, refit, percentile CI.

Pre-registered: primary `atc_coverage`; `mean_score` + `frame_coverage` secondary, Bonferroni over the
3-feature family (α = 0.0167). Detection (`pDetA`) only; no association claim.

## 2. The novelty issue (why it stays out of the dissertation)

The dissertation's contribution is a **before-running, label-free** transfer predictor for a promptable video
tracker, decomposed into detection vs association. This estimator conflicts with that on three counts:

1. **After-running, not before-running.** It requires *running SAM 3 first* — you have paid the inference cost
   (only the annotation cost is saved). That is a strictly weaker, different claim than deciding trust
   *before* spending any compute, which is the dissertation's whole premise.
2. **Confirmatory, not novel.** It is the established label-free-accuracy family (ATC / DoC / Accuracy- and
   Agreement-on-the-Line) applied almost unchanged. The only new wrinkle is "on a video tracker's detection
   half" — a low-novelty application, not a new idea.
3. **Mechanistically close to the target.** The feature and the target are **both functions of SAM 3's own
   output**. `conf_atc_coverage = 0` on 64 cells, and *all 64* have `pDetA = 0` — confidence-zero perfectly
   flags total-detection-misses. So it is nearer a *self-consistency check* than an independent predictor.
   (It is **not leakage** — `t` is frozen on train, the CV has a per-fold standardisation firewall, and the
   feature uses zero ground truth — but the closeness makes the "prediction" much less surprising than
   predicting from external properties.)

Putting it in the dissertation would invite the examiner to ask "isn't this just the model telling you it's
confident?" — which dilutes the sharper, harder, genuinely-novel before-running result. Better kept separate.

## 3. How it works (the mechanism)

On **positive** cells (the animal is present), whether SAM 3 *detects* it is strongly reflected in the
confidence and coverage of what it returns:

- If it returns few / low-confidence masklets → it mostly failed to find the animal → `pDetA` is low (and, in
  practice here, `pDetA = 0` whenever coverage is 0).
- If it returns confident, well-covered masklets → it found and followed the animal → `pDetA` is high.

The confidence↔accuracy link is a **calibration property of SAM 3**, so it is species- and location-agnostic —
which is exactly why it *transfers* to held-out species/locations (the ATC premise). The threshold `t`, frozen
on the seen reference, turns raw coverage into a calibrated accuracy estimate. Empirically the relationship is
near-deterministic at the bottom (no confident masklet ⇒ miss) and graded in the middle (more confidence ⇒
higher `pDetA`), which is why a single feature plus size/support controls already explains most of the
variance.

## 4. Results

Target `pDetA`, **346 positive test cells**, controls `log(n_frames)` + `log_area`. Baseline = mean predictor
(MAE 0.298).

**Out-of-sample gain (ΔMAE = baseline − model; both schemes must clear Bonferroni α = 0.0167):**

| feature | leave-species-out | leave-location-out | verdict |
|---|---|---|---|
| `atc_coverage` (primary) | ΔMAE **+0.143**, MAE 0.154, p = 0.0 | **+0.143**, MAE 0.154, p = 0.0 | ✅ validated |
| `mean_score` | +0.168, MAE 0.130, p = 0.0 | +0.168, MAE 0.129, p = 0.0 | ✅ |
| `frame_coverage` | +0.092, MAE 0.207, p = 0.0 | +0.092, MAE 0.206, p = 0.0 | ✅ |

**Full model (all 4 distances + all 4 confidence features + `log_area` + `log(support)`), group-cluster-bootstrap
95 % CIs, `pDetA` (pseudo-R² = 0.758):**

| predictor | coef | 95 % CI | excludes 0 |
|---|---|---|---|
| `conf_atc_coverage` | 9.37 | [7.78, 12.06] | ✅ |
| `conf_frame_coverage` | 0.56 | [0.29, 0.80] | ✅ |
| `conf_mean_score` | 0.23 | [−1.59, 2.16] | ✗ (collinear with the other conf features) |
| `taxonomic_distance` | −0.08 | [−0.53, 0.96] | ✗ (H0) |
| `environment_distance` | −0.02 | [−0.14, 0.10] | ✗ (H0) |
| `visual_distance` | 0.20 | [0.005, 0.41] | ✅ but the known size-confounded wrong-sign artefact |

**Diagnostic:** `pDetA = 0` on 18.8 % of cells; the 64 `atc_coverage = 0` cells are exactly those misses
(mean `pDetA` 0.000); the 282 `atc_coverage > 0` cells average `pDetA` 0.659. Correlations with `pDetA`:
`mean_score` 0.83, `atc_coverage` 0.75, `frame_coverage` 0.59, `log_area` 0.31, `n_frames` −0.02.

**Bottom line.** The confidence features clear **both** honest bars (clustered CI *and* grouped CV) that all
four before-running distances failed — but as an *after-running, confirmatory, mechanistically-close* estimator
for **detection only**. The `pAssA` numbers merely mirror `pDetA` (because `pAssA ≈ pDetA` on these
single-object cells); the association null stands. One-line honest summary: *label-free distances do not
predict SAM 3's transfer, but SAM 3's own detection confidence predicts its detection accuracy — a weaker,
after-running claim that we keep out of the dissertation to protect the before-running novelty.*
