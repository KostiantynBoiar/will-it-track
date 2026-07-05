# Roadmap — SAM 3 zero-shot transfer prediction

Phase summary of `.claude/IMPLEMENTATION_PLAN.md` (the authoritative per-task spec, with Goal /
Input / Output / Method / Done-when for every task). Verify-first: two decision gates.

- **Phase 0 — Setup & reference.** T0.1 reproducible env + SA-FARI + a working call to the official
  VEval on one clip. T0.2 **freeze the seen/unseen reference** (train species/locations + taxonomy
  manifest) with an asserted species+location disjointness test — the leakage firewall.
- **Phase 1 — Measurement.** T1.1 frozen SAM 3 inference over the test split (species-specific +
  generic prompts, hard negatives kept) → `outputs/predictions/`. T1.2 score with the **official
  VEval** → per-cell `pDetA`/`pAssA`/`pHOTA` + support → `outputs/scores.parquet`.
  **★ Gate: scores sensible and in the SA-FARI SAM 3 ballpark? Fix before feature work.**
- **Phase 2 — Distance features** (all label-free, against the frozen seen set). T2.1 taxonomic,
  T2.2 visual (DINOv2 mask-crop), T2.3 environment (scene + night/IR), T2.4 temporal, T2.5 SAM 3
  familiarity proxy, T2.6 assemble → `outputs/features.parquet` (+ a leakage unit test).
- **Phase 3 — Fit the law.** T3.1 per-target beta/logit GLM (support-weighted, `log(n_frames)`
  covariate) → `outputs/models/`. T3.2 variance partitioning (dominance/Shapley + VIF) + the
  detection↔species-novelty / association↔environment decomposition contrast.
- **Phase 4 — Out-of-sample validation.** T4.1 grouped CV (leave-species-out, leave-location-out) →
  `outputs/validation/`. T4.2 bootstrap CIs + the predictive-line figures → `outputs/figures/`.
  **★ Gate: any real out-of-sample signal? If not, invoke H0 and pivot to representational probing
  (does SAM 3 encode taxonomy?).**
- **Phase 5 — Ablations & robustness.** T5.1 factor ablations; T5.2 design/confound robustness
  (DINOv2 vs CLIP, cropped vs whole-frame, cell vs species aggregation, species vs generic prompt,
  ± support covariate).
- **Phase 6 — Tool & write-up.** T6.1 the reliability estimator (4 distances → `pDetA`/`pAssA` + CI).
  T6.2 the dissertation, every claim backed by a repo-generated figure/table.

## Dependency graph
```
T0.1 → T0.2 → {T2.1, T2.3, T2.4, T2.5}
T0.1 → T1.1 → T1.2 → {T2.2, T3.1}
{T2.1..T2.5} → T2.6 → T3.1 → T3.2 → {T4.1, T5.1, T5.2}
T4.1 → T4.2 → T6.1
{T3.2, T4.2, T5.1, T5.2, T6.1} → T6.2
```
