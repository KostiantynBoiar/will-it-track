# Q&A prep — questions to expect, and how to answer

Short, speakable answers. The number in **bold** is the one to quote if pushed.

---

## 1. The big one: "Isn't a null result just a failed experiment?"

**A.** No — a failed experiment is one you can't interpret. This one has a clear answer: label-free
distances don't predict transfer, and I can show the test wasn't broken, because the same test picks up
animal size out of sample (**ΔMAE +0.015 / +0.017, significant on both schemes**). A test that fires for one
thing and stays flat for another is telling you the signal isn't there.

> *If pushed:* the field currently assumes "far from training = unreliable". I showed that intuition doesn't
> hold for a promptable tracker. That's worth knowing before someone builds a deployment gate on it.

---

## 2. "How do you know it's not just underpowered?"

**A.** Three ways. (1) The positive control — size validates at the identical bar. (2) The machinery is
unit-tested to recover planted effects on synthetic data. (3) It replicates: same null on a second dataset
and two other trackers.

**Be honest about the limit:** I can exclude a *moderate* effect, not a small one. I don't have a formal
minimum-detectable-effect number — that's the main thing I'd add next.

---

## 3. "What is a species hold-out, and isn't constructing your own split cheating?"

**A.** I hide one whole species, build the reference from the rest, and ask how far that species is from what
I kept and whether the tracker does worse on it — repeated for every species.

I had to build it because SA-FARI's official split hides **locations, not species** — the same animals appear
on both sides, so novelty is ≈0 everywhere and H1 can't be tested at all.

**Why it's not cheating:** SAM 3 is frozen and never trained on any of SA-FARI. So "held out" isn't a claim
about what the model learned — it's only my choice of which species act as the reference for measuring
distance. I'm free to redraw that line, and I say so explicitly.

---

## 4. "You call the distances label-free, but you use ground-truth masks." ← hardest question

**A.** Fair, and I've scoped that explicitly. Label-free here means **no performance labels** — I never need
to know how well the tracker did in order to compute a distance. That's the property the research question
needs, because it's what makes the prediction *before-running*.

But you're right that the visual, environment and size features read GT masks to separate animal from scene.
That's a convenience of working on an annotated benchmark. In deployment you'd build the same prototype from
the handful of example images you already have for a named species. It's stated in §3.5 and in the
limitations rather than glossed over.

---

## 5. "What's section 4.2 actually for?"

**A.** It's the first and easiest of my two criteria: is each distance's effect distinguishable from zero at
all? Answer: no — **every confidence interval crosses zero**. So the distances fail at the softest hurdle,
before the out-of-sample test even starts.

---

## 6. "Why is the taxonomy interval so huge — [−3.87, +0.09]?"

**A.** Because on that split there's almost nothing to measure. The official split shares species, so
taxonomic distance is ≈0 in nearly every cell and the coefficient is estimated from essentially no variation.

**Key phrasing:** that's *measured imprecisely*, not *measured and refuted* — I count it as neither. Testing
species novelty properly is exactly what Split A exists for.

---

## 7. "Visual distance WAS significant on Split A. Isn't that a positive result?"

**A.** It's significant in the **wrong direction** — **+0.369**, meaning more-novel species are detected
*better*, the opposite of H1. And it's a confound: visually distinctive species are larger
(**r = 0.22 with log-area**), larger animals score higher (**r = 0.30**), and adding a size covariate shrinks
it from **+0.37 to +0.22** with the interval back across zero. So it was "bigger animals are easier"
wearing a novelty costume.

---

## 8. "Your positive control is animal size — but size is also your confound. Isn't that circular?"

**A.** They're different roles. As a *confound*, size explains away an apparent novelty effect. As a
*control*, it proves the pipeline can detect a real out-of-sample effect when one exists. The same variable
can do both without circularity.

**Concede the weak point:** size is a fairly mechanical effect (bigger = easier to segment), so it doesn't
prove I could have caught a *subtle* novelty effect. That's the power caveat again.

---

## 9. "You claim a detection/association decomposition, but pAssA ≈ pDetA. Is half your contribution empty?"

**A.** On SA-FARI, largely yes, and I say so: most cells contain a single object, so association tracks
detection almost exactly and there's little identity-specific variance for anything to predict. I report it
as a detection story rather than pretending it's two independent tests.

**The defence:** on BURST the association target *is* genuine — **corr(pDetA, pAssA) = 0.76**, with 71 of 132
cells multi-object — and the null still holds there. So the decomposition was tested where it's meaningful.

---

## 10. "Two datasets and three trackers — is that really 'task-general'?"

**A.** It's convergent, not proof, and I word it that way. The claim is narrow: the null isn't an artefact of
one dataset or one backbone, because it survives changing both. **GLEE +0.011 (p=0.088)** and
**Florence-2+SAM 2 +0.034 (p=0.030)** — both intervals span zero, neither clears Bonferroni (α=0.0125) —
while both trackers genuinely track BURST animals (**0.340 and 0.394**).

I explicitly don't claim independent statistical power: it's two models on one fair dataset.

---

## 11. "Why did you drop MammAlps? Convenient?"

**A.** Two reasons, both disclosed in the limitations. It's **~20 cells over 5 species** — too small to
certify a null either way. And its per-clip annotations aren't reproducible from the public release, so
nobody could verify the result. It was directionally consistent with the null, so dropping it doesn't help my
case — I report it as excluded rather than silently deleting it.

---

## 12. "The confidence estimator worked. Why isn't it in the dissertation — are you hiding a positive?"

**A.** No, it's stated openly in §5.4. An ATC-style estimator does clear the bar — but it's excluded for three
reasons: it's **after-running** (you must run the tracker first, which is a strictly weaker and different
question), it's confirmatory of an established literature rather than novel, and it's close to circular —
feature and target are both functions of the model's own output.

It's a real result at a different operating point, just not the before-running predictor this thesis asks
about.

---

## 13. "Surely novelty matters — a familiar animal is detected better than an alien one."

**A.** Agreed, and that's not in dispute. But that's the *coarse* claim. H1 is the far stronger *graded*
claim: that a continuous distance tells you where on the scale an unseen species lands, out of sample, beyond
size, well enough to act on.

They come apart because the useful variance is in the middle of the range — the easy extremes are already
predicted by the mean. So a real extreme-tail effect and a null graded predictor coexist fine.

---

## 14. "Why couldn't the model swap run on SA-FARI?"

**A.** Both challengers score ≈0 there, so there'd be no variance to model — a regression on flat scores
reports a fake "null". It's not the scorer: the zeros survive box-HOTA and a threshold-free AP metric with no
confidence gate at all.

The cause is the *domain*, not just the species — GLEE localises SA-FARI's everyday animals well (**IoU up to
0.90**) but its confidence collapses on dark camera-trap footage. BURST is the fair arena, and SAM 3 actually
scores *higher* there (**0.607 vs 0.537**), so there's no home-turf advantage.

---

## 15. "You measure distance from SA-FARI, not from SAM 3's real training data."

**A.** True, and it's the first limitation I list — SAM 3's corpus is undisclosed, so the null is scoped to
distance-from-SA-FARI.

**The hedge I ran:** the familiarity proxy reads separability from SAM 3's *own* features, and it's null too —
and it mostly re-encodes the visual distance (**r = 0.63–0.85**). So even model-internal familiarity doesn't
rescue prediction.

---

## 16. "So what should a conservation team actually do?"

**A.** Don't treat "close to training data" as a safety guarantee — it doesn't predict whether the tracker
will work. Check reliability with a small labelled spot-audit instead. If they're willing to run the model
first, its own confidence *is* informative.

---

## 17. "What would you do with more time?"

**A.** Three things: (1) a proper power analysis, so the null becomes "we exclude effects larger than X";
(2) a fuller representational probe — what the features encode structurally, not just whether they separate
species; (3) a larger fair mask-native dataset for the model swap.

---

## Numbers worth memorising

| Thing | Value |
|---|---|
| SA-FARI macro pDetA / pAssA / pHOTA | 0.527 / 0.546 / 0.533 (micro ≈0.65) |
| Cells | Split B: 1,447 (346 positive) · Split A: 1,025 over 99 species |
| Variance explained by all 4 distances | **≈3%**, every VIF ≈1 |
| Size (positive control) | ΔMAE **+0.015** LSO, **+0.017** LLO |
| Hallucination rate | **9.8%** on 1,486 hard negatives; r = −0.33; LSO p = **0.053** |
| BURST | 132 cells, 41 species, pHOTA 0.635 |
| Model swap | GLEE 0.340 (+0.011, p=0.088) · Florence-2+SAM 2 0.394 (+0.034, p=0.030) |
| Temporal gap | 32/1447 cells, all = 0 → untested |
