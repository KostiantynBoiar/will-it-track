# Q&A prep

Two questions, answered the way I'd say them out loud.

---

## Q1. "What is Section 4.2 for?"

This section is the first sanity check, before the real test.

For each distance I ask a very basic question: **is its effect distinguishable from zero at all?** The answer is no — every confidence interval crosses zero, so I can't tell any of them apart from having no effect whatsoever.

So the distances fail at the easiest hurdle. That's why the out-of-sample test in the next section isn't a surprise — nothing here ever looked promising enough to survive it.

**If she asks why the taxonomy row has that huge interval `[−3.87, +0.09]`:**

Short answer: that row isn't *measured and refuted* — it's barely measured at all, and the width of the interval is the model honestly telling me so.

Here's the mechanism. A regression coefficient answers "when this factor changes, how much does the score change?" — so it can only be estimated from cells where the factor actually *does* change. On this split it almost never does. Of the 840 cells that have a taxonomic distance at all, **778 of them — 92.6% — are exactly zero**. The variable takes only three distinct values across the whole split (0, 2 and 3), and just **62 cells are non-zero**.

Worse, taxonomic distance is constant within a species, so what matters isn't the cell count but how many *independent species* carry a non-zero value. That number is **7**. Four of them sit at distance 2 and three at distance 3. So the entire taxonomy coefficient is being estimated from seven species, spread over two levels.

That's exactly why the interval explodes. My confidence intervals resample whole species rather than individual cells (because cells within a species aren't independent), and when only seven species carry any signal, a resample will sometimes draw hardly any of them. The coefficient swings wildly from one resample to the next, and `[−3.87, +0.09]` is the honest record of that instability.

The reason it happens is structural, not accidental: SA-FARI's official split is disjoint by **location**, and deliberately shares species across train and test. Species novelty is the one thing that split holds fixed — so asking it to measure species novelty is asking the wrong question of the wrong data.

Which is precisely why I built the species hold-out. It repartitions the same videos so that species novelty genuinely varies, and on that split the taxonomy coefficient is estimated from real variation — and comes back flat, and visual distance comes back positive, the opposite of H1.

---

## Q2. "What is a species hold-out?"

The general idea: you hide part of the data, then test on the hidden part.

**A species hold-out** means the thing you hide is a whole species. I take one species out completely, build the reference from the remaining ones, and then ask: how far is that species from everything I kept, and does the tracker actually do worse on it? Then I repeat it for every species in turn.

**Why I had to build one.** SA-FARI's official split hides *locations*, not species — the same animals appear on both sides. So every species is already "seen", the novelty distance is ≈0 everywhere, and there's simply no variation to test H1 against.

**Why it isn't cheating.** SAM 3 is frozen — it never trained on any of SA-FARI. So "held out" isn't a claim about what the model learned; it's only my choice of which species count as the reference for measuring distance. I'm free to redraw that line, and I say so explicitly in the methodology.

**The punchline.** On that split, where novelty genuinely varies, the coefficients came back **positive** — more-novel species detected *better*, the opposite of H1 — and that turned out to be the animal-size confound.

---

## Q3. "What are the two splits, and where is that written?"

**Where it's documented.** The definition is in **§3.2, "Dataset, splits, and the analysis unit"** (pages 6–7) — the paragraph headed *"Two complementary splits."* It's referred to again in §3.4 (which axis is genuinely disjoint), §3.5.2 (the visual prototype is leave-species-out on Split A), and §3.6.3 (Split A is validated by leave-species-out CV). The Split A **results** are in **§4.5**, page 16.

**Split A — species hold-out (primary).** I hide one species at a time, so a held-out species' distance is measured to the nearest *other* species. Species novelty varies while the environment stays familiar, which is what isolates H1. **1,025 cells across all 99 species.**

**Split B — location hold-out (secondary).** The official SA-FARI train/test split, whose unseen camera sites isolate H2, and whose comparability with the published benchmark gives me the score-sanity check. **1,447 cells — 346 positive (the species is present, so the scores are defined) and 1,101 hard negatives**, drawn from 1,486 hard-negative probes across 92 unseen locations.

**Why there are two.** The shipped split is disjoint by *location* but shares species, so on it species-distance is ≈0 for every cell and it simply cannot test species novelty. The two splits are near-orthogonal probes of the same videos: one varies the species, the other varies the place.

**Why building Split A is allowed.** SAM 3 is frozen and was never trained on any of SA-FARI, so the official split reflects nothing the model learned — it only defines *my* reference distribution for measuring distance. That means I'm free to repartition the same videos. The dissertation says this outright: *"Split A is a constructed hold-out — legitimate because the model is frozen, and reported as such."* Inference is run **once**, over the union of both splits, so the two experiments are overlays on the same scored cells rather than two separate runs.

---

## Q4. "The Night/IR coefficient is the biggest in the table — doesn't that mean something?"

It means something physically, but it doesn't survive testing — and I chased it deliberately rather than letting it sit there.

**Why it's big.** −0.405 says detection gets worse at night and in infrared, which is entirely believable: dark, monochrome, low-contrast footage is genuinely harder. It's also the closest any well-measured factor comes to significance — the interval `[−0.94, +0.11]` only just clears zero.

**But it fails in three separate ways.**

First, it crosses zero, so by my own pre-registered rule it fails criterion (i). "Nearly significant" isn't a result.

Second, and decisively: when tested properly out of sample, it turns out to be animal size in disguise. That is exactly what the nested decomposition in Table 4.4 is for:

| Model | Leave-species-out | Leave-location-out |
|---|---|---|
| Low-light **alone** (no size) | +0.003 (p=0.233) ✗ | +0.004 (p=0.123) ✗ |
| Size **alone** | +0.015 (p=0.014) ✓ | +0.017 (p=0.001) ✓ |
| Low-light **+ size** | +0.017 (p=0.011) | +0.020 (p=0.000) |

Low-light on its own does not beat guessing the mean. Size on its own does. And low-light + size is barely better than size alone — so darkness adds essentially nothing of its own.

Third, the correlation that made it look exciting is an averaging artefact. Measured **per location** it's **r = −0.377**, which looks convincing. Measured at the **cell** level, where the model actually operates, it collapses to **−0.13**. Clutter does worse still: exactly **0.00**. That's the ecological-correlation fallacy, and §4.3 names it as such.

**The line to use:** "Night/IR is the strongest hint in the whole table, and the intrinsic-difficulty pivot in §4.3 is me going after it. Once I isolate it from animal size, low-light alone doesn't beat the baseline on either scheme, and the correlation that motivated the pivot shrinks from −0.377 to −0.13 at cell level. It's a real physical effect that isn't a usable predictor."

**Why this actually helps me.** The most promising-looking signal in the document still didn't survive scrutiny. That's a much stronger position than never having had a candidate at all — it shows I pursued the best lead and reported honestly when it dissolved.

**One thing to concede if pushed:** I show *that* darkness and animal size are entangled, but I don't have a verified mechanism for *why*. It's an empirical entanglement, not a story I've confirmed.
