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
