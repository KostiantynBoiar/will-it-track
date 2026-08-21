# Q&A prep

Two questions, answered the way I'd say them out loud.

---

## Q1. "What is Section 4.2 for?"

This section is the first sanity check, before the real test.

For each distance I ask a very basic question: **is its effect distinguishable from zero at all?** The answer
is no — every confidence interval crosses zero, so I can't tell any of them apart from having no effect
whatsoever.

So the distances fail at the easiest hurdle. That's why the out-of-sample test in the next section isn't a
surprise — nothing here ever looked promising enough to survive it.

**If she asks why the taxonomy row has that huge interval `[−3.87, +0.09]`:**

That one isn't measured and refuted — it's barely measured at all. This split shares species between train
and test, so taxonomic distance is near zero in almost every cell and there's nothing to estimate from. That's
exactly why I built the species hold-out.

---

## Q2. "What is a species hold-out?"

The general idea: you hide part of the data, then test on the hidden part.

**A species hold-out** means the thing you hide is a whole species. I take one species out completely, build
the reference from the remaining ones, and then ask: how far is that species from everything I kept, and does
the tracker actually do worse on it? Then I repeat it for every species in turn.

**Why I had to build one.** SA-FARI's official split hides *locations*, not species — the same animals appear
on both sides. So every species is already "seen", the novelty distance is ≈0 everywhere, and there's simply
no variation to test H1 against.

**Why it isn't cheating.** SAM 3 is frozen — it never trained on any of SA-FARI. So "held out" isn't a claim
about what the model learned; it's only my choice of which species count as the reference for measuring
distance. I'm free to redraw that line, and I say so explicitly in the methodology.

**The punchline.** On that split, where novelty genuinely varies, the coefficients came back **positive** —
more-novel species detected *better*, the opposite of H1 — and that turned out to be the animal-size
confound.
