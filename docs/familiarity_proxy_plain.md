# The "familiarity proxy" — what the hell it is, in plain English

A blunt, no-jargon explainer of the thing we just built (called **T2.5** in the project). For the whole-project
version, see [`project_in_plain_english.md`](project_in_plain_english.md).

---

## TL;DR

We tried one more way to guess whether SAM 3 will work on a new animal **before running it** — this time by
asking **SAM 3 itself** how well it already "knows" the animal. Result so far: **it doesn't help** — the same
"no" as every other before-running trick we tried.

## The hole we're plugging

Everywhere else in the project, "how novel is this animal?" means "how far is it from the animals in **our**
reference pile (SA-FARI)." But there's a hole in that logic:

> **We don't actually know what SAM 3 was trained on.** Meta never published SAM 3's full training data.

So "far from *our* pile" isn't the same as "far from what SAM 3 *actually* saw." SAM 3 has almost certainly
seen zebras from *somewhere* — we just can't list where. Every "distance from training" number in the project
therefore comes with an asterisk: it's distance from **our** reference, not from SAM 3's real (secret) training
set. This one experiment is the attempt to remove that asterisk.

## The trick

Instead of measuring distance to a pile *we* picked, **ask the model directly.** Push each animal's picture
through SAM 3's own "eyes" (its internal image encoder) and look at how that animal sits inside SAM 3's mental
map of the world:

- If SAM 3 has seen an animal a lot, it should hold it as a **clean, distinct blob** in its feature space — it
  "knows" it.
- If the animal is genuinely new to SAM 3, it should be **fuzzy, smeared into other animals** — it's unsure.

That "how cleanly does SAM 3 represent this animal" score is the **familiarity proxy.** It reads familiarity
*straight from the model*, so it doesn't care what we don't know about its training data.

## How it actually works

1. Take the ground-truth cut-outs of each animal, run them through SAM 3's encoder → a list of numbers (a
   "fingerprint") per picture.
2. Measure how **distinct** each species is, three different ways:
   - **Silhouette** — is the species' own cluster tight *and* far from every other species? (clean = familiar)
   - **Nearest-species** — how far is it from the closest *other* species? (this is basically our old "visual
     distance," but using SAM 3's eyes instead of a different vision model's)
   - **Density** — how typical is it of everything SAM 3 has seen?
3. Run the exact same brutal honesty test every other feature faced: hide whole species, train a tiny model to
   predict SAM 3's grade from the familiarity number, and check whether it (a) beats just guessing the average
   and (b) — the decisive bit — **adds anything the plain visual distance + animal size didn't already give.**

Everything is **before-running** (it's a property of the pictures, not of SAM 3's tracking output — so it's not
circular) and **label-free** (no answer key needed).

## What we found

**BURST (41 species): NO.** None of the three ways beats "visual distance + size." All three are 60–80% the
*same thing* as the old visual distance (correlations 0.63–0.80) — they mostly just re-measure "how different
does it look," which we already know doesn't predict transfer and is tangled up with animal size. One flicker:
the **silhouette** version got close (p = 0.02) and, interestingly, was *not* a size artefact — but it didn't
clear the (stricter, multiple-test-corrected) bar, and BURST is too small to trust a flicker.

**SA-FARI (the big, powered one): running now.** _[This line updates when it lands — pre-registered
expectation: the same "no."]_

## Why a "no" here is still worth having

This was the one honest before-running trick we hadn't tried yet. Now we have, and it **closes the loophole**:
even when you read familiarity *from the model itself* — sidestepping the "you don't know SAM 3's real training
data" objection — you *still* can't forecast its transfer. That makes the overall "no" much harder to argue
with.

## Bottom line

We asked SAM 3 "how well do you know this animal?" and used the answer to predict whether it'll track it well.
It **doesn't work**: the model's own sense of familiarity turns out to be just another way of measuring "how
visually distinct does it look" — which we already knew was a dead end (and a size confound in disguise). One
more honest "no," one more loophole closed.
