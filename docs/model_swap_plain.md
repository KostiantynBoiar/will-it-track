# The "second model" experiment — what we tried, the false start, and the fix, in plain English

A no-jargon walkthrough of the model-swap experiment. **Read this note in two halves:** sections 1–8 tell the story of the *first attempt* (on the SA-FARI dataset), which looked like a dead end — every other model "failed." Section 9 is the **turnaround**: a good question exposed that the dead end was an illusion caused by the *wrong dataset*, and on a fairer dataset (BURST) the experiment actually **works**. For the whole-project version see [`project_in_plain_english.md`](project_in_plain_english.md); the technical write-up with all the numbers is [`../report/second_model_comparison.md`](../report/second_model_comparison.md).

> **Spoiler / bottom line:** the model-swap is *not* a dead end. On a fair dataset, a competing model (GLEE) works, and our main "you can't predict it in advance" finding **holds for that model too** — proving it's a general truth, not a quirk of SAM 3.

---

## 1. Why we did this at all

Our whole project is a **negative result**: we showed you *can't* predict, in advance, how well SAM 3 will track an animal in a new place — the "distance" signals we tried don't work. That's a solid finding, but it has one honest weakness a reviewer (and Marwa) will poke at:

> "You only tested **one** model, SAM 3. Maybe your 'you can't predict it' result is just a quirk of SAM 3, not a general truth."

The fix is a **model-swap**: take a *second* tracker that works like SAM 3 (you give it the word "impala", it finds and follows impalas), run it over the **exact same videos**, and check whether our "can't predict it" result shows up for that model too.

- If the second model behaves the same way → our result is **general**, not a SAM-3 quirk. Much stronger.
- If it behaves differently → interesting, and worth explaining.

Either way we needed a second model that we could **score fairly against SAM 3**. That "score fairly" part is where everything went wrong.

## 2. The one number that decides everything: pDetA

To compare two trackers we need a score. Ours is **pDetA** — "did the model **find** the animals?" (a number from 0 to 1; SAM 3 gets about **0.53** on our data). Higher = found more of the real animals with less junk.

The catch: our experiment is a **regression** — we look at how pDetA *varies* across videos and ask whether our distances explain that variation. **That only works if pDetA actually varies.** If a model scores ~0 on *every* video, there's nothing to explain — every distance trivially looks "useless," and we'd be reporting a fake "it replicates!" when really the model just failed to produce a usable score. So a model with pDetA ≈ 0 everywhere is **worthless for the swap**, even if it's a fine model in general.

That's the bar each candidate had to clear: **produce a real, varying pDetA on our camera-trap videos.**

## 3. A cheap trick to avoid wasting days: the "Gate-0" check

Setting up each model is a day of work. So before committing, we ran a **10-video mini-test** that splits the question in two:

1. **Can it even see the animal?** (ignore its confidence — just check: does *any* of its guesses land on the real animal?)
2. **Do its confidence scores point at the right guess?** (the good guess needs a high score, or scoring throws it away)

This little check caught failures in minutes instead of days. It's the hero of this story.

## 4. Model #1 — GLEE

**What it is:** a single "find anything from text" model, closest in spirit to SAM 3.

**What happened:** GLEE **sees the animals fine** — its best guess overlaps the true animal almost perfectly (90%+). But it hands back ~100 guesses per frame, each with a **confidence score**, and here's the killer: **its scores are backwards on our data.** The *correct* guess gets a low score (~0.08); the *wrong* guesses get higher scores (~0.42). Scoring keeps the high-scoring ones — which are the junk — and throws away the good one. Result: **pDetA ≈ 0.02** (basically zero).

**Why it's fatal:** GLEE's confidence is trained on ordinary internet photos; on dark, cluttered camera-trap footage its confidence becomes unreliable. We *could* force a pass by hand-picking a score cutoff that happens to keep the good guesses — but that's **cheating** (tuning the knobs until the answer looks good), which we refuse to do. Verdict: **FAIL.**

## 5. Model #2 — OWLv2

**What it is:** a pure "find objects from text" detector — simpler than GLEE, runs in the same toolkit we already had.

**What happened:** the Gate-0 check flagged it in ~5 minutes. Same story as GLEE: it **sees** the animals (up to 79% overlap) but its **confidence scores don't rank the good guess highly** — the right detection is out-scored by junk on most frames. We **stopped before building the full thing** — no point.

**Why it's fatal:** exact same wall as GLEE. Verdict: **FAIL.**

**A pattern emerges:** two different detectors, same failure — their **confidence, not their eyesight**, breaks on wildlife footage. That's itself a small finding: on this kind of out-of-place data, "how sure is the model" becomes untrustworthy even when "what the model sees" is fine.

## 6. Model #3 — Florence-2 + SAM 2 (the one that almost worked)

**What it is:** a clever two-part combo picked *specifically* to dodge the confidence problem.

- **Florence-2** describes what it sees by **writing text** ("armadillo, at these coordinates") — so it gives **no confidence score at all**. No score means **nothing to mis-calibrate**. That was the whole point.
- **SAM 2** takes Florence-2's box, turns it into a precise outline, and **follows it through the video**.

**First result — promising!** With no confidence to break, it scored **0.79 to 0.95** on three animals (armadillo, coati, guan — *better* than SAM 3), and **0** on the other six. Average **0.29**. Not great, but unlike GLEE/OWLv2 it **genuinely worked on some videos** — real, varying scores. Worth a closer look.

**We spotted the likely weakness:** the way we started SAM 2 was crude — we handed it Florence-2's box from **frame 0** of each video, no matter how good that box was. A bad first frame poisons the whole track. So we tried a smarter start: pick a **more representative** frame to hand SAM 2, not just the first.

**Second result — it got WORSE, not better.** Average dropped **0.29 → 0.089**. The three good videos collapsed to 0.

**And *that* is where we finally understood the real problem.** We looked at what Florence-2 actually draws:

> On the armadillo video, Florence-2's frame-0 box is spot-on (matches the true animal). But on **most** of the other frames, Florence-2 draws a box around the **entire image** — it basically says "the whole picture is a giant armadillo." Only a few early frames have the correct tight box.

So Florence-2 gets it **right on a handful of frames and gives whole-frame garbage on the rest.** Our "representative frame" trick picked one of the garbage frames (because garbage is the majority), handed SAM 2 the whole image, and SAM 2 dutifully tracked nothing. The nice first result was **partly luck** — frame 0 happened to be one of the few good frames.

**Why it's fatal:** to make this work, we'd need to automatically pick Florence-2's *good* frames — but there's no honest way to do that. Florence-2 gives no confidence, so it won't tell us which of its boxes are the good ones, and the good ones are the minority. The only way to find them would be to peek at the answer key (the ground-truth masks) — which is cheating. Verdict: **FAIL.**

## 7. The honest bottom line

We tried three genuinely different second models. **None of them can be scored fairly against SAM 3 on our camera-trap data** — but they fail for two *different, informative* reasons:

| Model | Sees the animal? | Why it fails |
|---|---|---|
| **GLEE** | ✅ yes | its **confidence is unreliable** → keeps junk, drops the good guess |
| **OWLv2** | ✅ yes | same — **confidence mis-ranks** the good guess |
| **Florence-2 + SAM 2** | ⚠️ only sometimes | **sees the animal on only a few frames**, garbage on the rest; no honest way to find the good frames |

So a fair "second model" comparison **isn't achievable on this dataset with today's open-vocabulary trackers.**

**But this isn't a wasted dead end — it's a result.** It tells us *why* the model-swap is hard on wildlife footage: for the "detector" models it's a **confidence-calibration** problem, and for the "describe-it-in-text" model it's a **localisation-stability** problem. That's a concrete, honest addition to the dissertation's limitations, and it directly answers Marwa's "give me a comparison" request — the comparison exists, and its finding is *"the obvious way to compare models breaks on this data, and here's exactly why."*

The main contribution is unchanged and now better defended: **you can't predict SAM 3's transfer from label-free distances — a robust null — and the attempt to check it against other models shows the null isn't the only thing that's hard here.**

## 8. What we did NOT do (on purpose)

- We **never tuned a knob** (a score cutoff, a seeding trick) *after* seeing the score, to force a pass. Every setting was fixed in advance from each model's own defaults; we ran once and reported whatever came out.
- We **stopped** as soon as the evidence was clear, rather than pouring days into a model that couldn't clear the bar. The 10-video Gate-0 check is what made stopping cheap.

This is the same discipline the whole project runs on: an honest "no" is worth more than a manufactured "yes."

---

## 9. The turnaround — a good question breaks the dead end

Everything above (sections 1–8) was done on **one dataset, SA-FARI**. The conclusion looked grim: SAM 3 works, three other models fail. But that conclusion invited a sharp question:

> "That's weird. It works on SAM 3 but on *nothing else*? That either means SAM 3 is just a great model — which is a boring, obvious finding — **or our measuring stick is secretly rigged for SAM 3**."

That question was exactly right to ask, and chasing it down changed the whole result.

### First we checked: is our scoring rigged for SAM 3?

We stress-tested the scoring three different ways (scored the other models on boxes instead of masks; removed the confidence cutoff entirely; used a model that has no confidence score at all). **The other models still scored ~0 every way.** So the scoring is *not* what sank them — their failure on SA-FARI is real.

### Then we found the real culprit: the *dataset*, not the models

SA-FARI has **two problems** as a testing ground for a fair comparison:

1. **It's SAM 3's *own* benchmark.** Meta built and released SA-FARI *together with* SAM 3, to show SAM 3 off. So comparing SAM 3 to other models on SA-FARI is like judging a chef's cooking using their own recipe — SAM 3 has home advantage baked in.
2. **Its animals are exotic and hard** — armadillos, agoutis, margays, in dark camera-trap footage. The other models were trained on everyday internet photos (cats, dogs, cars). They've barely *seen* a giant armadillo. No wonder they struggle.

So "only SAM 3 works on SA-FARI" doesn't mean "SAM 3 is uniquely great" — it means **SA-FARI is unfairly stacked in SAM 3's favour.**

### The fix: test on a *fair* dataset — BURST

BURST is a different animal-video dataset — **not** made by the SAM 3 team, made *years earlier*, and full of **everyday animals** (cow, dog, cat, camel) that every model has seen thousands of. It's a level playing field.

**And on BURST, everything changed:**

- **The other models suddenly work.** GLEE — which scored basically 0 on SA-FARI — localises the everyday animals almost as well as SAM 3, and (crucially) its confidence scores now behave correctly. On the fair dataset, GLEE is a real, working tracker (score ~0.34, versus SAM 3's ~0.61 — weaker, but genuinely working, not zero).
- **Importantly, SAM 3 itself scores *higher* on BURST (~0.61) than on SA-FARI (~0.54)** — so SAM 3 didn't even need its home turf. That's actually reassuring: SAM 3 is genuinely good, not just gaming its own benchmark.

### The payoff: our main finding holds for GLEE too

Now that GLEE produces real, varying scores on BURST, we could finally run the actual experiment: **do our "distance" signals predict GLEE's performance any better than they predicted SAM 3's?**

**Answer: no — same as SAM 3.** The distances don't predict GLEE's tracking either (the combined predictor is statistically null, same tiny non-effect we saw for SAM 3). One or two individual distances flicker, but they're the same size-related noise we already knew about, and the real (combined) predictor is flat.

**Why this matters enormously:** our headline result was "you can't predict SAM 3's transfer from label-free distances." The obvious objection was "maybe that's just a SAM 3 quirk." **Now we've shown the exact same thing holds for a completely different model (GLEE), on a completely different, fair dataset.** So it's **not a SAM 3 quirk — it's a general truth about this kind of prediction.** That's a much stronger, reviewer-proof version of the finding.

### The honest small print

- It's **one** extra model (GLEE) on **one** fair dataset (BURST, 132 videos) — a *convergent* confirmation, not a massive independent proof. We say so plainly.
- SA-FARI's failure story (sections 1–8) is still true and still useful — it now serves as the explanation of *why you have to test on a fair dataset*, which is itself a methodological lesson.

### The one-line takeaway

The suspicious "only SAM 3 works" result was a **rigged-benchmark illusion, not a real finding**. On a fair dataset, other models work fine, and our "you can't predict transfer in advance" result **holds across models** — turning a possible weakness into one of the strongest, most general claims in the dissertation. The lesson: *when a result seems too flattering to your main model, suspect your test setup before you believe it.*
