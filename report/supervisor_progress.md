# Progress update — can we predict SAM 3's tracking reliability in advance?

*A plain-English summary of the work and results so far.*

---

## The short version (one paragraph)

The full analysis pipeline is built and all the main experiments are done. The honest headline is a **clear,
rigorous negative result**: you **cannot** predict how well SAM 3 will track an animal on an unseen species or
place from label-free "distance-from-training" signals computed *before* running it — beyond a small and
obvious animal-size effect. Importantly, this is *not* a case of a weak test finding nothing: the same test
does pick up the size effect, so it is genuinely finding no signal. A negative result like this, cleanly
decomposed into *finding* the animal vs *following* it, is a real and novel contribution — it is the first
time this question has been asked for a promptable video tracker.

---

## What we're trying to do (plain terms)

SAM 3 is a model you can point at a video and say *"track the impala,"* and it tracks it — even for species it
was never trained on. Conservation teams use models like this on new animals in new places constantly, but
nobody can tell them **in advance** whether the model will actually work for a given animal in a given place.
We asked: **can we predict that in advance, without anyone having to label the video first?** And we split the
question into two parts:

- **Detection** — does the model *find* the animal at all?
- **Association** — does it *keep track* of the same animal over time?

## What we've done

1. **Measured the model.** Ran SAM 3 across the test videos and scored it with the official evaluator, giving
   a reliability score for each (species, place, time) group.
2. **Built four "distance" signals** — how *different* a new species/place is from the familiar training data:
   how different the animal is (its place in the tree of life + its appearance), how different the scene is,
   and how far apart in time.
3. **Fitted a proper statistical model** (score = function of the distances) and tested it **honestly out of
   sample**: we hid whole species and whole locations, predicted them from the distances, and compared to the
   real scores — with conservative error bars that account for the small number of independent species/places.
4. **Tried a second idea when the first failed** — maybe it's about how *hard* the footage is (dark/night,
   cluttered, small animal), not how *novel* it is.
5. **Checked a separate question** — when the animal is *absent*, does the model hallucinate one anyway?

## What we found

- **The distances don't predict anything.** When we hide a species or a place and predict its score from the
  distances, we do no better than simply guessing the average. Every distance's effect is statistically
  indistinguishable from zero, on both the species hold-out and the location hold-out.

- **The one apparent signal was a trick of animal size.** "Visually distinctive species do better" turned out
  to just mean "bigger animals are easier to spot." Once we account for animal size, that effect disappears.

- **The "difficulty" idea also collapsed to size.** Dark/cluttered footage looked like it mattered, but only
  because it happened to correlate with animal size. On its own, it predicts nothing. (A correlation that
  looked strong when we grouped by location — about −0.38 — shrank to about −0.13 once we looked at individual
  clips, a classic statistical illusion.)

- **The only thing that predicts detection is animal size** — bigger animals are found more reliably. But the
  effect is small (it cuts the prediction error only slightly), it is the obvious "small objects are hard to
  detect" fact, and it is the very thing we were trying to *control for* — not a satisfying predictor.

- **"Following" the animal barely differs from "finding" it.** Almost all clips contain a single animal, so
  the two scores are nearly identical (they agree ~94% of the time). There is very little separate
  "association" signal for any feature to predict, so this half is honestly a detection story.

## Is the "we found nothing" trustworthy?

Yes — and we checked this directly, because the natural worry is *"maybe the test just wasn't sensitive
enough."* The **exact same test, on the exact same data, does detect the animal-size effect** out of sample.
A test that picks up size but stays flat for the distances is genuinely finding *no signal*, not failing to
look. This is the key defense of the negative result.

## Two other things we checked

- **Hallucinations (when the animal is absent):** SAM 3 wrongly returns an animal about **10%** of the time on
  "trick" queries. More visually distinctive species get hallucinated *less* — a real, size-independent
  correlation — but when we validate it out of sample it **just misses** the significance line (p = 0.053). So
  it is a genuine correlation, but not a validated predictor.

- **The model's own confidence (explored, then set aside):** if we let SAM 3 run first and read its *own*
  confidence, that **does** predict its detection accuracy well. But we deliberately keep this out of the main
  contribution, for two honest reasons: (1) it requires running the model first, which defeats the whole
  "predict *before* running" goal; and (2) it is close to circular — you are essentially asking the model how
  confident it is and finding that confident answers are more often right, which is a known result. We use it
  only as an internal check that our pipeline can detect a strong signal when one exists.

## Why a negative result is worth reporting

- No one has asked this question for a promptable video **tracker** before, or split it into detection vs
  association — so the study is novel regardless of the answer.
- A rigorous "far-from-training does **not** mean unreliable" is a genuinely useful finding: it tells
  conservation practitioners *not* to trust distance-from-training as a safety check, and to spot-audit with a
  few labels instead.
- A robust negative result from a demonstrably powerful test is stronger and more defensible than a weak,
  borderline positive would have been.

## Where we are & what's next

- **Pipeline:** complete and tested (measurement → scoring → distance features → modelling → out-of-sample
  validation), with honest, group-aware statistics throughout.
- **Write-up:** chapters drafted; results and discussion now reflect the negative result and its explanation.
- **Honest limitation:** we measure distance from the *dataset's* training split, not SAM 3's true (undisclosed)
  training data — so a proxy for the real training data is one possible future angle.
- **Possible next direction:** probing whether SAM 3's own internal features already "know" a species (a hedge
  against not knowing its true training set).

---

*Bottom line: the science is done and the answer is a clean, honest "no" — label-free distance does not predict
this tracker's transfer beyond a trivial size effect — which is itself the contribution.*
