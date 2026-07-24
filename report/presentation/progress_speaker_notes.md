# Speaker notes — progress presentation

*What to say for each slide, in plain language. Paraphrase freely; these match `progress.tex` (14 slides).*

---

### Slide 1 — Title
This is my MSc progress update. The project is about predicting how well SAM 3 — a promptable video tracker —
will work on animals and places it has never seen, *before* we run it. I'll walk through where I am and the
main finding, which is an honest negative result.

### Slide 2 — The result, in one sentence
Here's the headline up front. We *can* measure how reliable SAM 3 is on any given animal and place. But we *cannot* predict that reliability in advance from label-free "distance" signals — how different a new species or scene is from the training data. The only thing that predicts anything at all is animal size, and that's small and obvious. So the result is a clean negative one.

### Slide 3 — The question
To set it up: SAM 3 is "promptable" — you type a species name, say "impala", and it finds and follows that animal in a video, with no training on that species. Conservation teams point models like this at new animals
and regions constantly, but nobody can tell them in advance whether it will work. That's the question I'm answering.

### Slide 4 — What we've done
A quick summary of the work. 
- One: I ran SAM 3 over the test videos and scored it with the official scorer, giving a reliability number per species–place–time group. 
- Two: I built four "distance" signals that measure how novel a species or scene is. 
- Three: I fitted a model to predict the score from those distances and tested it honestly — hiding whole species and whole places and predicting them. 
- Four: when that failed, I tried a second idea — maybe it's about how *hard* the footage is, not how *novel*. 
- Five: I checked a side question about hallucinations.

### Slide 5 — What the pipeline sees
This is what the distances actually work with — a real camera-trap frame, here a peccary, a kind of wild pig.
On the left, the frame with the animal's outline. In the middle, the animal cut out on its own — that's what
the *visual* distance fingerprints. On the right, the same scene with the animal erased — that's what the
*environment* distance looks at. Plus the species' place on the tree of life and the timestamp. Importantly,
none of this uses SAM 3 — it's all computed beforehand, with no labels of the target.

### Slide 6 — The main result: the distances predict nothing
This is the key result. On the left: if you just guess the average score, your error is 0.30. Add the four
distances — still 0.30, no improvement. Add animal size — it drops a little, to 0.28. On the right is the same
thing as a picture: each dot is a held-out species or place; the x-axis is the true score, the y-axis is our
prediction. If the distances worked, the dots would follow the diagonal line. Instead they collapse onto a
flat horizontal band — the model just predicts the average no matter what. So the distances predict
essentially nothing.

### Slide 7 — The one "signal" was a trick of animal size
There was one distance — visual — that looked significant at first. But it had the *wrong* sign, and it turned
out to be a trick of size: "visually distinctive species do better" really just meant "bigger animals are
easier to spot." Once I control for animal size, that effect disappears. And the "difficulty" idea — that dark
or cluttered footage is harder — collapsed the same way: it only looked predictive because it tracks animal
size. On its own it predicts nothing.

### Slide 8 — Peeling it apart: only size survives
Here I pull that apart carefully — testing each thing on its own. Clip length: nothing. Animal size alone: a
small but real effect. Low-light alone, without size: nothing. Low-light plus size: the same as size alone. So
the size covariate carries the *entire* gain, and the difficulty signals only "passed" by riding on size. The
only label-free thing that predicts detection is object size — small, obvious, and the very thing I introduced
as a nuisance control.

### Slide 9 — "Following" barely differs from "finding"
On the association half — following the animal. Almost every clip has just one animal, so "finding" and
"following" come out nearly identical — they agree about 94% of the time. That means there's very little
separate "following" signal for anything to predict. So this half is honestly a detection story; I report the
association null with those numbers as the explanation, not as a separate failed search.

### Slide 10 — Is "we found nothing" trustworthy?
The obvious worry with any "we found nothing" is: maybe your test just wasn't sensitive enough. Here's my
answer. The exact same test, on the same data, *does* detect the size effect. So the test works — it fires
when there's a real signal. A test that catches size but stays flat for the distances is genuinely finding no
signal, not failing to look. This is the key defence of the negative result.

### Slide 11 — The null survives every check
I stress-tested it further. On the left: I drop each distance one at a time — the fit and the out-of-sample
result barely move; only adding animal size changes anything. On the right: all four distances together
explain only about 3% of the variance, no single factor stands out, and there's no collinearity hiding a real
effect. So the null isn't an accident of one factor or one modelling choice — it holds up.

### Slide 12 — Two other things we checked
Two side checks. First, hallucinations: when the animal is absent, SAM 3 wrongly returns one about 10% of the
time. More distinctive species hallucinate *less* — a real, size-independent correlation — but when I validate
it out of sample it *just* misses significance, p = 0.053. So it's a genuine correlation, but not a validated
predictor. Second, the model's own confidence: if you let SAM 3 run first and read its confidence, that *does*
predict accuracy — but I deliberately keep it out, because it needs running the model (which defeats the
"predict *before* running" goal) and it's near-circular. I use it only as an internal check that my pipeline
can detect a strong signal.

### Slide 13 — Why a negative result is a real contribution
Why is a negative result worth it? First, it's the first time anyone has asked this for a promptable video
*tracker*, split into detection versus association — so it's novel regardless of the answer. Second, it's
useful: it tells conservation teams *not* to trust "far from training means unreliable" as a safety check —
they should spot-check with a few labels instead. Third, a robust negative result from a test I've shown is
powerful is more defensible than a weak, borderline positive scraped past a threshold.

### Slide 14 — Where we are, and what's next
Finally, status. Done and validated: the measurement, the four distances, the modelling and out-of-sample
validation, the difficulty pivot, the false positives, and the ablations. The write-up reflects all of this
and the dissertation builds end-to-end. In progress: three robustness re-runs — a different visual encoder,
whole-frame crops, and a generic prompt — I'm waiting on a GPU for those, and I expect the null to hold. The
honest limitations and open directions are on the slide. Thanks — happy to take questions.
