# The whole MSc, in plain English

A five-minute, no-jargon tour of the entire project: the question we asked, everything we tried, and what we
found. (For *how the code does it*, see [`how_the_pipeline_works.md`](how_the_pipeline_works.md).)

---

## In one sentence

We asked whether you can **tell in advance whether an AI animal-tracker will work on a new animal or a new
place — without running it** — and after testing this every honest way we could, on three different datasets,
the answer is **no**. That clean, well-checked "no" *is* the result.

## The problem (why anyone cares)

Conservation teams point AI trackers at new species and new regions all the time. Nobody can tell them, ahead
of time, whether the AI will be trustworthy for *this* animal in *this* forest. There is no rule. So they
either waste effort double-checking everything, or they trust numbers that might be wrong. We tried to build
that missing rule.

## The AI we studied

**SAM 3** — a model you point at a video and give one word, like *"impala,"* and it finds and follows every
impala. It was never trained on our animals; it works "zero-shot." Think of it as a **contractor you hire for
one day**: you hand it a video and a word, it does the job, and you grade it afterwards. We never teach it or
change it — we only watch and grade.

## What "working" means — two separate grades

We split performance into two simple questions:

- **Did it FIND the animals?** — we call this *detection*.
- **Did it KEEP TRACK of each individual** as they move around? — *association*.

Grading these two things separately is one of the genuinely new parts of the project — nobody had done that for
a video tracker before.

## The idea we were testing

Could we predict those grades **before running the AI**, using only clues we can measure without any answer
key? We tried four "distance" clues — how *far* a new animal or place is from what the AI has already seen:

1. **Taxonomic distance** — how far on the tree of life (a close cousin of a known animal, or something totally
   different?).
2. **Visual distance** — how different it *looks*.
3. **Environment distance** — how different the *scene* looks (jungle vs. snow vs. night).
4. **Time distance** — how many years apart the footage is.

The bet (our hypothesis): the *further* a new animal or place is, the *worse* the AI should do.

## Everything we tried (the journey)

1. **Graded the AI once** over a big wildlife dataset (~100 species) and checked the grades match the published
   numbers — so we know the measurement is real.
2. **Tested the four distance clues** against the grades — twice: holding out whole unseen *species*, and whole
   unseen *places*.
3. **Split it into "find" vs. "follow"** and checked each on its own.
4. **Hunted confounds** — made sure a result wasn't secretly just "big animals are easy to see."
5. **Tried scene difficulty** instead of novelty — darkness, clutter, night footage.
6. **Checked hallucinations** — does the AI invent animals that aren't there, and is *that* predictable?
7. **Tried a different kind of clue** — reading the AI's *own confidence* after it runs.
8. **Stress-tested** — re-ran everything while swapping the vision model, the crop, the wording of the prompt,
   and the distance maths, to be sure the answer wasn't a fluke of one choice.
9. **Repeated the whole thing on two completely different datasets** — MammAlps (Alpine camera traps) and BURST
   (internet videos, 41 species) — to see if the answer holds elsewhere.

## What we found — a clean "no"

None of the before-running distance clues predict the AI's grade. Across **three different datasets** the story
is the same:

| Dataset | Species | Do the distance clues predict the grade? |
|---|---|---|
| SA-FARI | ~100 | **No** |
| MammAlps | 5 (small) | **No** — but too small to be fully sure |
| BURST | 41 | **No** — bigger test, still no |

The one clue that ever *looked* like it worked (visual distance) turned out to point **the opposite way** — it
said more-unusual-looking animals are *easier* to find, the reverse of the theory — and that was really just
"bigger animals are easier to see" wearing a disguise. So it is not a real predictor.

## The one thing that *did* work — with an asterisk

If you let the AI **run first** and then read **its own confidence**, that *does* track how well it did. But:

- It's an **after-running** clue, not a before-running one — a weaker, less useful claim, because you've
  already spent the effort.
- It's almost **circular** — the AI grading its own homework.

So we report it honestly as a known, non-novel side-result — not the prize we were after.

## Why a "no" is a real result (not a failure)

Two things make the "no" trustworthy rather than lazy:

1. **A positive control.** We checked that our test *can* catch a signal when one truly exists (it catches the
   AI's own confidence, and it catches animal size). So a flat result for the distance clues means "there is
   genuinely nothing there," not "our test was too weak to see it."
2. **We tried hard to break it.** We swapped every ingredient, controlled for the obvious confounds, and
   repeated the entire pipeline on new data. It held every single time. That is a *robust* null, not a
   convenient one.

## The bottom line — what this MSc contributes

- You **cannot** forecast this AI's success on a new species or place from simple "distance-from-training"
  clues. Transfer is not that simple.
- This is the **first** careful, honest test of that idea for a **video tracker**, split into *find* vs.
  *follow*, and confirmed across **three** datasets.
- The contribution is the **rigorous, honest "no"** — which is a genuine scientific result, and a useful
  warning to anyone who assumed these shortcuts would work.

---

### The 30-second version

We tried to predict whether a zero-shot AI tracker would work on new wildlife *before running it*, using
label-free "distance" clues. It doesn't work — cleanly, and on three datasets. The only thing that predicts
success is the AI's own after-the-fact confidence, which is both circular and less useful. The honest,
well-tested "no" is the finding.
