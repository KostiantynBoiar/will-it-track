# Q&A prep

Four questions, written the way I'd actually say them out loud.

---

## Q1. "What is Section 4.2 for?"

It's the first check I do, before the real test.

The question there is dead simple: does any of these distances have an effect I can even tell apart from zero? And the answer is no — every single interval includes zero, so I can't distinguish any of them from having no effect at all.

So they fall over at the easiest possible hurdle. That's why the proper out-of-sample test in the next section isn't a surprise — nothing here ever looked good enough to survive it.

**If she asks about the taxonomy row with the huge interval `[−3.87, +0.09]`:**

Honestly, that row isn't really a result. I just couldn't measure it properly on this split, and the width of the interval is the model telling me exactly that.

The reason is that a coefficient only means something if the thing you're measuring actually varies — and here it barely does. Out of the 840 cells that have a taxonomic distance at all, 778 of them are exactly zero. That's 92.6%. There are only three different values in the entire split: 0, 2 and 3. So only 62 cells are non-zero.

And it gets worse, because taxonomic distance is identical for every cell of the same species. So what actually matters isn't how many cells I have, it's how many *species* have a non-zero value. That number is seven. Seven animals, spread across two values. That's the whole basis for the coefficient.

Which is why the interval blows up. When I build these intervals I resample whole species rather than individual cells, because cells from the same species aren't independent. And when only seven species carry anything at all, plenty of those resamples come back with almost none of them — so the estimate jumps around wildly from one to the next. That interval is just an honest record of the jumping.

And it's not bad luck, it's baked into the split. SA-FARI's official split separates by location and deliberately keeps the same species on both sides. Species novelty is the one thing it holds constant. So asking it about species novelty is asking the wrong question of the wrong data — which is exactly why I built the species hold-out.

---

## Q2. "What is a species hold-out?"

The basic idea is that you hide part of your data and then test on the hidden part.

With a species hold-out, the thing you hide is a whole species. So I take one species out completely — say impala — build my reference from the other 98, and then ask: how far is impala from everything I kept, and does SAM 3 actually do worse on it? Then I do the same for the next species, and so on through all of them.

I had to build it because SA-FARI's official split hides locations, not species. The same animals turn up on both sides. So every species is already "seen", novelty comes out as basically zero everywhere, and there's nothing for H1 to be tested against.

And it isn't cheating, because SAM 3 is frozen — it never trained on any of this data. So "held out" isn't a statement about what the model has or hasn't seen. It's just my choice of which species I treat as the reference when I measure distance, and I'm free to draw that line wherever it makes sense. I say that explicitly in the methodology.

The interesting bit is what came out of it. On that split, where novelty actually varies, the coefficients came back positive — so more unusual species were detected *better*, not worse. That's the opposite of what H1 predicts. And when I dug into it, it was animal size doing the work.

---

## Q3. "What are the two splits, and where is that written?"

It's all in Section 3.2, pages 6 to 7 — the paragraph headed "Two complementary splits". It comes up again in 3.4, 3.5.2 and 3.6.3, and the Split A results are in Section 4.5 on page 16.

There are two of them because they answer different questions.

Split A is the species hold-out, and it's the primary one. I hide a species at a time, so its distance is measured to the nearest *other* species. Species novelty varies while the environment stays familiar — that's what isolates H1. It's 1,025 cells across all 99 species.

Split B is the official SA-FARI train/test split, and it's the secondary one. There the unseen thing is the camera site, so it isolates H2, and because it's the published split it also lets me check my scores line up with the benchmark. That one's 1,447 cells — 346 where the species is actually present, so the scores are defined, and 1,101 hard negatives, across 92 unseen locations.

The reason I need both is that the official split shares species between train and test. So on it, species-distance is basically zero for every cell and it simply can't test species novelty. The two splits end up being near-orthogonal views of the same videos: one varies the animal, the other varies the place.

And building my own is fine because SAM 3 is frozen and never trained on any of SA-FARI. The official split doesn't reflect anything the model learned — it only defines which species I'm treating as my reference. So I can repartition the same videos. I put that in the text directly: it's a constructed hold-out, legitimate because the model is frozen, and reported as such. Inference only runs once, over both splits together, so the two experiments are two views of the same scored cells rather than two separate runs.

---

## Q4. "The Night/IR coefficient is the biggest in the table — doesn't that mean something?"

It means something physically, but it doesn't hold up when I test it — and I did go after it rather than leave it sitting there.

It's big because it's believable. Minus 0.4 says detection gets worse at night and on infrared, and of course it does — dark, monochrome, low-contrast footage is harder. It's also the closest thing in the table to being significant; the interval only just clears zero.

But it falls down three times over.

The obvious one is that it still crosses zero, so by the rule I set myself in advance, it fails. "Nearly significant" isn't a result.

The one that actually matters is that when I test it out of sample, it turns out to be animal size wearing a disguise. That's what the nested comparison in Table 4.4 is for:

| Model | Leave-species-out | Leave-location-out |
|---|---|---|
| Low-light **alone** (no size) | +0.003 (p=0.233) ✗ | +0.004 (p=0.123) ✗ |
| Size **alone** | +0.015 (p=0.014) ✓ | +0.017 (p=0.001) ✓ |
| Low-light **+ size** | +0.017 (p=0.011) | +0.020 (p=0.000) |

Low-light on its own doesn't beat just guessing the average. Size on its own does. And putting them together is barely better than size by itself — so darkness isn't contributing anything of its own.

And the third thing is that the correlation which made it look exciting in the first place was an averaging artefact. Measured per location it's −0.377, which looks convincing. Measured per cell, which is where the model actually operates, it drops to −0.13. Clutter is worse — it's exactly zero.

So the way I'd put it: night/IR is the strongest hint in the whole table, and the difficulty pivot in Section 4.3 is me chasing it. Once I separate it from animal size, low-light on its own doesn't beat the baseline on either scheme, and the correlation that motivated the pivot shrinks from −0.377 to −0.13. It's a real physical effect that isn't a usable predictor.

And I'd argue that helps me rather than hurts. The most promising-looking signal in the document still didn't survive scrutiny — that's a better position than never having had a candidate at all, because it shows I went after the best lead and reported honestly when it fell apart.

The one thing I'd concede if she pushes: I can show darkness and animal size are entangled, but I don't have a verified explanation for *why*. That's an empirical finding, not a mechanism I've confirmed.
