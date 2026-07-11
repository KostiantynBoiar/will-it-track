# The distance features — in plain English

This folder answers one question for every test case: **"how far is this animal/place from what we've
already seen?"** The whole thesis is that these "how far" numbers can predict whether SAM 3 will work.
Nothing here runs SAM 3 — these are the cheap, label-free measurements we take *before* trusting the model.

---

## 1. The big picture (read this first)

We have a pile of camera-trap videos of animals. We split them into two piles:

- **Reference** ("what we've seen") — the anchor.
- **Probe** ("the thing we're testing") — the target.

For every probe, we compute **four distances** to the reference:

| Feature | Plain-English question | Answer is per… |
|---|---|---|
| **Taxonomic** | How far apart are these two animals on the tree of life? | species |
| **Visual** | How *different does the animal look* to a vision model? | species |
| **Environment** | How *different does the place look* (the background scene)? | location |
| **Temporal** | How many *years* apart was the footage filmed? | cell (species+place+year) |

Later, a regression asks: *do these distances predict SAM 3's score?* If a big distance reliably means a
low score, we've built a "will it work here?" tool.

### Two experiments (this is the key idea)

Because we can pick what counts as "reference", we run two setups:

- **Split A — species hold-out** (the primary one). We hide one species at a time and ask "how novel is it
  compared to *all the other* species?" This is **leave-one-species-out**: an animal is never compared to
  itself. This tests whether *unfamiliar animals* trip up detection.
- **Split B — location hold-out** (the secondary one). Reference = one geographic region, probe = a
  totally different region. This tests whether *unfamiliar places* trip up tracking.

The same code serves both — a `Partition` object just says "here are the reference species/places and here
are the probe ones", and the features don't care which experiment they're in.

---

## 2. The four features, plainly

### Taxonomic distance (`taxonomic.py`)
Count the steps up the family tree until two animals share an ancestor. Same genus = 1 step, same family =
2, same order = 3, and so on. A leopard is 1 step from a lion, but ~4 steps from a frog. Pure lookup, no
images. (Capped low here because SA-FARI's species are fairly clustered — so this signal is real but modest,
which is exactly why we also need the visual one.)

### Temporal gap (`temporal.py`)
Just: how many years between this footage and the nearest reference footage? A 2023 clip vs a 2014 reference
= a gap of 9. Trivial arithmetic on the video timestamps.

### Visual distance (`visual.py`) — the interesting one
"How unfamiliar does this animal *look*?" Steps:
1. For a species, grab a handful of frames where the animal is masked (we have ground-truth outlines).
2. **Cut the animal out** using its mask (background blacked out — so we measure the *animal*, not the bush
   behind it).
3. Feed each cutout to **DINOv2** (a frozen, pretrained vision model) → each becomes a list of ~768 numbers
   (a "fingerprint" of how it looks).
4. Average all a species' fingerprints into one **prototype** (its typical appearance).
5. The distance = how different this species' prototype is from the *nearest other* species' prototype
   (cosine distance: 0 = identical-looking, bigger = more different).

So "pig = 0.49" means the pig looked more distinct from everything else than "raccoon = 0.21" did.

### Environment distance (`environment.py`)
Same trick, but for the *place* instead of the animal:
1. Take frames and **erase the animal** (fill the animal's pixels with the average background colour), so
   only the scene is left.
2. DINOv2 → a fingerprint of the *scene*.
3. Average per location → each camera site gets a "typical scene" prototype.
4. Distance = how different a probe site looks from the nearest reference site.
5. Bonus: it also flags **night/IR footage** by noticing the frame is basically grayscale
   (`is_night_ir`), and a rough **clutter** score (how busy/textured the scene is).

---

## 3. How the machinery fits together (the plumbing)

The visual and environment features both need the same three helpers, so those live in separate files:

- **`frames.py`** — the "images" toolbox. It figures out which frames actually have the animal outlined,
  downloads only those frames on demand (fetch-or-skip — a missing frame is skipped, never a crash), cuts
  the animal out, erases the animal for the background version, and does the night/IR + clutter colour
  checks. **No AI here — just PIL/numpy.**
- **`embed.py`** — the "fingerprint" machine. Loads DINOv2 (or CLIP) once, turns a batch of images into
  normalised number-vectors. It also has a **cache**: once we've fingerprinted a frame, we save the vector
  to disk, so re-running doesn't re-download or re-compute. This is the only file that touches the heavy ML
  library (torch).
- **`splits.reference_records` / `probe_records`** (in `../splits.py`) — hand a feature the right pile of
  videos: "give me the reference-side records" vs "give me the probe-side records".

### The flow, once, in words
> For each species → find its videos → for each video, sample a few outlined frames → download them →
> cut out the animal → DINOv2 fingerprint (or read it from cache) → average into a prototype → compare each
> probe species' prototype to the reference prototypes → that comparison is the distance.

Environment is the same sentence with "location" instead of "species" and "erase the animal" instead of
"cut out the animal".

---

## 4. Things worth knowing (honesty section)

- **We don't use every frame.** There are ~880k outlined frames; embedding them all would mean downloading
  ~5 GB and hours of compute. So we **sample** (a few frames per masklet, a cap per species/location). The
  numbers are stable well before we use everything, but the caps are knobs in the config.
- **The cache is what makes it affordable.** First run downloads frames + fingerprints them; every run after
  is basically instant because the vectors are on disk.
- **"Leave-one-out" is the anti-cheating rule.** On Split A a species is never allowed to be its own
  reference — otherwise every distance would trivially be 0. (There's a test that checks this.)
- **Same species can still have a non-zero visual distance on Split B**, because the *same* animal at a
  *new* location looks a bit different (lighting, IR, pose). That's expected, not a bug.
- **Missing yet:** the SAM-3 "familiarity" distance (`familiarity.py`) needs SAM 3, and `assemble.py` will
  later glue all the distances into one table. Both are stubs.

---

## 5. How the shared machinery is organised (after the refactor)

The skeleton that visual and environment share now lives in one place, so each feature is thin:

- **`pipeline.py`** holds the shared engine: `embed_crops` (fetch → load → crop → embed, cache-first),
  `prototype` (re-normalised mean), `nearest_distance` (`1 − max cosine`, with self-exclusion),
  `record_annotations` (the `SAFARI(origin)` + `annotations_by_video` + category-filter lookup), and
  `safari_by_origin`.
- **`frames.py`** holds the pure image/colour utilities, including one even-spacing sampler
  (`sample_evenly`, wrapped by `sample_frame_indices`).
- **`embed.py`** is the model + on-disk vector cache.
- **`visual.py` / `environment.py`** each just build their items (with their own crop function) and group
  by their own key — species for visual, location for environment.

So the only real difference between the two features is *which crop* they embed and *what they group by*.
Everything is covered by tests, and the refactor was verified to reproduce the exact same distances.
