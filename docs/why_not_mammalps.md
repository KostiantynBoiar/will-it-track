# Why MammAlps is not used as a replication dataset

MammAlps ([Zenodo record 15588220](https://zenodo.org/records/15588220)) — an Alpine camera-trap
tracking dataset — was an early candidate for an independent cross-dataset replication. We ran it once,
but it is **not** part of the final dissertation, and the code for it has been **removed** from this
repository. This note records why, honestly, so the decision is not mistaken for an oversight.

## The two reasons

**1. It is underpowered.** MammAlps yields only **~20 scored cells over 5 species** (red deer, roe deer,
hare, wolf, and a single-cell fox) across 9 cameras. At that scale the out-of-sample test cannot certify
power: not even the animal-size control — the one before-running quantity that *does* validate on the much
larger SA-FARI sample — clears the bar on MammAlps. A flat distance result there is therefore only a weak
*consistency check*, never independent confirmation. The taxonomic and temporal axes are also degenerate
(5 near-related species; a 6-week window), so it effectively re-tests only the environment axis.

**2. Its input is not reproducible from the public release.** This is the decisive reason. The pipeline
needs per-clip **box/track/species** records. The public Zenodo release ships **segmentation maps**
(`benchmark_1/segmaps/*.npz`) and video metadata (`raw_videos_mammalps_v1.csv`) — **not** the per-frame
bounding-box / track / species JSONs the adapter would consume. The dense annotations used in our original
one-off run came from an **offline conversion that was never committed**, and we cannot regenerate them
from the public artifacts alone. So a MammAlps result cannot be reproduced from this repository by anyone.

## What we do instead

The independent cross-dataset replication uses **BURST** (mask-native, 41 species, 132 cells), whose
annotations *are* fully reproducible from its committed acquisition. BURST is both better powered and
irreproachably reproducible, so it replaces MammAlps in every role:

| | MammAlps | BURST |
|---|---|---|
| Cells / species | ~20 / 5 | 132 / 41 |
| Powered? | No | Better (still can't exclude a *small* effect) |
| Ground truth | Boxes (filled-rectangle hack needed) | Native masks |
| Reproducible from public release? | **No** (dense prep lost) | **Yes** |

The cross-*model* swap (GLEE, Florence-2 + SAM 2) likewise runs on BURST, not MammAlps, for the same
reproducibility reason.

## What was removed

The whole MammAlps subsystem was deleted, since it was dead code relative to the shipped dissertation and
its input could not be rebuilt:

- `src/adapters/mammalps.py`, `src/adapters/boxes.py` (box→filled-rectangle RLE, used only by MammAlps),
  `src/adapters/mammalps_taxonomy.csv`
- `configs/mammalps.yaml`, `tests/test_mammalps_adapter.py`
- `scripts/{run_mammalps.sh, extract_mammalps_frames.py, mammalps_replication_analysis.py}`
- `docs/mammalps_replication.md`, the `MammAlpsConfig` block in `src/config.py`

The dissertation's replication story is therefore **two datasets** — SA-FARI (primary) and BURST
(independent) — plus the cross-model swap on BURST.
