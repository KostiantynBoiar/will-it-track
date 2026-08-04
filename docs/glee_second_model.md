# GLEE as the second promptable tracker — video/text-path findings (Item 1)

Verification note written *before* any GPU spend, per the model-swap plan. Question it answers: **does GLEE do
zero-shot, text-prompted _video_ instance tracking with masks, from a scriptable path, on an _arbitrary_ species
noun?** Short answer: **yes — go.** Details and the adapter contract this implies are below.

Sources: [GLEE repo](https://github.com/FoundationVision/GLEE) ·
[GLEE paper (CVPR 2024, arXiv:2312.09158)](https://arxiv.org/abs/2312.09158) ·
[MinVIS (arXiv:2208.02245)](https://arxiv.org/abs/2208.02245).
All facts below are from the GLEE source (`projects/GLEE/glee/models/glee_model.py`), `assets/{TEST,MODEL_ZOO}.md`,
and the paper — read, not run (the CUDA op needs a GPU).

## Why GLEE, not SAM 2

The swap must keep SAM 3's contract: **text prompt in → tracked masks out, zero-shot, open-vocab, one model.**
SAM 2 is a *visual*-prompt propagator (points/boxes/mask on frame 1) with no language input — to use it you'd hand
it a first-frame GT mask, which *hands it the pDetA answer* and collapses the detection experiment. GLEE is the
only other single unified model with SAM 3's exact contract. (Grounded-SAM-2 / DEVA are two-stage
detector+tracker *pipelines*, so any gap confounds detector vs tracker — rejected as the primary.)

## (a) The scriptable video + text entry point

GLEE's `GLEE_Model.forward` signature:

```python
def forward(self, images, prompts, task, targets=None, batch_name_list=None,
            is_train=True, visual_prompt_type='scribble')
```

- **Arbitrary text, open-vocab, at inference.** `batch_name_list` takes a plain list of category-name strings;
  they are CLIP-encoded on the fly (`token_x = self.text_encoder(*texts)['last_hidden_state']` →
  `@ self.lang_projection`) and matched against object queries. The vocabulary is **not** fixed to a dataset —
  `assert batch_name_list` for detection/VIS tasks means *you* supply the names. So a **single-element
  `batch_name_list=[prompt]`** (e.g. `["impala"]`) is the exact analogue of SAM 3's `add_text_prompt(text=prompt)`.
- **Video task switch.** The `task` argument selects the mode (`'vis'`, `'ovis'`, `'ytvis19'`, … vs image
  detection / `'grounding'` / `'rvos'`). For our use: run per-frame in the VIS/detection mode with our one name.
- **Outputs.** The forward returns dicts with **`pred_masks`**, **`pred_logits`** (query↔text similarity → the
  score), and **`pred_track_embed`** (the per-query embedding used for cross-frame association).
- **Documented run path** (fixed-vocab benchmark eval): `python3 projects/GLEE/train_net.py --config-file
  projects/GLEE/configs/video/Lite/ytvis19_base.yaml --eval-only ... MODEL.WEIGHTS <ckpt>`. **Caveat:** `TEST.md`
  documents only *benchmark* eval with dataset vocabularies — arbitrary-noun **video** prompting is a supported
  *capability of the model* (via `batch_name_list`) but **not a shipped demo command**. The adapter therefore
  drives the model/predictor directly rather than through `train_net.py`.

## (b) How identities are linked over time (the association step)

GLEE follows **MinVIS**: it does **not** need video-based training. Object queries trained to be discriminative
within a frame are temporally consistent, so tracks are formed by **bipartite matching of object queries
(`pred_track_embed`) between consecutive frames** (near-online). Contrastive track loss tightens same-instance
embeddings. Concretely, the adapter runs GLEE per frame, then Hungarian-matches each frame's queries to the
running tracks by `pred_track_embed` cosine similarity → one persistent track id per instance.

## (c) Verdict: arbitrary-noun video open-vocab — **YES**

The capability is real (open-vocab text list consumed at inference + MinVIS query association for video). The only
friction is that it's **undocumented as a one-liner** — we script the predictor directly. **Go** for the GPU
phase, subject to the Gate-1 pDetA sanity check (a strong capability can still detect camera-trap animals poorly).

## (d) Adapter steps this implies (`GleeTracker._run`)

Per clip, with `batch_name_list=[prompt]`:
1. For each frame: GLEE forward in VIS/detection mode → `pred_masks [Q,H,W]`, `pred_logits [Q]` (score vs the one
   name), `pred_track_embed [Q,D]`.
2. Threshold queries by score (`glee_score_threshold`); keep candidate instances per frame.
3. **MinVIS association:** bipartite-match kept queries across frames by `pred_track_embed` → one track per
   instance, with a per-frame mask (or absent) and one aggregate score.
4. Per track → `Masklet`: `segmentations` = list of **exactly `len(frames)`**, `encode_rle(mask@originalHW)` where
   present else `None`; `score` = one scalar. **No `obj_id`** (track id = list membership, discarded, as
   `Sam3Tracker` does). **Empty list = hard negative.**

This is a pure function of GLEE's raw per-frame output → `list[Masklet]`, so it is **factored out and
hermetically unit-tested on synthetic tensors** without a GPU (`_masklets_from_glee`).

## (e) Checkpoint + install facts

- **Checkpoint (zero-shot):** `GLEE_Lite_scaleup.pth` (ResNet-50; `-scaleup` = extra SA1B/GRIT auto-annotated data,
  best zero-shot), HuggingFace `Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Lite_scaleup.pth` (+ CLIP text
  weights). Lite is <20 GB and the cleanest "different architecture" contrast; Plus (Swin-L) only if Lite's pDetA
  is too weak.
- **Deps:** detectron2 + a **custom CUDA op** (MSDeformAttn, `sh make.sh` against **torch 2.1.0 + CUDA 12.1** — the
  #1 install failure; use a CUDA-12.1 base image). MIT licence.
- **Local status:** torch/transformers/detectron2 present locally, but the CUDA op won't build/run without a GPU —
  so this note is code+docs-derived, and the live model path is exercised only in the (deferred) GPU phase.

## Framing (state in the write-up)

A **controlled model-swap**, *not* independent replication: same SA-FARI cells, same GT, same four distances, same
GLM/CV — only the tracker changes. It tests whether the label-free null is **SAM-3-specific or task-general**. A
GLEE that is *also* unpredictable-from-distances is the strong result — regardless of whether GLEE tracks better or
worse than SAM 3 overall. Guardrail: never tune `glee_score_threshold` to flatter the null; keep the after-running
confidence positive control as the liveness check.
