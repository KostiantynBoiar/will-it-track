"""Frozen GLEE promptable video tracking — the second tracker for the model-swap experiment.

GLEE (CVPR 2024) has the same contract as SAM 3: a text prompt in -> zero-shot detect, segment and track
every matching instance across a video, masks out. Running it over the same SA-FARI cells tests whether the
label-free null is SAM-3-specific or task-general. See docs/glee_second_model.md.

Only the GPU-only backend (torch + GLEE + detectron2, absent from the local analysis env) is imported
lazily in load(); everything else is imported at the top. So this module imports on the CPU analysis env
and the tests never touch a GPU. Model-touching code lives in load()/_run(); the pure output-to-Masklet
assembly is _masklets_from_glee(), unit-tested without any model. Masklet and encode_rle are reused from
sam3_tracker — one masklet per track, no obj_id (identity is list membership, as SAM 3 does).
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from src.config import Config
from src.inference.sam3_tracker import Masklet, encode_rle


def _masklets_from_glee(
    tracks: dict[int, list[tuple[int, np.ndarray, float]]],
    n_frames: int,
    threshold: float = 0.0,
) -> list[Masklet]:
    """Assemble one Masklet per GLEE track from its per-frame detections (pure, model-free, tested).

    Args:
        tracks: track_id -> list of (frame_index, bool_mask, score) for the frames where the track is
            present. Masks must already be at original frame resolution; absent frames are omitted.
        n_frames: Total frames in the clip; each masklet's segmentations is padded to this length.
        threshold: Drop a track whose max-over-frames score is below this (GLEE's per-query pre-filter).

    Returns:
        One Masklet per kept track, segmentations of length n_frames (RLE where present, None where absent)
        and one scalar score. An empty list means nothing found (the hard-negative case), never a fake mask.
    """
    out: list[Masklet] = []
    for _track_id, detections in tracks.items():
        if not detections:
            continue
        score = max(float(s) for _fi, _mask, s in detections)
        if score < threshold:
            continue
        segs: list[dict | None] = [None] * n_frames
        for frame_index, mask, _s in detections:
            if not 0 <= frame_index < n_frames:
                raise ValueError(f"frame_index {frame_index} out of range for {n_frames}-frame clip")
            segs[frame_index] = encode_rle(np.asarray(mask, dtype=bool))
        out.append(Masklet(segmentations=segs, score=score))
    return out


def _glee_out_dict(raw):  # noqa: ANN001, ANN202
    """Reach GLEE's pred_* dict from forward()'s return, tolerant of the tuple-vs-dict shape.

    GLEE's forward may return the dict directly (app.py: (outputs, _) = ...; outputs['pred_masks']) or a
    nested tuple ((out_dict, mask_dict), ...); peel until a dict with 'pred_masks' is found.
    """
    obj = raw
    for _ in range(4):
        if isinstance(obj, dict) and "pred_masks" in obj:
            return obj
        if isinstance(obj, (tuple, list)) and obj:
            obj = obj[0]
        else:
            break
    raise RuntimeError(f"could not locate GLEE pred dict in forward() output of type {type(raw)}")


class GleeTracker:
    """Frozen GLEE video tracker (loads on first track); satisfies the Tracker protocol.

    The real GLEE forward + MinVIS association live in _run and need a GPU (GLEE's MSDeformAttn CUDA op);
    only the output post-processing (_masklets_from_glee) runs off-GPU.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (the model is not loaded until the first track)."""
        self.config = config or Config()
        self._model = None
        self._torch = None
        self._dtype = None

    def load(self) -> None:
        """Lazily build the frozen GLEE predictor on the configured device (GPU box only).

        Imports torch + GLEE + detectron2 here, mirroring Sam3Tracker.load. The model/config/weights come
        from inference.glee_{config,weights,model}; raises if they are unset (never hit in the tests).
        """
        if self._model is not None:
            return
        import torch  # lazy: GPU (pod) env only

        inf = self.config.inference
        if not inf.glee_config or not inf.glee_weights or not inf.glee_repo:
            raise RuntimeError(
                "GleeTracker needs inference.glee_repo, glee_config and glee_weights set "
                "(the GLEE checkout root + model YAML + .pth). See docs/glee_second_model.md for staging."
            )
        self._model = self._build_glee(inf.glee_config, inf.glee_weights, inf.device)
        self._dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(inf.precision, torch.float32)
        self._torch = torch

    def _build_glee(self, config_path: str, weights_path: str, device: str):  # noqa: ANN001, ANN202
        """Construct the frozen GLEE_Model from its detectron2 cfg + checkpoint on device (GPU box only).

        Mirrors GLEE's app.py: get_cfg -> add_glee_config -> merge_from_file -> GLEE_Model(cfg, None,
        device, None, True) -> load_state_dict(strict=False) -> eval(). strict=False is deliberate — the
        checkpoint carries training-only keys (matcher, track-loss) absent at inference. The GLEE checkout
        is a source tree, so glee_repo (the dir containing projects/) is put on sys.path first, and model
        construction runs with the cwd at glee_repo because GLEE loads CLIP weights from the relative path
        projects/GLEE/clip_vit_base_patch32.
        """
        import os

        import torch  # torch + detectron2 + GLEE are GPU-backend only -> lazy
        from detectron2.config import get_cfg

        glee_repo = self.config.inference.glee_repo
        if glee_repo and glee_repo not in sys.path:
            sys.path.insert(0, glee_repo)

        # GLEE reads several data files (dataset descriptions, CLIP text weights) from paths relative to
        # the repo root, at BOTH import time and construction time, so chdir there for the whole block.
        prev_cwd = os.getcwd()
        try:
            if glee_repo:
                os.chdir(glee_repo)
            from projects.GLEE.glee.config import add_glee_config
            from projects.GLEE.glee.models.glee_model import GLEE_Model

            cfg = get_cfg()
            add_glee_config(cfg)
            cfg.merge_from_file(config_path)
            cfg.freeze()

            model = GLEE_Model(cfg, None, device, None, True).to(device)
            state = torch.load(weights_path, map_location=device)
            state = state.get("model", state)  # detectron2 ckpts wrap weights under "model"
            # Some GLEE checkpoints (e.g. *_scaleup) prefix every key with "glee." while GLEE_Model's own
            # modules are unprefixed (backbone.*, lang_projection, ...); strip it so the weights map on.
            if state and all(k.startswith("glee.") for k in state):
                state = {k[len("glee."):]: v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
        finally:
            os.chdir(prev_cwd)
        print(f"  GLEE weights loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        model.eval()

        self._pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
        self._pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)
        self._size_divisibility = 32
        self._infer_short_side = 800
        return model

    def track(self, frames: list[Image.Image], prompt: str) -> list[Masklet]:
        """Run GLEE open-vocab video tracking; one Masklet per kept track.

        Mirrors Sam3Tracker.track's OOM guard: a clip that OOMs the GPU retries once on CPU rather than
        crashing a multi-hour batch.
        """
        self.load()
        try:
            return self._run(frames, prompt, self.config.inference.device)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            self._torch.cuda.empty_cache()
            return self._run(frames, prompt, "cpu")  # rare oversized clip: CPU fallback

    def _run(self, frames: list[Image.Image], prompt: str, processing_device: str) -> list[Masklet]:
        """One open-vocab video pass: GLEE per-frame forward + MinVIS association -> tracks -> masklets.

        The GLEE call and association are finalised on the GPU box (see docs/glee_second_model.md); the
        schema _infer_tracks must produce for _masklets_from_glee is fixed and tested here.
        """
        tracks = self._infer_tracks(frames, prompt, processing_device)
        return _masklets_from_glee(tracks, len(frames), self.config.inference.glee_score_threshold)

    def _infer_tracks(
        self, frames: list[Image.Image], prompt: str, processing_device: str
    ) -> dict[int, list[tuple[int, np.ndarray, float]]]:
        """GLEE per-frame forward + MinVIS track-embed association -> per-track detections at original HW.

        Per frame: preprocess (RGB, ImageNet mean/std at 0-255 scale, no /255; resize short side to 800,
        pad to a multiple of 32); forward in category mode with batch_name_list=[prompt]; read
        pred_masks/pred_logits/pred_track_embed; keep top-k queries above the score threshold; upsample
        masks to the original (H, W) and binarise at logit 0; then link detections to running tracks by
        Hungarian matching on track-embed cosine similarity (MinVIS), creating a new track for each
        unmatched detection. Returns the dict schema _masklets_from_glee consumes.
        """
        import torch  # torch/torchvision are GPU-backend only -> lazy (scipy is a top-level import)
        import torch.nn.functional as F
        import torchvision

        model = self._model
        device = self.config.inference.device
        thr = self.config.inference.glee_score_threshold
        keep_topk = self.config.inference.glee_topk
        match_thr = self.config.inference.glee_match_threshold
        resizer = torchvision.transforms.Resize(self._infer_short_side, antialias=True)
        pixel_mean = self._pixel_mean.to(processing_device)
        pixel_std = self._pixel_std.to(processing_device)
        d = self._size_divisibility

        tracks: dict[int, list[tuple[int, np.ndarray, float]]] = {}
        track_embeds: dict[int, np.ndarray] = {}  # track_id -> last-seen L2-normed embed (running prototype)
        next_track_id = 0

        for frame_idx, frame in enumerate(frames):
            img = np.asarray(frame.convert("RGB"))  # HWC uint8 RGB (no BGR swap)
            ori_h, ori_w = img.shape[:2]

            chw = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))).to(processing_device).float()
            chw = (chw - pixel_mean) / pixel_std  # 0-255 scale, no /255
            resized = resizer(chw[None])  # [1,3,rh,rw]
            re_h, re_w = resized.shape[-2:]
            pad_h = (re_h + d - 1) // d * d
            pad_w = (re_w + d - 1) // d * d
            infer_image = torch.zeros(1, 3, pad_h, pad_w, device=processing_device)
            infer_image[0, :, :re_h, :re_w] = resized[0]  # top-left; zeros pad bottom/right
            infer_image = infer_image.to(device)

            with torch.no_grad():
                raw = model(infer_image, [], task="coco", batch_name_list=[prompt], is_train=False)
            out = _glee_out_dict(raw)

            scores_q = out["pred_logits"][0].sigmoid().max(-1)[0]  # [Q]
            mask_pred = out["pred_masks"][0]  # [Q, h', w'] logits
            track_embed = out["pred_track_embed"][0]  # [Q, D]

            k = min(keep_topk, scores_q.shape[0])
            top_scores, top_idx = scores_q.topk(k)
            sel = top_idx[top_scores >= thr]
            if sel.numel() == 0:
                continue

            m = mask_pred[sel][None]  # [1, K, h', w']
            m = F.interpolate(m, size=(pad_h, pad_w), mode="bilinear", align_corners=False)
            m = m[:, :, :re_h, :re_w]  # crop the zero pad -> resized-image extent
            m = F.interpolate(m, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
            masks_np = (m[0] > 0.0).detach().cpu().numpy().astype(bool)  # [K, ori_h, ori_w]
            emb = F.normalize(track_embed[sel], dim=-1).detach().cpu().numpy()  # [K, D] L2-normed
            det_scores = scores_q[sel].detach().cpu().numpy()

            if track_embeds:
                tids = list(track_embeds.keys())
                prev = np.stack([track_embeds[t] for t in tids])  # [T, D] already normed
                sim = emb @ prev.T  # [K, T] cosine
                row, col = linear_sum_assignment(-sim)  # maximise similarity
                matched = set()
                for r, c in zip(row, col, strict=False):
                    if sim[r, c] >= match_thr:
                        tid = tids[c]
                        tracks[tid].append((frame_idx, masks_np[r], float(det_scores[r])))
                        track_embeds[tid] = emb[r]  # near-online: update to the latest frame
                        matched.add(r)
                unmatched = [r for r in range(len(sel)) if r not in matched]
            else:
                unmatched = list(range(len(sel)))

            for r in unmatched:
                tracks[next_track_id] = [(frame_idx, masks_np[r], float(det_scores[r]))]
                track_embeds[next_track_id] = emb[r]
                next_track_id += 1

        return tracks
