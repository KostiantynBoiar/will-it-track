"""Frozen GLEE promptable video tracking — the second tracker for the model-swap experiment.

GLEE (CVPR 2024) has the same contract as SAM 3: a text prompt in -> zero-shot detect, segment and track
every matching instance across a video, masks out. Running it over the same SA-FARI cells tests whether the
label-free null is SAM-3-specific or task-general. See docs/glee_second_model.md.

The heavy backend (torch + GLEE + detectron2) is imported lazily in load(), so this module imports on the
CPU analysis env and the tests never touch a GPU. Model-touching code lives in load()/_run(); the pure
output-to-Masklet assembly is _masklets_from_glee(), unit-tested without any model. Masklet and encode_rle
are reused from sam3_tracker — one masklet per track, no obj_id (identity is list membership, as SAM 3 does).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

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
        if not inf.glee_config or not inf.glee_weights:
            raise RuntimeError(
                "GleeTracker needs inference.glee_config and inference.glee_weights set "
                "(the GLEE model YAML + .pth). See docs/glee_second_model.md for staging."
            )
        # Build the GLEE model from its config + weights. Finalised on the GPU box against the GLEE repo
        # checkout (projects/GLEE), per docs/glee_second_model.md.
        self._model = self._build_glee(inf.glee_config, inf.glee_weights, inf.device)
        self._dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(inf.precision, torch.float32)
        self._torch = torch

    def _build_glee(self, config_path: str, weights_path: str, device: str):  # noqa: ANN001, ANN202
        """Construct the GLEE model from its config + weights on device (GPU box only)."""
        from detectron2.config import get_cfg  # lazy: GLEE/detectron2 env only
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        return DefaultPredictor(cfg)

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
        """GLEE forward + MinVIS association -> track_id -> [(frame_index, bool_mask@HW, score), ...].

        Finalised against the GLEE checkout on the GPU box (docs/glee_second_model.md): per frame, run the
        forward with batch_name_list=[prompt] to get pred_masks/pred_logits/pred_track_embed; keep queries
        above the score threshold; Hungarian-match pred_track_embed across frames into persistent tracks;
        upsample each kept mask to (H, W) and binarise. Isolated so _masklets_from_glee stays GPU-free.
        """
        raise NotImplementedError(
            "GleeTracker._infer_tracks is finalised on the GPU box against the GLEE repo checkout; "
            "the CPU groundwork (contract + _masklets_from_glee) is complete and tested."
        )
