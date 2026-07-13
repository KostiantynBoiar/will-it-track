"""Frozen SAM 3 promptable video tracking — the only model/torch-touching module.

The heavy backend (``transformers`` SAM 3 + torch) is imported *lazily* inside ``load()``, so this
module imports on the CPU analysis env (no GPU deps) and the tests can run. A :class:`FakeTracker` with
the same interface drives those tests; the real :class:`Sam3Tracker` runs on the GPU box (Colab).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pydantic import BaseModel

from src.config import Config


def encode_rle(mask: np.ndarray) -> dict:
    """COCO-RLE encode a boolean mask into a JSON-serialisable ``{size, counts}`` dict."""
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


class Masklet(BaseModel):
    """One tracked object: a per-frame RLE mask (``None`` where absent) + a confidence score."""

    segmentations: list[dict | None]
    score: float


class Tracker(Protocol):
    """The interface the harness depends on (satisfied by both trackers below)."""

    def track(self, frames: list[Image.Image], prompt: str) -> list[Masklet]: ...


class Sam3Tracker:
    """Frozen SAM 3 video tracker via ``transformers`` ``Sam3VideoModel`` (loads on first ``track``)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (the model is not loaded until the first ``track``)."""
        self.config = config or Config()
        self._model = None
        self._processor = None
        self._torch = None

    def load(self) -> None:
        """Lazily build the frozen predictor on the configured device."""
        if self._model is not None:
            return
        import torch  # lazy: only available in the GPU (Colab) env
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        name = self.config.inference.sam3_model
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            self.config.inference.precision, torch.float32
        )
        self._model = (
            Sam3VideoModel.from_pretrained(name, dtype=dtype)
            .to(self.config.inference.device)
            .eval()
        )
        self._processor = Sam3VideoProcessor.from_pretrained(name)
        self._torch = torch

    def track(self, frames: list[Image.Image], prompt: str) -> list[Masklet]:
        """Run promptable tracking and return one :class:`Masklet` per kept object."""
        self.load()
        session = self._processor.init_video_session(
            video=frames, inference_device=self.config.inference.device
        )
        self._processor.add_text_prompt(session, text=prompt)

        n = len(frames)
        segs: dict[int, list[dict | None]] = {}
        scores: dict[int, float] = {}
        with self._torch.no_grad():
            for output in self._model.propagate_in_video_iterator(session):
                processed = self._processor.postprocess_outputs(session, output)
                for obj_id, mask, score in _objects(processed):
                    segs.setdefault(obj_id, [None] * n)[output.frame_idx] = encode_rle(mask)
                    scores[obj_id] = max(scores.get(obj_id, 0.0), float(score))

        threshold = self.config.inference.score_threshold
        return [
            Masklet(segmentations=segs[oid], score=scores[oid])
            for oid in segs
            if scores[oid] >= threshold
        ]


def _objects(processed):  # noqa: ANN001 - transformers SAM 3 postprocess dict
    """Yield ``(obj_id, bool_mask, score)`` from one frame's postprocessed SAM 3 output.

    ``Sam3VideoProcessor.postprocess_outputs`` returns a dict of parallel tensors --- ``object_ids``
    ``(N,)``, ``scores`` ``(N,)`` and ``masks`` ``(N, H, W)`` (binary at original resolution) --- not an
    iterable of per-object dicts. Comparing ``> 0.5`` is a no-op on an already-binary mask and a safe
    threshold should a build return float probabilities instead.
    """
    obj_ids = processed["object_ids"].tolist()
    scores = processed["scores"].tolist()
    masks = processed["masks"]
    for i, obj_id in enumerate(obj_ids):
        mask = (masks[i] > 0.5).detach().cpu().numpy().astype(bool)
        yield int(obj_id), mask, float(scores[i])


class FakeTracker:
    """Deterministic stand-in for the harness tests — no model, no GPU.

    Returns ``masklets_per_call`` synthetic objects (a small mask on the first frame); set it to ``0`` to
    simulate a hard negative (nothing found).
    """

    def __init__(self, config: Config | None = None, masklets_per_call: int = 1) -> None:
        """Initialize."""
        self.config = config or Config()
        self.masklets_per_call = masklets_per_call

    def track(self, frames: list[Image.Image], prompt: str) -> list[Masklet]:
        """Return synthetic masklets aligned to ``frames``."""
        if not frames or self.masklets_per_call <= 0:
            return []
        width, height = frames[0].size
        mask = np.zeros((height, width), dtype=bool)
        mask[: max(1, height // 4), : max(1, width // 4)] = True
        out = []
        for _ in range(self.masklets_per_call):
            segs: list[dict | None] = [None] * len(frames)
            segs[0] = encode_rle(mask)
            out.append(Masklet(segmentations=segs, score=0.9))
        return out
