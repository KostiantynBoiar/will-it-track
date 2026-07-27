"""Box → filled-rectangle COCO-RLE, for box-only datasets (e.g. MammAlps).

A dataset whose ground truth is bounding boxes rather than masks is made to fit the mask-based pipeline by
emitting each box as a filled-rectangle RLE mask: :func:`src.dataset.SAFARI.mask_at` decodes it unchanged, so
the visual (box-crop), size (box area) and environment (box-erased scene) features run as *box*-based
features, and the scorer is told to read ``bbox_*`` HOTA (``eval.prefer_bbox``) so a tight predicted mask is
not penalised against the rectangle GT.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as coco_mask


def box_to_rle(x: float, y: float, w: float, h: float, height: int, width: int) -> dict | None:
    """COCO-RLE ``{size, counts}`` for the axis-aligned box ``[x, y, w, h]`` on an ``(height, width)`` frame.

    Coordinates are clamped to the frame; ``counts`` is an ASCII string (the schema ``mask_at`` reads).
    Returns ``None`` for an empty/degenerate box (so the frame is treated as unlabelled).
    """
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(width, int(round(x + w))), min(height, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    rle = coco_mask.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle
