"""Frame + crop utilities for the visual/environment features (T2.2/T2.3).

Pure PIL/numpy — no torch. Resolves the annotated frames of a masklet, pulls them on demand
(fetch-or-skip via :class:`~src.acquire.FrameFetcher`, never crashing a feature), and produces the two
crop kinds: the mask-cropped animal (visual) and the animal-masked-out background (environment). Also
holds the colour-only heuristics (night/IR, clutter) so their tests need neither frames nor a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image

from src.acquire import FrameFetcher
from src.config import Config


def annotated_frame_indices(annotation: dict) -> list[int]:
    """Frame indices of a masklet that carry a mask (``segmentations[i] is not None``)."""
    segs = annotation.get("segmentations") or []
    return [i for i, seg in enumerate(segs) if seg is not None]


def sample_frame_indices(annotation: dict, n: int, seed: int = 0) -> list[int]:
    """Up to ``n`` evenly-spaced annotated frame indices (deterministic)."""
    idx = annotated_frame_indices(annotation)
    if n <= 0 or len(idx) <= n:
        return idx
    positions = np.linspace(0, len(idx) - 1, n).round().astype(int)
    return sorted({idx[p] for p in positions})


def ensure_frames(file_names: Sequence[str], split: str, config: Config | None = None) -> int:
    """Fetch the given frames if missing (batched); return how many were newly pulled.

    Fetch-or-skip: a network/GCS failure logs nothing but returns 0 rather than crashing the caller.
    """
    cfg = config or Config()
    try:
        return FrameFetcher(cfg).fetch(list(file_names), split)
    except Exception:  # noqa: BLE001 - missing frames must not crash a feature; caller skips them
        return 0


def load_frame(file_name: str, config: Config | None = None) -> Image.Image | None:
    """Load a local frame as RGB, or ``None`` if it is missing/unreadable (call ``ensure_frames`` first)."""
    cfg = config or Config()
    path = cfg.paths.data_root / cfg.data.frames_subdir / file_name
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001 - a corrupt frame is skipped, not fatal
        return None


def _fit_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to ``shape`` (H, W) with nearest-neighbour if it does not already match."""
    if mask.shape == shape:
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (shape[1], shape[0]), Image.NEAREST
    )
    return np.asarray(resized) > 127


def animal_crop(
    frame: Image.Image,
    mask: np.ndarray,
    mask_crop: bool = True,
    min_px: int = 64,
) -> Image.Image | None:
    """Crop the animal to its mask bounding box; ``None`` if the mask is empty/too small.

    The crop box is derived from ``np.where(mask)`` (not the stored COCO bbox) so it always matches the
    decoded pixels. When ``mask_crop`` the background is zeroed *before* cropping (§9 DO), so appearance —
    not scene — drives the embedding.
    """
    arr = np.asarray(frame.convert("RGB"))
    mask = _fit_mask(mask.astype(bool), arr.shape[:2])
    if int(mask.sum()) < min_px:
        return None
    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if mask_crop:
        arr = np.where(mask[:, :, None], arr, 0)
    return Image.fromarray(arr.astype(np.uint8)).crop(box)


def masked_background(
    frame: Image.Image,
    masks: Iterable[np.ndarray],
    fill: str = "mean",
) -> Image.Image:
    """Whole frame with the animal (union of masks) neutralised — mean-inpainted, not a black hole."""
    arr = np.asarray(frame.convert("RGB")).astype(np.uint8).copy()
    union = np.zeros(arr.shape[:2], dtype=bool)
    for mask in masks:
        if mask is not None:
            union |= _fit_mask(mask.astype(bool), arr.shape[:2])
    if union.any() and (~union).any():
        fill_value = arr[~union].mean(axis=0) if fill == "mean" else np.zeros(3)
        arr[union] = fill_value.astype(np.uint8)
    return Image.fromarray(arr)


def frame_achromatic(frame: Image.Image, threshold: float = 12.0) -> bool:
    """True if the frame is near-grayscale (IR/night): mean per-pixel channel spread below ``threshold``."""
    arr = np.asarray(frame.convert("RGB")).astype(np.int16)
    spread = arr.max(axis=2) - arr.min(axis=2)
    return float(spread.mean()) < threshold


def laplacian_variance(frame: Image.Image) -> float:
    """Variance of a 4-neighbour Laplacian over the interior — a cheap clutter/sharpness proxy.

    Computed on interior pixels only (each has all four neighbours) so a flat frame gives exactly 0.
    """
    g = np.asarray(frame.convert("L"), dtype=np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = -4.0 * g[1:-1, 1:-1] + g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2]
    return float(lap.var())
