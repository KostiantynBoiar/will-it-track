"""Generate the 'distances in action' figure images from a real frame + GT mask (reproducible).

Writes ``feat_overlay/animal/background/original.png`` to ``report/dissertation/figures/`` for
Figure~\\ref{fig:features}: the frame with its ground-truth mask, the mask-cropped animal (the *visual*
distance input), and the animal-erased background (the *environment* distance input). Torch/PIL-only, no
SAM 3. The subject is a fixed, hand-picked daytime frame (a collared peccary) whose frames are local.

Run: ``PYTHONPATH=. python -m src.analysis.figures_features``
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.config import Config
from src.dataset import SAFARI

# Hand-picked subject: a large, daytime (colour) collared peccary whose frame is local.
_SPLIT, _VIDEO_ID, _CATEGORY_ID, _FRAME_INDEX = "test", "583", "43570", 0
_FRAME_FILE = "sa_fari_000584/00000.jpg"
_ORANGE = np.array([214, 104, 26])  # the deck/dissertation accent


def _fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    r = Image.fromarray((mask.astype(np.uint8) * 255)).resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(r) > 127


def generate(config: Config | None = None) -> None:
    """Write the four feature-illustration PNGs."""
    cfg = config or Config()
    froot = cfg.paths.data_root / cfg.data.frames_subdir
    outdir = cfg.paths.outputs_root.parent / "report" / "dissertation" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    fpath = froot / _FRAME_FILE
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not local — fetch it first (src.acquire) or pick another subject")

    safari = SAFARI(_SPLIT, cfg)
    ann = next(
        a for a in safari.annotations_by_video()[_VIDEO_ID]
        if str(a["category_id"]) == _CATEGORY_ID and (a.get("segmentations") or [])[_FRAME_INDEX] is not None
    )
    frame = Image.open(fpath).convert("RGB")
    arr = np.asarray(frame)
    mask = _fit(safari.mask_at(ann, _FRAME_INDEX).astype(bool), arr.shape[:2])

    frame.save(outdir / "feat_original.png")
    # (a) frame + mask overlay (orange)
    ov = arr.copy()
    ov[mask] = (0.45 * ov[mask] + 0.55 * _ORANGE).astype(np.uint8)
    Image.fromarray(ov).save(outdir / "feat_overlay.png")
    # (b) visual: mask-cropped animal (background zeroed, cropped to the mask bbox)
    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    Image.fromarray(np.where(mask[:, :, None], arr, 0).astype(np.uint8)).crop(box).save(outdir / "feat_animal.png")
    # (c) environment: animal mean-inpainted
    bg = arr.copy()
    bg[mask] = arr[~mask].mean(0).astype(np.uint8)
    Image.fromarray(bg).save(outdir / "feat_background.png")
    print(f"feature figures -> {outdir} (subject: {safari._species_name(_CATEGORY_ID)}, mask {int(mask.sum())} px)")


if __name__ == "__main__":
    generate()
