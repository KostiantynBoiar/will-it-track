"""Pure crop / colour-heuristic tests for the visual+environment features (no frames, no model)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.features.frames import (
    animal_crop,
    annotated_frame_indices,
    frame_achromatic,
    laplacian_variance,
    masked_background,
    sample_frame_indices,
)


def _solid(value: int, size: int = 20) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8))


def test_animal_crop_box_from_mask() -> None:
    """The crop box is the mask's bounding box (PIL size is (w, h))."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 6:12] = True  # h=5, w=6
    crop = animal_crop(_solid(200), mask, mask_crop=True, min_px=4)
    assert crop is not None
    assert crop.size == (6, 5)
    assert (np.asarray(crop) == 200).all()  # full mask over the box → all foreground


def test_animal_crop_zeros_background() -> None:
    """With ``mask_crop`` the non-mask pixels inside the box are zeroed."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 6:12] = True
    mask[5, 6] = False  # a hole at the box's top-left corner
    crop = np.asarray(animal_crop(_solid(200), mask, mask_crop=True, min_px=4))
    assert (crop[0, 0] == 0).all()
    assert (crop[-1, -1] == 200).all()


def test_animal_crop_empty_returns_none() -> None:
    """An empty/too-small mask yields no crop."""
    assert animal_crop(_solid(0), np.zeros((20, 20), dtype=bool), min_px=4) is None


def test_masked_background_neutralises_union() -> None:
    """The animal union is mean-inpainted (not left as the original values); shape is preserved."""
    arr = np.full((20, 20, 3), 50, dtype=np.uint8)
    arr[0:10] = 200
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:3, 1:3] = True
    out = np.asarray(masked_background(Image.fromarray(arr), [mask], fill="mean"))
    assert out.shape == (20, 20, 3)
    assert not (out[1:3, 1:3] == 200).all()  # replaced by the scene mean


def test_frame_achromatic() -> None:
    """Grayscale frames read as night/IR; saturated ones do not."""
    gray = _solid(120)
    color = Image.fromarray(
        np.dstack([np.full((10, 10), 220), np.full((10, 10), 20), np.full((10, 10), 20)]).astype(
            np.uint8
        )
    )
    assert frame_achromatic(gray) is True
    assert frame_achromatic(color) is False


def test_sample_frame_indices() -> None:
    """Only non-None segmentation frames are sampled; at most ``n`` of them."""
    ann = {"segmentations": [None, {}, {}, None, {}, {}]}
    assert annotated_frame_indices(ann) == [1, 2, 4, 5]
    sampled = sample_frame_indices(ann, 2)
    assert len(sampled) <= 2
    assert set(sampled) <= {1, 2, 4, 5}


def test_laplacian_variance() -> None:
    """Flat frames have ~0 clutter; textured frames have positive clutter."""
    assert laplacian_variance(_solid(100)) == 0.0
    noisy = Image.fromarray((np.random.RandomState(0).rand(10, 10, 3) * 255).astype(np.uint8))
    assert laplacian_variance(noisy) > 0
