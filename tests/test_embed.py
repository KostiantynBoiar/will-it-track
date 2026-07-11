"""Model-dependent smokes for the embedding pipeline.

Gated behind ``RUN_MODEL_TESTS=1`` so a routine ``pytest`` never triggers a weight download or a frame
pull; run ``RUN_MODEL_TESTS=1 pytest`` once the DINOv2 weights are cached.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from src.config import Config
from src.dataset import SAFARI
from src.features.embed import Embedder
from src.features.visual import VisualDistance
from src.splits import build_species_partition

_CFG = Config()
_ANN = SAFARI("train", _CFG).ann_path.exists() and SAFARI("test", _CFG).ann_path.exists()
_needs_model = pytest.mark.skipif(
    not os.environ.get("RUN_MODEL_TESTS"), reason="set RUN_MODEL_TESTS=1 (uses DINOv2 weights)"
)
_needs_ann = pytest.mark.skipif(not _ANN, reason="SA-FARI annotations not fetched")


def _solids(*values: int) -> list[Image.Image]:
    return [Image.fromarray(np.full((40, 40, 3), v, dtype=np.uint8)) for v in values]


@_needs_model
def test_embedder_smoke() -> None:
    """DINOv2 embeddings are ``(N, D)`` float32, L2-normalised, and deterministic."""
    vecs = Embedder("dinov2", _CFG).embed(_solids(30, 200))
    assert vecs.shape[0] == 2 and vecs.shape[1] > 0
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert np.allclose(vecs, Embedder("dinov2", _CFG).embed(_solids(30, 200)), atol=1e-4)


@_needs_model
@_needs_ann
def test_visual_distance_subset() -> None:
    """Visual distance over a 4-species leave-one-species-out subset (pulls their frames)."""
    part = build_species_partition(_CFG)
    subset = part.reference_species[:4]
    mini = part.model_copy(update={"reference_species": subset, "probe_species": subset})
    series = VisualDistance(_CFG).compute(mini)

    assert set(series.index) == set(subset)
    valid = series.dropna()
    assert valid.between(0.0, 2.0).all()  # cosine distance range
