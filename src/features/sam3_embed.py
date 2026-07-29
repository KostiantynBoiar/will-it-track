"""SAM 3's own vision-encoder embeddings for the familiarity proxy (T2.5).

Unlike the frozen DINOv2/CLIP encoders in :mod:`src.features.embed`, this reads representations from **SAM 3
itself** --- one pooled vector per mask-cropped animal --- so the familiarity proxy can measure how separable a
species is in SAM 3's own feature space. It satisfies the same duck-typed ``embed(images) -> (N, D)`` contract
as :class:`~src.features.embed.Embedder`, so it drops straight into :func:`~src.features.pipeline.embed_crops`.

The heavy backend (``transformers`` SAM 3 + torch) is imported **lazily** inside ``_load`` --- exactly like
:class:`~src.inference.sam3_tracker.Sam3Tracker` --- so this module imports on the CPU analysis env and the
hermetic tests (which substitute a fake embedder) never touch it. Runs on ``inference.device`` /
``inference.precision`` (GPU / bf16), not the ``features.embed_device`` used for DINOv2. Image embeddings come
from ``Sam3Model.get_vision_features(pixel_values=...)`` --- a pure image-encoder forward, no video tracking.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.config import Config


class Sam3Embedder:
    """SAM 3 vision-encoder embedder: ``embed(images) -> (N, D)`` L2-normalised float32 (loads on first use)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize (the model is not loaded until the first ``embed``)."""
        self.config = config or Config()
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._dtype = None

    def _load(self) -> None:
        """Lazily build the frozen SAM 3 image encoder on the configured GPU/precision.

        Tries the standalone single-image ``Sam3Model`` first (the direct image API); falls back to the
        detector inside the video model if the checkpoint only ships the video wrapper.
        """
        if self._model is not None:
            return
        import torch  # lazy: transformers/torch SAM 3 only in the GPU env
        from transformers import Sam3Processor, Sam3VideoModel

        name = self.config.inference.sam3_model
        device = self.config.inference.device
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            self.config.inference.precision, torch.float32
        )
        try:  # preferred: the standalone image model exposes get_vision_features directly
            from transformers import Sam3Model

            model = Sam3Model.from_pretrained(name, dtype=dtype)
        except Exception as exc:  # noqa: BLE001 - fall back to the detector inside the video model
            video = Sam3VideoModel.from_pretrained(name, dtype=dtype)
            model = getattr(video, "detector_model", None)
            if model is None or not hasattr(model, "get_vision_features"):
                raise RuntimeError(
                    "could not reach a SAM 3 image encoder with get_vision_features "
                    f"(standalone Sam3Model failed: {exc})"
                ) from exc
        self._model = model.to(device).eval()
        self._processor = Sam3Processor.from_pretrained(name)
        self._torch, self._device, self._dtype = torch, device, dtype

    def _pool(self, out) -> np.ndarray:  # noqa: ANN001 - Sam3VisionEncoderOutput
        """Reduce a vision-encoder output to one vector per crop per ``features.familiarity_pooling``."""
        if self.config.features.familiarity_pooling == "patch_mean":
            return out.last_hidden_state.mean(dim=1).float().cpu().numpy()
        return out.pooler_output.float().cpu().numpy()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Embed crops to L2-normalised float32 vectors ``(len(images), D)`` via SAM 3's vision encoder."""
        if not images:
            return np.zeros((0, 0), dtype="float32")
        self._load()
        batch = max(1, self.config.inference.batch_frames)
        chunks = []
        with self._torch.no_grad():
            for start in range(0, len(images), batch):
                px = self._processor(images=images[start : start + batch], return_tensors="pt")[
                    "pixel_values"
                ].to(self._device, self._dtype)
                chunks.append(self._pool(self._model.get_vision_features(pixel_values=px)))
        vecs = np.concatenate(chunks, axis=0).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / np.clip(norms, 1e-12, None)).astype("float32")
