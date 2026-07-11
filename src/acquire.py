"""Acquire SA-FARI: gated annotations from Hugging Face + public 6 fps frames from GCS.

Annotations (``facebook/SA-FARI``) are **gated** — accept the license and ``huggingface-cli login``
first (see the README "Data access" section). Frames (``gs://cxl-public-camera-trap``) are **public**
— anonymous ``gcsfs``, no auth. Only a minimal slice is pulled (the annotation JSONs + a few clips'
frames); the full 6 fps set is a later background job.

Usage::

    PYTHONPATH=. .venv/bin/python -m src.acquire --list             # probe the public GCS layout
    PYTHONPATH=. .venv/bin/python -m src.acquire --annotations      # HF snapshot (needs login)
    PYTHONPATH=. .venv/bin/python -m src.acquire --frames --n-clips 3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gcsfs
from huggingface_hub import constants as hf_constants
from huggingface_hub import hf_hub_download

from src.config import Config
from src.dataset import SAFARI

# Fine-grained HF tokens 403 on the Xet download backend; force plain HTTPS. `is_xet_available()`
# reads this flag live, so overriding it here (after import) is enough.
hf_constants.HF_HUB_DISABLE_XET = True


class AnnotationFetcher:
    """Download the gated SA-FARI annotation JSONs from Hugging Face."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize.

        Args:
            config: Project config (``data.hf_repo``, ``paths.data_root``, ``data.annotations_subdir``).
        """
        self.config = config or Config()

    def fetch(self) -> Path:
        """Download the two ``_ext`` annotation JSONs into ``data/annotations/`` (flat).

        The repo stores them under an ``annotation/`` prefix; we place them flat, where the loader
        (:class:`src.dataset.SAFARI`) expects ``data_root/annotations_subdir/<name>``.

        Returns:
            The annotations directory.

        Raises:
            Exception: If the HF login/license gate is not satisfied (surfaced from the hub).
        """
        dest = self.config.paths.data_root / self.config.data.annotations_subdir
        dest.mkdir(parents=True, exist_ok=True)
        for name in (self.config.data.train_ann, self.config.data.test_ann):
            cached = hf_hub_download(
                repo_id=self.config.data.hf_repo,
                repo_type="dataset",
                filename=f"annotation/{name}",
                token=True,  # reads ~/.cache/huggingface/token or $HF_TOKEN
            )
            shutil.copyfile(cached, dest / name)
        return dest


class FrameFetcher:
    """Pull a subset of 6 fps frames from the public GCS bucket (anonymous)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the anonymous GCS filesystem.

        Args:
            config: Project config (``data.gcs_bucket``, ``paths.data_root``, ``data.frames_subdir``).
        """
        self.config = config or Config()
        self.bucket = self.config.data.gcs_bucket
        self.fs = gcsfs.GCSFileSystem(token="anon")

    def list_layout(self, depth: int = 2, per_level: int = 6) -> list[str]:
        """Probe the top of the bucket to confirm public access + the frame layout.

        Args:
            depth: Levels to descend.
            per_level: Entries to show per level.

        Returns:
            Indented ``gs://`` paths.
        """
        lines: list[str] = []

        def walk(prefix: str, remaining: int, indent: int) -> None:
            try:
                entries = self.fs.ls(prefix)[:per_level]
            except Exception as exc:  # noqa: BLE001 - report, don't crash the probe
                lines.append("  " * indent + f"{prefix}  <error: {exc}>")
                return
            for entry in entries:
                lines.append("  " * indent + f"gs://{entry}")
                if remaining > 1:
                    walk(entry, remaining - 1, indent + 1)

        walk(self.bucket, depth, 0)
        return lines

    def _frame_uri(self, split: str, file_name: str) -> str:
        """Full bucket URI for one ``file_names`` entry under the split's frame root."""
        base = self.config.data.frames_gcs_dir.format(split=split)
        return f"{self.bucket}/{base}/{file_name}"

    def fetch(self, file_names: list[str], split: str) -> int:
        """Download the given frame paths into ``data/frames/`` (skipping ones already present).

        Args:
            file_names: Annotation ``file_names`` (``<video_name>/<frame>.jpg``).
            split: The source split (``"train"`` / ``"test"``) — selects the bucket frame root.

        Returns:
            The number of files newly downloaded.

        Raises:
            FileNotFoundError: If the first frame does not resolve (wrong split/template).
        """
        if not file_names:
            return 0
        if not self.fs.exists(self._frame_uri(split, file_names[0])):
            raise FileNotFoundError(
                f"cannot locate {file_names[0]!r} at {self._frame_uri(split, file_names[0])}"
            )
        dest_root = self.config.paths.data_root / self.config.data.frames_subdir
        pulled = 0
        for name in file_names:
            dest = dest_root / name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.fs.get_file(self._frame_uri(split, name), str(dest))
            pulled += 1
        return pulled


def main() -> None:
    """CLI: ``--list`` / ``--annotations`` / ``--frames --n-clips N``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("--list", action="store_true", help="probe the public GCS layout")
    ap.add_argument("--annotations", action="store_true", help="download the gated HF annotations")
    ap.add_argument(
        "--frames", action="store_true", help="pull frames for the first --n-clips videos"
    )
    ap.add_argument("--split", default="test", choices=("train", "test"), help="split for --frames")
    ap.add_argument("--n-clips", type=int, default=3)
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if args.list:
        for line in FrameFetcher(cfg).list_layout():
            print(line)
    if args.annotations:
        print("annotations ->", AnnotationFetcher(cfg).fetch())
    if args.frames:
        records = SAFARI(args.split, cfg).records()[: args.n_clips]
        file_names = [name for r in records for name in r.file_names]
        pulled = FrameFetcher(cfg).fetch(file_names, args.split)
        print(f"frames -> pulled {pulled} files for {len(records)} {args.split} clips")


if __name__ == "__main__":
    main()
