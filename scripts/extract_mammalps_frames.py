"""Extract subsampled frames for the MammAlps probe clips from the big ``mammalps_v1.zip``.

Reads the converted probe JSON (``data.test_ann``) and, for each clip's ``source_video`` + ``sample_step``,
pulls just that MP4 out of the ~82 GiB zip and ffmpeg-extracts every ``step``-th frame to
``data_root/frames/<file_id>/<i>.jpg`` — aligned to the clip's ``file_names``. Only the capped probe clips are
extracted (a few hundred MP4s), not the whole archive. Idempotent: a clip whose frames already exist is
skipped.

Run: ``PYTHONPATH=. python scripts/extract_mammalps_frames.py --config configs/mammalps.yaml --zip <path>``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from src.config import Config


def _zip_member(names_by_base: dict[str, str], source_video: str) -> str | None:
    """The zip entry for a clip, matched by basename (the archive's internal prefix is undocumented)."""
    return names_by_base.get(Path(source_video).name)


def extract(config: Config, zip_path: str) -> None:
    """Extract every clip's subsampled frames from the zip into ``data_root/frames/<file_id>/``."""
    ann_path = config.paths.data_root / config.data.annotations_subdir / config.data.test_ann
    videos = json.loads(ann_path.read_text())["videos"]
    frames_root = config.paths.data_root / config.data.frames_subdir
    with zipfile.ZipFile(zip_path) as zf:
        names_by_base = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".mp4")}
        ok = miss = skip = 0
        for v in videos:
            file_id, step, n = v["video_name"], int(v.get("sample_step", 10)), len(v["file_names"])
            outdir = frames_root / file_id
            if outdir.exists() and len(list(outdir.glob("*.jpg"))) >= n:
                skip += 1
                continue
            member = _zip_member(names_by_base, v.get("source_video", ""))
            if member is None:
                miss += 1
                print(f"MISS {v.get('source_video')}", flush=True)
                continue
            outdir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                mp4 = Path(td) / "clip.mp4"
                with zf.open(member) as src, open(mp4, "wb") as dst:
                    dst.write(src.read())
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                     "-vf", f"select='not(mod(n\\,{step}))'", "-vsync", "vfr",
                     "-frames:v", str(n), "-start_number", "0", str(outdir / "%06d.jpg")],
                    check=True,
                )
            ok += 1
            if ok % 20 == 0:
                print(f"extracted {ok} clips ...", flush=True)
        print(f"DONE extracted={ok} skipped={skip} missing={miss}", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--zip", required=True, help="path to mammalps_v1.zip")
    args = ap.parse_args()
    extract(Config.load(args.config), args.zip)


if __name__ == "__main__":
    main()
