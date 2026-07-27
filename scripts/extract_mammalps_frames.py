"""Extract subsampled frames for the MammAlps probe clips — from a local zip OR a remote (range) zip.

Reads the converted probe JSON (``data.test_ann``) and, for each clip's ``source_video`` + ``sample_step``,
pulls just that MP4 and ffmpeg-extracts every ``step``-th frame to ``data_root/frames/<file_id>/<i>.jpg`` —
aligned to the clip's ``file_names``. Only the capped probe clips (a few hundred MP4s) are touched, not the
whole 82 GiB archive; each MP4 is fetched to a temp file and discarded, so the only persistent storage is the
frames (~5 GB). With ``--remote-url`` it uses ``remotezip`` (HTTP range) to avoid ever downloading the full
zip. Idempotent: a clip whose frames already exist is skipped.

Run (remote): ``python scripts/extract_mammalps_frames.py --config configs/mammalps.yaml
--remote-url https://zenodo.org/records/15588220/files/mammalps_v1.zip``
Run (local):  ``... --zip /workspace/data_mammalps/mammalps_v1.zip``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from src.config import Config


def _member_map(names) -> dict[str, str]:
    """``basename → full zip entry`` for MP4s (the archive's internal prefix is undocumented)."""
    return {Path(n).name: n for n in names if n.lower().endswith(".mp4")}


def _ffmpeg(mp4: Path, outdir: Path, step: int, n: int) -> None:
    """Extract every ``step``-th frame (first ``n``) of ``mp4`` to ``outdir/%06d.jpg`` (0-indexed)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-vf", f"select='not(mod(n\\,{step}))'", "-vsync", "vfr",
         "-frames:v", str(n), "-start_number", "0", str(outdir / "%06d.jpg")],
        check=True,
    )


def extract(config: Config, zip_path: str | None, remote_url: str | None) -> None:
    """Extract every clip's subsampled frames (from a local zip or a remote range-zip) into ``frames/``."""
    ann_path = config.paths.data_root / config.data.annotations_subdir / config.data.test_ann
    videos = json.loads(ann_path.read_text())["videos"]
    frames_root = config.paths.data_root / config.data.frames_subdir

    if remote_url:
        from remotezip import RemoteZip
        zf = RemoteZip(remote_url)
    else:
        zf = zipfile.ZipFile(zip_path)
    members = _member_map(zf.namelist())
    ok = miss = skip = 0
    try:
        for v in videos:
            file_id, step, n = v["video_name"], int(v.get("sample_step", 10)), len(v["file_names"])
            outdir = frames_root / file_id
            if outdir.exists() and len(list(outdir.glob("*.jpg"))) >= n:
                skip += 1
                continue
            member = members.get(Path(v.get("source_video", "")).name)
            if member is None:
                miss += 1
                print(f"MISS {v.get('source_video')}", flush=True)
                continue
            outdir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                mp4 = Path(td) / "clip.mp4"
                with zf.open(member) as src, open(mp4, "wb") as dst:
                    dst.write(src.read())  # one MP4 at a time, then discarded
                _ffmpeg(mp4, outdir, step, n)
            ok += 1
            if ok % 10 == 0:
                print(f"extracted {ok} clips ...", flush=True)
    finally:
        zf.close()
    print(f"DONE extracted={ok} skipped={skip} missing={miss}", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--zip", default=None, help="path to a local mammalps_v1.zip")
    ap.add_argument("--remote-url", default=None, help="Zenodo zip URL (HTTP-range via remotezip)")
    args = ap.parse_args()
    if not (args.zip or args.remote_url):
        ap.error("one of --zip or --remote-url is required")
    extract(Config.load(args.config), args.zip, args.remote_url)


if __name__ == "__main__":
    main()
