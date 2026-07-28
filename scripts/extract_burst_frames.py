"""Extract the annotated frames for the BURST probe videos — streamed from the HF TAO-Amodal frame zips.

Reads the converted probe JSON (``data.test_ann``) and, for each video's ``source_dataset`` + ``source_seq``,
pulls just that video's annotated frames from ``frames/<split>/<source>.zip`` on the gated HF dataset
``chengyenhsieh/TAO-Amodal`` (HTTP range via ``remotezip`` + a bearer token) and writes them to
``data_root/frames/<video_name>/<frame>``, aligned to the video's ``file_names``. Only the capped
animal-subset frames are touched (a few thousand JPEGs), never the whole multi-GB archive. Idempotent: a video
whose frames already exist is skipped.

Frame paths inside each source zip are ``<seq_name>/<frame>.jpg``; the probe's ``video_name`` is
``<source>__<seq_name>`` so the two never collide across sources.

Run: ``PYTHONPATH=. python scripts/extract_burst_frames.py --config configs/burst.yaml --split val``
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from src.config import Config

_HF_URL = "https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal/resolve/main/frames/{split}/{source}.zip"


def _token() -> str:
    """Read the cached Hugging Face token (the frame repo is gated)."""
    tok = os.environ.get("HF_TOKEN") or Path("~/.cache/huggingface/token").expanduser().read_text().strip()
    return tok


def extract(config: Config, split: str) -> None:
    """Extract every probe video's annotated frames from the per-source HF zips into ``frames/``."""
    from remotezip import RemoteZip

    ann = config.paths.data_root / config.data.annotations_subdir / config.data.test_ann
    videos = json.loads(ann.read_text())["videos"]
    frames_root = config.paths.data_root / config.data.frames_subdir
    headers = {"Authorization": f"Bearer {_token()}"}

    by_source: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        by_source[v["source_dataset"]].append(v)

    ok = miss = skip = 0
    for source, vids in sorted(by_source.items()):
        # Which zip members (<seq_name>/<frame>) does this source need, and where do they go?
        wanted: dict[str, Path] = {}
        for v in vids:
            outdir = frames_root / v["video_name"]
            if outdir.exists() and len(list(outdir.glob("*.jpg"))) >= len(v["file_names"]):
                skip += 1
                continue
            for fn in v["file_names"]:  # "<source>__<seq>/<frame>"
                member = f"{v['source_seq']}/{Path(fn).name}"
                wanted[member] = frames_root / fn
        if not wanted:
            continue
        print(f"[{source}] pulling {len(wanted)} frames for {len(vids)} videos ...", flush=True)
        with RemoteZip(_HF_URL.format(split=split, source=source), headers=headers) as zf:
            present = set(zf.namelist())
            for member, dst in wanted.items():
                if member not in present:
                    miss += 1
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as s, open(dst, "wb") as d:
                    d.write(s.read())
                ok += 1
                if ok % 200 == 0:
                    print(f"  extracted {ok} frames ...", flush=True)
    print(f"DONE extracted={ok} skipped_videos={skip} missing={miss}", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val", help="BURST/TAO split the frames come from (val/test)")
    args = ap.parse_args()
    extract(Config.load(args.config), args.split)


if __name__ == "__main__":
    main()
