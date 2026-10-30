#!/usr/bin/env python3
"""Download missing registered ASRank inputs into project-local data/."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.data.downloader import download_required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="YYYYMMDD")
    parser.add_argument("--files-root", type=Path)
    parser.add_argument("--page-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    manifests = download_required(
        args.date,
        files_root=args.files_root,
        page_size=args.page_size,
        workers=args.workers,
        progress=lambda text: print(text, file=sys.stderr, flush=True),
    )
    print(json.dumps({"date": args.date, "downloads": manifests}, indent=2))


if __name__ == "__main__":
    main()
