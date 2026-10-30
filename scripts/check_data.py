#!/usr/bin/env python3
"""Inspect a registered experiment's exact input dates and readiness."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.data import check_date


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="YYYYMMDD")
    parser.add_argument("--files-root", type=Path)
    parser.add_argument(
        "--deep", action="store_true", help="Verify all ASRank rows, dates, identities and hashes"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_date(args.date, files_root=args.files_root, deep=args.deep)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["raw_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
