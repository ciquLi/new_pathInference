#!/usr/bin/env python3
"""Build PathRadar-compatible raw AS-path input from local MetaGMPI path datasets."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path


def is_ipv4_prefix(value: str) -> bool:
    try:
        network = ipaddress.ip_network(value, strict=False)
    except ValueError:
        return False
    return network.version == 4


def normalize_path(line: str) -> str | None:
    parts = [part.strip() for part in line.strip().split("->") if part.strip()]
    if len(parts) < 3:
        return None
    prefix = parts[-1]
    ases = parts[:-1]
    if not is_ipv4_prefix(prefix):
        return None
    if any(not asn.isdigit() for asn in ases):
        return None

    deduped: list[str] = []
    for asn in ases:
        if not deduped or deduped[-1] != asn:
            deduped.append(asn)
    if len(deduped) < 2:
        return None
    return "|".join(deduped)


def build_raw_aspaths(input_dir: Path, output_file: Path) -> dict[str, int]:
    files = sorted(input_dir.glob("*.csv"))
    stats = {
        "files": len(files),
        "lines_seen": 0,
        "lines_written": 0,
        "invalid_lines": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fout:
        for csv_file in files:
            with csv_file.open("r", encoding="utf-8", errors="ignore") as fin:
                for raw_line in fin:
                    stats["lines_seen"] += 1
                    normalized = normalize_path(raw_line)
                    if normalized is None:
                        stats["invalid_lines"] += 1
                        continue
                    fout.write(normalized + "\n")
                    stats["lines_written"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare raw PathRadar AS-path input from local MetaGMPI data."
    )
    parser.add_argument("--date", required=True, help="Dataset date, e.g. 20251201")
    parser.add_argument(
        "--files-root",
        default="/media/lxl/data/asinfer/files",
        help="Root directory that contains local MetaGMPI data.",
    )
    parser.add_argument(
        "--source",
        choices=["bgp", "path", "path_without_prefix"],
        default="path_without_prefix",
        help=(
            "Input source. "
            "'path' reads path_by_vantage/<date>/path/*.csv and is usually much smaller; "
            "'path_without_prefix' reads path_by_vantage/<date>/path_without_prefix/*.csv "
            "and is the fastest option for fair-comparison experiments; "
            "'bgp' reads data/aspath/bgp/<date>/*.csv."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output file path. Defaults to ./aspaths_<date>_raw.txt",
    )
    args = parser.parse_args()

    root = Path(args.files_root)
    if args.source == "bgp":
        input_dir = root / "data" / "aspath" / "bgp" / args.date
    elif args.source == "path":
        input_dir = root / "data" / "aspath" / "path_by_vantage" / args.date / "path"
    else:
        input_dir = root / "data" / "aspath" / "path_by_vantage" / args.date / "path_without_prefix"
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")

    output_file = Path(args.output) if args.output else Path(f"aspaths_{args.date}_raw.txt")
    stats = build_raw_aspaths(input_dir, output_file)

    print(f"source={args.source}")
    print(f"input_dir={input_dir}")
    print(f"output_file={output_file}")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
