#!/usr/bin/env python3
"""Split PathRadar datasets using MetaGMPI's existing train/test path files."""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from pathlib import Path

REPO_DIR = Path(os.environ.get("PATHRADAR_RUN_DIR", Path(__file__).resolve().parents[2]))
INTERMEDIATE_DIR = REPO_DIR / "intermediate"


def convert_path(path: str) -> str:
    return path.strip().replace("->", "|")


def load_path_set(directory: Path) -> set[str]:
    paths: set[str] = set()
    csv_files = sorted(directory.glob("*.csv"))
    print(f"[load] {directory} files={len(csv_files)}", flush=True)
    for index, csv_file in enumerate(csv_files, start=1):
        file_paths = 0
        with csv_file.open("r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    paths.add(convert_path(line))
                    file_paths += 1
        if index % 10 == 0 or index == len(csv_files):
            print(
                f"[load] {directory.name} processed_files={index}/{len(csv_files)} "
                f"last_file_paths={file_paths} unique_paths={len(paths)}",
                flush=True,
            )
    print(f"[load] {directory.name} completed unique_paths={len(paths)}", flush=True)
    return paths


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def build_auxiliary_files(base_paths: list[str], output_dir: Path, suffix: str) -> None:
    linkfreq = defaultdict(int)
    linkorder = defaultdict(list)
    prev_frequency = defaultdict(int)
    next_frequency = defaultdict(int)
    spread = defaultdict(set)
    graph = defaultdict(set)

    total = len(base_paths)
    print(f"[aux] building suffix={suffix} paths={total}", flush=True)
    for index, line in enumerate(base_paths, start=1):
        path = line.split("|")[::-1]
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            graph[src].add(dst)
            graph[dst].add(src)
            linkfreq[(src, dst)] += 1
            linkfreq[(dst, src)] += 1
        if len(path) > 1:
            for i in range(len(path)):
                prev_frequency["|".join(path[: i + 1])] += 1
                for j in range(i, len(path)):
                    next_frequency["|".join(path[i : j + 1])] += 1
        for asn in path:
            spread[asn].update(path)
        if index % 500000 == 0 or index == total:
            print(
                f"[aux] suffix={suffix} processed={index}/{total} "
                f"linkfreq={len(linkfreq)} prev={len(prev_frequency)} next={len(next_frequency)}",
                flush=True,
            )

    with (output_dir / f"linkfreq_{suffix}.txt").open("w", encoding="utf-8") as fout:
        for (src, dst), freq in linkfreq.items():
            fout.write(f"{src}|{dst} {freq}\n")

    with (output_dir / f"linkorder_{suffix}.txt").open("w", encoding="utf-8") as fout:
        for node in sorted(graph):
            neighbors = sorted(
                ((neighbor, linkfreq[(node, neighbor)]) for neighbor in graph[node]),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            ordered = [neighbor for neighbor, _ in neighbors]
            linkorder[node] = ordered
            fout.write(f"{node} {'|'.join(ordered)}\n")

    with (output_dir / f"joinfreq_{suffix}.txt").open("w", encoding="utf-8") as fout:
        for key, num in prev_frequency.items():
            fout.write(f"prev {key} {num}\n")
        for key, num in next_frequency.items():
            fout.write(f"next {key} {num}\n")

    if suffix == "data":
        with (output_dir / "core_shell.txt").open("w", encoding="utf-8") as fout:
            for key in sorted(spread):
                fout.write(f"{key} {len(spread[key])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create PathRadar split files aligned with MetaGMPI train/test directories."
    )
    parser.add_argument("--date", required=True, help="Dataset date, e.g. 20251201")
    parser.add_argument(
        "--files-root",
        default="/media/lxl/data/asinfer/files",
        help="Root directory that contains path_by_vantage/<date>/",
    )
    parser.add_argument(
        "--done-rib",
        default=str(INTERMEDIATE_DIR / "done_rib.txt"),
        help="Valley-free PathRadar path file to split.",
    )
    parser.add_argument(
        "--validation-ratio",
        default=0.5,
        type=float,
        help="Ratio of non-test paths assigned to rib_validation.txt.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for train/validation partition.",
    )
    args = parser.parse_args()

    data_root = (
        Path(args.files_root)
        / "data"
        / "aspath"
        / "path_by_vantage"
        / args.date
        / "path_without_prefix"
    )
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"missing train/test directories under {data_root}")

    done_rib_path = Path(args.done_rib)
    if not done_rib_path.is_file():
        raise SystemExit(f"done_rib file not found: {done_rib_path}")

    start = time.time()
    print(f"[start] date={args.date}", flush=True)
    local_train = load_path_set(train_dir)
    local_test = load_path_set(test_dir)
    overlap = local_train & local_test
    if overlap:
        raise SystemExit(f"train/test overlap detected: {len(overlap)} paths")

    random.seed(args.seed)
    rib_data: list[str] = []
    rib_train: list[str] = []
    rib_validation: list[str] = []
    rib_test: list[str] = []

    unmatched = 0
    seen = 0
    print(f"[scan] matching {done_rib_path}", flush=True)
    with done_rib_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            path = line.strip()
            if not path:
                continue
            seen += 1
            if path in local_test:
                rib_test.append(path)
            elif path in local_train:
                rib_data.append(path)
                if random.random() < args.validation_ratio:
                    rib_validation.append(path)
                else:
                    rib_train.append(path)
            else:
                unmatched += 1
            if seen % 1000000 == 0:
                print(
                    f"[scan] seen={seen} rib_data={len(rib_data)} "
                    f"rib_train={len(rib_train)} rib_validation={len(rib_validation)} "
                    f"rib_test={len(rib_test)} unmatched={unmatched}",
                    flush=True,
                )

    print("[write] writing rib files", flush=True)
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    write_lines(INTERMEDIATE_DIR / "rib_data.txt", rib_data)
    write_lines(INTERMEDIATE_DIR / "rib_train.txt", rib_train)
    write_lines(INTERMEDIATE_DIR / "rib_validation.txt", rib_validation)
    write_lines(INTERMEDIATE_DIR / "rib_test.txt", rib_test)
    (INTERMEDIATE_DIR / "rib_origintest.txt").write_text("", encoding="utf-8")

    build_auxiliary_files(rib_data, INTERMEDIATE_DIR, "data")
    build_auxiliary_files(rib_train, INTERMEDIATE_DIR, "train")

    print(f"[done] elapsed_sec={time.time() - start:.1f}", flush=True)
    print(f"done_rib_seen={seen}")
    print(f"rib_data={len(rib_data)}")
    print(f"rib_train={len(rib_train)}")
    print(f"rib_validation={len(rib_validation)}")
    print(f"rib_test={len(rib_test)}")
    print(f"unmatched_done_rib={unmatched}")
    print(f"local_train_paths={len(local_train)}")
    print(f"local_test_paths={len(local_test)}")


if __name__ == "__main__":
    main()
