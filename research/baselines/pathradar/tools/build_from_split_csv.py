#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path


def convert_path(line: str) -> str:
    return line.strip().replace("->", "|")


def load_paths(csv_dir: Path) -> list[str]:
    paths: set[str] = set()
    csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"[load] {csv_dir} csv_files={len(csv_files)}", flush=True)
    for index, csv_file in enumerate(csv_files, start=1):
        with csv_file.open("r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    paths.add(convert_path(line))
        if index % 25 == 0 or index == len(csv_files):
            print(f"[load] processed={index}/{len(csv_files)} unique_paths={len(paths)}", flush=True)
    return sorted(paths)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        description="Build PathRadar training intermediates from an existing train/test csv split."
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-dir", default="")
    parser.add_argument("--asrel-file", default="")
    parser.add_argument("--validation-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    source_dir = run_dir / "source_data" / "path_without_prefix"
    train_dir = Path(args.train_dir) if args.train_dir else source_dir / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"missing train csv dir: {train_dir}")

    intermediate_dir = run_dir / "intermediate"
    raw_data_dir = run_dir / "raw_data"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    asrel = Path(args.asrel_file) if args.asrel_file else raw_data_dir / "asrel.txt"
    if not asrel.is_file():
        raise SystemExit(f"missing asrel file: {asrel}")
    shutil.copy2(asrel, intermediate_dir / "asrel_rib.txt")

    start = time.time()
    random.seed(args.seed)
    train_paths = load_paths(train_dir)
    rib_train: list[str] = []
    rib_validation: list[str] = []
    for path in train_paths:
        if random.random() < args.validation_ratio:
            rib_validation.append(path)
        else:
            rib_train.append(path)

    write_lines(intermediate_dir / "done_rib.txt", train_paths)
    write_lines(intermediate_dir / "rib_data.txt", train_paths)
    write_lines(intermediate_dir / "rib_train.txt", rib_train)
    write_lines(intermediate_dir / "rib_validation.txt", rib_validation)
    write_lines(intermediate_dir / "rib_test.txt", [])
    write_lines(intermediate_dir / "rib_origintest.txt", [])

    build_auxiliary_files(train_paths, intermediate_dir, "data")
    build_auxiliary_files(rib_train, intermediate_dir, "train")

    print(f"[done] date={args.date} elapsed_sec={time.time() - start:.1f}", flush=True)
    print(f"rib_data={len(train_paths)}")
    print(f"rib_train={len(rib_train)}")
    print(f"rib_validation={len(rib_validation)}")


if __name__ == "__main__":
    main()
