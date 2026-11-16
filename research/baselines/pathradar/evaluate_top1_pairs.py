#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pathradar_prediction import REPO_DIR, TRAINED_DIR, Routingtree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PathRadar as a top-1 single-path predictor on AS pairs."
    )
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--date", default="20251201")
    parser.add_argument(
        "--files-root",
        default="/media/lxl/data/asinfer/files",
        help="Root that contains data/aspath/path_by_vantage/<date>/path_without_prefix/test",
    )
    parser.add_argument(
        "--test-dir",
        default="",
        help="Optional direct path to the test csv directory. Overrides --files-root.",
    )
    parser.add_argument(
        "--pairs-file",
        default="",
        help="Optional file of AS pairs. Each line may be 'src dst', 'src,dst', or 'src|dst'.",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--per-source-sample-size",
        type=int,
        default=0,
        help="If > 0, sample up to this many AS pairs for each source AS separately.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=0,
        help="Process ground-truth AS pairs in batches of this size within each source evaluation.",
    )
    parser.add_argument(
        "--include-pair-details",
        type=int,
        default=1,
        help="Whether to include per-pair prediction details in the output JSON (1=yes, 0=no).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. Defaults to repo/testing/top1_eval_round<r>_sample<n>.json",
    )
    return parser.parse_args()


def load_ground_truth(date: str, files_root: str, test_dir_arg: str = "") -> dict[tuple[str, str], set[tuple[str, ...]]]:
    if test_dir_arg:
        test_dir = Path(test_dir_arg)
    else:
        test_dir = (
            Path(files_root)
            / "data"
            / "aspath"
            / "path_by_vantage"
            / date
            / "path_without_prefix"
            / "test"
        )
    if not test_dir.is_dir():
        raise SystemExit(f"missing test directory: {test_dir}")

    ground_truth: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    csv_files = sorted(
        p
        for p in test_dir.glob("*.csv")
        if "_country" not in p.name and "_continent" not in p.name
    )
    if not csv_files:
        raise SystemExit(f"no test csv files found under {test_dir}")

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", errors="ignore") as fin:
            for raw_line in fin:
                line = raw_line.strip()
                if not line:
                    continue
                path = tuple(line.split("->"))
                if len(path) < 2:
                    continue
                ground_truth[(path[0], path[-1])].add(path)
    return ground_truth


def parse_pairs_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [token for token in re.split(r"[\s,|]+", line) if token]
        if len(parts) < 2:
            continue
        pairs.append((parts[0], parts[1]))
    return pairs


def load_models(rounds: int) -> dict[str, object]:
    model = {}
    for cate in ("core", "shell"):
        for num in ("1", "2", "3"):
            model_name = cate + num
            with open(TRAINED_DIR / "models" / f"xgboost_{rounds}_{model_name}.model", "rb") as f:
                model[model_name] = pickle.load(f)
    return model


def jaccard_similarity(path1: tuple[str, ...], path2: tuple[str, ...]) -> float:
    set1 = set(path1)
    set2 = set(path2)
    union = set1 | set2
    if not union:
        return 1.0
    return len(set1 & set2) / len(union)


def top1_prediction(rt: Routingtree, src: str) -> dict[str, tuple[str, ...]]:
    rt.simulate(src)
    top1: dict[str, tuple[str, ...]] = {}
    for dst, paths in rt.AS_path.items():
        if dst == src or not paths:
            continue
        best = max(paths, key=lambda path: rt.spread_score.get(path, float("-inf")))
        top1[dst] = best
    return top1


def evaluate_source_task(payload: tuple[int, float, str, dict[str, list[list[str]]], int, int]):
    rounds, threshold, src, gt_for_src, include_pair_details, pair_batch_size = payload
    rt = Routingtree(rounds, threshold, "test", 0, 1)
    rt.model = load_models(rounds)
    pred_by_dst = top1_prediction(rt, src)

    source_exact_matches = 0
    source_covered_pairs = 0
    source_jaccard_total = 0.0
    source_jaccard_total_covered = 0.0
    pair_details = [] if include_pair_details else None

    dest_items = sorted(gt_for_src.items())
    step = pair_batch_size if pair_batch_size and pair_batch_size > 0 else len(dest_items)
    for start in range(0, len(dest_items), step):
        batch = dest_items[start : start + step]
        for dst, gt_paths_raw in batch:
            gt_paths = [tuple(path) for path in gt_paths_raw]
            pred_path = pred_by_dst.get(dst)
            covered = pred_path is not None
            exact = False
            best_jaccard = 0.0
            matched_gt: tuple[str, ...] | None = None
            if pred_path is not None:
                source_covered_pairs += 1
                for gt_path in gt_paths:
                    score = jaccard_similarity(pred_path, gt_path)
                    if score > best_jaccard:
                        best_jaccard = score
                        matched_gt = gt_path
                    if pred_path == gt_path:
                        exact = True
                if exact:
                    source_exact_matches += 1
                source_jaccard_total_covered += best_jaccard
            source_jaccard_total += best_jaccard
            if include_pair_details:
                pair_details.append(
                    {
                        "src": src,
                        "dst": dst,
                        "covered": covered,
                        "exact_match": exact,
                        "jaccard": best_jaccard,
                        "predicted_path": list(pred_path) if pred_path is not None else None,
                        "matched_ground_truth": list(matched_gt) if matched_gt is not None else None,
                        "ground_truth_paths": [list(path) for path in gt_paths],
                    }
                )

    return {
        "src": src,
        "requested_pairs": len(gt_for_src),
        "coverage_count": source_covered_pairs,
        "exact_match_count": source_exact_matches,
        "jaccard_total": source_jaccard_total,
        "jaccard_total_covered": source_jaccard_total_covered,
        "pair_details": pair_details if include_pair_details else [],
    }


def main() -> None:
    args = parse_args()
    gt = load_ground_truth(args.date, args.files_root, args.test_dir)

    if args.pairs_file:
        requested_pairs = parse_pairs_file(Path(args.pairs_file))
    else:
        if args.per_source_sample_size > 0:
            pairs_by_src_all: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for pair in sorted(gt):
                pairs_by_src_all[pair[0]].append(pair)
            requested_pairs = []
            random.seed(args.seed)
            for src in sorted(pairs_by_src_all):
                source_pairs = pairs_by_src_all[src]
                if len(source_pairs) > args.per_source_sample_size:
                    requested_pairs.extend(random.sample(source_pairs, args.per_source_sample_size))
                else:
                    requested_pairs.extend(source_pairs)
        else:
            requested_pairs = sorted(gt)
            if args.sample_size > 0 and len(requested_pairs) > args.sample_size:
                random.seed(args.seed)
                requested_pairs = random.sample(requested_pairs, args.sample_size)

    if not requested_pairs:
        raise SystemExit("no AS pairs selected for evaluation")

    pairs_with_gt = [pair for pair in requested_pairs if pair in gt]
    missing_gt = [pair for pair in requested_pairs if pair not in gt]

    pairs_by_src: dict[str, dict[str, list[list[str]]]] = defaultdict(dict)
    for src, dst in pairs_with_gt:
        pairs_by_src[src][dst] = [list(path) for path in sorted(gt[(src, dst)])]

    exact_matches = 0
    covered_pairs = 0
    jaccard_total = 0.0
    jaccard_total_covered = 0.0
    include_pair_details = 1 if args.include_pair_details else 0
    pair_details = [] if include_pair_details else None
    source_breakdown = []

    tasks = [
        (
            args.rounds,
            args.threshold,
            src,
            pairs_by_src[src],
            include_pair_details,
            args.pair_batch_size,
        )
        for src in sorted(pairs_by_src)
    ]
    max_workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for source_result in executor.map(evaluate_source_task, tasks):
            exact_matches += source_result["exact_match_count"]
            covered_pairs += source_result["coverage_count"]
            jaccard_total += source_result["jaccard_total"]
            jaccard_total_covered += source_result["jaccard_total_covered"]
            if include_pair_details:
                pair_details.extend(source_result["pair_details"])
            requested_for_src = source_result["requested_pairs"]
            source_breakdown.append(
                {
                    "src": source_result["src"],
                    "requested_pairs": requested_for_src,
                    "coverage_count": source_result["coverage_count"],
                    "coverage_rate": source_result["coverage_count"] / requested_for_src if requested_for_src else 0.0,
                    "exact_match_count": source_result["exact_match_count"],
                    "exact_match_rate": source_result["exact_match_count"] / requested_for_src if requested_for_src else 0.0,
                    "mean_jaccard": source_result["jaccard_total"] / requested_for_src if requested_for_src else 0.0,
                }
            )
            print(
                "source="
                + source_result["src"]
                + f" requested={requested_for_src}"
                + f" coverage={source_result['coverage_count'] / requested_for_src if requested_for_src else 0.0:.6f}"
                + f" exact={source_result['exact_match_count'] / requested_for_src if requested_for_src else 0.0:.6f}"
                + f" jaccard={source_result['jaccard_total'] / requested_for_src if requested_for_src else 0.0:.6f}",
                flush=True,
            )

    total_pairs = len(requested_pairs)
    total_pairs_with_gt = len(pairs_with_gt)
    results = {
        "rounds": args.rounds,
        "threshold": args.threshold,
        "date": args.date,
        "seed": args.seed,
        "workers": max(1, min(args.workers, len(tasks))) if pairs_by_src else 0,
        "per_source_sample_size": args.per_source_sample_size,
        "pair_batch_size": args.pair_batch_size,
        "include_pair_details": bool(include_pair_details),
        "requested_pairs": total_pairs,
        "pairs_with_ground_truth": total_pairs_with_gt,
        "pairs_missing_ground_truth": len(missing_gt),
        "coverage_count": covered_pairs,
        "coverage_rate_over_requested": covered_pairs / total_pairs if total_pairs else 0.0,
        "coverage_rate_over_gt_pairs": covered_pairs / total_pairs_with_gt if total_pairs_with_gt else 0.0,
        "exact_match_count": exact_matches,
        "exact_match_rate_over_requested": exact_matches / total_pairs if total_pairs else 0.0,
        "exact_match_rate_over_gt_pairs": exact_matches / total_pairs_with_gt if total_pairs_with_gt else 0.0,
        "exact_match_rate_over_covered": exact_matches / covered_pairs if covered_pairs else 0.0,
        "mean_jaccard_over_requested": jaccard_total / total_pairs if total_pairs else 0.0,
        "mean_jaccard_over_gt_pairs": jaccard_total / total_pairs_with_gt if total_pairs_with_gt else 0.0,
        "mean_jaccard_over_covered": jaccard_total_covered / covered_pairs if covered_pairs else 0.0,
        "notes": [
            "Ground truth is read from the original 10 MetaGMPI test-vantage path files.",
            "Top-1 prediction keeps only the highest spread_score path for each AS pair.",
            "Jaccard is computed on the set of AS nodes in the path.",
            "If multiple ground-truth paths exist for one AS pair, exact match succeeds if any path matches and Jaccard uses the best-matching ground-truth path.",
        ],
        "source_breakdown": sorted(source_breakdown, key=lambda item: item["src"]),
        "missing_ground_truth_pairs": [{"src": src, "dst": dst} for src, dst in missing_gt],
        "pair_details": pair_details if include_pair_details else [],
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            REPO_DIR
            / "testing"
            / (
                f"top1_eval_round{args.rounds}_persrc{args.per_source_sample_size}_"
                f"sample{total_pairs}.json"
                if args.per_source_sample_size > 0
                else f"top1_eval_round{args.rounds}_sample{total_pairs}.json"
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"output={output_path}")
    print(f"requested_pairs={total_pairs}")
    print(f"pairs_with_ground_truth={total_pairs_with_gt}")
    print(f"coverage_count={covered_pairs}")
    print(f"coverage_rate_over_requested={results['coverage_rate_over_requested']}")
    print(f"exact_match_rate_over_requested={results['exact_match_rate_over_requested']}")
    print(f"mean_jaccard_over_requested={results['mean_jaccard_over_requested']}")
    if covered_pairs:
        print(f"exact_match_rate_over_covered={results['exact_match_rate_over_covered']}")
        print(f"mean_jaccard_over_covered={results['mean_jaccard_over_covered']}")


if __name__ == "__main__":
    main()
