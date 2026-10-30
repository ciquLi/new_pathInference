"""Score frozen AS-pair predictions with a fixed independent denominator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from pathveil.utils import file_record, project_root, utc_now, write_json

FROZEN_TRUTH_IDENTITIES = {
    "20261015": {
        "pairs": 1890,
        "unique_paths": 2198,
        "sha256": "40db9e7f1cfc5228858724701e15e6995cd53d423f9bf776dfee14f69215b254",
        "pair_inventory_sha256": "025f94a957d04aa98e9ceae7d94b9f62ac357c15f78f7486260620d4939c76d0",
    }
}


def asn(value):
    text = str(value).strip()
    if not text.isascii() or not text.isdecimal() or not 0 < int(text) < 2**32:
        raise ValueError("Invalid ASN: " + repr(value))
    return str(int(text))


def read_truth(path):
    pairs, rows = defaultdict(set), 0
    with Path(path).open(newline="") as handle:
        for number, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            if len(row) != 1:
                raise ValueError("Expected one complete AS path at row " + str(number))
            nodes = tuple(asn(node) for node in row[0].split("->"))
            # A same-AS source/destination query legitimately has one ASN.
            if not nodes or len(nodes) != len(set(nodes)):
                raise ValueError("Reference path is empty or contains a loop")
            pairs[(nodes[0], nodes[-1])].add(nodes)
            rows += 1
    return dict(pairs), rows


def read_predictions(path):
    pairs, statuses = {}, Counter()
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair = (asn(row["source_asn"]), asn(row["destination_asn"]))
            if pair in pairs:
                raise ValueError("Duplicate prediction AS pair at row " + str(number))
            malformed = False
            try:
                if not isinstance(row.get("predicted_path"), list):
                    raise ValueError("Predicted path must be a list")
                nodes = tuple(asn(node) for node in row["predicted_path"])
            except (TypeError, ValueError):
                nodes, malformed = (), True
            valid = bool(nodes) and (nodes[0], nodes[-1]) == pair
            pairs[pair] = {
                "path": nodes,
                "endpoint_valid": valid,
                "malformed": malformed,
                "status": row.get("status", ""),
            }
            statuses[row.get("status", "")] += 1
    return pairs, {
        "rows": len(pairs),
        "endpoint_valid_pairs": sum(row["endpoint_valid"] for row in pairs.values()),
        "status_counts": dict(statuses),
    }


def pair_inventory_sha256(truth):
    pairs = sorted(truth, key=lambda pair: tuple(map(int, pair)))
    text = "\n".join("->".join(pair) for pair in pairs) + "\n"
    return hashlib.sha256(text.encode()).hexdigest()


def set_jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 0.0


def score_predictions(truth_path, prediction_path, output_dir, date="20261015", identity=None):
    identity = identity or FROZEN_TRUTH_IDENTITIES.get(str(date))
    if not identity:
        raise ValueError("A frozen independent truth identity is required for this date")
    provenance = {"truth": file_record(truth_path), "predictions": file_record(prediction_path)}
    truth, raw_rows = read_truth(truth_path)
    unique_paths = sum(map(len, truth.values()))
    if (
        len(truth) != identity["pairs"]
        or unique_paths != identity["unique_paths"]
        or provenance["truth"]["sha256"] != identity["sha256"]
        or pair_inventory_sha256(truth) != identity["pair_inventory_sha256"]
    ):
        raise ValueError("Fixed independent traceroute query/path inventory changed")
    predictions, input_stats = read_predictions(prediction_path)
    pair_rows, path_rows = [], []
    covered_pairs = covered_paths = matched_pairs = missing = empty = malformed = wrong = 0
    total_score = rounded_total = path_total = 0.0
    statuses = Counter()
    for pair in sorted(truth, key=lambda item: tuple(map(int, item))):
        observed, record = truth[pair], predictions.get(pair)
        prediction = record["path"] if record else ()
        valid = bool(record and record["endpoint_valid"])
        status = record["status"] if record else "missing_prediction_row"
        statuses[status] += 1
        missing += record is None
        malformed += bool(record and record["malformed"])
        empty += bool(record and not record["malformed"] and not prediction)
        wrong += bool(record and prediction and not valid)
        exact = int(valid and prediction in observed)
        score = max(set_jaccard(prediction, path) for path in observed) if valid else 0.0
        total_score += score
        rounded_total += round(score, 3)
        covered_pairs += valid
        covered_paths += len(observed) if valid else 0
        matched_pairs += exact
        pair_rows.append(
            dict(
                source_asn=pair[0],
                destination_asn=pair[1],
                observed_unique_paths=len(observed),
                prediction_status=status,
                endpoint_valid=int(valid),
                predicted_path="->".join(prediction),
                any_truth_exact=exact,
                jaccard_best_observed=score,
                jaccard_best_observed_rounded3=round(score, 3),
            )
        )
        for path in sorted(observed, key=lambda item: tuple(map(int, item))):
            value = set_jaccard(prediction, path) if valid else 0.0
            path_total += value
            path_rows.append(
                dict(
                    source_asn=pair[0],
                    destination_asn=pair[1],
                    observed_path="->".join(path),
                    predicted_path="->".join(prediction),
                    endpoint_valid=int(valid),
                    ordered_exact=int(valid and path == prediction),
                    set_jaccard=value,
                )
            )
    count = len(truth)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    metrics = dict(
        all_truth_pairs=count,
        all_unique_truth_paths=unique_paths,
        endpoint_valid_truth_pairs=covered_pairs,
        unique_truth_paths_in_endpoint_valid_pairs=covered_paths,
        missing_prediction_rows=missing,
        empty_prediction_pairs=empty,
        malformed_prediction_pairs=malformed,
        wrong_endpoint_prediction_pairs=wrong,
        pairs_matching_any_truth=matched_pairs,
        exactly_matched_unique_truth_paths=matched_pairs,
        endpoint_valid_pair_coverage=ratio(covered_pairs, count),
        unique_path_exact_recall=ratio(matched_pairs, unique_paths),
        pair_any_truth_exact_rate=ratio(matched_pairs, count),
        covered_pair_any_truth_exact_rate=ratio(matched_pairs, covered_pairs),
        covered_unique_path_exact_recall=ratio(matched_pairs, covered_paths),
        single_prediction_path_recall_ceiling=ratio(count, unique_paths),
        pair_jaccard_accuracy=ratio(total_score, count),
        pair_jaccard_accuracy_rounded3=ratio(rounded_total, count),
        covered_pair_jaccard_accuracy=ratio(total_score, covered_pairs),
        unique_path_jaccard_mean=ratio(path_total, unique_paths),
        truth_pair_status_counts=dict(statuses),
        prediction_file=input_stats,
    )
    summary = dict(
        date=str(date),
        created_at_utc=utc_now(),
        evaluation="frozen_single_prediction_same_as_gap_jaccard_and_ordered_exact_v2",
        input_provenance=provenance,
        truth=dict(
            input_rows=raw_rows,
            duplicate_rows_removed=raw_rows - unique_paths,
            unique_paths=unique_paths,
            unique_pairs=count,
        ),
        fixed_truth_identity=identity,
        metrics=metrics,
        matching_policy={
            "path_transformations": "none; no reversal, truncation or wildcard matching",
            "truth_used_to_change_predictions": False,
            "missing_prediction_score": 0,
        },
        measurement_assumptions=[
            "Same-AS bounded unknown runs were removed during reference construction.",
            "This assumption does not exclude an unobserved detour or prove router ownership.",
            "The strict no-unknown-hop subset is empty for the 20261015 campaign.",
            "Counts weight distinct AS pairs/paths, not trace records, IPs, domains or traffic.",
        ],
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("per_pair.csv", pair_rows), ("per_path.csv", path_rows)]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "summary.json", summary)
    accuracy = metrics["pair_jaccard_accuracy_rounded3"]
    (output / "gmpi_result.txt").write_text(
        "metric:fixed-pair max-truth AS-set Jaccard; round3 per pair; missing=0\n"
        + "testDataNum:%d,total_rwd:%.3f,accuracy:%.3f,N=1\n" % (count, rounded_total, accuracy)
        + "accuracy_full_precision:%.15f\n" % accuracy
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261015")
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    truth = (
        args.truth
        or project_root()
        / "data/traceroute"
        / args.date
        / "measurement/as_paths_path_inference.csv"
    )
    print(json.dumps(score_predictions(truth, args.predictions, args.outdir, args.date), indent=2))


if __name__ == "__main__":
    main()
