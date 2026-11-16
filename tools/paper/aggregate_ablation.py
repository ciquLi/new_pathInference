"""Aggregate dated ablations on their fixed query universes."""

import csv
import json
from collections import defaultdict
from pathlib import Path


RUNS = Path(__file__).resolve().parents[2] / "runs"
DATES = ("20260201", "20260301", "20260401", "20260501", "20260601")
VARIANTS = ("current", "legacy_generator", "no_recursion", "all_recursion")


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def candidate_scores(path):
    scores = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            pair = row["pair"]
            assert pair not in scores, (path, pair)
            scores[pair] = float(row["oracle_jaccard"])
    return scores


def prediction_scores(path):
    scores = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            pair = row["pair"]
            assert pair not in scores, (path, pair)
            scores[pair] = float(row["jaccard"])
    return scores


def aggregate():
    for date in DATES:
        query_sets = {}
        counts = set()
        for variant in VARIANTS:
            run = RUNS / date / "generator_recursion_v1" / variant
            metrics_files = sorted((run / "diagnostics").glob("as*/METRICS.json"))
            assert len(metrics_files) == 10, (date, variant, len(metrics_files))
            totals = defaultdict(float)
            for metrics_file in metrics_files:
                source = metrics_file.parent.name[2:]
                metrics = read_json(metrics_file)
                n = metrics["queries"]
                if variant == "current":
                    summary = read_json(run / "T-PG-B/independent_evaluation" / ("source_" + source + ".json"))
                    prediction_file = Path(summary["prediction_file"])
                else:
                    evaluation = run / "T-PG-B/evaluations" / ("as" + source)
                    summary = read_json(evaluation / "evaluation_summary.json")
                    prediction_file = evaluation / "pair_predictions.csv"
                assert n == summary["query_count"], (date, variant, source)
                candidates = candidate_scores(metrics_file.parent / "candidate_oracle.csv")
                predictions = prediction_scores(prediction_file)
                pairs = set(candidates)
                assert len(pairs) == n and set(predictions) == pairs, (date, variant, source)
                assert query_sets.setdefault(source, pairs) == pairs, (date, variant, source)
                oracle_sum = sum(candidates.values())
                final_sum = sum(predictions.values())
                assert abs(oracle_sum / n - metrics["oracle_jaccard"]) < 1e-9
                assert abs(final_sum - summary["total_jaccard_reward"]) < 1e-6
                totals["queries"] += n
                totals["candidate_esr"] += sum(score == 1.0 for score in candidates.values())
                totals["oracle_accuracy"] += oracle_sum
                totals["final_accuracy"] += final_sum
                totals["final_esr"] += sum(score == 1.0 for score in predictions.values())
            n = int(totals["queries"])
            counts.add(n)
            values = [100 * totals[name] / n for name in (
                "candidate_esr", "oracle_accuracy", "final_accuracy", "final_esr"
            )]
            print(date, variant, n, *("%.2f" % value for value in values), flush=True)
        assert len(counts) == 1, (date, counts)


if __name__ == "__main__":
    aggregate()
