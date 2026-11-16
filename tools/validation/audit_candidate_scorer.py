#!/usr/bin/env python3
"""Audit completed candidate-scorer sources without changing frozen experiments."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pathInference.experiment.contracts import require, verify_files
from pathInference.utils import object_hash, read_json, sha256, utc_now, write_json


def audit(run, rescore=False):
    scope = ROOT / "research/candidate_scorer_v1/FULL_FEBRUARY.json"
    cfg = read_json(scope)
    os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
    identity = read_json(run / "IDENTITY.json")
    verify_files(identity["code_sha256"])
    full = read_json(run / "FULL_FEBRUARY_IDENTITY.json")
    require(full["scope_sha256"] == sha256(scope), "Scope changed")
    require(full["original_identity_sha256"] == sha256(run / "IDENTITY.json"), "Original identity changed")
    verify_files({str(ROOT / "research/candidate_scorer_v1/full_february.py"): full["launcher_sha256"],
                  str(ROOT / "research/candidate_scorer_v1/resume.py"): full["publication_repair_sha256"]})
    config_path = ROOT / "research/candidate_scorer_v1/CONFIG.json"
    require(sha256(config_path) == identity["config_sha256"], "Numerical configuration changed")
    baseline = ROOT / read_json(config_path)["baseline"]
    old = {str(row["source_asn"]): row for row in read_json(baseline / "RESULTS.json")["per_source"]}
    require(set(cfg["sources"]) == set(old), "Baseline source scope differs")
    checked = []
    pending = []
    for source in cfg["sources"]:
        directory = run / ("as" + source)
        result_path = directory / "RESULTS.json"
        if not result_path.is_file():
            pending.append(source)
            continue
        for stage in ("prepare", "train", "generate", "predict", "evaluate"):
            task_path = run / "tasks" / (source + "_" + stage + ".json")
            require(task_path.is_file(), "Result lacks stage receipt: " + str(task_path))
            task = read_json(task_path)
            require(task["status"] == "PASS" and task["source"] == source and task["stage"] == stage,
                    "Invalid stage receipt")
            require(task["config_sha256"] == identity["config_sha256"], "Stage configuration differs")
            verify_files(task["outputs_sha256"])
            if task.get("timing_file"):
                timing = read_json(task["timing_file"])
                require(timing["source"] == source and timing["stage"] == stage, "Timing stage differs")
                for key in ("wall_seconds", "cpu_seconds"):
                    require(math.isfinite(timing[key]) and timing[key] >= 0, "Invalid measured time")
                if stage == "train":
                    for key in ("lightgbm_fit_wall_seconds", "train_features_wall_seconds", "validation_features_wall_seconds"):
                        require(0 <= timing[key] <= timing["wall_seconds"], "Training timing is incomplete")
        reuse = read_json(directory / "REUSE.json")
        verify_files(reuse["inputs_sha256"])
        split = read_json(Path(reuse["split"]) / "split_manifest.json")
        require(set(split["train_pairs"]).isdisjoint(split["validation_pairs"]), "Training/validation overlap")
        prepared = read_json(reuse["prepared"])
        require(prepared["admitted_library_path_set_sha256"][source] == reuse["admitted_library_path_set_sha256"],
                "Admitted library differs")
        model = read_json(directory / "MODEL_FROZEN.json")
        prediction = read_json(directory / "PREDICTIONS_FROZEN.json")
        require(model["feature_schema_sha256"] == object_hash(model["feature_schema"]), "Feature schema differs")
        require(model["pair_disjoint"] and not model["final_labels_read"] and not prediction["final_labels_read"],
                "Label boundary declaration differs")
        require(model["parameters"]["num_threads"] == cfg["threads"] == 2, "CPU thread setting differs")
        require(model["model_sha256"] == prediction["model_sha256"], "Prediction uses another model")
        verify_files({model["model"]: model["model_sha256"], prediction["predictions"]: prediction["predictions_sha256"]})
        marker = read_json(run / "T-PG-B/pg_inputs" / ("as" + source) / "control/pathgen_completed" / ("test_inputs_" + source + ".json"))
        require(marker["sha256"] == prediction["candidate_sha256"], "Prediction uses another candidate set")
        dispatch = marker["label_blind_compilation"]
        verify_files({marker["output"]: marker["sha256"], dispatch["dispatch_status_file"]: dispatch["dispatch_status_sha256"]})
        with Path(prediction["predictions"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        pairs = [row["pair"] for row in rows]
        inventory = read_json(prepared["fixed_test_pairs_inventory"][source]["path"])
        require(len(pairs) == len(set(pairs)) and set(pairs) == set(inventory["pairs"]), "Prediction endpoints differ")
        result = read_json(result_path)
        previous = old[source]
        require(result["query_count"] == result["prediction_rows"] == prediction["queries"] == previous["query_count"] == len(pairs),
                "Evaluation denominator differs")
        require(result["model_sha256"] == model["model_sha256"] and result["prediction_sha256"] == prediction["predictions_sha256"],
                "Evaluation identity differs")
        require(result["fixed_identity"] == previous["fixed_identity"], "Evaluation reference identity differs")
        require(result["baseline_gmpi_jaccard"] == previous["jaccard_accuracy"] and result["baseline_gmpi_exact"] == previous["ordered_exact_accuracy"],
                "Baseline metrics differ")
        require(abs(result["jaccard_change"] - (result["jaccard_accuracy"] - previous["jaccard_accuracy"])) < 1e-12,
                "Jaccard difference inconsistent")
        require(result["candidate_diagnostic_mismatches"] == 0 and result["diagnostic_comparison_passed"],
                "Candidate diagnostic comparison failed")
        require(result["missing_predictions"] == result["invalid_nonempty_predictions"] == 0,
                "Missing or invalid prediction rows")
        require(result["ordered_exact_accuracy"] <= result["candidate_exact_available"] + 1e-12
                and result["jaccard_accuracy"] <= result["oracle_jaccard"] + 1e-12, "Score exceeds candidate oracle")
        if rescore:
            from pathInference.downstream.evaluation import score_source
            independent = score_source(previous["reference_file"], prediction["predictions"], source, previous["fixed_identity"])
            for key in ("query_count", "ordered_exact_matches", "total_jaccard_reward", "jaccard_accuracy", "ordered_exact_accuracy"):
                require(abs(independent[key] - result[key]) < 1e-10, "Independent rescore differs: " + key)
        checked.append(dict(source=source, status="PASS", result_sha256=sha256(result_path), query_count=len(pairs),
                            jaccard=result["jaccard_accuracy"], gmpi_jaccard=previous["jaccard_accuracy"],
                            candidate_diagnostic_mismatches=0, independent_rescore=rescore))
    aggregate = run / "RESULTS.json"
    if aggregate.is_file():
        rows = read_json(aggregate)
        require(len(rows) == len(checked) and {str(row["source"]) for row in rows} == {row["source"] for row in checked},
                "Aggregate result scope differs")
        for row in rows:
            require(row == read_json(run / ("as" + str(row["source"])) / "RESULTS.json"),
                    "Aggregate metrics differ from source result")
    return dict(status="PASS" if not pending else "INCOMPLETE", checked_utc=utc_now(), completed_sources=checked,
                pending_sources=pending, historical_candidate_identity_limitation="Only counts, exact availability and oracle can be compared; historical full candidates were cleaned")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/20260201/candidate_scorer_v1")
    parser.add_argument("--rescore", action="store_true", help="Read final labels only for already frozen, completed sources")
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.rescore)
    write_json(args.run_dir / "COMPLETION_AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
