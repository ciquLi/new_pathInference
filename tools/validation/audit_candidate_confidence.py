#!/usr/bin/env python3
"""Independent checks for the controlled frozen-candidate confidence experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from pathInference.experiment.contracts import require, signature, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json


def top1(labels, predictions, groups):
    offset, reward = 0, 0.0
    for size in groups:
        end = offset + int(size)
        reward += float(labels[offset + int(np.argmax(predictions[offset:end]))])
        offset = end
    require(offset == len(labels) == len(predictions), "Validation group offsets differ")
    return reward / len(groups)


def check_barrier(run, name, sources):
    barrier = read_json(run / name)
    require(barrier["sources"] == sources and barrier["snapshot_sha256"] == sha256(run / "SNAPSHOT.json"),
            "Barrier scope/snapshot differs")
    verify_files(barrier["receipts_sha256"])


def audit(run, rescore=False, validate_models=False, check_candidates=False):
    cfg_file = ROOT / "research/candidate_confidence_v2/CONFIG.json"
    cfg, snap = read_json(cfg_file), read_json(run / "SNAPSHOT.json")
    os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
    verify_files(snap["files_sha256"])
    protocol = read_json(run / "PROTOCOL.json")
    require(protocol["configuration"] == cfg and protocol["boundary"] == "after candidate generation",
            "Protocol boundary differs")
    require(not snap["final_route_labels_read"] and not snap["candidates_regenerated"]
            and snap["training_feature_matrices_reused"], "Upstream freeze declaration differs")
    reference_run = ROOT / cfg["reference_run"]
    sources = cfg["sources"]
    require(set(sources) == set(snap["sources"]) and len(sources) == len(set(sources)) == 10,
            "Ten-source inventory differs")
    if (run / "ALL_MODELS_FROZEN.json").exists():
        check_barrier(run, "ALL_MODELS_FROZEN.json", sources)
    if (run / "ALL_PREDICTIONS_FROZEN.json").exists():
        check_barrier(run, "ALL_PREDICTIONS_FROZEN.json", sources)
    pending, checked, validation, membership = [], [], [], []
    for source in sources:
        directory = run / ("as" + source)
        original = reference_run / ("as" + source)
        binding = snap["sources"][source]
        reuse = binding["binding"]
        split = read_json(Path(reuse["split"]) / "split_manifest.json")
        require(set(split["train_pairs"]).isdisjoint(split["validation_pairs"]), "Pair overlap")
        prepared = read_json(reuse["prepared"])
        admission = read_json(reference_run / "augmentation" / ("as" + source) / "ADMISSION.json")
        with Path(admission["path"]).open(newline="") as stream:
            library = [tuple(row[0].split("->")) for row in csv.reader(stream)]
        require(signature(library) == admission["path_set_sha256"] == reuse["admitted_library_path_set_sha256"]
                == prepared["admitted_library_path_set_sha256"][source], "Admitted library differs")
        if not (directory / "MODEL_FROZEN.json").exists():
            pending.append(dict(source=source, stage="train")); continue
        model = read_json(directory / "MODEL_FROZEN.json")
        ref = read_json(original / "MODEL_FROZEN.json")
        verify_files({model["model"]: model["model_sha256"], ref["model"]: ref["model_sha256"]})
        expected = dict(ref["parameters"], objective="query_soft_label_cross_entropy")
        require(model["parameters"] == expected and model["feature_schema"] == ref["feature_schema"],
                "More than the objective changed")
        require(model["snapshot_sha256"] == sha256(run / "SNAPSHOT.json")
                and model["reference_model_sha256"] == ref["model_sha256"]
                and model["label_temperature"] == cfg["label_temperature"]
                and model["final_labels_read"] is False, "Model identity/label boundary differs")
        row = dict(source=source, reference_validation_jaccard=model["diagnostics"]["reference_validation_jaccard"],
                   ranking_validation_jaccard=model["validation_jaccard"], best_iteration=model["best_iteration"],
                   validation_oracle=model["diagnostics"]["validation_oracle"], fit_wall_seconds=model["fit_wall_seconds"])
        if validate_models:
            import lightgbm as lgb
            with np.load(original / "validation.npz", allow_pickle=False) as cache:
                X, labels, groups = cache["X"], cache["y"], cache["groups"]
            for record, expected_score in ((model, model["validation_jaccard"]),
                                           (ref, row["reference_validation_jaccard"])):
                booster = lgb.Booster(model_file=record["model"])
                require(booster.feature_name() == record["feature_schema"], "Actual model schema differs")
                score = top1(labels, booster.predict(X, num_threads=2), groups)
                require(abs(score - expected_score) < 1e-7, "Independent validation reward differs")
            row["independent_validation"] = "PASS"
            # Diagnose learning against the admitted weak labels without opening test routes.
            statistics = {}
            for split_name in ("train", "validation"):
                with np.load(original / (split_name + ".npz"), allow_pickle=False) as cache:
                    matrix, targets, sizes = cache["X"], cache["y"], cache["groups"]
                starts = np.r_[0, np.cumsum(sizes)[:-1]]
                best = np.maximum.reduceat(targets, starts)
                worst = np.minimum.reduceat(targets, starts)
                statistics[split_name] = dict(queries=len(sizes), candidates=len(targets),
                    mean_candidates=float(np.mean(sizes)), median_candidates=float(np.median(sizes)),
                    fraction_candidates_label_one=float(np.mean(targets == 1.)),
                    fraction_queries_have_label_one=float(np.mean(best == 1.)),
                    fraction_queries_all_labels_equal=float(np.mean(best == worst)),
                    mean_query_label_range=float(np.mean(best-worst)), oracle=float(np.mean(best)))
                for label, record in (("regression", ref), ("ranking", model)):
                    booster = lgb.Booster(model_file=record["model"])
                    predictions = booster.predict(matrix, num_threads=2)
                    statistics[split_name][label + "_top1_jaccard"] = top1(targets, predictions, sizes)
                del matrix, targets, sizes
            row["weak_label_diagnostics"] = statistics
        validation.append(row)
        for stage in ("train", "predict", "evaluate"):
            receipt = run / "tasks" / (source + "_" + stage + ".json")
            if receipt.is_file():
                task = read_json(receipt)
                require(task["status"] == "PASS" and task["configuration_sha256"] == sha256(cfg_file),
                        "Stage config/status differs")
                verify_files(task["code_sha256"]); verify_files(task["outputs_sha256"])
                timing = read_json(directory / ("TIMING_" + stage + ".json"))
                require(timing["source"] == source and timing["stage"] == stage, "Timing identity differs")
                for key in ("wall_seconds", "cpu_seconds"):
                    require(math.isfinite(timing[key]) and timing[key] >= 0, "Invalid measured time")
        if not (directory / "RESULTS.json").exists():
            pending.append(dict(source=source, stage="predict_or_evaluate")); continue
        require((run / "ALL_PREDICTIONS_FROZEN.json").is_file(), "Premature final evaluation")
        pred = read_json(directory / "PREDICTIONS_FROZEN.json")
        require(not pred["final_labels_read"] and pred["model_sha256"] == model["model_sha256"]
                and pred["candidate_sha256"] == binding["candidate_sha256"], "Prediction binding differs")
        verify_files({pred["predictions"]: pred["predictions_sha256"]})
        if check_candidates:
            from audit_gmpi_confidence import candidate_membership
            original_pred = read_json(original / "PREDICTIONS_FROZEN.json")
            verify_files({original_pred["predictions"]: original_pred["predictions_sha256"]})
            vocabulary = read_json(original / "VOCABULARY.json")
            membership.append(dict(source=source, **candidate_membership(binding, vocabulary, {
                "regression": original_pred["predictions"], "ranking": pred["predictions"]}, None)))
        result, old = read_json(directory / "RESULTS.json"), read_json(original / "RESULTS.json")
        require(result["query_count"] == pred["queries"] == old["query_count"]
                and result["prediction_sha256"] == pred["predictions_sha256"]
                and result["fixed_identity"] == old["fixed_identity"], "Evaluation denominator/identity differs")
        require(result["candidate_sha256"] == binding["candidate_sha256"]
                and result["regression_jaccard"] == old["jaccard_accuracy"]
                and result["oracle_jaccard"] == old["oracle_jaccard"], "Controlled baseline differs")
        require(result["jaccard_accuracy"] <= old["oracle_jaccard"] + 1e-10
                and result["ordered_exact_accuracy"] <= old["candidate_exact_available"] + 1e-10,
                "Accuracy exceeds candidate oracle")
        require(result["missing_predictions"] == result["invalid_nonempty_predictions"] == 0,
                "Missing/invalid predictions")
        if rescore:
            from pathInference.downstream.evaluation import score_source
            independent = score_source(old["reference_file"], pred["predictions"], source, old["fixed_identity"])
            for key in ("query_count", "ordered_exact_matches", "total_jaccard_reward", "jaccard_accuracy"):
                require(abs(independent[key] - result[key]) < 1e-10, "Independent final rescore differs")
        checked.append(dict(source=source, status="PASS", query_count=result["query_count"],
                            jaccard=result["jaccard_accuracy"], independent_rescore=rescore))
    if not pending:
        require((run / "SUMMARY.json").is_file() and (run / "REPORT.md").is_file(), "Missing final deliverables")
        aggregate = read_json(run / "RESULTS.json")
        require({r["source"] for r in aggregate} == set(sources) and len(aggregate) == 10, "Aggregate scope differs")
        for row in aggregate:
            require(row == read_json(run / ("as" + row["source"]) / "RESULTS.json"), "Aggregate row differs")
        summary = read_json(run / "SUMMARY.json")
        count = sum(row["query_count"] for row in aggregate)
        require(summary["query_count"] == count, "Aggregate denominator differs")
        for key, value in summary["micro"].items():
            expected = math.fsum(row[key] * row["query_count"] for row in aggregate) / count
            require(abs(value - expected) < 1e-12, "Aggregate metric differs")
    return dict(status="PASS" if not pending else "INCOMPLETE", checked_utc=utc_now(),
        completed_sources=checked, pending=pending, validation=validation, candidate_membership=membership,
        final_labels_read=bool(rescore and checked), candidate_identity="Exact bytes equal frozen v1 cache",
        historical_GMPI_candidate_limitation="Historical full candidate sets were deleted; counts/exact/oracle match only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/20260201/candidate_confidence_v2")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--validate-models", action="store_true")
    parser.add_argument("--check-candidates", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.rescore, args.validate_models, args.check_candidates)
    name = "COMPLETION_AUDIT.json" if result["status"] == "PASS" else "PRELIMINARY_AUDIT.json"
    write_json(args.run_dir / name, result)
    if args.validate_models:
        write_json(args.run_dir / "WEAK_LABEL_DIAGNOSTICS.json", dict(
            checked_utc=result["checked_utc"], final_test_labels_read=False, sources=result["validation"]))
        lines = ["# 候选打分：训练与验证审计", "",
                 "全部十源的模型及验证分数已独立复核。以下标签均来自冻结增强库，不是最终测试真值。", "",
                 "| AS | 训练查询 | 验证查询 | 回归验证 Jaccard | 排序验证 Jaccard | 排序最佳轮次 |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in result["validation"]:
            stats = row.get("weak_label_diagnostics", {})
            lines.append(f"| {row['source']} | {stats.get('train', {}).get('queries', '—')} | "
                         f"{stats.get('validation', {}).get('queries', '—')} | "
                         f"{row['reference_validation_jaccard']:.6f} | "
                         f"{row['ranking_validation_jaccard']:.6f} | {row['best_iteration']} |")
        lines += ["", "高验证分数只说明模型能复现增强标签里的偏好，不能代替真实路径测试精度。",
                  "本轮全部模型和预测冻结后才评测 test；现阶段没有按测试结果调整配置。"]
        (args.run_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
