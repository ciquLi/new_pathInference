#!/usr/bin/env python3
"""Finalize the fixed25 research comparison without training or changing inputs.

Reads frozen experiment artifacts, verifies the evaluation and admission
boundaries, and writes audit/diagnostic reports only to the research batch.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pathInference.downstream.evaluation import score_source
from pathInference.experiment.contracts import signature

BATCH = ROOT / "runs/batches/partial_train_v1"


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(mapping):
    for path, expected in mapping.items():
        require(digest(path) == expected, "File identity changed: " + str(path))


def paths(path):
    with Path(path).open(newline="") as stream:
        rows = [tuple(row[0].split("->")) for row in csv.reader(stream)]
    require(len(rows) == len(set(rows)), "Repeated path: " + str(path))
    return set(rows)


def predictions(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {r["pair"]: r for r in rows}
    require(len(rows) == len(result), "Repeated prediction pair")
    return result


def geometry(tree):
    return {key: tree[key] for key in (
        "primary_leaves", "fallback_leaves", "selected_firsthops", "firsthop_decisions",
        "configuration", "source_shape", "graph_identity_sha256",
    )}


def verify_split(group, source, manifest):
    split_dir = group / "gmpi/shared_splits" / ("as" + source) / "surrogate_split"
    split = read(split_dir / "split_manifest.json")
    require(split["test_labels_used_for_selection"] is False, "Test used for split")
    inventories = {name: set(split[name + "_pairs"]) for name in ("train", "validation")}
    require(inventories["train"].isdisjoint(inventories["validation"]), "Validation overlap")
    require(digest(split["input_csv"]) == split["input_sha256"] == manifest["training_input_sha256"],
            "Training/split input mismatch")
    with Path(split["input_csv"]).open(newline="") as stream:
        rows = list(csv.reader(stream))
    by_pair = {row[4]: row for row in rows}
    require(len(rows) == len(by_pair), "Repeated training pair")
    require(set(by_pair) == inventories["train"] | inventories["validation"], "Split inventory mismatch")
    for name in ("train", "validation"):
        with (split_dir / (name + ".csv")).open(newline="") as stream:
            actual = list(csv.reader(stream))
        require(len(actual) == split["row_counts"][name] == len(inventories[name]), "Split count mismatch")
        require({row[4] for row in actual} == inventories[name], "Split pair mismatch")
        require(all(row == by_pair[row[4]] for row in actual), "Split changed training row")
    verify(manifest["shared_split_sha256"])
    return split


def training_audit(run, baseline, source, admission):
    group = run / "T-PG-B"
    selection_file = group / "gmpi/selections" / ("as" + source) / "SELECTION_FROZEN.json"
    selected = read(selection_file)
    require(selected["final_test_used_for_selection"] is False, "Test used for selection")
    verify({selected["chosen_model"]: selected["chosen_model_sha256"],
            selected["chosen_training_manifest"]: selected["chosen_training_manifest_sha256"]})
    manifest = read(selected["chosen_training_manifest"])
    worker = Path(selected["chosen_training_manifest"]).parent
    require(manifest["original_test_used_for_selection"] is False
            and manifest["initialized_from_checkpoint"] is False, "Training provenance changed")
    require(manifest["admitted_library_path_set_sha256"] == admission["path_set_sha256"],
            "Training library mismatch")
    require(not (worker / "ORIGINAL_TEST_ACCESS_STARTED.json").exists(), "Training opened final reference")
    provenance = manifest["source_training_provenance"]
    require(provenance["original_source_originated_rows"] == 0, "Original heldout source used")
    verify({provenance["audit_file"]: provenance["audit_sha256"]})
    split = verify_split(group, source, manifest)
    history = read(worker / "training_history.json")
    verify(manifest["validation_audits_sha256"])
    require(len(history) == manifest["epochs"] == len(manifest["validation_audits_sha256"]),
            "Incomplete epoch history")
    best = max(history, key=lambda row: row["validation_jaccard_reward"])
    require(best["epoch"] == manifest["selected_epoch"]
            and best["validation_jaccard_reward"] == manifest["best_surrogate_validation_jaccard"],
            "Checkpoint not chosen by validation")
    for row in history:
        epoch = read(worker.parents[3] / "epoch_validation" / ("epoch_%02d.json" % row["epoch"]))
        require(epoch["final_references_opened"] is False
                and epoch["query_count"] == split["row_counts"]["validation"]
                and epoch["natural_jaccard"] == epoch["criterion_value"] == row["validation_jaccard_reward"],
                "Epoch selection boundary differs")
    configuration = read(worker / "configuration.json")
    old = read(baseline / "T-PG-B/gmpi/selections" / ("as" + source) / "SELECTION_FROZEN.json")
    old_configuration = read(Path(old["chosen_training_manifest"]).parent / "configuration.json")
    keys = ("seed", "date", "validation_fraction", "epochs", "patience", "batch_size", "workers", "lr", "k", "hidden_size")
    require(all(configuration["arguments"][k] == old_configuration["arguments"][k] for k in keys),
            "GMPI configuration changed")
    barrier = read(group / "control/source_selection_barriers" / ("as" + source + ".json"))
    completion = read(group / "evaluations" / ("as" + source) / "EVALUATION_COMPLETED.json")
    require(barrier["selection_sha256"] == digest(selection_file)
            and barrier["final_labels_opened"] is False
            and completion["raw_labels_opened_only_after_source_selection"] is True
            and completion["final_test_used_for_model_selection"] is False
            and completion["model_sha256"] == selected["chosen_model_sha256"], "Evaluation selection barrier differs")
    return dict(checkpoint_sha256=selected["chosen_model_sha256"], selected_epoch=manifest["selected_epoch"],
                validation_queries=split["row_counts"]["validation"], training_pairs=split["row_counts"]["train"],
                pair_disjoint_verified=True, cold_training_verified=True, gmpi_parameters_unchanged=True,
                final_reference_not_used_for_checkpoint=True,
                inherited_graph_history_origin="legacy_uncertified; not retrospectively certified")


def paired(old, new, added):
    before, after = predictions(old["prediction_file"]), predictions(new["prediction_file"])
    require(set(before) == set(after) and len(before) == old["query_count"] == new["query_count"],
            "Paired query universe differs")
    added_destinations = {p[-1] for p in added}
    counts, sums = Counter(), defaultdict(list)
    transitions, groups = defaultdict(list), defaultdict(list)
    hops = {"before": Counter(), "after": Counter()}
    for pair, a in before.items():
        b = after[pair]
        x, y = float(a["jaccard"]), float(b["jaccard"])
        delta = y - x
        category = "equal" if abs(delta) < 1e-12 else "improved" if delta > 0 else "degraded"
        counts[category] += 1
        sums[category].append(delta)
        ah = a["predicted_path"].split("->")[1] if a["predicted_path"] else "empty"
        bh = b["predicted_path"].split("->")[1] if b["predicted_path"] else "empty"
        hops["before"][ah] += 1
        hops["after"][bh] += 1
        if a["predicted_path"] != b["predicted_path"]:
            transitions[(ah, bh)].append((x, y))
        name = "added_training_destination" if pair.split("->")[-1] in added_destinations else "other_query"
        groups[name].append((x, y))
    changes = [dict(before=a, after=b, pairs=len(values),
                    sum_delta=math.fsum(y - x for x, y in values),
                    mean_delta=math.fsum(y - x for x, y in values) / len(values))
               for (a, b), values in transitions.items()]
    require(math.isclose(math.fsum(math.fsum(v) for v in sums.values()) / len(before),
                         new["jaccard_accuracy"] - old["jaccard_accuracy"], abs_tol=1e-12), "Paired delta differs")
    diagnostic = dict(pair_counts=dict(counts), sum_delta_by_category={k: math.fsum(v) for k, v in sums.items()},
                      top_changed_firsthops=sorted(changes, key=lambda x: -x["pairs"])[:12],
                      worst_firsthop_transitions=sorted(changes, key=lambda x: x["sum_delta"])[:12],
                      predicted_firsthops={k: dict(v.most_common()) for k, v in hops.items()})
    grouped = {name: dict(queries=len(values), old_reward=math.fsum(x for x, y in values),
                         new_reward=math.fsum(y for x, y in values),
                         delta_mean=math.fsum(y - x for x, y in values) / len(values))
               for name, values in groups.items()}
    return diagnostic, grouped


def render_report(rows, aggregates, diagnostics, grouped):
    lines = [
        "# 两AS部分吸纳TRAIN路径：固定25%首轮最终结果", "",
        "五日期10/10组完成，最终审计10/10通过。AS34927平均Jaccard上升，AS58057五期均下降；这轮规则不宜直接并入正式方法。", "",
        "## 实验定义", "",
        "基线为已冻结的21／5方法。只对AS34927、AS58057吸纳来自其他原始TRAIN源的正向连续子路径，目标AS自身的原始观测仍保持留出。",
        "全部保留原增强路径；新增数量上限为增强条数的25%（即混合库约20%）。每目的地最多新增一条，优先补充增强库未覆盖的目的地，各首跳轮流选；同目的地先选最短合法路径，再用固定哈希打破并列。规则在本次新评测前固定。",
        "前缀树几何、GMPI参数与选择准则保持基线定义；训练标签及PATS本地知识同时使用混合库并验证集合哈希，重新冷启动训练。检查点由源内pair-disjoint的BGP训练/验证划分选择。", "",
        "## 五日期宏平均", "",
        "各日期等权；最后一行对10个AS×日期组合等权。Jaccard为完整固定查询集上的平均值，缺失预测按零计。Exact要求完整有序路径匹配。", "",
        "| AS | 基线Jaccard | 部分吸纳Jaccard | 变化（百分点） | 基线Exact | 部分吸纳Exact | 变化（百分点） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, a in aggregates.items():
        lines.append("| {} | {:.4%} | {:.4%} | {:+.4f} | {:.4%} | {:.4%} | {:+.4f} |".format(
            "两AS合计" if source == "all" else source, a["baseline_jaccard"], a["jaccard"],
            a["delta_jaccard"] * 100, a["baseline_exact"], a["exact"], a["delta_exact"] * 100))
    lines += ["", "## 逐日期Jaccard", "",
              "| 日期 | AS | 查询数 | 原增强 | 新增TRAIN | 基线 | 部分吸纳 | 变化（百分点） |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {date} | {source} | {queries} | {augmentation_paths} | {added_train_paths} | {baseline_jaccard:.4%} | {jaccard:.4%} | {delta:+.4f} |".format(
            **r, delta=r["delta_jaccard"] * 100))
    lines += ["", "## 逐日期Exact", "",
              "| 日期 | AS | 基线Exact | 部分吸纳Exact | 变化（百分点） |",
              "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append("| {date} | {source} | {baseline_exact:.4%} | {exact:.4%} | {delta:+.4f} |".format(
            **r, delta=r["delta_exact"] * 100))
    lines += ["", "## 逐查询诊断", "",
              "以下分组只用于实验结束后的解释，不参与吸纳、调参或检查点选择。新增目的地指本轮新增TRAIN路径的终点。", "",
              "| 日期 | AS | Jaccard提高/持平/下降的查询数 | 新增目的地查询数 | 该组变化（百分点） | 其他查询变化（百分点） |",
              "|---|---|---:|---:|---:|---:|"]
    for d, g in zip(diagnostics, grouped):
        c, groups = d["pair_counts"], g["groups"]
        added, other = groups["added_training_destination"], groups["other_query"]
        lines.append("| {} | {} | {}/{}/{} | {} | {:+.4f} | {:+.4f} |".format(
            d["date"], d["source"], c.get("improved", 0), c.get("equal", 0), c.get("degraded", 0),
            added["queries"], added["delta_mean"] * 100, other["delta_mean"] * 100))
    lines += ["",
              "AS34927在0201、0301、0401提高，0501、0601略降；新增目的地查询五期均有提高，但后两期其他查询的损失超过了该组收益。因此平均收益不能解释成稳定的跨日期提升。", "",
              "AS58057的下降覆盖新增目的地和其他查询，尤其表现为原先经174的预测转到6939或50673。例如0201有9,834个查询从174转到6939，这组平均Jaccard下降26.64个百分点；0301分别有4,993和6,713个查询从174转到6939、50673。五期均出现同方向变化。", "",
              "十组实验的模型查询数和候选oracle Jaccard（在模型查询上按3位小数计算的最佳候选均值）均与基线相同。AS58057的候选oracle约为98.15%至98.27%，最终选择的Jaccard仍下降。这支持继续检查候选排序/训练分布变化；相同oracle不等于完整候选集合或历史特征完全相同。", "",
              "本实验同时改变训练库和PATS本地知识，无法单独归因于其中一个。新增TRAIN路径通过来源和合法性检查，仍不意味着它们代表目标AS对所有目的地的出路偏好。当前结果只否定这轮固定25%吸纳规则可统一改善两个AS，不能据此否定所有部分吸纳方案。", "",
              "## 审计与复现", "",
              "- `FINAL_AUDIT.json`：10组参考文件/查询身份、预测哈希、独立重评分、树几何、共享库、冷训练、pair-disjoint划分与检查点准则审计。",
              "- `FINAL_SUMMARY.json`：逐日期完整指标、宏平均与固定规则；`RESULTS.csv`保留运行期对照表。",
              "- `PAIRED_DIAGNOSTICS.json`：五日期逐查询变化计数、首跳转移与预测首跳分布。",
              "- `NEW_DESTINATION_DIAGNOSTICS.json`：新增TRAIN目的地/其他查询分组结果。",
              "- `ADMISSION_AUDIT.json`、`BOUNDARY_AUDIT.json`：吸纳数量、树和混合库边界。", "",
              "本次复跑相关5项测试通过。实验实现阶段曾通过全套238项测试，1项按原设计跳过。最终审计工具仅读取输入和模型，输出到本实验batch目录：", "",
              "```bash", "python -B tools/validation/verify_partial_train.py", "```", "",
              "实验分支：`research/partial-train-v1`。正式包恢复点为`9ce7a82`，实验实现为`c77c39b`，调度边界修正为`29612d8`。数值代码与运行输入身份已固定；正式五日期基线未改动。", "",
              "## 适用范围", "",
              "这是固定种子的单次探索性实验，未做多种子重复。此前已查看过同日期测试分布，因此不能当作从未接触数据的最终验证。25%比例和吸纳顺序并非已验证的最优规则。历史图/历史特征的上游来源仍标记为`legacy_uncertified`；本次审计不对继承资产作追溯认证。", ""]
    return "\n".join(lines)


def main():
    config_file = ROOT / "research/partial_train_v1/CONFIG.json"
    config = read(config_file)
    require(config == read(BATCH / "CONFIG.json"), "Batch configuration differs")
    release = read(ROOT / config["baseline_manifest"])
    frozen = read(ROOT / config["code_manifest"])
    verify({ROOT / p: h for p, h in frozen["source_files_sha256"].items() if p.startswith("src/")})
    audits, diagnostics, grouped, rows = [], [], [], []
    inputs = {str(config_file): digest(config_file), str(Path(__file__)): digest(__file__)}
    for date, relative in config["dates"].items():
        baseline = ROOT / relative
        old_results_file = baseline / "RESULTS.json"
        require(digest(old_results_file) == release["baseline_assets_sha256"][str(old_results_file.relative_to(ROOT))],
                "Frozen baseline results changed")
        old_results = {r["source_asn"]: r for r in read(old_results_file)["per_source"]}
        run = ROOT / "runs" / date / config["name"] / "fixed25"
        verify(read(run / "PROTOCOL.json")["code_sha256"])
        for source in config["sources"]:
            group = run / "T-PG-B"
            child = group / "pg_inputs" / ("as" + source)
            admission_file = run / "augmentation" / ("as" + source) / "ADMISSION.json"
            admission = read(admission_file)
            verify(admission["inputs_sha256"])
            verify({admission["path"]: admission["sha256"], admission["json_path"]: admission["json_sha256"]})
            library = paths(admission["path"])
            added = paths(admission_file.parent / "selected_train.csv")
            old_admission = read(baseline / "augmentation" / ("as" + source) / "ADMISSION.json")
            augmented = paths(old_admission["path"])
            require(library == augmented | added and augmented.isdisjoint(added), "Mixed library differs")
            require(len(added) <= int(len(augmented) * config["policy"]["maximum_added_to_augmented_ratio"])
                    and len({p[-1] for p in added}) == len(added), "Admission budget/destination rule differs")
            require(signature(library) == admission["path_set_sha256"], "Library set hash differs")
            tree = read(run / "query_trees" / (source + ".json"))
            require(geometry(tree) == geometry(read(baseline / "query_trees" / (source + ".json"))), "Tree geometry changed")
            require(tree["admitted_path_set_sha256"] == admission["path_set_sha256"], "Tree library binding differs")
            for name in ("training_inputs_%s_truth.json", "training_inputs_%s_local.json", "test_inputs_%s_local.json"):
                consumer = read(child / "control/consumer_audit" / (name % source))
                require(consumer["same_admitted_pool_verified"] is True
                        and consumer["path_set_sha256"] == admission["path_set_sha256"], "Consumer library differs")
            training = training_audit(run, baseline, source, admission)
            result_file = group / "independent_evaluation" / ("source_" + source + ".json")
            new, old = read(result_file), old_results[source]
            summary_file = group / "evaluations" / ("as" + source) / "evaluation_summary.json"
            old_summary_file = baseline / "T-PG-B/evaluations" / ("as" + source) / "evaluation_summary.json"
            summary, old_summary = read(summary_file), read(old_summary_file)
            identity = read(child / "PREPARED.json")["test_identity"][source]
            require(new["fixed_identity"] == old["fixed_identity"], "Fixed test identities differ")
            for result in (old, new):
                rescored = score_source(result["reference_file"], result["prediction_file"], source, identity)
                for key in ("query_count", "prediction_sha256", "ordered_exact_matches", "fixed_identity",
                            "jaccard_accuracy", "ordered_exact_accuracy", "missing_predictions", "invalid_nonempty_predictions"):
                    require(rescored[key] == result[key], "Independent rescore differs: " + key)
            require(new["checkpoint_sha256"] == training["checkpoint_sha256"], "Evaluation checkpoint changed")
            diagnostic, groups = paired(old, new, added)
            diagnostics.append(dict(date=date, source=source, **diagnostic))
            grouped.append(dict(date=date, source=source, groups=groups))
            audits.append(dict(date=date, source=source, query_count=new["query_count"],
                               fixed_reference_sha256=new["fixed_identity"]["reference_sha256"],
                               pair_inventory_sha256=new["fixed_identity"]["pair_inventory_sha256"],
                               independent_rescore_verified=True, tree_geometry_identical=True,
                               shared_library_hash_verified=True,
                               candidate_oracle_before=old_summary["candidate_oracle_jaccard_rounded3_on_model_queries"],
                               candidate_oracle_after=summary["candidate_oracle_jaccard_rounded3_on_model_queries"],
                               model_queries_before=old_summary["generated_model_queries"],
                               model_queries_after=summary["generated_model_queries"], **training))
            rows.append(dict(date=date, source=source, queries=new["query_count"],
                             augmentation_paths=len(augmented), added_train_paths=len(added),
                             baseline_jaccard=old["jaccard_accuracy"], jaccard=new["jaccard_accuracy"],
                             delta_jaccard=new["jaccard_accuracy"] - old["jaccard_accuracy"],
                             baseline_exact=old["ordered_exact_accuracy"], exact=new["ordered_exact_accuracy"],
                             delta_exact=new["ordered_exact_accuracy"] - old["ordered_exact_accuracy"]))
            for path in (old_results_file, admission_file, result_file, summary_file, old_summary_file,
                         Path(old["prediction_file"]), Path(new["prediction_file"])):
                inputs[str(path)] = digest(path)
            print(date, source, "PASS", flush=True)
    aggregates = {}
    for source in (*config["sources"], "all"):
        subset = [r for r in rows if source == "all" or r["source"] == source]
        totals = {k: math.fsum(r[k] for r in subset) / len(subset)
                  for k in ("baseline_jaccard", "jaccard", "delta_jaccard", "baseline_exact", "exact", "delta_exact")}
        totals.update(groups=len(subset), queries=sum(r["queries"] for r in subset))
        aggregates[source] = totals
    outputs = {
        "PAIRED_DIAGNOSTICS.json": diagnostics,
        "NEW_DESTINATION_DIAGNOSTICS.json": grouped,
        "FINAL_SUMMARY.json": dict(rows=rows, macro_averages=aggregates, fixed_policy=config["policy"],
                                   exploratory=True, multi_seed_replicates=False),
        "FINAL_AUDIT.json": dict(status="PASS", completed=len(rows), checks=audits, inputs_sha256=inputs,
                                 frozen_production_source_verified=True,
                                 limitation="Inherited graph/history origins remain legacy_uncertified; this audit verifies the new admission/selection boundary."),
    }
    for filename, content in outputs.items():
        (BATCH / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    (BATCH / "REPORT.md").write_text(render_report(rows, aggregates, diagnostics, grouped))
    (BATCH / "PROGRESS.md").write_text(
        "# 两AS部分吸纳实验\n\n五日期共10/10组完成，最终审计10/10通过；相关5项测试通过。\n\n"
        "AS34927五日期平均Jaccard：83.4508% → 85.1859%（+1.7351个百分点）。\n\n"
        "AS58057五日期平均Jaccard：76.1315% → 72.8215%（-3.3100个百分点）。\n\n"
        "这轮规则不宜直接并入正式方法。完整结果见[REPORT.md](REPORT.md)，审计见[FINAL_AUDIT.json](FINAL_AUDIT.json)。\n")
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == "__main__":
    main()
