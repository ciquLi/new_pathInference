"""Frozen-tree partial admission of audited original TRAIN forward subpaths.

Production numerical APIs are called without modifying their source or relaxing
their guards. Every run is independent of the five-date production baselines.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pathInference.downstream import common
from pathInference.experiment.contracts import (
    forbid_formal_references, freeze, freeze_bytes, path_bytes, require, signature, verify_files,
)
from pathInference.utils import object_hash, project_root, read_json, sha256, utc_now, write_json

ROOT = project_root()
NAME = "partial_train_v1"
SOURCES = ["34927", "58057"]
CONFIG = ROOT / "research/partial_train_v1/CONFIG.json"
BATCH = ROOT / "runs/batches/partial_train_v1"


def stable_key(path, seed):
    return hashlib.sha256((str(seed) + ":" + "->".join(path)).encode()).hexdigest()


def choose_paths(original, augmented, prefixes, role, ratio=0.25, seed=20261015):
    """At most one new path/destination; balance first hops by round robin.

    A destination absent from the augmentation library takes priority. Within a
    destination choose the shortest eligible TRAIN path, resolving ties by a
    stable hash. Branch scheduling never uses final labels or predicted mass.
    """
    augmented = set(map(tuple, augmented))
    prefixes = set(map(tuple, prefixes))
    require(augmented and 0 < ratio <= 1, "Invalid augmentation or admission ratio")
    source = next(iter(augmented))[0]
    require(all(p[0] == source for p in augmented | prefixes), "Mixed source admission")
    from pathInference.prefix_tree.public_prefix_tree import commercial_phases

    counts = Counter()
    by_destination = {}
    for path in sorted(set(map(tuple, original))):
        counts["original_unique"] += 1
        if path in augmented:
            counts["already_augmented"] += 1
            continue
        if not (2 <= len(path) <= 20 and path[0] == source and len(set(path)) == len(path)
                and all(n.isascii() and n.isdecimal() and str(int(n)) == n
                        and 0 < int(n) < 2**32 for n in path)):
            counts["invalid"] += 1
            continue
        if not any(path[:depth] in prefixes for depth in (2, 3) if len(path) >= depth):
            counts["outside_frozen_tree"] += 1
            continue
        if commercial_phases([role(a, b) for a, b in zip(path, path[1:])])["status"] == "incompatible":
            counts["known_commercial_conflict"] += 1
            continue
        counts["eligible_unique"] += 1
        previous = by_destination.get(path[-1])
        if previous is None or (len(path), stable_key(path, seed)) < (len(previous), stable_key(previous, seed)):
            by_destination[path[-1]] = path
    covered = {p[-1] for p in augmented}
    branches = defaultdict(list)
    for path in by_destination.values():
        branches[path[1]].append(path)
    for values in branches.values():
        values.sort(key=lambda p: (p[-1] in covered, stable_key(p, seed)))
    offsets = Counter()
    selected = []
    budget = int(len(augmented) * ratio)
    # Finish the new-destination tier before supplementing covered endpoints.
    for already_covered in (False, True):
        active = True
        while active and len(selected) < budget:
            active = False
            for branch in sorted(branches, key=int):
                offset = offsets[branch]
                values = branches[branch]
                if offset < len(values) and (values[offset][-1] in covered) == already_covered:
                    selected.append(values[offset])
                    offsets[branch] += 1
                    active = True
                    if len(selected) == budget:
                        break
    counts.update(eligible_destinations=len(by_destination), budget=budget, selected=len(selected),
                  newly_covered_destinations=sum(p[-1] not in covered for p in selected))
    return sorted(selected), dict(counts)


def current_code():
    return {str(p): sha256(p) for p in sorted((ROOT / "src/pathInference").rglob("*.py"))}


def verified_baseline(path, release):
    path = Path(path).resolve()
    key = str(path.relative_to(ROOT))
    expected = release["baseline_assets_sha256"].get(key)
    require(expected is not None, "Artifact absent from frozen baseline inventory: " + key)
    verify_files({str(path): expected})
    return path


def prepare(date):
    config = read_json(CONFIG)
    require(date in config["dates"], "Unregistered experiment date")
    run = ROOT / "runs" / date / NAME / "fixed25"
    ready = run / "PREPARATION_READY.json"
    if ready.exists():
        result = read_json(ready)
        require(result["config_sha256"] == sha256(CONFIG), "Research configuration changed")
        verify_files(result["outputs_sha256"])
        verify_files(read_json(run / "PROTOCOL.json")["code_sha256"])
        return run
    release = read_json(ROOT / config["baseline_manifest"])
    baseline = ROOT / config["dates"][date]
    run.mkdir(parents=True, exist_ok=True)
    # Verify the fixed production modules, without requiring this research
    # branch to masquerade as the historical production source inventory.
    frozen_code = read_json(ROOT / config["code_manifest"])["source_files_sha256"]
    verify_files({str(ROOT / p): h for p, h in frozen_code.items() if p.startswith("src/")})
    old_proto = read_json(verified_baseline(baseline / "PROTOCOL.json", release))
    require(old_proto["date"] == date, "Baseline date differs")
    protocol = dict(version=NAME, date=date, sources=SOURCES, common9=SOURCES,
                    method=old_proto["method"], public_features={s: old_proto["public_features"][s] for s in SOURCES},
                    baseline_run=str(baseline), baseline_protocol_sha256=sha256(baseline / "PROTOCOL.json"),
                    policy=config["policy"], config_sha256=sha256(CONFIG),
                    code_sha256=current_code(), final_labels_used_for_admission=False)
    freeze(run / "PROTOCOL.json", protocol)
    freeze(run / "UPSTREAM_BOUND.json", read_json(verified_baseline(baseline / "UPSTREAM_BOUND.json", release)))
    source_manifest = baseline / "data/data_manifest.json"
    # The shared numerical environment uses the same prepared routing overlay.
    freeze(run / "data/data_manifest.json", read_json(verified_baseline(source_manifest, release)))
    group_method = read_json(baseline / "T-PG-B/METHOD_FROZEN.json")
    group_method.update(version=NAME, sources=SOURCES, public_features=protocol["public_features"],
                        protocol_sha256=sha256(run / "PROTOCOL.json"), partial_train_policy=config["policy"])
    freeze(run / "T-PG-B/METHOD_FROZEN.json", group_method)
    freeze(run / "EVALUATION_POLICY_AMENDMENT.json", dict(
        selection_scope="single_source_generator", training_and_selection_unchanged=True,
        final_scores_used_for_further_training=False))
    freeze(run / "query_trees/METHOD_FROZEN.json", dict(
        version=NAME, date=date, sources=SOURCES, baseline=str(baseline),
        prefix_geometry="exact unchanged baseline tree; admitted library binding changes",
        protocol_sha256=sha256(run / "PROTOCOL.json")))
    outputs = {}
    from pathInference.prefix_tree.registered_topology import RegisteredPublicPrefixTopology
    with forbid_formal_references():
        topology = RegisteredPublicPrefixTopology.from_manifest(run / "data/data_manifest.json")
        for source in SOURCES:
            old_child = baseline / "T-PG-B/pg_inputs" / ("as" + source)
            old_prepared_path = verified_baseline(old_child / "PREPARED.json", release)
            prepared = read_json(old_prepared_path)
            verify_files(prepared["prepared_files"])
            for name, state in prepared["base_file_state"].items():
                actual = Path(name).stat()
                require(actual.st_size == state["size"] and actual.st_mtime_ns == state["mtime_ns"], "Baseline input changed")
            audit_path = Path(prepared["derived_source_audits"][source])
            audit = read_json(audit_path)
            require(audit["test_paths_read"] is False and audit["train_merged_source_origin_rows"] == 0,
                    "Original source holdout not established")
            verify_files({audit["source_csv"]: audit["source_csv_sha256"]})
            original = common.csv_paths(audit["source_csv"], source)
            require(len(original) == audit["source_csv_unique_paths"], "Audited TRAIN size differs")
            admission_path = verified_baseline(baseline / "augmentation" / ("as" + source) / "ADMISSION.json", release)
            old_admission = read_json(admission_path)
            verify_files({old_admission["path"]: old_admission["sha256"], old_admission["json_path"]: old_admission["json_sha256"]})
            augmented = common.csv_paths(old_admission["path"], source)
            tree_path = verified_baseline(baseline / "query_trees" / (source + ".json"), release)
            tree = read_json(tree_path)
            prefixes = {tuple(x["prefix"]) for x in tree["primary_leaves"] + tree["fallback_leaves"]}
            selected, counts = choose_paths(original, augmented, prefixes, topology.public_role,
                                            config["policy"]["maximum_added_to_augmented_ratio"], config["policy"]["seed"])
            library = sorted(set(map(tuple, augmented)) | set(selected))
            directory = run / "augmentation" / ("as" + source)
            directory.mkdir(parents=True, exist_ok=True)
            csv_bytes, json_bytes = path_bytes(library)
            freeze_bytes(directory / "paths.csv", csv_bytes)
            freeze_bytes(directory / "paths.json", json_bytes)
            selected_bytes, _ = path_bytes(selected)
            freeze_bytes(directory / "selected_train.csv", selected_bytes)
            provenance = {str(admission_path): sha256(admission_path), str(tree_path): sha256(tree_path),
                          str(old_prepared_path): sha256(old_prepared_path), str(audit_path): sha256(audit_path),
                          audit["source_csv"]: audit["source_csv_sha256"]}
            item = dict(status="FROZEN", date=date, source=source, strategy=NAME + "_fixed25_tree_compatible",
                        rho=old_admission["rho"], rows=len(library), destinations=len({p[-1] for p in library}),
                        path=str(directory / "paths.csv"), json_path=str(directory / "paths.json"),
                        sha256=sha256(directory / "paths.csv"), json_sha256=sha256(directory / "paths.json"),
                        path_set_sha256=signature(library), protocol_sha256=sha256(run / "PROTOCOL.json"),
                        baseline_augmented_paths=len(augmented), selected_train_paths=len(selected), selection=counts,
                        inputs_sha256=provenance, final_labels_read=False)
            freeze(directory / "ADMISSION.json", item)
            tree.update(admitted_path_set_sha256=item["path_set_sha256"], library_csv=item["path"],
                        library_csv_sha256=item["sha256"], construction_method_sha256=sha256(run / "query_trees/METHOD_FROZEN.json"))
            tree["data_code_input_sha256"] = {
                p: h for p, h in tree["data_code_input_sha256"].items() if not p.endswith(".py")}
            tree["data_code_input_sha256"].update(provenance)
            tree["data_code_input_sha256"][str(directory / "ADMISSION.json")] = sha256(directory / "ADMISSION.json")
            tree["research_library_binding"] = NAME
            new_tree = run / "query_trees" / (source + ".json")
            freeze(new_tree, tree)
            require(prefixes == {tuple(x["prefix"]) for x in tree["primary_leaves"] + tree["fallback_leaves"]}, "Tree geometry changed")
            child = run / "T-PG-B/pg_inputs" / ("as" + source)
            # Reuse verified immutable metadata, with a new explicit prepared
            # identity. Historical PREPARED files are never overwritten.
            files = dict(prepared["prepared_files"])
            for relative in ("data/data_manifest.json", "neighbor_selection/reviewed_selected_neighbors.csv",
                             "neighbor_selection/recursive_secondhop_anchors.csv"):
                target = child / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                freeze_bytes(target, (old_child / relative).read_bytes())
                files[str(target)] = sha256(target)
            for suffix in (".csv", ".json"):
                target = child / "path_augmentation_output/train" / (source + "_augmented_paths" + suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                freeze_bytes(target, (directory / ("paths" + suffix)).read_bytes())
                files[str(target)] = sha256(target)
            subset = dict(status="PASS", date=date, sources=[source], files_by_source={source: item},
                          outputs_sha256={item["path"]: item["sha256"], item["json_path"]: item["json_sha256"]})
            for key, filename in (("selected_neighbors_csv", "reviewed_selected_neighbors.csv"),
                                  ("recursive_anchors_csv", "recursive_secondhop_anchors.csv")):
                subset[key] = str(child / "neighbor_selection" / filename)
            freeze(child / "upstream/RELEASE.json", subset)
            prepared.update(data_manifest=str(child / "data/data_manifest.json"), code_sha256=current_code(),
                            admitted_pool_manifest_sha256=sha256(child / "upstream/RELEASE.json"),
                            admitted_pool={source: dict(paths=len(library), csv_sha256=item["sha256"], json_sha256=item["json_sha256"])},
                            admitted_library_path_set_sha256={source: item["path_set_sha256"]},
                            public_tree_spec_files={source: dict(path=str(new_tree), sha256=sha256(new_tree))},
                            research_baseline_prepared=dict(path=str(old_prepared_path), sha256=sha256(old_prepared_path)))
            prepared["binding"] = dict(data_manifest_sha256=sha256(child / "data/data_manifest.json"),
                                       release_sha256=sha256(child / "upstream/RELEASE.json"))
            prepared["tree_B_generator"]["code_sha256"] = current_code()
            for path in (new_tree, directory / "ADMISSION.json", run / "PROTOCOL.json", child / "upstream/RELEASE.json"):
                files[str(path)] = sha256(path)
            prepared["prepared_files"] = files
            freeze(child / "PREPARED.json", prepared)
            common.verify_runtime_authorization(child)
            outputs.update({str(p): sha256(p) for p in (child / "PREPARED.json", new_tree, directory / "ADMISSION.json",
                                                        directory / "paths.csv", directory / "paths.json", directory / "selected_train.csv")})
            print(json.dumps(dict(stage="admission_prepared", date=date, source=source, **counts)), flush=True)
    freeze(ready, dict(status="PASS", config_sha256=sha256(CONFIG), outputs_sha256=outputs))
    return run


def report():
    config = read_json(CONFIG)
    rows = []
    for date, baseline_relative in config["dates"].items():
        baseline_rows = {r["source_asn"]: r for r in read_json(ROOT / baseline_relative / "RESULTS.json")["per_source"]}
        run = ROOT / "runs" / date / NAME / "fixed25"
        for source in SOURCES:
            path = run / "T-PG-B/independent_evaluation" / ("source_" + source + ".json")
            if path.is_file():
                result = read_json(path)
                admission = read_json(run / "augmentation" / ("as" + source) / "ADMISSION.json")
                base = baseline_rows[source]
                rows.append(dict(date=date, source=source, baseline_jaccard=base["jaccard_accuracy"],
                                 jaccard=result["jaccard_accuracy"], delta_jaccard=result["jaccard_accuracy"] - base["jaccard_accuracy"],
                                 baseline_exact=base["ordered_exact_accuracy"], exact=result["ordered_exact_accuracy"],
                                 queries=result["query_count"], augmentation_paths=admission["baseline_augmented_paths"],
                                 added_train_paths=admission["selected_train_paths"], run_dir=str(run)))
    BATCH.mkdir(parents=True, exist_ok=True)
    write_json(BATCH / "RESULTS.json", dict(status="COMPLETED" if len(rows) == 10 else "RUNNING", rows=rows, updated_utc=utc_now()))
    lines = ["# 两AS部分吸纳TRAIN路径：固定25%首轮", "", "每个目的地最多一条新增路径，前缀树和GMPI配置保持基线定义。", "",
             "| 日期 | AS | 增强 | 新增TRAIN | 原Jaccard | 新Jaccard | 变化(百分点) |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {date} | {source} | {augmentation_paths} | {added_train_paths} | {baseline_jaccard:.4%} | {jaccard:.4%} | {delta:+.4f} |".format(**row, delta=row["delta_jaccard"] * 100))
    (BATCH / "REPORT.md").write_text("\n".join(lines) + "\n")
    if rows:
        with (BATCH / "RESULTS.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_command(command, log, description):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            state = dict(status="RUNNING", task=description, pid=process.pid, log=str(log), updated_utc=utc_now())
            write_json(BATCH / "STATUS.json", state)
            completed = len(report())
            tail = log.read_text(errors="replace")[-1600:]
            progress = ["# 两AS部分吸纳实验进度", "", "更新时间：" + state["updated_utc"], "",
                        "当前任务：" + description, "PID：" + str(process.pid), "已完成：%d/10" % completed,
                        "日志：`" + str(log.relative_to(ROOT)) + "`", "", "最近日志：", "```text", tail, "```", "",
                        "结果表：REPORT.md / RESULTS.csv。日期顺序执行；候选生成只处理两个AS。"]
            (BATCH / "PROGRESS.md").write_text("\n".join(progress) + "\n")
            print(json.dumps(state), flush=True)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        require(process.returncode == 0, "Research stage failed: " + description + "; log=" + str(log))


def batch():
    BATCH.mkdir(parents=True, exist_ok=True)
    with (BATCH / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(CONFIG)
        freeze(BATCH / "CONFIG.json", config)
        try:
            for date in config["dates"]:
                log_dir = BATCH / "logs"
                run_command([sys.executable, "-B", "-m", "pathInference.research.partial_train", "--stage", "prepare", "--date", date],
                            log_dir / (date + "_prepare.log"), date + " 准备混合训练库")
                run = ROOT / "runs" / date / NAME / "fixed25"
                run_command([sys.executable, "-B", "-m", "pathInference.pats.batch_pathgen", "--run-dir", str(run)],
                            log_dir / (date + "_pathgen.log"), date + " 两AS候选生成")
                for source in SOURCES:
                    for stage in ("train", "annotate", "evaluate", "rescore"):
                        run_command([sys.executable, "-B", "-m", "pathInference.experiment.worker", "--run", str(run),
                                     "--stage", stage, "--source", source, "--device", "cuda", "--cpu-threads", "4"],
                                    log_dir / (date + "_" + source + "_" + stage + ".log"), date + " AS" + source + " " + stage)
                report()
            write_json(BATCH / "STATUS.json", dict(status="COMPLETED", completed=10, updated_utc=utc_now()))
            (BATCH / "PROGRESS.md").write_text("# 两AS部分吸纳实验\n\n已完成五日期共10/10组。结果见 REPORT.md、RESULTS.csv。\n")
        except BaseException as error:
            write_json(BATCH / "STATUS.json", dict(status="FAILED", error=repr(error), updated_utc=utc_now()))
            with (BATCH / "PROGRESS.md").open("a") as stream:
                stream.write("\n任务失败：" + repr(error) + "\n")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("batch", "prepare", "report"), default="batch")
    parser.add_argument("--date")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.date)
    elif args.stage == "report":
        report()
    else:
        batch()


if __name__ == "__main__":
    main()
