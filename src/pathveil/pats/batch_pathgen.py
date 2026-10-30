"""Batch PATS candidate generation for one date.

The batch process owns one shared numerical environment for all registered
sources. It publishes source-scoped outputs only after both stages finish.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from pathveil.experiment import compile_queries
from pathveil.experiment.contracts import freeze, require, verify_files
from pathveil.experiment.worker import receipt_path
from pathveil.utils import read_json, sha256, utc_now, write_json
from pathveil.downstream import common

from .policy import POLICY


STAGE_FILES = {
    "training_inputs": (
        "train",
        "_train_extra_data_generated_data.csv",
        "pathgen_meta_results/PG-train-result.csv",
    ),
    "test_inputs": (
        "test",
        "_generated_data.csv",
        "pathgen_results/PG-test-result.csv",
    ),
}


def _copy(source, target):
    source, target = Path(source), Path(target)
    require(source.is_file(), "Batch source file missing: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(sha256(source) == sha256(target), "Batch output changed: " + str(target))
    else:
        shutil.copyfile(source, target)
    return target


def _merge_prepared(run):
    run = Path(run)
    proto = read_json(run / "PROTOCOL.json")
    sources = list(map(str, proto["sources"]))
    children = {
        source: run / "T-PG-B/pg_inputs" / ("as" + source)
        for source in sources
    }
    prepared_by_source = {
        source: common.verify_runtime_authorization(child)
        for source, child in children.items()
    }
    first = copy.deepcopy(prepared_by_source[sources[0]])
    first["sources"] = sources
    first["common9"] = sources
    first["data_manifest"] = str(run / "data/data_manifest.json")
    first["parent_run"] = str(run)
    first["batch_pathgen"] = True
    for key in (
        "admitted_pool",
        "admitted_library_path_set_sha256",
        "test_identity",
        "fixed_test_pairs_inventory",
        "derived_source_audits",
        "public_tree_spec_files",
    ):
        merged = {}
        for source in sources:
            merged.update(prepared_by_source[source].get(key, {}))
        first[key] = merged
    prepared_files = {}
    base_state = {}
    code_sha = {}
    for value in prepared_by_source.values():
        prepared_files.update(value.get("prepared_files", {}))
        base_state.update(value.get("base_file_state", {}))
        code_sha.update(value.get("code_sha256", {}))
    prepared_files[str(run / "data/data_manifest.json")] = sha256(run / "data/data_manifest.json")
    first["prepared_files"] = prepared_files
    first["base_file_state"] = base_state
    first["code_sha256"] = code_sha
    batch = run / "control/pats_batch_pathgen"
    batch.mkdir(parents=True, exist_ok=True)
    write_json(batch / "PREPARED.json", first)
    common.verify_runtime_authorization(batch)
    return batch, first, prepared_by_source


def _stage_arguments(batch, prepared, stage, experiment_dir=None, sources=None):
    run = Path(batch).parents[1]
    experiment_dir = Path(experiment_dir or batch)
    sources = ",".join(sources or prepared["sources"])
    if stage == "training_inputs":
        thread_count = POLICY["pathgen_train_threads"]
    else:
        thread_count = 1
    return [
        "--stage", stage,
        "--experiment_dir", str(experiment_dir),
        "--date", prepared["date"],
        "--target_vases", sources,
        "--meta_test_as_num", str(len(sources.split(","))),
        "--as_type", "devoid",
        "--num_thread", str(thread_count),
        "--k", str(POLICY["query_capacity"]),
        "--oppNum", "200",
        "--pg_local_knowledge_mode", "augmentation_only",
        "--train-extra-data-mode", "augmentation_only",
        "--support_data_dir", str(experiment_dir / "path_augmentation_output"),
        "--graph_selected_neighbors_csv", str(experiment_dir / "neighbor_selection/reviewed_selected_neighbors.csv"),
        "--graph_recursive_anchors_csv", str(experiment_dir / "neighbor_selection/recursive_secondhop_anchors.csv"),
        "--prefix-tree-dir", str(run / "query_trees"),
    ]


def _write_existing_task_receipt(run, source, stage="test_inputs"):
    child = Path(run) / "T-PG-B/pg_inputs" / ("as" + str(source))
    marker = child / "control/pathgen_completed" / (stage + "_" + str(source) + ".json")
    value = read_json(marker)
    output = Path(value["output"])
    outputs = {str(marker): sha256(marker), str(output): sha256(output)}
    for candidate in (
        child / "control/label_blind_test_inputs" / (str(source) + ".json"),
        child / "control/consumer_audit" / (stage + "_" + str(source) + "_local.json"),
    ):
        if candidate.is_file():
            outputs[str(candidate)] = sha256(candidate)
    receipt = {
        "status": "PASS",
        "stage": "pg_test",
        "source": str(source),
        "date": read_json(Path(run) / "PROTOCOL.json")["date"],
        "protocol_sha256": sha256(Path(run) / "PROTOCOL.json"),
        "prepared_sha256": sha256(child / "PREPARED.json"),
        "outputs_sha256": outputs,
    }
    freeze(receipt_path(run, "pg_test", source), receipt)


def _run_parallel_test_stage(run, batch, prepared, prepared_by_source):
    pending = []
    for source in prepared["sources"]:
        child = Path(run) / "T-PG-B/pg_inputs" / ("as" + source)
        marker = child / "control/pathgen_completed" / ("test_inputs_" + source + ".json")
        if marker.is_file():
            _write_existing_task_receipt(run, source)
        elif (Path(batch) / "control/pathgen_completed" / ("test_inputs_" + source + ".json")).is_file():
            # The interrupted shared batch may have completed this source
            # before failing while publishing a later source. Publish that
            # durable result instead of regenerating its 60k+ queries.
            one = copy.deepcopy(prepared)
            one["sources"] = [source]
            _publish_stage(run, batch, one, prepared_by_source, "test_inputs")
        else:
            pending.append(source)
    active = {}
    limit = POLICY["pathgen_test_threads"]
    while pending or active:
        while pending and len(active) < limit:
            source = pending.pop(0)
            child = Path(run) / "T-PG-B/pg_inputs" / ("as" + source)
            log = Path(run) / "logs" / ("pats_test_" + source + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("ab")
            arguments = _stage_arguments(
                batch,
                prepared_by_source[source],
                "test_inputs",
                experiment_dir=child,
                sources=[source],
            )
            command = [
                sys.executable,
                "-B",
                "-c",
                "from pathveil.experiment import compile_queries; raise SystemExit(compile_queries.main())",
                *arguments,
            ]
            env = dict(
                os.environ,
                PATHVEIL_PROJECT_ROOT=str(Path(run).parents[3]),
                PATHVEIL_STAGE_GUARD_ROOT=str(Path(run).resolve()),
                PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            process = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[source] = (process, stream)
        for source, (process, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del active[source]
            if code:
                raise RuntimeError("PATS test path generation failed for " + source)
            _write_existing_task_receipt(run, source)
        if active:
            time.sleep(2)


def _prepare_batch_inputs(run, batch, prepared):
    run = Path(run)
    support = batch / "path_augmentation_output/train"
    for source in prepared["sources"]:
        for suffix in (".csv", ".json"):
            source_path = run / "augmentation" / ("as" + source) / ("paths" + suffix)
            _copy(source_path, support / (source + "_augmented_paths" + suffix))
    upstream = read_json(run / "UPSTREAM_BOUND.json")
    release = read_json(Path(upstream["run"]) / "upstream/RELEASE.json")
    for key, name in (
        ("selected_neighbors_csv", "reviewed_selected_neighbors.csv"),
        ("recursive_anchors_csv", "recursive_secondhop_anchors.csv"),
    ):
        _copy(release[key], batch / "neighbor_selection" / name)


def _publish_stage(run, batch, prepared, prepared_by_source, stage):
    folder, suffix, report_rel = STAGE_FILES[stage]
    for source in prepared["sources"]:
        child = Path(run) / "T-PG-B/pg_inputs" / ("as" + source)
        child_marker = child / "control/pathgen_completed" / (stage + "_" + source + ".json")
        # A previous batch may have published this source before a later
        # source failed.  Frozen markers are immutable, so verify and reuse
        # the complete source instead of trying to overwrite it.
        if child_marker.exists():
            existing = read_json(child_marker)
            verify_files(existing.get("outputs_sha256", {}))
            continue
        batch_output = Path(batch) / "model_input_data_PG" / folder / "meta-devoid" / (source + suffix)
        child_output = child / "model_input_data_PG" / folder / "meta-devoid" / (source + suffix)
        _copy(batch_output, child_output)
        batch_marker = Path(batch) / "control/pathgen_completed" / (stage + "_" + source + ".json")
        marker = read_json(batch_marker)
        marker["output"] = str(child_output)
        marker["prepared_sha256"] = sha256(child / "PREPARED.json")
        child_marker = child / "control/pathgen_completed" / (stage + "_" + source + ".json")
        freeze(child_marker, marker)
        for consumer in (("truth", "local") if stage == "training_inputs" else ("local",)):
            name = stage + "_" + source + "_" + consumer + ".json"
            _copy(Path(batch) / "control/consumer_audit" / name, child / "control/consumer_audit" / name)
        if stage == "test_inputs":
            label = Path(batch) / "control/label_blind_test_inputs" / (source + ".json")
            if label.exists():
                value = read_json(label)
                value["prepared_sha256"] = sha256(child / "PREPARED.json")
                freeze(child / "control/label_blind_test_inputs" / label.name, value)
        report = Path(batch) / report_rel
        _copy(report, child / report_rel)
        task_receipt = {
            "status": "PASS",
            "stage": "pg_train" if stage == "training_inputs" else "pg_test",
            "source": source,
            "date": prepared["date"],
            "protocol_sha256": sha256(Path(run) / "PROTOCOL.json"),
            "prepared_sha256": sha256(child / "PREPARED.json"),
            "outputs_sha256": {
                str(child_output): sha256(child_output),
                str(child_marker): sha256(child_marker),
            },
        }
        for consumer in (("truth", "local") if stage == "training_inputs" else ("local",)):
            audit = child / "control/consumer_audit" / (stage + "_" + source + "_" + consumer + ".json")
            task_receipt["outputs_sha256"][str(audit)] = sha256(audit)
        freeze(receipt_path(run, task_receipt["stage"], source), task_receipt)


def _published_stage_complete(run, prepared, stage):
    """Return true only when every source has a verified published marker."""
    folder, suffix, _ = STAGE_FILES[stage]
    for source in prepared["sources"]:
        child = Path(run) / "T-PG-B/pg_inputs" / ("as" + source)
        marker = child / "control/pathgen_completed" / (stage + "_" + source + ".json")
        output = child / "model_input_data_PG" / folder / "meta-devoid" / (source + suffix)
        if not marker.is_file() or not output.is_file():
            return False
        value = read_json(marker)
        verify_files(value.get("outputs_sha256", {}))
    return True


def run_batch(run):
    run = Path(run)
    receipt = run / "PATS_PATHGEN_COMPLETED.json"
    if receipt.exists():
        value = read_json(receipt)
        verify_files(value["outputs_sha256"])
        return value
    batch, prepared, prepared_by_source = _merge_prepared(run)
    _prepare_batch_inputs(run, batch, prepared)
    outputs = []
    for stage in ("training_inputs", "test_inputs"):
        if not _published_stage_complete(run, prepared, stage):
            if stage == "test_inputs":
                _run_parallel_test_stage(run, batch, prepared, prepared_by_source)
            else:
                compile_queries.main(_stage_arguments(batch, prepared, stage))
                _publish_stage(run, batch, prepared, prepared_by_source, stage)
        folder, suffix, _ = STAGE_FILES[stage]
        outputs.extend(
            str(Path(run) / "T-PG-B/pg_inputs" / ("as" + source) / "model_input_data_PG" / folder / "meta-devoid" / (source + suffix))
            for source in prepared["sources"]
        )
    value = dict(
        status="COMPLETED",
        version=POLICY["version"],
        algorithm=POLICY["path_constructor"],
        date=prepared["date"],
        sources=prepared["sources"],
        shared_process=True,
        stages=["training_inputs", "test_inputs"],
        batch_prepared=str(batch / "PREPARED.json"),
        outputs_sha256={path: sha256(path) for path in outputs},
        completed_utc=utc_now(),
    )
    freeze(receipt, value)
    return value


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run one shared PATS path-generation batch")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_batch(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
