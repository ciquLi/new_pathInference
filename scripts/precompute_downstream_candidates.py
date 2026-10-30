#!/usr/bin/env python3
"""Prepare all formal PG inputs before GPU training, with isolated AS workers.

Calls the frozen pathgen module unchanged. Each AS has its own cache, report
and output directory; verified results are published under a single lock.
"""
from __future__ import annotations

import csv
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from pathveil.utils import read_json, write_json, sha256, utc_now
from pathveil.downstream import common
from pathveil.downstream.runner import _execute, _adopt_existing
from build_training_cases import WorkerMemoryCoordinator
from run_priority_parallel import gpu_idle

GIB = 1024**3
STAGES = ("training_inputs", "test_inputs")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage_paths(run, source, stage):
    kind = "train" if stage == "training_inputs" else "test"
    suffix = (
        "_train_extra_data_generated_data.csv"
        if kind == "train"
        else "_generated_data.csv"
    )
    return (
        run / "model_input_data_PG" / kind / "meta-devoid" / (source + suffix),
        run / "control/pathgen_completed" / (stage + "_" + source + ".json"),
    )


def report_path(run, stage):
    if stage == "training_inputs":
        return run / "pathgen_meta_results/PG-train-result.csv"
    return run / "pathgen_results/PG-test-result.csv"


def command(run, source, stage, prepared):
    return [
        sys.executable,
        "-B",
        "-m",
        "pathveil.downstream.pathgen",
        "--stage",
        stage,
        "--experiment_dir",
        str(run),
        "--date",
        prepared["date"],
        "--target_vases",
        source,
        "--meta_test_as_num",
        str(len(prepared["sources"])),
        "--as_type",
        "devoid",
        "--num_thread",
        "1",
        "--k",
        "200",
        "--oppNum",
        "200",
        "--pg_local_knowledge_mode",
        "augmentation_only",
        "--train-extra-data-mode",
        "augmentation_only",
        "--support_data_dir",
        str(run / "path_augmentation_output"),
        "--graph_selected_neighbors_csv",
        str(run / "neighbor_selection/reviewed_selected_neighbors.csv"),
        "--graph_recursive_anchors_csv",
        str(run / "neighbor_selection/recursive_secondhop_anchors.csv"),
    ]


def environment(prepared):
    data = read_json(prepared["data_manifest"])
    env = dict(
        os.environ,
        PATHVEIL_DATA_MANIFEST=prepared["data_manifest"],
        PATHVEIL_PROJECT_ROOT=str(PROJECT),
        ASINFER_FILES_ROOT=data["source_data_root"],
        PATHVEIL_GRAPH_DIR=data["graph_dir"],
        PATHVEIL_ASRANK_LINKS=data["public_files"]["asrank_links"],
        PYTHONHASHSEED=str(prepared["configuration"]["hash_seed"]),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(PROJECT / "src"),
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(
            sha256(source) == sha256(target), "Isolated input changed: " + str(target)
        )
    else:
        shutil.copyfile(source, target)


def isolated_run(run, source, prepared):
    subrun = run / "control/pg_precompute/workers" / ("as" + source)
    subrun.mkdir(parents=True, exist_ok=True)
    snapshot = dict(prepared)
    snapshot["prepared_files"] = dict(prepared["prepared_files"])
    for suffix in (".csv", ".json"):
        source_path = (
            run
            / "path_augmentation_output/train"
            / (source + "_augmented_paths" + suffix)
        )
        target = subrun / "path_augmentation_output/train" / source_path.name
        copy_verified(source_path, target)
        snapshot["prepared_files"][str(target)] = sha256(target)
    for name in ("reviewed_selected_neighbors.csv", "recursive_secondhop_anchors.csv"):
        target = subrun / "neighbor_selection" / name
        copy_verified(run / "neighbor_selection" / name, target)
        snapshot["prepared_files"][str(target)] = sha256(target)
    snapshot.update(
        parallel_precompute_source=source,
        parent_run=str(run),
        parent_prepared_sha256=sha256(run / "PREPARED.json"),
    )
    path = subrun / "PREPARED.json"
    if path.exists():
        require(read_json(path) == snapshot, "Isolated worker inputs changed")
    else:
        write_json(path, snapshot)
    return subrun


def verify_completion(run, source, stage, prepared, published_output=None):
    output, marker = stage_paths(run, source, stage)
    if not output.is_file() and published_output is not None:
        output = published_output
    require(
        marker.is_file() and output.is_file(),
        "PG stage did not finish: " + stage + "/" + source,
    )
    receipt = read_json(marker)
    require(
        receipt["source"] == source
        and receipt["mode"] == stage
        and receipt["admitted_pool_manifest_sha256"]
        == prepared["admitted_pool_manifest_sha256"]
        and receipt["sha256"] == sha256(output),
        "PG completion identity changed",
    )
    if stage == "test_inputs":
        require(
            receipt["dispatch_version"] == 1 and receipt["date"] == prepared["date"],
            "Missing original graph dispatch",
        )
    return output, marker, receipt


def merge_report(source, target, asn):
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if row["vantage"] == asn]
    require(fields and len(rows) == 1, "Expected exactly one PG report row per AS")
    existing = []
    if target.exists():
        with target.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == fields, "PG report schema changed")
            existing = list(reader)
    old = [row for row in existing if row["vantage"] == asn]
    if old:
        require(old == rows, "Conflicting existing PG report row")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".publishing")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + rows)
    temporary.replace(target)


def publish(run, subrun, source, stage, prepared, actual_command):
    target, marker = stage_paths(run, source, stage)
    # A crash after the atomic move can leave the original receipt in the
    # worker directory. Recover only from its matching published CSV hash.
    output, producer_marker, receipt = verify_completion(
        subrun, source, stage, prepared, target
    )
    directory = run / "control/pg_precompute"
    with (directory / ".publish.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merge_report(
            report_path(subrun, stage),
            report_path(run, stage),
            source,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha256(target) == receipt["sha256"], "Existing PG input differs")
        else:
            # Same filesystem: keep one large CSV and archive its producer proof.
            output.replace(target)
        archive = directory / "producers" / (stage + "_" + source + ".json")
        audits = {}
        for path in (subrun / "control/consumer_audit").glob(
            stage + "_" + source + "_*.json"
        ):
            destination = run / "control/consumer_audit" / path.name
            copy_verified(path, destination)
            audits[str(destination)] = sha256(destination)
        write_json(
            archive,
            dict(
                actual_command=actual_command,
                producer_receipt=receipt,
                producer_receipt_sha256=sha256(producer_marker),
                published_csv=str(target),
                published_csv_sha256=sha256(target),
                consumer_audits_sha256=audits,
                parent_prepared_sha256=sha256(run / "PREPARED.json"),
            ),
        )
        receipt = dict(
            receipt,
            output=str(target),
            parallel_producer=str(archive),
            parallel_producer_sha256=sha256(archive),
        )
        write_json(marker, receipt)
        _adopt_existing(
            run,
            stage + "_" + source,
            command(run, source, stage, prepared),
            [target, marker],
            sha256(run / "PREPARED.json"),
        )


def cached(run, source, stage, prepared):
    output, completion = stage_paths(run, source, stage)
    marker = run / "control/stages" / (stage + "_" + source + ".json")
    if marker.exists():
        _execute(
            run,
            stage + "_" + source,
            command(run, source, stage, prepared),
            [output, completion],
            {},
            sha256(run / "PREPARED.json"),
        )
        return True
    if completion.exists():
        verify_completion(run, source, stage, prepared)
        _adopt_existing(
            run,
            stage + "_" + source,
            command(run, source, stage, prepared),
            [output, completion],
            sha256(run / "PREPARED.json"),
        )
        return True
    return False


def gpu_groups():
    groups = set()
    for status_file in (PROJECT / "runs").glob("*/unified255/*/status.json"):
        try:
            status = read_json(status_file)
            pid = int(status["pid"])
            if status["stage"] not in ("downstream", "pg_precompute", "waiting_gpu"):
                continue
            proc = Path("/proc", str(pid))
            text = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if (
                str(status_file.parent) in text
                and "run_registered_experiment.py" in text
                and os.getpgid(pid) == pid
            ):
                groups.add(pid)
        except (OSError, ValueError, KeyError):
            continue
    return groups


def prepare_all(run_dir):
    run = common.validate_run(run_dir)
    directory = run / "control/pg_precompute"
    request = read_json(directory / "REQUEST.json")
    prepared = common.verify_runtime_authorization(run)
    identity = read_json(run / "RUN_IDENTITY.json")
    require(
        identity["code_sha256"]
        == {
            str(p.relative_to(PROJECT)): sha256(p)
            for p in sorted((PROJECT / "src/pathveil").rglob("*.py"))
        },
        "Frozen numerical source changed",
    )
    require(
        request["date"] == prepared["date"] == identity["date"],
        "CPU-first date mismatch",
    )
    require(
        request["prepared_sha256"] == sha256(run / "PREPARED.json"),
        "CPU-first input identity changed",
    )
    workers = int(request.get("workers", 3))
    require(1 <= workers <= 3, "PG workers must be 1..3")
    expected = int(request.get("worker_memory_gib", 18)) * GIB
    reserve = int(request.get("min_available_gib", 12)) * GIB
    require(
        expected >= 18 * GIB and reserve >= 12 * GIB, "PG memory budget was reduced"
    )
    coordinator = WorkerMemoryCoordinator(PROJECT / "runs")
    env = environment(prepared)
    sources = list(prepared["sources"])
    active = {}
    with (run / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(stage, **values):
            write_json(
                run / "status.json",
                dict(
                    date=prepared["date"],
                    pid=os.getpid(),
                    run_dir=str(run),
                    stage=stage,
                    status="FAILED" if stage == "failed" else "running",
                    updated_utc=utc_now(),
                    **values,
                ),
            )

        status("pg_precompute")
        try:
            previous = directory / "COMPLETED.json"
            if previous.exists():
                proof = read_json(previous)
                require(
                    proof["status"] == "PASS"
                    and proof["prepared_sha256"] == sha256(run / "PREPARED.json"),
                    "Completed CPU-first identity changed",
                )
                common.verify_files(proof["outputs_sha256"])
            else:
                require(
                    not any(
                        (run / "control/gmpi").glob("*/seed*/training_manifest.json")
                    ),
                    "CPU-first preparation must precede model fitting",
                )
            completed = set()
            for source in sources:
                for stage in STAGES:
                    if cached(run, source, stage, prepared):
                        completed.add((source, stage))
                        continue
                    subrun = directory / "workers" / ("as" + source)
                    if stage_paths(subrun, source, stage)[1].exists():
                        publish(
                            run,
                            subrun,
                            source,
                            stage,
                            prepared,
                            command(subrun, source, stage, prepared),
                        )
                        completed.add((source, stage))
            while len(completed) < 2 * len(sources):
                for key, task in list(active.items()):
                    code = task["process"].poll()
                    if code is None:
                        continue
                    coordinator.release(task["process"])
                    task["log"].close()
                    require(code == 0, "Parallel PG worker failed: " + str(key))
                    publish(
                        run, task["subrun"], key[0], key[1], prepared, task["command"]
                    )
                    completed.add(key)
                    del active[key]
                busy_sources = {key[0] for key in active}
                pending = [
                    (s, stage)
                    for s in sources
                    for stage in STAGES
                    if (s, stage) not in completed
                    and (s, stage) not in active
                    and s not in busy_sources
                    and (
                        stage == "training_inputs"
                        or (s, "training_inputs") in completed
                    )
                ]
                for source, stage in pending:
                    if len(active) >= workers:
                        break
                    subrun = isolated_run(run, source, prepared)
                    cmd = command(subrun, source, stage, prepared)
                    log_path = subrun / "logs" / (stage + ".log")
                    log_path.parent.mkdir(exist_ok=True)
                    log = log_path.open("ab")
                    process = coordinator.spawn(
                        lambda: subprocess.Popen(
                            cmd,
                            cwd=PROJECT,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        ),
                        expected,
                        reserve,
                    )
                    if process is None:
                        log.close()
                        break
                    active[source, stage] = dict(
                        process=process, log=log, subrun=subrun, command=cmd
                    )
                write_json(
                    directory / "STATUS.json",
                    dict(
                        status="running",
                        updated_utc=utc_now(),
                        completed_stages=len(completed),
                        total_stages=2 * len(sources),
                        completed_test_sources=sum(
                            (s, "test_inputs") in completed for s in sources
                        ),
                        active=[
                            dict(source=s, stage=stage, pid=t["process"].pid)
                            for (s, stage), t in active.items()
                        ],
                        requested_workers=workers,
                        shared_memory=coordinator.snapshot(),
                    ),
                )
                time.sleep(2)
            # Freeze every completed candidate CSV and original dispatch receipt
            # before the first new GPU model is fitted.
            outputs = {
                str(path): sha256(path)
                for s in sources
                for stage in STAGES
                for path in stage_paths(run, s, stage)
            }
            outputs.update(
                {
                    str(report_path(run, stage)): sha256(report_path(run, stage))
                    for stage in STAGES
                }
            )
            write_json(
                directory / "COMPLETED.json",
                dict(
                    status="PASS",
                    date=prepared["date"],
                    sources=sources,
                    prepared_sha256=sha256(run / "PREPARED.json"),
                    outputs_sha256=outputs,
                    all_candidates_precede_GPU_training=True,
                    original_numerical_API="pathveil.downstream.pathgen",
                    updated_utc=utc_now(),
                ),
            )
            write_json(
                directory / "STATUS.json",
                dict(
                    status="PASS",
                    completed_stages=len(completed),
                    total_stages=len(completed),
                    completed_test_sources=len(sources),
                    active=[],
                    updated_utc=utc_now(),
                ),
            )
            while not gpu_idle(12288, allowed_groups=gpu_groups()):
                status("waiting_gpu", candidates_complete=True)
                time.sleep(5)
        except BaseException as error:
            status("failed", error=repr(error))
            raise
        finally:
            for task in active.values():
                if task["process"].poll() is None:
                    task["process"].terminate()
                    task["process"].wait()
                coordinator.release(task["process"])
                task["log"].close()
