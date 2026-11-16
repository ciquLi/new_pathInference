#!/usr/bin/env python3
"""Two-CPU independent ablation queue; no dependency on PathRadar finishing."""
from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pathInference.research import ablation as experiment
from pathInference.experiment.contracts import freeze, require, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json

BATCH = experiment.BATCH
STAGES = ("prepare", "pg_train", "pg_test", "train", "annotate", "evaluate", "rescore", "metrics")
GMPI_STAGES = ("train", "annotate", "evaluate", "rescore")


def stage_command(key):
    if key[3] in GMPI_STAGES:
        # GMPI installs a process-lifetime audit hook scoped to T-PG-B.
        # Publish the enclosing research receipt in the dispatcher process.
        return [sys.executable, "-B", "-u", "-c",
                "import sys; from pathInference.research.ablation import gmpi_stage; gmpi_stage(*sys.argv[1:])",
                *key]
    return [sys.executable, "-B", "-u", "-m", "pathInference.research.ablation",
            "--date", key[0], "--variant", key[1], "--source", key[2], "--stage", key[3]]


def recover_gmpi_task(key):
    """Register only a hash-verified numerical receipt; never rerun its stage."""
    if key[3] not in GMPI_STAGES or task_path(*key).exists():
        return False
    from pathInference.experiment import worker
    run = experiment.run_dir(*key[:2])
    receipt = worker.receipt_path(run, key[3], key[2])
    if not receipt.is_file():
        return False
    worker.verify(run, key[3], key[2])
    freeze(task_path(*key), dict(
        status="PASS", date=key[0], variant=key[1], source=key[2], stage=key[3],
        code_sha256=sha256(experiment.__file__), config_sha256=sha256(experiment.CONFIG),
        outputs_sha256={str(receipt): sha256(receipt)},
        registration="dispatcher verified numerical receipt; no numerical work repeated",
        elapsed_seconds=None, peak_rss_kib=None, completed_utc=utc_now()))
    return True


def stage_environment(env, run, stage):
    # The GMPI adapter opens final labels only for guards whose root equals
    # its T-PG-B group after checkpoint verification. Candidate workers use
    # the enclosing run because they publish source and construction records.
    guard_root = run / "T-PG-B" if stage in ("train", "annotate", "evaluate", "rescore") else run
    return dict(env, PATHVEIL_STAGE_GUARD_ROOT=str(guard_root))


class AdoptedProcess:
    """Observe an existing worker without interrupting or duplicating it."""
    def __init__(self, item):
        self.pid = item["pid"]
        self.started = item["proc_start_ticks"]
        self.key = tuple(item[k] for k in ("date", "variant", "source", "stage"))

    def poll(self):
        try:
            fields = Path("/proc") .joinpath(str(self.pid), "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            fields = []
        if fields and fields[19] == self.started and fields[0] != "Z":
            return None
        return 0 if task_path(*self.key).is_file() else 1


def stages(variant):
    return ("prepare", "pg_test", "metrics") if variant == "current" else STAGES


def task_path(date, variant, source, stage):
    return experiment.run_dir(date, variant) / "control/research_tasks" / (stage + "_" + source + ".json")


def verify_task(path):
    value = read_json(path)
    require(value["status"] == "PASS" and value["config_sha256"] == sha256(experiment.CONFIG)
            and value["code_sha256"] == sha256(experiment.__file__), "Task code/config identity differs")
    verify_files(value["outputs_sha256"])


def source_done(date, variant, source):
    return experiment.run_dir(date, variant) / "control/source_done" / (source + ".json")


def finish_source(date, variant, source):
    """Publish a compact durable receipt before pruning only reproducible copies."""
    run = experiment.run_dir(date, variant)
    done = source_done(date, variant, source)
    if done.exists():
        value = read_json(done)
        require(value["config_sha256"] == sha256(experiment.CONFIG)
                and value["code_sha256"] == sha256(experiment.__file__), "Completed source identity differs")
        verify_files(value["outputs_sha256"])
        return
    outputs = [run / "diagnostics" / ("as" + source) / "METRICS.json",
               run / "diagnostics" / ("as" + source) / "candidate_oracle.csv",
               run / "augmentation" / ("as" + source) / "ADMISSION.json",
               run / "augmentation" / ("as" + source) / "paths.csv",
               run / "augmentation" / ("as" + source) / "paths.json"]
    if variant == "current":
        original = experiment.asset(experiment.baseline(date) / "T-PG-B/independent_evaluation" / ("source_" + source + ".json"))
        score = read_json(original)
        result = run / "T-PG-B/independent_evaluation" / ("source_" + source + ".json")
        freeze(result, dict(score, reused_frozen_result=dict(path=str(original), sha256=sha256(original))))
        outputs.append(result)
    else:
        from pathInference.experiment import train
        selection = train.verify_selection(source, run / "T-PG-B")
        outputs += [Path(selection["chosen_model"]), Path(selection["chosen_training_manifest"]),
                    run / "T-PG-B/independent_evaluation" / ("source_" + source + ".json")]
    freeze(done, dict(status="PASS", date=date, variant=variant, source=source,
                      config_sha256=sha256(experiment.CONFIG), code_sha256=sha256(experiment.__file__),
                      outputs_sha256={str(p): sha256(p) for p in outputs}, completed_utc=utc_now()))
    cleanup = []
    child = run / "T-PG-B/pg_inputs" / ("as" + source)
    for relative in ("model_input_data_PG", "runtime"):
        cleanup.append(child / relative)
    if variant != "current":
        p0 = run / "T-PG-B/gmpi/p0" / ("as" + source)
        evaluation = run / "T-PG-B/evaluations" / ("as" + source) / "evaluation_run"
        for base in (p0, evaluation):
            for relative in ("model_input_data_PG", "pathgen_results", "runtime"):
                cleanup.append(base / relative)
    removed = []
    for directory in cleanup:
        require(run.resolve() in directory.resolve().parents and not directory.is_symlink(), "Cleanup escaped experiment")
        if directory.is_dir():
            shutil.rmtree(directory)
            removed.append(str(directory))
    freeze(run / "control/source_cleanup" / (source + ".json"), dict(
        completed_source_receipt_sha256=sha256(done), removed_directories=removed,
        preserved="results, models, training manifests, augmentation, identities, pair oracle"))


def publish(config, active, failures):
    now = datetime.now(timezone(timedelta(hours=8)))
    lines = ["# 生成器与递归策略消融队列", "", "更新时间（北京时间）：" + now.strftime("%Y-%m-%d %H:%M:%S"),
             "", "与 PathRadar 并行；CPU核6、7，最多两个CPU任务、一个GMPI GPU任务。",
             "", "| 日期 | 配置 | 已训练AS | 已完成评测AS | 当前任务 |", "|---|---|---:|---:|---|"]
    rows = []
    for date in config["dates"]:
        sources = read_json(experiment.baseline(date) / "PROTOCOL.json")["sources"]
        for variant in config["variants"]:
            completed = sum(source_done(date, variant, s).exists() for s in sources)
            running = ["AS" + k[2] + " " + k[3] + " (PID " + str(v[0].pid) + ")"
                       for k, v in active.items() if k[:2] == (date, variant)]
            failed = ["AS" + k[2] + " " + k[3] + " 失败，见日志" for k in failures if k[:2] == (date, variant)]
            state = "; ".join(running + failed) or ("已完成" if completed == len(sources) else "排队")
            run = experiment.run_dir(date, variant)
            trained = len(list((run / "control/research_tasks").glob("train_*.json")))
            trained_text = "复用原模型" if variant == "current" else f"{trained}/{len(sources)}"
            lines.append(f"| {date} | {variant} | {trained_text} | {completed}/{len(sources)} | {state} |")
            for source in sources:
                metric = run / "diagnostics" / ("as" + source) / "METRICS.json"
                score = run / "T-PG-B/independent_evaluation" / ("source_" + source + ".json")
                if metric.is_file() and score.is_file():
                    m, s = read_json(metric), read_json(score)
                    rows.append(dict(date=date, variant=variant, source=source, rho=s["rho"],
                                     candidate_coverage=m["candidate_coverage"], exact_available=m["exact_available"],
                                     oracle_jaccard=m["oracle_jaccard"], final_jaccard=s["jaccard_accuracy"],
                                     ordered_exact_accuracy=s["ordered_exact_accuracy"]))
    lines += ["", "旧生成器只与current比较；无递归、全部递归只与current比较。",
              "current最终准确率复用冻结结果；已清理的候选只为补齐oracle诊断而再生成。",
              "详细日志：各日期配置下 logs/；队列状态：STATUS.json；结果：RESULTS.csv。", ""]
    if failures:
        lines += ["失败详情：", ""] + [str(k) + "：" + str(v) for k, v in failures.items()]
    temporary = BATCH / "PROGRESS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(BATCH / "PROGRESS.md")
    write_json(BATCH / "STATUS.json", dict(
        status="RUNNING" if active else "QUEUED", updated_utc=utc_now(),
        active=[dict(date=k[0], variant=k[1], source=k[2], stage=k[3], pid=v[0].pid,
                     log=str(v[2]), elapsed_seconds=time.monotonic() - v[3]) for k, v in active.items()],
        failures={"/".join(k): v for k, v in failures.items()}))
    if rows:
        with (BATCH / "RESULTS.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    slot = now.strftime("%Y-%m-%d %H:") + str(now.minute // 15)
    slot_file = BATCH / ".bulletin_slot"
    if not slot_file.exists() or slot_file.read_text() != slot:
        with (BATCH / "BULLETINS.md").open("a") as stream:
            stream.write("\n" + "\n".join(lines) + "\n")
        slot_file.write_text(slot)


def main():
    BATCH.mkdir(parents=True, exist_ok=True)
    config = read_json(experiment.CONFIG)
    os.sched_setaffinity(0, config["cpu_affinity"])
    lock = (BATCH / ".dispatch.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze(BATCH / "CONFIG.json", config)
    queue_identity = dict(
        config_sha256=sha256(experiment.CONFIG), numerical_code_sha256=sha256(experiment.__file__),
        dispatcher_sha256=sha256(__file__))
    identity_file = BATCH / "QUEUE_IDENTITY.json"
    if identity_file.exists():
        previous = read_json(identity_file)
        require(previous["config_sha256"] == queue_identity["config_sha256"]
                and previous["numerical_code_sha256"] == queue_identity["numerical_code_sha256"],
                "A dispatch update must preserve numerical/config identities")
        if previous != queue_identity:
            freeze(BATCH / ("DISPATCH_AMENDMENT_" + sha256(__file__)[:16] + ".json"), dict(
                previous_queue_identity_sha256=sha256(identity_file), current=queue_identity,
                scope="dispatch guard boundaries, worker adoption and verified receipt publication only; numerical code unchanged"))
    else:
        freeze(identity_file, queue_identity)
    write_json(BATCH / "DISPATCHER.json", dict(pid=os.getpid(), started_utc=utc_now(), cpu_affinity=sorted(os.sched_getaffinity(0))))
    production = read_json(ROOT / config["code_manifest"])["source_files_sha256"]
    verify_files({str(ROOT / p): h for p, h in production.items() if p.startswith("src/")})
    env = dict(os.environ, PATHVEIL_PROJECT_ROOT=str(ROOT), PYTHONPATH=str(ROOT / "src"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(config["seed"]),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PATHVEIL_PATS_PATHGEN_TRAIN_THREADS="1", PATHVEIL_PATS_PATHGEN_TEST_THREADS="1")
    # Historical source/resume bypasses are never inherited by research jobs.
    env.pop("PATHVEIL_RESUME_RUN", None)
    active, failures = {}, {}
    handover = BATCH / "HANDOVER.json"
    if handover.exists():
        for item in read_json(handover)["workers"]:
            process = AdoptedProcess(item)
            if process.poll() is None:
                key = process.key
                log = Path(item["log"])
                active[key] = (process, log.open("ab"), log, time.monotonic() - item["elapsed_seconds"])
    try:
        for date in config["dates"]:
            sources = read_json(experiment.baseline(date) / "PROTOCOL.json")["sources"]
            sources = sorted(sources, key=lambda s: config["source_order"].index(s))
            while True:
                for key, (process, stream, log, start) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    del active[key]
                    if recover_gmpi_task(key):
                        code = 0
                    if code:
                        failures[key] = dict(exit_code=code, log=str(log))
                    else:
                        verify_task(task_path(*key))
                        if key[3] == "metrics":
                            finish_source(*key[:3])
                pending = []
                for source in sources:
                    for variant in config["variants"]:
                        if source_done(date, variant, source).exists():
                            continue
                        if any(k[:3] == (date, variant, source) for k in failures):
                            continue
                        for stage in stages(variant):
                            key = (date, variant, source, stage)
                            marker = task_path(*key)
                            if key not in active:
                                recover_gmpi_task(key)
                            if marker.exists():
                                verify_task(marker)
                                if stage == "metrics":
                                    finish_source(date, variant, source)
                                continue
                            pending.append(key)
                            break
                for key in pending:
                    if len(active) >= config["cpu_slots"]:
                        break
                    if key in active:
                        continue
                    if key[3] == "train" and any(k[3] in ("train", "evaluate") for k in active):
                        continue
                    if key[3] == "evaluate" and any(k[3] in ("train", "evaluate") for k in active):
                        continue
                    if shutil.disk_usage(ROOT).free < config["minimum_free_disk_gib"] * 1024**3:
                        raise RuntimeError("Insufficient disk space; preserve caches and resume after freeing space")
                    run = experiment.run_dir(*key[:2])
                    logs = run / "logs"
                    logs.mkdir(parents=True, exist_ok=True)
                    log = logs / (key[3] + "_" + key[2] + ".log")
                    stream = log.open("ab")
                    stage_env = stage_environment(env, run, key[3])
                    command = stage_command(key)
                    process = subprocess.Popen(command, cwd=ROOT, env=stage_env, stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = (process, stream, log, time.monotonic())
                publish(config, active, failures)
                if not pending and not active:
                    break
                time.sleep(10)
        publish(config, active, failures)
        state = read_json(BATCH / "STATUS.json")
        state["status"] = "FAILED" if failures else "COMPLETED"
        write_json(BATCH / "STATUS.json", state)
        return 1 if failures else 0
    except BaseException:
        write_json(BATCH / "ERROR.json", dict(error=traceback.format_exc(), updated_utc=utc_now()))
        raise
    finally:
        for process, stream, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
