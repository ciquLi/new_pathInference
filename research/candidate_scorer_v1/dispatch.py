#!/usr/bin/env python3
"""Resumable sequential pilot, at most two CPU cores and no GPU."""
from __future__ import annotations

import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pathInference.research import candidate_scorer as experiment
from pathInference.experiment.contracts import freeze, require, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json

STAGES = ("prepare", "train", "generate", "predict", "evaluate")


def report(active=None, error=None):
    run = experiment.RUN
    lines = ["# 候选路径独立打分试验", "", "更新时间：" + utc_now(), "",
             "20260201；AS 顺序执行；最多 2 CPU 线程；不使用 GPU。", "",
             "| AS | prepare | train | generate | predict | evaluate | 当前状态 |",
             "|---|---|---|---|---|---|---|"]
    results = []
    for source in experiment.config()["sources"]:
        state = run / ("as" + source) / "STATUS.json"
        text = str(read_json(state)) if state.is_file() else "等待"
        cells = []
        for stage in STAGES:
            file = run / "tasks" / (source + "_" + stage + ".json")
            cells.append("完成" if file.is_file() else "运行中" if active == (source, stage) else "等待")
        lines.append("| " + " | ".join([source, *cells, text.replace("|", "/")]) + " |")
        result = run / ("as" + source) / "RESULTS.json"
        if result.is_file():
            results.append(read_json(result))
    if error:
        lines += ["", "失败后已停止：" + error]
    if results:
        lines += ["", "| AS | rho | GMPI Jaccard | 新模型 Jaccard | 变化 | 新模型完整匹配率 | 候选诊断通过 |",
                  "|---|---|---|---|---|---|---|"]
        for r in results:
            lines.append(f"| {r['source']} | {r['rho']:.4f} | {r['baseline_gmpi_jaccard']:.6f} | "
                         f"{r['jaccard_accuracy']:.6f} | {r['jaccard_change']:+.6f} | "
                         f"{r['ordered_exact_accuracy']:.6f} | {r['diagnostic_comparison_passed']} |")
        write_json(run / "RESULTS.json", results)
    temporary = run / "PROGRESS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(run / "PROGRESS.md")
    write_json(run / "STATUS.json", dict(updated_utc=utc_now(), active=active, error=error,
                                         completed_sources=len(results), dispatcher_pid=os.getpid()))


def outputs(source, stage):
    run = experiment.RUN
    directory = run / ("as" + source)
    if stage == "prepare":
        files = [run / "IDENTITY.json", directory / "REUSE.json", directory / "VOCABULARY.json"]
    elif stage == "train":
        files = [directory / "MODEL_FROZEN.json", directory / "model.txt"]
    elif stage == "generate":
        file = run / "T-PG-B/pg_inputs" / ("as" + source) / "control/pathgen_completed" / ("test_inputs_" + source + ".json")
        files = [file, Path(read_json(file)["output"])]
    elif stage == "predict":
        files = [directory / "PREDICTIONS_FROZEN.json", directory / "pair_predictions.csv"]
    else:
        files = [directory / "RESULTS.json"]
    return {str(p): sha256(p) for p in files}


def main():
    run = experiment.RUN
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".dispatch.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This pilot is already running")
    cfg = experiment.config()
    os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PATHVEIL_PROJECT_ROOT=str(ROOT))
    (run / "logs").mkdir(exist_ok=True)
    (run / "tasks").mkdir(exist_ok=True)
    for source in cfg["sources"]:
        for stage in STAGES:
            key = (source, stage)
            receipt = run / "tasks" / (source + "_" + stage + ".json")
            if receipt.is_file():
                record = read_json(receipt)
                require(record["config_sha256"] == sha256(experiment.CONFIG), "Cached configuration changed")
                verify_files(record["outputs_sha256"])
                if (run / "IDENTITY.json").is_file():
                    verify_files(read_json(run / "IDENTITY.json")["code_sha256"])
                continue
            report(key)
            log = run / "logs" / (source + "_" + stage + ".log")
            command = [sys.executable, "-B", "-u", "-m", "pathInference.research.candidate_scorer",
                       "--source", source, "--stage", stage]
            started = time.monotonic()
            with log.open("a") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                write_json(run / "ACTIVE.json", dict(source=source, stage=stage, worker_pid=child.pid,
                                                      log=str(log), started_utc=utc_now()))
                while child.poll() is None:
                    report(key)
                    time.sleep(30)
            if child.returncode:
                error = f"AS{source} {stage} exit={child.returncode}; 日志：{log}"
                report(error=error)
                raise SystemExit(error)
            freeze(receipt, dict(status="PASS", source=source, stage=stage,
                                 config_sha256=sha256(experiment.CONFIG), outputs_sha256=outputs(source, stage),
                                 elapsed_seconds=time.monotonic() - started, completed_utc=utc_now()))
            report()
    report()


if __name__ == "__main__":
    main()
