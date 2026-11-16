#!/usr/bin/env python3
"""Low-resource queue; all upstream artifacts and candidate bytes stay frozen."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/pathinference-confidence-matplotlib"
sys.path.insert(0, str(ROOT / "src"))
from pathInference.research import candidate_confidence as experiment
from pathInference.experiment.contracts import freeze, require, verify_files
from pathInference.utils import read_json, sha256, write_json, utc_now


def paths(source, stage):
    if stage == "snapshot":
        return [experiment.RUN / "SNAPSHOT.json", experiment.RUN / "PROTOCOL.json"]
    directory = experiment.source_dir(source)
    names = {"train": ["MODEL_FROZEN.json", "model.txt"],
             "predict": ["PREDICTIONS_FROZEN.json", "ranked_predictions.csv"],
             "evaluate": ["RESULTS.json"]}
    return [directory / name for name in names[stage]]


def receipt_path(source, stage):
    return experiment.RUN / "tasks" / ((source or "all") + "_" + stage + ".json")


def checked(source, stage):
    file = receipt_path(source, stage)
    if not file.is_file():
        return False
    record = read_json(file)
    require(record["configuration_sha256"] == sha256(experiment.CONFIG), "Stage config changed")
    require(record["code_sha256"] == code_identity(), "Stage code changed")
    verify_files(record["outputs_sha256"])
    return True


def code_identity():
    files = [Path(__file__).resolve(), Path(experiment.__file__).resolve()]
    return {str(file): sha256(file) for file in files}


def report(active=None, error=None):
    run, cfg = experiment.RUN, experiment.config()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["# 20260201：冻结候选后的路径置信度评估", "", "更新时间：" + now, "",
        "范围：全部 10 个 AS；最多 2 CPU，无 GPU。直接复用已生成的候选和训练特征，不重新生成路径。",
        "目标函数对照：LightGBM 平方误差回归（已有模型）与查询内 soft-label 排序（新模型）。",
        "先冻结全部模型，再预测全部源，最后统一评测；训练和选轮次不读 test 路径。", "",
        "| AS | 输入审计 | 排序模型 | 候选预测 | 最终评测 | 实际状态 |",
        "|---|---|---|---|---|---|"]
    rows = []
    for source in cfg["sources"]:
        directory = run / ("as" + source)
        state = read_json(directory / "STATUS.json") if (directory / "STATUS.json").is_file() else {}
        cells = []
        for stage in ("snapshot", "train", "predict", "evaluate"):
            done = receipt_path(None if stage == "snapshot" else source, stage).is_file()
            cells.append("完成" if done else "运行中" if active == (source, stage)
                         or (stage == "snapshot" and active == (None, stage)) else "等待")
        detail = state.get("stage", "等待")
        if state.get("committed_queries") is not None:
            detail += "；已持久化 " + str(state["committed_queries"]) + " 个查询"
        if state.get("candidate_bytes"):
            detail += "；候选读取 {:.1%}".format(state["candidate_byte_offset"] / state["candidate_bytes"])
        if state.get("best_iteration") is not None:
            detail += "；最佳轮次 " + str(state["best_iteration"])
        lines.append("| " + " | ".join([source, *cells, detail]) + " |")
        if (directory / "RESULTS.json").is_file():
            rows.append(read_json(directory / "RESULTS.json"))
    if active:
        lines += ["", "当前阶段：" + str(active)]
    if error:
        lines += ["", "已停止，错误：" + error]
    lines += ["", "输入/候选身份：SNAPSHOT.json；运行日志：logs/；逐 AS 模型、预测、计时：as<ASN>/。",
              "置信度分数只用于排序，尚未校准为真实概率。"]
    text = "\n".join(lines) + "\n"
    temporary = run / "PROGRESS.md.tmp"
    temporary.write_text(text); temporary.replace(run / "PROGRESS.md")
    write_json(run / "STATUS.json", dict(updated_utc=utc_now(), active=active, error=error,
        completed_sources=len(rows), expected_sources=len(cfg["sources"]), dispatcher_pid=os.getpid()))
    if rows:
        write_json(run / "RESULTS.json", rows)
        fields = ["source", "rho", "query_count", "gmpi_jaccard", "regression_jaccard", "jaccard_accuracy",
                  "objective_change_jaccard", "gmpi_exact", "regression_exact", "ordered_exact_accuracy", "oracle_jaccard"]
        with (run / "RESULTS.csv").open("w", newline="") as stream:
            writer = __import__("csv").DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    if len(rows) == len(cfg["sources"]):
        metrics = ["gmpi_jaccard", "regression_jaccard", "jaccard_accuracy", "gmpi_exact",
                   "regression_exact", "ordered_exact_accuracy", "oracle_jaccard"]
        total = sum(row["query_count"] for row in rows)
        micro = {key: sum(row[key] * row["query_count"] for row in rows) / total for key in metrics}
        write_json(run / "SUMMARY.json", dict(query_count=total, micro=micro, sources=len(rows),
            interpretation="Only LightGBM objective is controlled; GMPI is historical context, not an objective-only comparison",
            historical_test_results_already_known=True))
        detail = ["", "| AS | GMPI Jaccard | 回归 Jaccard | 排序 Jaccard | 排序减回归 |",
                  "|---|---|---|---|---|"]
        for row in rows:
            detail.append(f"| {row['source']} | {row['gmpi_jaccard']:.6f} | {row['regression_jaccard']:.6f} | "
                          f"{row['jaccard_accuracy']:.6f} | {row['objective_change_jaccard']:+.6f} |")
        detail += ["", "全部查询加权：" + json.dumps(micro, ensure_ascii=False), "",
            "最终参考路径仅在全部模型和预测固定后用于评测。既往测试结果已知，本轮属于探索性对照。"]
        (run / "REPORT.md").write_text(text + "\n".join(detail) + "\n")


def worker(source, stage):
    cfg = experiment.config()
    os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
    if source:
        experiment.source_dir(source).mkdir(parents=True, exist_ok=True)
    started, cpu = time.monotonic(), time.process_time()
    try:
        if stage == "snapshot":
            experiment.snapshot()
        else:
            getattr(experiment, stage)(source)
    finally:
        directory = experiment.source_dir(source) if source else experiment.RUN
        write_json(directory / ("TIMING_" + stage + ".json"), dict(
            source=source, stage=stage, finished_utc=utc_now(), wall_seconds=time.monotonic()-started,
            cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def run_stage(source, stage):
    if checked(source, stage):
        return
    run = experiment.RUN
    active = (source, stage)
    name = (source or "all") + "_" + stage
    report(active=active)
    arguments = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--stage", stage]
    if source:
        arguments += ["--source", source]
    started = time.monotonic()
    with (run / "logs" / (name + ".log")).open("a") as stream:
        child = subprocess.Popen(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        write_json(run / "ACTIVE.json", dict(source=source, stage=stage, worker_pid=child.pid, started_utc=utc_now()))
        while child.poll() is None:
            report(active=active)
            time.sleep(10)
    require(child.returncode == 0, name + " failed; exit=" + str(child.returncode))
    outputs = {str(file): sha256(file) for file in paths(source, stage)}
    freeze(receipt_path(source, stage), dict(source=source, stage=stage, status="PASS",
        configuration_sha256=sha256(experiment.CONFIG), code_sha256=code_identity(), outputs_sha256=outputs,
        wall_seconds=time.monotonic()-started, completed_utc=utc_now()))
    report()


def dispatch():
    run, cfg = experiment.RUN, experiment.config()
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True); (run / "tasks").mkdir(exist_ok=True)
    lock = (run / ".dispatch.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run_stage(None, "snapshot")
        for stage in ("train", "predict", "evaluate"):
            for source in cfg["sources"]:
                run_stage(source, stage)
            if stage in ("train", "predict"):
                barrier = "ALL_MODELS_FROZEN.json" if stage == "train" else "ALL_PREDICTIONS_FROZEN.json"
                freeze(run / barrier, dict(sources=cfg["sources"], snapshot_sha256=sha256(run / "SNAPSHOT.json"),
                    receipts_sha256={str(receipt_path(source, stage)): sha256(receipt_path(source, stage))
                                     for source in cfg["sources"]}))
        report()
    except BaseException as error:
        report(error=str(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--stage", choices=("snapshot", "train", "predict", "evaluate"))
    args = parser.parse_args()
    if args.stage:
        worker(args.source, args.stage)
    else:
        dispatch()
