#!/usr/bin/env python3
"""Continue the ten-source February queue with measured stage and fitting time."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone, timedelta
import fcntl
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pathInference.research import candidate_scorer as experiment
from pathInference.experiment.contracts import freeze, require, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json

HERE = Path(__file__).resolve().parent
SCOPE = HERE / "FULL_FEBRUARY.json"
STAGES = ("prepare", "train", "generate", "predict", "evaluate")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity():
    frozen = read_json(experiment.RUN / "IDENTITY.json")
    verify_files(frozen["code_sha256"])
    require(frozen["config_sha256"] == sha256(experiment.CONFIG), "Numerical configuration changed")
    cfg = read_json(SCOPE)
    baseline_proto = read_json(experiment.baseline() / "PROTOCOL.json")
    require(set(cfg["sources"]) == set(baseline_proto["sources"]) and len(cfg["sources"]) == 10,
            "Full scope differs from the ten baseline sources")
    require(cfg["threads"] == experiment.config()["threads"] == 2, "CPU budget differs")
    value = dict(scope_sha256=sha256(SCOPE), launcher_sha256=sha256(__file__),
                 publication_repair_sha256=sha256(HERE / "resume.py"),
                 original_identity_sha256=sha256(experiment.RUN / "IDENTITY.json"),
                 numerical_configuration_sha256=sha256(experiment.CONFIG),
                 scope=cfg, only_method_change="none; source scope expanded, execution time recorded")
    freeze(experiment.RUN / "FULL_FEBRUARY_IDENTITY.json", value)
    return cfg


def expanded_prepare():
    """Extend only the pilot source-membership check; frozen method stays identical."""
    tree = ast.parse(inspect.getsource(experiment.prepare))
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "require"
                and len(node.args) == 2 and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "Unregistered pilot source"):
            node.args[0] = ast.parse("source in FULL_SOURCES", mode="eval").body
            count += 1
    require(count == 1, "Unexpected preparation source-check layout")
    ast.fix_missing_locations(tree)
    namespace = dict(vars(experiment), FULL_SOURCES=read_json(SCOPE)["sources"])
    exec(compile(tree, str(Path(__file__).resolve()) + ":expanded_source_check", "exec"), namespace)
    return namespace["prepare"]


def instrument_training(timing):
    """Measure the fit itself separately from feature building and loading."""
    import lightgbm as lgb
    original_fit = lgb.train
    original_matrix = experiment.training_matrix
    def fit(*args, **kwargs):
        start, cpu = time.monotonic(), time.process_time()
        try:
            return original_fit(*args, **kwargs)
        finally:
            timing["lightgbm_fit_wall_seconds"] = time.monotonic() - start
            timing["lightgbm_fit_cpu_seconds"] = time.process_time() - cpu
    def matrix(source, name, *args, **kwargs):
        cached = (experiment.RUN / ("as" + source) / (name + "_features.json")).is_file()
        start, cpu = time.monotonic(), time.process_time()
        try:
            return original_matrix(source, name, *args, **kwargs)
        finally:
            timing[name + "_features_wall_seconds"] = time.monotonic() - start
            timing[name + "_features_cpu_seconds"] = time.process_time() - cpu
            timing[name + "_features_reused"] = cached
    lgb.train, experiment.training_matrix = fit, matrix


def stage(source, name):
    cfg = identity()
    require(source in cfg["sources"], "Unknown source")
    os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
    repair = load_module("candidate_scorer_repair", HERE / "resume.py")
    experiment.code_identity = lambda: read_json(experiment.RUN / "IDENTITY.json")["code_sha256"]
    experiment.prepare = expanded_prepare()
    timing = dict(source=source, stage=name, started_utc=utc_now(), measurement="process wall and CPU clocks")
    if name == "train":
        instrument_training(timing)
    started, cpu = time.monotonic(), time.process_time()
    try:
        repair.stage(source, name)
    finally:
        timing.update(wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, finished_utc=utc_now())
        write_json(experiment.RUN / ("as" + source) / ("TIMING_" + name + ".json"), timing)


def prior_timing(source):
    """Historical fits lack clocks; preserve bounds instead of inventing exact time."""
    values = {}
    for name in STAGES:
        path = experiment.RUN / ("as" + source) / ("TIMING_" + name + ".json")
        task = experiment.RUN / "tasks" / (source + "_" + name + ".json")
        if path.is_file():
            values[name] = read_json(path)
        elif task.is_file():
            receipt = read_json(task)
            values[name] = dict(wall_seconds=receipt["elapsed_seconds"], cpu_seconds=None,
                                measurement="historical dispatcher elapsed; includes polling delay up to 30 seconds")
    file = experiment.RUN / "logs" / (source + "_train.log")
    if "train" in values and "lightgbm_fit_wall_seconds" not in values["train"] and file.is_file():
        events = []
        for line in file.read_text().splitlines():
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict) and row.get("stage") in ("training", "model_frozen"):
                events.append(row)
        starts = [r for r in events if r["stage"] == "training"]
        ends = [r for r in events if r["stage"] == "model_frozen"]
        if starts and ends:
            values["train"]["fit_and_model_save_log_interval_seconds"] = (
                datetime.fromisoformat(ends[-1]["updated_utc"]) -
                datetime.fromisoformat(starts[-1]["updated_utc"])).total_seconds()
            values["train"]["lightgbm_fit_wall_seconds"] = None
    recovery = experiment.RUN / ("as" + source) / "PREDICTIONS_FROZEN.json"
    if ("predict" in values and recovery.is_file()
            and read_json(recovery).get("recovered_existing_complete_csv")):
        # The successful receipt measures only output recovery. Keep the
        # earlier prediction work visible rather than reporting recovery as inference.
        values["predict"]["publication_recovery_wall_seconds"] = values["predict"]["wall_seconds"]
        values["predict"]["wall_seconds"] = None
        values["predict"]["measurement"] = "full historical prediction stage unavailable; recovery reported separately"
        events = []
        log = experiment.RUN / "logs" / (source + "_predict.log")
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict) and row.get("stage") == "predicting":
                events.append(row)
        if len(events) >= 2:
            values["predict"]["prediction_logged_work_lower_bound_seconds"] = (
                datetime.fromisoformat(events[-1]["updated_utc"]) -
                datetime.fromisoformat(events[0]["updated_utc"])).total_seconds()
    return values


def report(waiting=False, active=None, error=None):
    run = experiment.RUN
    sources = read_json(SCOPE)["sources"]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# 20260201 全部 10 个 AS：独立候选打分实验", "", "更新时间（北京时间）：" + stamp,
             "", "最多 2 CPU，AS 顺序执行，无 GPU；模型与数据均按冻结身份复用。", "",
             "| AS | 准备 | 训练 | 候选生成 | 预测 | 评测 | 当前状态 |",
             "|---|---|---|---|---|---|---|"]
    results = []
    timings = {}
    for source in sources:
        state = run / ("as" + source) / "STATUS.json"
        detail = str(read_json(state)) if state.is_file() else "排队"
        cells = ["完成" if (run / "tasks" / (source + "_" + s + ".json")).is_file()
                 else "运行中" if active == (source, s) else "等待" for s in STAGES]
        lines.append("| " + " | ".join([source, *cells, detail.replace("|", "/")]) + " |")
        file = run / ("as" + source) / "RESULTS.json"
        if file.is_file():
            results.append(read_json(file))
        timings[source] = prior_timing(source)
    if waiting:
        lines += ["", "接续原三源队列，等待其完成；其当前工作继续复用，不重复启动计算。"]
    if error:
        lines += ["", "失败后已停止：" + error]
    if results:
        lines += ["", "| AS | GMPI Jaccard | 新模型 Jaccard | 变化（百分点） | GMPI 完整匹配率 | 新模型完整匹配率 | 候选诊断通过 |",
                  "|---|---|---|---|---|---|---|"]
        for r in results:
            lines.append(f"| {r['source']} | {r['baseline_gmpi_jaccard']:.6f} | {r['jaccard_accuracy']:.6f} | "
                         f"{100*r['jaccard_change']:+.3f} | {r['baseline_gmpi_exact']:.6f} | "
                         f"{r['ordered_exact_accuracy']:.6f} | {r['diagnostic_comparison_passed']} |")
    lines += ["", "| AS | 训练阶段总耗时（秒） | LightGBM 拟合（秒） | 候选生成（秒） | 预测（秒） |",
              "|---|---|---|---|---|"]
    def number(value):
        return f"{value:.3f}" if value is not None else "—"
    for source in sources:
        times = timings[source]
        tr = times.get("train", {})
        fit = tr.get("lightgbm_fit_wall_seconds")
        bound = tr.get("fit_and_model_save_log_interval_seconds")
        fit_text = number(fit) if fit is not None else f"≤{bound:.3f}（日志区间，含模型保存）" if bound is not None else "—"
        pred = times.get("predict", {})
        lower = pred.get("prediction_logged_work_lower_bound_seconds")
        pred_text = number(pred.get("wall_seconds")) if lower is None else f"≥{lower:.3f}（日志下界）"
        lines.append("| " + " | ".join([source, number(tr.get("wall_seconds")), fit_text,
                     number(times.get("generate", {}).get("wall_seconds")),
                     pred_text]) + " |")
    lines += ["", "训练阶段总耗时含加载证据、训练/验证特征计算、拟合、模型保存。历史阶段总耗时含调度轮询延迟；新阶段同时记录 CPU 时间与峰值内存。",
              "已完成的模型不为补计时而重训；历史拟合只能记录训练开始至模型保存的日志区间。"]
    path = run / ("PROGRESS_ALL.md" if waiting else "PROGRESS.md")
    temp = path.with_suffix(".md.full.tmp"); temp.write_text("\n".join(lines) + "\n"); temp.replace(path)
    write_json(run / "TIMINGS.json", timings)
    if not waiting:
        write_json(run / "RESULTS.json", results)
        write_json(run / "STATUS.json", dict(updated_utc=utc_now(), active=active, error=error,
                   completed_sources=len(results), expected_sources=10, dispatcher_pid=os.getpid()))
        if len(results) == 10:
            (run / "REPORT.md").write_text("\n".join(lines) + "\n")
            import csv
            with (run / "RESULTS.csv").open("w", newline="") as stream:
                keys = ["source", "rho", "query_count", "baseline_gmpi_jaccard", "jaccard_accuracy", "jaccard_change",
                        "baseline_gmpi_exact", "ordered_exact_accuracy", "oracle_jaccard", "diagnostic_comparison_passed"]
                writer = csv.DictWriter(stream, keys, extrasaction="ignore"); writer.writeheader(); writer.writerows(results)


def dispatch():
    run = experiment.RUN
    cfg = identity()
    os.sched_setaffinity(0, set(cfg["cpu_affinity"]))
    own = (run / ".full_february.lock").open("a+")
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = (run / ".dispatch.lock").open("a+")
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            report(waiting=True); time.sleep(30)
    driver = load_module("scorer_driver", HERE / "dispatch.py")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
               PATHVEIL_PROJECT_ROOT=str(ROOT))
    for source in cfg["sources"]:
        for name in STAGES:
            task = run / "tasks" / (source + "_" + name + ".json")
            if task.is_file():
                value = read_json(task)
                require(value["config_sha256"] == sha256(experiment.CONFIG), "Cached configuration differs")
                verify_files(value["outputs_sha256"])
                continue
            report(active=(source, name))
            started = time.monotonic()
            with (run / "logs" / (source + "_" + name + ".log")).open("a") as stream:
                child = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                          "--source", source, "--stage", name], cwd=ROOT, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT)
                write_json(run / "ACTIVE.json", dict(source=source, stage=name, worker_pid=child.pid,
                                                      started_utc=utc_now()))
                while child.poll() is None:
                    report(active=(source, name)); time.sleep(10)
            if child.returncode:
                error = f"AS{source} {name} exit={child.returncode}"
                report(error=error); raise SystemExit(error)
            freeze(task, dict(status="PASS", source=source, stage=name,
                              config_sha256=sha256(experiment.CONFIG), outputs_sha256=driver.outputs(source, name),
                              elapsed_seconds=time.monotonic()-started, completed_utc=utc_now(),
                              full_scope_sha256=sha256(SCOPE), timing_file=str(run / ("as"+source) / ("TIMING_"+name+".json"))))
            report()
    report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.stage:
        stage(args.source, args.stage)
    else:
        dispatch()
