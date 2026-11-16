"""Publish live progress for the five-date external baseline reruns."""

import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/baselines"
FILES = ROOT.parent / "files"
DATES = ("20260201", "20260301", "20260401", "20260501", "20260601")


def expected_pairs(date):
    config = json.loads((ROOT / "configs/dates" / (date + ".json")).read_text())
    metadata = FILES / "data/aspath/path_by_vantage" / date / "RL_Graph/vantage_dst_test.json"
    with metadata.open() as stream:
        queries = json.loads(json.load(stream))
    return sum(len(set(queries[source])) for source in config["test_sources"])


def knownpath_count(date):
    folder = RUNS / "knownpath" / date / "results" / date
    pairs = 0
    destinations = 0
    for path in folder.glob("*-knownpath-result.csv"):
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) == 6:
                    try:
                        pairs += int(row[1])
                    except ValueError:
                        continue
                    destinations += 1
    return destinations, pairs


def pathradar_stage(date):
    folder = RUNS / "pathradar" / date
    prepared = folder / "PREPARED.json"
    rib_count = (json.loads(prepared.read_text())["dataset_counts"].get("rib")
                 if prepared.is_file() else None)

    def prediction_progress(round_number):
        output = folder / "testing" / ("result_round%d" % round_number)
        done = sum(1 for _ in output.glob("*.txt")) if output.is_dir() else 0
        return "第 %d 轮训练集预测 %s/%s" % (
            round_number, f"{done:,}", f"{rib_count:,}" if rib_count else "?")

    if (folder / "COMPLETE.json").is_file():
        record = json.loads((folder / "COMPLETE.json").read_text())
        if record.get("round") == 2:
            return "完成"
    models = folder / "trained/models"
    if all((models / ("xgboost_2_%s%s.model" % (kind, index))).is_file()
           for kind in ("core", "shell") for index in (1, 2, 3)):
        return "第 2 轮评估"
    if (folder / "PREDICT_TRAIN_ROUND1.json").is_file():
        return "第 2 轮训练"
    if (folder / "logs/predict_train_1.log").is_file():
        return prediction_progress(1)
    if (folder / "PREDICT_TRAIN_ROUND0.json").is_file():
        return "第 1 轮训练"
    if (folder / "logs/predict_train_0.log").is_file():
        return prediction_progress(0)
    for stage in ("train_0", "train", "migrate", "shortest", "combine", "build"):
        if (folder / "logs" / (stage + ".log")).is_file():
            return stage
    return "等待"


def publish():
    now = datetime.now(timezone(timedelta(hours=8)))
    lines = ["# 外部 baseline 全量重跑进度", "",
             "更新时间（北京时间）：" + now.strftime("%Y-%m-%d %H:%M:%S"), "",
             "| 日期 | KnownPath 目的地 | KnownPath 已评分/总查询 | PathRadar |",
             "| --- | ---: | ---: | --- |"]
    all_done = True
    for date in DATES:
        destinations, scored = knownpath_count(date)
        expected = expected_pairs(date)
        kp_done = (RUNS / "knownpath" / date / "COMPLETE.json").is_file()
        pr_stage = pathradar_stage(date)
        all_done &= kp_done and pr_stage == "完成"
        lines.append("| %s | %s | %s/%s | %s |" % (
            date, f"{destinations:,}", f"{scored:,}", f"{expected:,}", pr_stage))
    lines += ["", "KnownPath 和 PathRadar 按日期顺序运行；1015 暂不纳入。",
              "详细错误见各方法的 `BATCH.log` 和日期目录下的 `logs/`。", ""]
    output = RUNS / "PROGRESS.md"
    temporary = output.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(output)
    bulletin = RUNS / "BULLETINS.md"
    last_slot = bulletin.with_suffix(".slot")
    slot = now.strftime("%Y-%m-%d %H:") + str(now.minute // 15)
    if not last_slot.is_file() or last_slot.read_text() != slot:
        with bulletin.open("a") as stream:
            stream.write("\n## " + now.strftime("%Y-%m-%d %H:%M") + "\n\n")
            stream.write("\n".join(lines[4:12]) + "\n")
        last_slot.write_text(slot)
    return all_done


if __name__ == "__main__":
    while True:
        if publish():
            break
        time.sleep(60)
