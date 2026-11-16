"""Summarize completed full-query KnownPath and PathRadar runs."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/baselines"
DATES = ("20260201", "20260301", "20260401", "20260501", "20260601")


def knownpath(date):
    folder = RUNS / "knownpath" / date / "results" / date
    score = 0.0
    count = 0
    sources = {}
    for path in folder.glob("*-knownpath-detail-result.csv"):
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 4:
                    raise ValueError("Malformed KnownPath detail in " + str(path))
                value = float(row[3])
                score += value
                count += 1
                subtotal, n = sources.get(row[1], (0.0, 0))
                sources[row[1]] = (subtotal + value, n + 1)
    return {"pairs": count, "mean_jaccard": score / count,
            "sources": {source: {"pairs": n, "mean_jaccard": subtotal / n}
                        for source, (subtotal, n) in sorted(sources.items())}}


def main():
    rows = []
    for date in DATES:
        kp_complete = json.loads((RUNS / "knownpath" / date / "COMPLETE.json").read_text())
        kp = knownpath(date)
        if kp["pairs"] != kp_complete["expected_pairs"]:
            raise ValueError("KnownPath detail coverage mismatch for " + date)
        pr = json.loads((RUNS / "pathradar" / date / "testing/top1_eval_round2_full.json").read_text())
        if pr["requested_pairs"] != kp["pairs"]:
            raise ValueError("Baseline query counts differ for " + date)
        rows.append({"date": date, "pairs": kp["pairs"],
                     "knownpath_jaccard": kp["mean_jaccard"],
                     "pathradar_jaccard": pr["mean_jaccard_over_requested"],
                     "knownpath_sources": kp["sources"],
                     "pathradar_sources": pr["source_breakdown"]})
    (RUNS / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# 外部 baseline 全量重跑结果", "",
             "| 日期 | 查询数 | KnownPath Jaccard | PathRadar Jaccard |",
             "| --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| %s | %s | %.4f%% | %.4f%% |" % (
            row["date"], f"{row['pairs']:,}",
            row["knownpath_jaccard"] * 100, row["pathradar_jaccard"] * 100))
    lines += ["", "PathRadar 采用第 2 轮模型（阈值 0.7）；两种方法均使用相同日期的完整测试源目的对。",
              "KnownPath 对每个查询取其预测路径与参考路径的最大 Jaccard；",
              "PathRadar 使用原始评估器的 `mean_jaccard_over_requested`。", ""]
    (RUNS / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
