"""Publish each independently rescored AS and both aggregate definitions."""
from __future__ import annotations

import csv
import fcntl
import math
from pathlib import Path

from pathveil.utils import read_json, sha256, utc_now, write_json


def publish(run):
    run = Path(run)
    protocol = read_json(run / "PROTOCOL.json")
    with (run / ".results.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for source in protocol["sources"]:
            path = run / "T-PG-B/independent_evaluation" / ("source_" + source + ".json")
            if path.exists():
                row = read_json(path)
                if sha256(row["prediction_file"]) != row["prediction_sha256"]:
                    raise ValueError("Published prediction bytes changed")
                rows.append(row)
        total = sum(r["query_count"] for r in rows)
        value = dict(
            status="COMPLETED" if len(rows) == len(protocol["sources"]) else "RUNNING",
            date=protocol["date"],
            completed_sources=len(rows),
            expected_sources=len(protocol["sources"]),
            query_count=total,
            macro_jaccard=math.fsum(r["jaccard_accuracy"] for r in rows) / len(rows)
            if rows
            else None,
            query_weighted_jaccard=math.fsum(r["total_jaccard_reward"] for r in rows) / total
            if total
            else None,
            per_source=rows,
            updated_utc=utc_now(),
            uncompleted_sources_excluded_from_partial_average=True,
        )
        common_sources = protocol.get("common9", protocol["sources"])
        common_rows = [r for r in rows if r["source_asn"] in common_sources]
        common_queries = sum(r["query_count"] for r in common_rows)
        value["common9"] = dict(
            expected_sources=common_sources,
            completed_sources=len(common_rows),
            query_count=common_queries,
            macro_jaccard=math.fsum(r["jaccard_accuracy"] for r in common_rows) / len(common_rows)
            if common_rows
            else None,
            query_weighted_jaccard=math.fsum(r["total_jaccard_reward"] for r in common_rows)
            / common_queries
            if common_queries
            else None,
        )
        write_json(run / "RESULTS.json", value)
        fields = [
            "source_asn",
            "rho",
            "admission_strategy",
            "admitted_paths",
            "admitted_destinations",
            "selected_objective",
            "query_count",
            "jaccard_accuracy",
            "ordered_exact_accuracy",
            "coverage",
        ]
        temporary = run / "RESULTS.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(run / "RESULTS.csv")
        lines = [
            "# " + protocol["date"] + "：ρ分组路径推断",
            "",
            "逐AS完成后发布；测试缺候选计0，未完成AS不计入阶段平均。",
            "",
            "| AS | ρ | 增强路径 | 查询 | Jaccard | 完整有序匹配 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        by_source = {r["source_asn"]: r for r in rows}
        for source in protocol["sources"]:
            row = by_source.get(source)
            if row:
                lines.append(
                    "| {} | {:.4f} | {} | {} | {:.4%} | {:.4%} |".format(
                        source,
                        row["rho"],
                        row["admitted_paths"],
                        row["query_count"],
                        row["jaccard_accuracy"],
                        row["ordered_exact_accuracy"],
                    )
                )
            else:
                lines.append(
                    "| {} | {} | 待完成 | — | — | — |".format(
                        source, protocol["public_features"][source]["rho"]
                    )
                )
        if rows:
            lines += [
                "",
                "已完成AS等权平均：{:.4%}；按查询加权：{:.4%}。".format(
                    value["macro_jaccard"], value["query_weighted_jaccard"]
                ),
            ]
        lines += ["", "每个日期使用其自身数据和同日上游模型；正式参考只用于冻结模型后的评测。", "Jaccard、完整有序路径匹配与增强库观测支持率属于不同指标。"]
        temporary = run / "REPORT.md.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(run / "REPORT.md")
        return value
