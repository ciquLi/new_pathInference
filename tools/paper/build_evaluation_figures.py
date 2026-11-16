"""Build paper evaluation summaries and vector figures from frozen runs."""

from __future__ import annotations

import csv
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/IEEE-PathVeil/IEEE Conference Template"
RUNS = {
    "20260201": "run_3e92673fd15d85e4",
    "20260301": "run_e3f6274a5f3c9ec8",
    "20260401": "run_ab7ab2f546f7279d",
    "20260501": "run_834f604486bd8597",
    "20260601": "run_e3376ed94b5612d8",
}


def source_stage(date: str, run: Path, source: str) -> dict:
    tree = json.loads((run / "query_trees" / f"{source}.json").read_text())
    selected = set(map(str, tree["selected_firsthops"]))
    references: set[str] = set()
    by_destination: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    observed: set[str] = set()
    reference_file = (
        ROOT.parent / "files/data/aspath/path_by_vantage" / date
        / "path_without_prefix/test" / f"{source}.csv"
    )
    with reference_file.open() as stream:
        for line in stream:
            path = line.strip()
            if not path or path in references:
                continue
            parts = path.split("->")
            if len(parts) < 2 or parts[0] != source:
                raise ValueError(f"Invalid reference path in {reference_file}")
            references.add(path)
            observed.add(parts[1])
            counts = by_destination[parts[-1]]
            counts[0] += 1
            counts[1] += parts[1] in selected

    admitted: set[str] = set()
    admitted_file = run / "augmentation" / f"as{source}" / "paths.csv"
    with admitted_file.open() as stream:
        for line in stream:
            path = line.strip()
            if path:
                admitted.add(path)
    admitted_neighbors = {path.split("->", 2)[1] for path in admitted}
    selected_observed = selected & observed
    represented = selected_observed & admitted_neighbors
    evaluable = {path for path in admitted if path.rsplit("->", 1)[-1] in by_destination}
    supported = evaluable & references
    pair_coverage_sum = sum(hit / total for total, hit in by_destination.values())
    return {
        "source": source,
        "reference_pairs": len(by_destination),
        "reference_unique_paths": len(references),
        "selected_neighbors": len(selected),
        "selected_observed_neighbors": len(selected_observed),
        "represented_selected_observed_neighbors": len(represented),
        "pair_coverage_sum": pair_coverage_sum,
        "firsthop_coverage_pct": 100 * pair_coverage_sum / len(by_destination),
        "branch_representation_pct": 100 * len(represented) / len(selected_observed)
        if selected_observed else None,
        "admitted_paths": len(admitted),
        "admitted_on_reference_pairs": len(evaluable),
        "admitted_outside_reference_pairs": len(admitted) - len(evaluable),
        "ordered_reference_matches": len(supported),
        "complete_path_support_pct": 100 * len(supported) / len(evaluable)
        if evaluable else None,
    }


def knownpath(date: str) -> dict:
    base = ROOT / "runs/baselines/knownpath" / date
    complete = json.loads((base / "COMPLETE.json").read_text())
    count, score, jaccard_one = 0, 0.0, 0
    detail_files = sorted((base / "results" / date).glob("*-knownpath-detail-result.csv"))
    if not detail_files:
        raise ValueError(f"Missing KnownPath detail for {date}")
    for file in detail_files:
        with file.open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 4:
                    raise ValueError(f"Malformed KnownPath row in {file}")
                count += 1
                value = float(row[3])
                score += value
                jaccard_one += value == 1.0
    if count != complete["expected_pairs"]:
        raise ValueError(f"KnownPath query count mismatch for {date}")
    return {
        "query_count": count,
        "jaccard_pct": 100 * score / count,
        "jaccard_one_pct": 100 * jaccard_one / count,
    }


def pathveil_jaccard_one(run: Path, sources: list[str], expected: int) -> float:
    count = matches = 0
    for source in sources:
        prediction = run / "T-PG-B/evaluations" / f"as{source}" / "pair_predictions.csv"
        summary = json.loads(
            (run / "T-PG-B/independent_evaluation" / f"source_{source}.json").read_text()
        )
        source_count = 0
        score_sum = 0.0
        with prediction.open(newline="") as stream:
            for row in csv.DictReader(stream):
                value = float(row["jaccard"])
                source_count += 1
                score_sum += value
                matches += value == 1.0
        if source_count != summary["query_count"] or not math.isclose(
            score_sum / source_count, summary["jaccard_accuracy"], abs_tol=1e-9
        ):
            raise ValueError(f"PathVeil scored predictions mismatch for {source}")
        count += source_count
    if count != expected:
        raise ValueError("PathVeil Jaccard-one query count mismatch")
    return 100 * matches / count


def pathradar(date: str, expected: int) -> dict | None:
    file = (
        ROOT / "runs/baselines/pathradar" / date
        / "testing/top1_eval_round2_full.json"
    )
    if not file.exists():
        return None
    value = json.loads(file.read_text())
    if value["requested_pairs"] != expected:
        raise ValueError(f"PathRadar query count mismatch for {date}")
    return {
        "query_count": expected,
        "jaccard_pct": 100 * value["mean_jaccard_over_requested"],
    }


def summarize(date: str, run_id: str) -> dict:
    run = ROOT / "runs" / date / "pats_rho_v1" / run_id
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    policy = protocol["method"]["selection_policy"]
    if policy["recursion_topk"] != 5 or policy["low_root_limit"] != 21:
        raise ValueError(f"Mismatched method setting for {date}")
    with (run / "RESULTS.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    sources = [str(row["source_asn"]) for row in rows]
    if len(sources) != 10 or len(set(sources)) != 10:
        raise ValueError(f"Unexpected source cohort for {date}")
    count = sum(int(row["query_count"]) for row in rows)
    def weighted(key: str) -> float:
        return 100 * sum(
            int(row["query_count"]) * float(row[key]) for row in rows
        ) / count
    stages = [source_stage(date, run, source) for source in sources]
    for row, stage in zip(rows, stages):
        if int(row["admitted_paths"]) != stage["admitted_paths"]:
            raise ValueError(f"Admitted-library count mismatch for {date}/{row['source_asn']}")
        if int(row["query_count"]) != stage["reference_pairs"]:
            raise ValueError(f"Reference-pair count mismatch for {date}/{row['source_asn']}")
    stage_pairs = sum(value["reference_pairs"] for value in stages)
    represented = sum(value["represented_selected_observed_neighbors"] for value in stages)
    observed = sum(value["selected_observed_neighbors"] for value in stages)
    supported = sum(value["ordered_reference_matches"] for value in stages)
    evaluable = sum(value["admitted_on_reference_pairs"] for value in stages)
    kp = knownpath(date)
    if kp["query_count"] != count or stage_pairs != count:
        raise ValueError(f"Fixed query universe mismatch for {date}")
    return {
        "date": date,
        "run_id": run_id,
        "sources": sources,
        "query_count": count,
        "admitted_paths": sum(value["admitted_paths"] for value in stages),
        "pathveil_jaccard_pct": weighted("jaccard_accuracy"),
        "pathveil_jaccard_one_pct": pathveil_jaccard_one(run, sources, count),
        "pathveil_exact_pct": weighted("ordered_exact_accuracy"),
        "pathveil_coverage_pct": weighted("coverage"),
        "pathveil_source_mean_jaccard_pct": 100 * sum(
            float(row["jaccard_accuracy"]) for row in rows
        ) / len(rows),
        "knownpath": kp,
        "pathradar": pathradar(date, count),
        "stage": {
            "firsthop_coverage_pct": 100 * sum(
                value["pair_coverage_sum"] for value in stages
            ) / stage_pairs,
            "represented_neighbors": represented,
            "selected_observed_neighbors": observed,
            "branch_representation_pct": 100 * represented / observed,
            "supported_paths": supported,
            "reference_evaluable_paths": evaluable,
            "outside_reference_pairs": sum(
                value["admitted_outside_reference_pairs"] for value in stages
            ),
            "complete_path_support_pct": 100 * supported / evaluable,
        },
        "source_stage": stages,
    }


def plot_main(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    y = list(reversed(range(len(rows))))
    kp = [row["knownpath"]["jaccard_pct"] for row in rows]
    ours = [row["pathveil_jaccard_pct"] for row in rows]
    for position, left, right in zip(y, kp, ours):
        ax.plot([left, right], [position, position], color="#c9d0d3", lw=1.5, zorder=1)
    ax.scatter(kp, y, color="#6c747c", marker="s", s=31, label="KnownPath", zorder=3)
    ax.scatter(ours, y, color="#007f76", marker="o", s=37, label="PathVeil", zorder=4)
    for position, row in zip(y, rows):
        if row["pathradar"] is not None:
            ax.scatter(row["pathradar"]["jaccard_pct"], position,
                       color="#bf5a36", marker="^", s=36,
                       label="PathRadar" if not any(
                           artist.get_label() == "PathRadar" for artist in ax.collections
                       ) else "_nolegend_", zorder=4)
    for position, value in zip(y, ours):
        ax.annotate(f"{value:.1f}", (value, position), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.8)
    ax.set_yticks(y, ["Feb", "Mar", "Apr", "May", "Jun"])
    ax.set_xlim(60, 91)
    ax.set_xticks([60, 65, 70, 75, 80, 85, 90])
    ax.set_xlabel("Mean AS-set Jaccard (%)")
    ax.set_ylabel("2026 snapshot")
    ax.grid(axis="x", color="#e5e9ea", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0),
               frameon=False, fontsize=7.8, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.8)
    fig.savefig(PAPER / "figures/fig_eval_main.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures/fig_eval_main_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stage(rows: list[dict]) -> None:
    regular = []
    sparse = []
    for row in rows:
        for source in row["source_stage"]:
            (sparse if source["source"] == "328840" else regular).append(source)
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    ax.scatter([r["firsthop_coverage_pct"] for r in regular],
               [r["complete_path_support_pct"] for r in regular],
               color="#6c747c", marker="o", s=22, alpha=0.75,
               label="Other source snapshots")
    ax.scatter([r["firsthop_coverage_pct"] for r in sparse],
               [r["complete_path_support_pct"] for r in sparse],
               color="#bf5a36", marker="D", s=36, label="AS328840")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("First-hop coverage (%)")
    ax.set_ylabel("Complete-path support (%)")
    ax.grid(color="#e5e9ea", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    fig.tight_layout(pad=0.8)
    fig.savefig(PAPER / "figures/fig_eval_stage.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures/fig_eval_stage_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    rows = [summarize(date, run_id) for date, run_id in RUNS.items()]
    (PAPER / "analysis/evaluation_5dates.json").write_text(
        json.dumps({"metric": "raw AS-set Jaccard and Jaccard=1; missing queries score zero",
                    "pathradar_round": 2, "dates": rows}, indent=2) + "\n"
    )
    plot_main(rows)
    plot_stage(rows)
    for row in rows:
        print(row["date"], "queries", row["query_count"],
              "PathVeil", f"{row['pathveil_jaccard_pct']:.4f}",
              "KnownPath", f"{row['knownpath']['jaccard_pct']:.4f}",
              "PathRadar", row["pathradar"], "stage", row["stage"])


if __name__ == "__main__":
    main()
