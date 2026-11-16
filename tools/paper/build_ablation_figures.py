"""Build source-level component comparisons from the five dated ablations."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/IEEE-PathVeil/IEEE Conference Template"
DATES = ("20260201", "20260301", "20260401", "20260501", "20260601")
VARIANTS = ("current", "legacy_generator", "no_recursion", "all_recursion")


def source_scores(date: str, variant: str, source: str) -> dict:
    base = ROOT / "runs" / date / "generator_recursion_v1" / variant
    metrics = json.loads((base / "diagnostics" / f"as{source}" / "METRICS.json").read_text())
    if variant == "current":
        summary = json.loads((base / "T-PG-B/independent_evaluation"
                              / f"source_{source}.json").read_text())
        prediction = Path(summary["prediction_file"])
    else:
        prediction = base / "T-PG-B/evaluations" / f"as{source}" / "pair_predictions.csv"
        summary = json.loads((prediction.parent / "evaluation_summary.json").read_text())
    n = int(metrics["queries"])
    if n != summary["query_count"]:
        raise ValueError(f"Query count mismatch for {date}/{variant}/{source}")
    count = esr = 0
    total = 0.0
    with prediction.open(newline="") as stream:
        for row in csv.DictReader(stream):
            score = float(row["jaccard"])
            count += 1
            total += score
            esr += score == 1
    if (count != n or not math.isclose(total, summary["total_jaccard_reward"],
                                     rel_tol=0, abs_tol=1e-6)):
        raise ValueError(f"Prediction mismatch for {date}/{variant}/{source}")
    return {"queries": n, "accuracy_sum": total, "esr_count": esr,
            "oracle_accuracy": float(metrics["oracle_jaccard"])}


def load() -> dict:
    by_date = {}
    for date in DATES:
        base = ROOT / "runs" / date / "generator_recursion_v1/current/diagnostics"
        sources = sorted(path.parent.name[2:] for path in base.glob("as*/METRICS.json"))
        if len(sources) != 10:
            raise ValueError(f"Expected ten target sources for {date}")
        by_date[date] = {}
        for source in sources:
            variants = {name: source_scores(date, name, source) for name in VARIANTS}
            if len({v["queries"] for v in variants.values()}) != 1:
                raise ValueError(f"Inconsistent query counts for {date}/{source}")
            by_date[date][source] = variants
    return by_date


def aggregate_by_source(by_date: dict) -> dict:
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    dates = defaultdict(list)
    for date, sources in by_date.items():
        for source, variants in sources.items():
            dates[source].append(date)
            for name, value in variants.items():
                totals = sums[source][name]
                totals["queries"] += value["queries"]
                totals["accuracy_sum"] += value["accuracy_sum"]
                totals["esr_count"] += value["esr_count"]
                totals["oracle_accuracy_sum"] += value["oracle_accuracy"] * value["queries"]
    output = {}
    for source, variants in sums.items():
        output[source] = {"dates": dates[source], "variants": {}}
        for name, value in variants.items():
            n = value["queries"]
            output[source]["variants"][name] = {
                "queries": int(n),
                "accuracy_pct": 100 * value["accuracy_sum"] / n,
                "esr_pct": 100 * value["esr_count"] / n,
                "oracle_accuracy_pct": 100 * value["oracle_accuracy_sum"] / n,
            }
    return output


def plot(by_source: dict, comparison: tuple[str, ...], filename: str) -> None:
    sources = sorted(by_source, key=lambda asn: int(asn))
    labels = [f"AS{source}" + (f" ({len(by_source[source]['dates'])})"
                                  if len(by_source[source]["dates"]) < 5 else "")
              for source in sources]
    y = list(reversed(range(len(sources))))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.65), sharey=True)
    styles = (("#b45536", "o"), ("#127a70", "s"))
    names = {"legacy_generator": "High-frequency generator",
             "no_recursion": "vs. no recursion",
             "all_recursion": "vs. full recursion"}
    x_label = ("Conditional recursion minus comparator (pp)"
               if filename == "fig_ablation_recursion"
               else "Prefix-tree generator gain (pp)")
    for ax, (metric, title) in zip(axes, (("accuracy_pct", "Accuracy"),
                                          ("esr_pct", "ESR"))):
        for index, variant in enumerate(comparison):
            color, marker = styles[index]
            gains = [by_source[source]["variants"]["current"][metric]
                     - by_source[source]["variants"][variant][metric]
                     for source in sources]
            offsets = [position + (0.13 if index == 0 and len(comparison) > 1
                                   else -0.13 if index == 1 else 0)
                       for position in y]
            ax.scatter(gains, offsets, color=color, marker=marker, s=23,
                       label=names[variant], zorder=3)
        ax.axvline(0, color="#43494a", linewidth=0.85)
        ax.grid(axis="x", color="#e1e5e6", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(x_label)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelleft=True)
        ax.set_yticks(y, labels, fontsize=8.0)
        limits = ax.get_xlim()
        margin = max(0.55, (limits[1] - limits[0]) * 0.03)
        ax.set_xlim(limits[0] - margin, limits[1] + margin)
    if len(comparison) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, -0.045), frameon=False, fontsize=8.0)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(PAPER / "figures" / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures" / f"{filename}_preview.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42})
    by_date = load()
    by_source = aggregate_by_source(by_date)
    output = {"aggregation": "query-weighted within each source across available dates",
              "by_date": by_date, "by_source": by_source}
    (PAPER / "analysis/ablation_by_source.json").write_text(
        json.dumps(output, indent=2) + "\n")
    plot(by_source, ("no_recursion", "all_recursion"), "fig_ablation_recursion")
    plot(by_source, ("legacy_generator",), "fig_ablation_generator")
    for source, value in by_source.items():
        print(source, len(value["dates"]),
              *(f"{value['variants']['current'][metric] - value['variants']['legacy_generator'][metric]:.2f}"
                for metric in ("accuracy_pct", "esr_pct")))


if __name__ == "__main__":
    main()
