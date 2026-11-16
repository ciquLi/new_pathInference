"""Plot five dated fixed-query Jaccard survival distributions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/IEEE-PathVeil/IEEE Conference Template"
RUNS = {
    "20260201": ("run_3e92673fd15d85e4", 685_820),
    "20260301": ("run_e3f6274a5f3c9ec8", 685_765),
    "20260401": ("run_ab7ab2f546f7279d", 688_302),
    "20260501": ("run_834f604486bd8597", 692_950),
    "20260601": ("run_e3376ed94b5612d8", 692_497),
}
PATHRADAR_DATES = frozenset(RUNS)
BIN_LABELS = ("0", "(0, 0.5)", "[0.5, 0.75)", "[0.75, 1)", "1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bin_index(score: float) -> int:
    if not 0 <= score <= 1:
        raise ValueError(f"Invalid Jaccard score: {score}")
    if score == 0:
        return 0
    if score < 0.5:
        return 1
    if score < 0.75:
        return 2
    if score < 1:
        return 3
    return 4


class Counts:
    def __init__(self) -> None:
        self.bins = [0] * len(BIN_LABELS)
        self.frequencies: Counter[float] = Counter()
        self.total = 0.0

    def add(self, score: float) -> None:
        self.bins[bin_index(score)] += 1
        self.frequencies[score] += 1
        self.total += score

    def result(self, files: list[Path], expected: int) -> dict:
        count = sum(self.bins)
        if count != expected:
            raise ValueError(f"Expected {expected} scored queries, got {count}")
        return {
            "queries": count,
            "mean_jaccard_pct": 100 * self.total / count,
            "esr_pct": 100 * self.bins[-1] / count,
            "bins": dict(zip(BIN_LABELS, self.bins)),
            "score_frequencies": sorted([score, n] for score, n in self.frequencies.items()),
            "source_files": [{"path": str(p), "sha256": sha256(p)} for p in files],
        }


def knownpath(date: str, expected: int) -> dict:
    directory = ROOT / "runs/baselines/knownpath" / date / "results" / date
    files = sorted(directory.glob("*-knownpath-detail-result.csv"))
    if len(files) != 8:
        raise ValueError(f"Expected eight KnownPath detail files for {date}")
    counts = Counts()
    for path in files:
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 4:
                    raise ValueError(f"Malformed KnownPath row in {path}")
                counts.add(float(row[3]))
    return counts.result(files, expected)


def pathveil(date: str, run_id: str, expected: int) -> dict:
    directory = (ROOT / "runs" / date / "pats_rho_v1" / run_id
                 / "T-PG-B/evaluations")
    files = sorted(directory.glob("as*/pair_predictions.csv"))
    if len(files) != 10:
        raise ValueError(f"Expected ten PathVeil source files for {date}")
    counts = Counts()
    for path in files:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                counts.add(float(row["jaccard"]))
    return counts.result(files, expected)


def pathradar(date: str, expected: int) -> dict:
    folder = ROOT / "runs/baselines/pathradar" / date / "testing"
    stem = ("top1_eval_round2_full_reevaluated"
            if date in ("20260201", "20260301", "20260401")
            else "top1_eval_round2_full")
    path = folder / f"{stem}.pairs.csv"
    audit_file = folder / f"{stem}.pairs.audit.json"
    evaluation_file = folder / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"PathRadar per-query scores are required: {path}")
    audit = json.loads(audit_file.read_text())
    if audit["status"] != "PASS" or audit["records"] != expected:
        raise ValueError(f"PathRadar query audit failed for {date}")
    if audit["details_sha256"] != sha256(path):
        raise ValueError(f"PathRadar detail hash changed for {date}")
    with evaluation_file.open(encoding="utf-8") as stream:
        header_lines = []
        for line in stream:
            if '"pair_details": [' in line:
                header_lines.append(line.split('"pair_details": [', 1)[0])
                break
            header_lines.append(line)
        else:
            raise ValueError(f"PathRadar evaluation lacks pair_details: {date}")
    summary = json.loads("".join(header_lines) + '"pair_details": []\n}')
    if summary["requested_pairs"] != expected or not summary["include_pair_details"]:
        raise ValueError(f"PathRadar evaluation identity differs for {date}")
    counts = Counts()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            counts.add(float(row["jaccard"]))
    result = counts.result([path], expected)
    if not math.isclose(result["mean_jaccard_pct"] / 100,
                        summary["mean_jaccard_over_requested"], abs_tol=1e-9):
        raise ValueError("PathRadar detail scores differ from their summary")
    result["evaluation_file"] = {
        "path": str(evaluation_file), "sha256": sha256(evaluation_file)
    }
    return result


def survival(values: dict) -> tuple[list[float], list[float]]:
    remaining = values["queries"]
    xs, ys = [], []
    for score, count in values["score_frequencies"]:
        xs.append(score)
        ys.append(100 * remaining / values["queries"])
        remaining -= count
    if xs[0] != 0:
        xs.insert(0, 0.0)
        ys.insert(0, 100.0)
    if remaining != 0 or xs[-1] != 1:
        raise ValueError("Incomplete Jaccard distribution")
    return xs, ys


def plot(rows: dict[str, dict]) -> None:
    styles = {
        "PathVeil": ("#116d68", "-", "o"),
        "KnownPath": ("#575e66", "--", "s"),
        "PathRadar": ("#b65737", ":", "^"),
    }
    fig, axes = plt.subplots(1, 5, figsize=(7.15, 2.42),
                             sharex=True, sharey=True)
    for ax, (date, row) in zip(axes, rows.items()):
        for name, values in row["methods"].items():
            xs, ys = survival(values)
            color, style, marker = styles[name]
            ax.step(xs, ys, where="pre", color=color, linestyle=style,
                    linewidth=1.55, label=name)
            ax.plot(1, values["esr_pct"], color=color, marker=marker,
                    markersize=3.5)
        ax.set_title(date[4:6] + "/" + date[6:], fontsize=8.5)
        ax.set_xlim(0, 1.015)
        ax.set_ylim(0, 101)
        ax.set_xticks((0, 0.5, 1), ("0", "0.5", "1"))
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.grid(color="#e5e9e9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.0)
    axes[0].set_ylabel("Queries with $J \geq t$ (%)", fontsize=8.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=8.5)
    fig.supxlabel("AS-set Jaccard threshold $t$", y=0.13, fontsize=8.5)
    fig.subplots_adjust(left=0.087, right=0.995, top=0.85, bottom=0.34,
                        wspace=0.16)
    fig.savefig(PAPER / "figures/fig_query_jaccard_distribution.pdf",
                bbox_inches="tight")
    fig.savefig(PAPER / "figures/fig_query_jaccard_distribution_preview.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42})
    rows = {}
    for date, (run_id, expected) in RUNS.items():
        methods = {"PathVeil": pathveil(date, run_id, expected),
                   "KnownPath": knownpath(date, expected)}
        if date in PATHRADAR_DATES:
            methods["PathRadar"] = pathradar(date, expected)
        rows[date] = {"queries": expected, "methods": methods}
        print(date, *(f"{name}: {v['mean_jaccard_pct']:.2f}/{v['esr_pct']:.2f}"
                      for name, v in methods.items()), flush=True)
    expected_means = {
        "20260201": (84.52, 71.78), "20260301": (84.30, 70.52),
        "20260401": (83.17, 71.42), "20260501": (83.14, 70.58),
        "20260601": (83.12, 71.06),
    }
    for date, (ours, baseline) in expected_means.items():
        if (abs(rows[date]["methods"]["PathVeil"]["mean_jaccard_pct"] - ours) >= 0.01
                or abs(rows[date]["methods"]["KnownPath"]["mean_jaccard_pct"]
                       - baseline) >= 0.01):
            raise ValueError(f"Distribution disagrees with Table III on {date}")
    expected_pathradar = {
        "20260201": (68.93, 26.84),
        "20260301": (67.86, 25.26),
        "20260401": (69.99, 28.05),
        "20260501": (71.00, 28.96),
        "20260601": (70.08, 26.82),
    }
    for date, (accuracy, esr) in expected_pathradar.items():
        result = rows[date]["methods"]["PathRadar"]
        if abs(result["mean_jaccard_pct"] - accuracy) >= 0.01 or abs(result["esr_pct"] - esr) >= 0.01:
            raise ValueError(f"PathRadar distribution disagrees with Table III on {date}")
    output = {"definition": "survival P(J >= t); J=1 is ESR, area is mean Jaccard",
              "missing_prediction_score": 0, "dates": rows}
    (PAPER / "analysis/query_jaccard_distribution_5dates.json").write_text(
        json.dumps(output, indent=2) + "\n")
    plot(rows)


if __name__ == "__main__":
    main()
