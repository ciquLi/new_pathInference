"""Remove reproducible intermediates from completed PATS runs."""

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
DATES = ("20260201", "20260301", "20260401", "20260501", "20260601")
DIR_NAMES = {"pg_inputs", "model_input_data_PG", "runtime_cache"}


def size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def targets(run):
    directories = []
    for path in run.rglob("*"):
        if path.is_dir() and path.name in DIR_NAMES:
            if not any(parent in directories for parent in path.parents):
                directories.append(path)
    files = [path for path in run.rglob("*_queries.jsonl")
             if "tree_B" in path.parts and not any(parent in directories for parent in path.parents)]
    return directories + files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = []
    for date in DATES:
        for run in sorted((ROOT / "runs" / date / "pats_rho_v1").glob("run_*")):
            if not (run / "REPORT.md").is_file() or not (run / "RESULTS.json").is_file():
                continue
            if (run / ".run.lock").exists():
                # A stale lock file is expected; only a live process matters.
                pass
            candidates = targets(run)
            manifest.append({"run": str(run.relative_to(ROOT)),
                             "paths": [str(path.relative_to(ROOT)) for path in candidates],
                             "bytes": sum(size(path) for path in candidates)})
    total = sum(item["bytes"] for item in manifest)
    print("runs=%d paths=%d bytes=%d (%.1f GiB)" % (
        len(manifest), sum(len(item["paths"]) for item in manifest),
        total, total / (1024 ** 3)), flush=True)
    if not args.execute:
        return
    for item in manifest:
        for relative in item["paths"]:
            path = ROOT / relative
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        print(item["run"], "removed %.1f GiB" % (item["bytes"] / (1024 ** 3)), flush=True)
    report = ROOT / "runs" / "PATS_CLEANUP.json"
    report.write_text(json.dumps({"preserved": ["results", "models", "augmentation",
                                               "pair_predictions", "query_trees", "manifests"],
                                  "removed": manifest, "total_bytes": total}, indent=2) + "\n")


if __name__ == "__main__":
    main()
