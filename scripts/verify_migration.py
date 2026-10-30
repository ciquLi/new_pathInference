#!/usr/bin/env python3
"""Recompute the March frozen model scores and selected libraries."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.data.prepare import prepare_date
from pathveil.upstream.replay import run_baseline
from pathveil.utils import project_root, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    root = project_root()
    run = (args.run_dir or root / "runs/20260301/unified255/migration_verification").resolve()
    manifest = prepare_date("20260301", run)
    method = read_json(root / "configs/method_unified255.json")
    path = run_baseline(manifest, run, method, project_root=root)
    print(json.dumps({"status": "PASS", "release": str(path)}, indent=2))


if __name__ == "__main__":
    main()
