import argparse
import os
from pathlib import Path

from .workflow import prepare, execute, build_all


def main():
    parser = argparse.ArgumentParser(description="Registered-date PATS workflow")
    parser.add_argument("date")
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("PATHVEIL_PROJECT_ROOT", Path(__file__).resolve().parents[3])))
    parser.add_argument("--stage", choices=["prepare", "run", "build_all", "source"], default="run")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cuda")
    args = parser.parse_args()
    os.environ["PATHVEIL_PROJECT_ROOT"] = str(args.project_root.resolve())
    root = args.project_root.resolve()
    if args.stage == "prepare":
        prepare(root, args.date)
        return 0
    run = args.run_dir or prepare(root, args.date)
    if args.stage == "source":
        from .construction import construct
        if not args.source:
            parser.error("--source is required")
        construct(run, args.source)
        return 0
    if args.stage == "build_all":
        build_all(run)
        return 0
    execute(run, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
