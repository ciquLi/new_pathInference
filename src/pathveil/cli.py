"""One date selects the registered data, method and evaluation workflow."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

from pathveil.utils import (
    object_hash,
    project_root,
    read_json,
    sha256,
    utc_now,
    validate_date,
    write_json,
)


def run_identity(date, registry, method):
    root = project_root()
    sources = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted((root / "src/pathveil").rglob("*.py"))
    }
    return dict(
        date=date,
        registry_sha256=sha256(registry["registry_path"]),
        method_config=method,
        code_sha256=sources,
    )


def load_method(registry, root=None, override=None):
    """Resolve the explicit per-date method, retaining the frozen default."""
    root = Path(root or project_root())
    path = Path(override or registry.get("method_config", "configs/method_unified255.json"))
    return read_json(path if path.is_absolute() else root / path)


def choose_run(root, date, identity, requested=None):
    if requested:
        run = Path(requested).resolve()
        if root / "runs" not in run.parents:
            raise ValueError("--run-dir must be inside this project's runs/")
    else:
        parent = (
            root
            / "runs"
            / date
            / identity.get("method_config", {}).get("workflow_id", "unified255")
        )
        current = parent / "current.json"
        previous = read_json(current) if current.is_file() else None
        if previous and previous.get("identity_sha256") == object_hash(identity):
            run = Path(previous["run_dir"])
        else:
            stamp = utc_now().replace("-", "").replace(":", "").replace(".", "_")
            run = parent / stamp
        write_json(current, dict(run_dir=str(run), identity_sha256=object_hash(identity)))
    run.mkdir(parents=True, exist_ok=True)
    manifest = run / "RUN_IDENTITY.json"
    if manifest.is_file() and read_json(manifest) != identity:
        raise ValueError("Code/config changed; choose a fresh run directory")
    write_json(manifest, identity)
    return run


def execute(args):
    from pathveil.data.registry import load_registry
    from pathveil.data.preflight import check_date

    root = project_root()
    registry = load_registry(args.date)
    if registry.get("workflow_id") == "rho_adaptive_full_tree_v2" and not args.migration_replay:
        from pathveil.rho_adaptive.full_tree import execute as execute_full_tree

        return execute_full_tree(args, registry)
    if registry.get("workflow_id") == "pats_rho_v1" and not args.migration_replay:
        from pathveil.pats.runner import execute as execute_pats

        return execute_pats(args, registry)
    if registry.get("workflow_id") == "rho_adaptive_v1" and not args.migration_replay:
        from pathveil.rho_adaptive.runner import execute as execute_adaptive

        return execute_adaptive(args, registry)
    if registry.get("workflow_id") == "rho_prefix_v1" and not args.migration_replay:
        from pathveil.experiment.runner import execute as execute_rho

        return execute_rho(args, registry)
    if args.check_only:
        report = check_date(args.date, deep=args.deep_check)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["raw_ready"] else 2
    method = load_method(registry, root, getattr(args, "method_config", None))
    if method.get("runtime_resources"):
        from pathveil.upstream.resources import runtime_resources

        threads = str(runtime_resources(method)["cpu_threads"])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = threads
    identity = run_identity(args.date, registry, method)
    run = choose_run(root, args.date, identity, args.run_dir)
    write_json(run / "method_config.json", method)
    for key, folder in {
        "TMPDIR": "tmp",
        "MPLCONFIGDIR": "matplotlib",
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch",
    }.items():
        directory = run / "runtime" / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    with (run / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This experiment is already running: " + str(run)) from error

        def status(stage, **details):
            value = dict(
                date=args.date,
                stage=stage,
                updated_utc=utc_now(),
                pid=os.getpid(),
                run_dir=str(run),
                **details,
            )
            write_json(run / "status.json", value)
            print(json.dumps(value, ensure_ascii=False), flush=True)

        try:
            from pathveil.data.downloader import download_required
            from pathveil.data.prepare import prepare_date

            status("data_check", status="running")
            report = check_date(args.date, deep=args.deep_check)
            write_json(run / "data/preflight.json", report)
            if report["required_downloads"]:
                status("data_download", status="running", datasets=report["required_downloads"])
                download_required(args.date, progress=lambda text: print(text, flush=True))
            status("data_prepare", status="running")
            # A quick status check may sample headers; a runnable experiment
            # must validate complete public snapshots before fitting models.
            data = prepare_date(args.date, run, deep=True)
            manifest_path = run / "data/data_manifest.json"
            if args.prepare_only:
                status("data_prepare", status="complete", data_manifest=str(manifest_path))
                return 0
            status("upstream", status="running")
            if args.migration_replay:
                from pathveil.upstream.replay import run_baseline

                release_path = run_baseline(data, run, method, project_root=root)
            else:
                from pathveil.upstream.entry import run_upstream

                release_path = run_upstream(data, run, method, project_root=root)
            release_path = Path(release_path)
            release = read_json(release_path)
            if release.get("status") != "PASS":
                raise ValueError("Upstream did not release a verified route library")
            if args.upstream_only:
                status("upstream", status="complete", release=str(release_path))
                return 0
            status("downstream", status="running")
            if registry["mode"] == "traceroute":
                from pathveil.traceroute.runner import run_traceroute

                trace_result = run_traceroute(run, manifest_path, release_path, device=args.device)
                summary = trace_result["summary"]
            else:
                from pathveil.downstream.runner import run_downstream

                summary = run_downstream(run, manifest_path, release_path, device=args.device)
            status("complete", status="PASS", summary=str(summary))
            return 0
        except BaseException as error:
            status("failed", status="FAILED", error=repr(error))
            with (run / "error.txt").open("w") as stream:
                traceback.print_exc(file=stream)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", type=validate_date, help="Registered YYYYMMDD")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--upstream-only", action="store_true")
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate dated inputs and print the execution plan without training",
    )
    mode.add_argument(
        "--augmentation-only",
        action="store_true",
        help="Complete the frozen augmentation libraries without GMPI training",
    )
    parser.add_argument(
        "--migration-replay",
        action="store_true",
        help="Explicit March cached-feature regression, not new candidate generation",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--fresh-run", action="store_true", help="Keep existing artifacts and start a new run"
    )
    parser.add_argument("--method-config", type=Path, help="Explicit method configuration override")
    parser.add_argument("--deep-check", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        code = execute(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
