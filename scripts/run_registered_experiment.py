#!/usr/bin/env python3
"""Run the frozen CLI in this PID, with one narrowly audited metadata repair.

All experiment arguments go unchanged to pathveil.cli.main(). Only a fresh,
exact model-interface protocol gate failure may invoke the external normalizer;
its successful audit receipt permits one retry of the original CLI invocation.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
MODEL_GATE = "Only explicit date-local source-disjoint models are accepted"
MODEL_GATE_REPR = "UpstreamError(" + repr(MODEL_GATE) + ")"
FULL_PROTOCOL = "date_local_all_eligible_source_disjoint_v2"
SOURCE_POLICY = "all_canonical_unique_valid_paths_gt_threshold_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def locate_run(arguments, root):
    """Read routing arguments only; the frozen parser owns their validation."""
    if any(
        flag in arguments
        for flag in (
            "--check-only",
            "--prepare-only",
            "--migration-replay",
            "--help",
            "-h",
        )
    ):
        return None
    boolean = {"--upstream-only", "--deep-check"}
    valued = {"--run-dir", "--method-config", "--device"}
    positional, values = [], {}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        option, equals, value = argument.partition("=")
        if option in valued:
            if not equals:
                index += 1
                if index >= len(arguments):
                    return None
                value = arguments[index]
            values[option] = value
        elif argument in boolean:
            pass
        elif argument.startswith("-"):
            return None
        else:
            positional.append(argument)
        index += 1
    if len(positional) != 1:
        return None
    date = positional[0]
    if len(date) != 8 or not date.isascii() or not date.isdecimal():
        return None
    try:
        datetime.strptime(date, "%Y%m%d")
        requested = values.get("--run-dir")
        registry_file = root / "configs/dates" / (date + ".json")
        workflow = (
            read_json(registry_file).get("workflow_id", "unified255")
            if registry_file.exists()
            else "unified255"
        )
        run = (
            Path(requested)
            if requested is not None
            else Path(read_json(root / "runs" / date / workflow / "current.json")["run_dir"])
        ).resolve()
        if root / "runs" not in run.parents:
            return None
        return date, run
    except (OSError, ValueError, KeyError, TypeError):
        return None


def status_stamp(selection):
    if selection is None:
        return None
    path = selection[1] / "status.json"
    try:
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def eligible_failure(error, arguments, root, previous_status):
    if isinstance(error, SystemExit):
        if error.code is None or error.code == 0:
            return None
    elif not (type(error).__name__ == "UpstreamError" and str(error) == MODEL_GATE):
        return None
    selection = locate_run(arguments, root)
    if selection is None or status_stamp(selection) == previous_status:
        return None
    date, run = selection
    try:
        status = read_json(run / "status.json")
        identity = read_json(run / "RUN_IDENTITY.json")
        config = identity["method_config"]
        bundle = read_json(run / "upstream/neighbor_models/neighbor_bundle.json")
        if not (
            status.get("pid") == os.getpid()
            and status.get("date") == date
            and status.get("run_dir") == str(run)
            and status.get("stage") == "failed"
            and status.get("status") == "FAILED"
            and status.get("error") == MODEL_GATE_REPR
            and identity.get("date") == date
            and config.get("protocol") == FULL_PROTOCOL
            and config.get("source_selection", {}).get("policy") == SOURCE_POLICY
            and read_json(run / "method_config.json") == config
            and bundle.get("status") == "PASS"
            and bundle.get("date") == date
            and bundle.get("model_fit_date") == date
            and bundle.get("protocol") == FULL_PROTOCOL
        ):
            return None
        return run
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def normalize_run(run_dir, project_root):
    scripts = str(project_root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from normalize_neighbor_protocol import normalize_run as normalize

    return normalize(run_dir, project_root=project_root)


def repaired_route(arguments, root):
    selection = locate_run(arguments, root)
    if selection is None or "--upstream-only" in arguments:
        return None
    date, run = selection
    receipt = run / "control/derived_csv_repair/REPAIR.json"
    if not receipt.is_file():
        return None
    if read_json(receipt).get("date") != date:
        raise ValueError("Repaired input run belongs to another date")
    if any(arg.split("=", 1)[0] == "--method-config" for arg in arguments):
        raise ValueError("Repaired downstream uses its frozen registered method")
    device = "cuda"
    for index, arg in enumerate(arguments):
        if arg == "--device":
            device = arguments[index + 1]
        elif arg.startswith("--device="):
            device = arg.partition("=")[2]
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid training device")
    return run, device


def precompute_route(arguments, root):
    selection = locate_run(arguments, root)
    if selection is None or "--upstream-only" in arguments:
        return None
    date, run = selection
    request = run / "control/pg_precompute/REQUEST.json"
    if not request.is_file():
        return None
    if read_json(request).get("date") != date:
        raise ValueError("Candidate preparation belongs to another date")
    if any(arg.split("=", 1)[0] == "--method-config" for arg in arguments):
        raise ValueError("Candidate preparation uses its frozen registered method")
    return run


def main(argv=None, *, cli_main=None, normalizer=None, project_root=None):
    """Dependency injection is for tiny tests, never a command-line option."""
    root = Path(project_root or os.environ.get("PATHVEIL_PROJECT_ROOT", PROJECT)).resolve()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if cli_main is None:
        candidate_run = precompute_route(arguments, root)
        if candidate_run is not None:
            scripts = str(root / "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            from precompute_downstream_candidates import prepare_all

            prepare_all(candidate_run)
        route = repaired_route(arguments, root)
        if route is not None:
            scripts = str(root / "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            from repair_downstream_inputs import resume_run

            return resume_run(route[0], device=route[1])
    if cli_main is None:
        source_path = str(root / "src")
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        from pathveil.cli import main as cli_main
    normalizer = normalizer or normalize_run
    previous_status = status_stamp(locate_run(arguments, root))
    original_argv = sys.argv
    original_project_root = os.environ.get("PATHVEIL_PROJECT_ROOT")
    sys.argv = [str(PROJECT / "scripts/run_registered_experiment.py"), *arguments]
    os.environ["PATHVEIL_PROJECT_ROOT"] = str(root)
    try:
        try:
            return cli_main()
        except (SystemExit, Exception) as error:
            run = eligible_failure(error, arguments, root, previous_status)
            if run is None:
                raise
            try:
                receipt = Path(normalizer(run, project_root=root))
                if read_json(receipt).get("status") != "PASS":
                    raise RuntimeError("Protocol metadata normalization has no PASS receipt")
            except (SystemExit, Exception) as normalization_error:
                print(
                    "Protocol metadata normalization failed: " + str(normalization_error),
                    file=sys.stderr,
                    flush=True,
                )
                raise error from None
            print(
                json.dumps(
                    dict(
                        event="neighbor_protocol_metadata_normalized",
                        pid=os.getpid(),
                        run_dir=str(run),
                        receipt=str(receipt),
                    )
                ),
                flush=True,
            )
        # The retry intentionally sits outside the recovery handler. It cannot
        # normalize again and preserves the frozen CLI's own result/exception.
        return cli_main()
    finally:
        sys.argv = original_argv
        if original_project_root is None:
            os.environ.pop("PATHVEIL_PROJECT_ROOT", None)
        else:
            os.environ["PATHVEIL_PROJECT_ROOT"] = original_project_root


if __name__ == "__main__":
    raise SystemExit(main())
