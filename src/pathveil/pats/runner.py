"""Bridge the formal date entry to the frozen PATS numerical workflow."""
from __future__ import annotations

import json
from pathlib import Path

from pathveil.data.preflight import check_date
from pathveil.utils import project_root, read_json, utc_now

from .policy import POLICY, runtime_policy
from .workflow import prepare, execute as run


def execute(args, registry):
    root = project_root()
    if args.check_only:
        report = check_date(args.date, deep=args.deep_check)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["raw_ready"] else 2
    method_path = Path(getattr(args, "method_config", None) or registry["workflow_config"])
    if not method_path.is_absolute():
        method_path = root / method_path
    method = read_json(method_path)
    policy = runtime_policy(method)
    if args.plan_only:
        baseline_workflow = registry.get("baseline_workflow", "rho_prefix_v1")
        current = root / "runs" / args.date / baseline_workflow / "current.json"
        if not current.is_file() or not (Path(read_json(current)["run_dir"]) / "PROTOCOL.json").is_file():
            print(json.dumps(dict(date=args.date, workflow=POLICY["version"],
                status="INPUT_PREPARATION_REQUIRED", selection_policy=policy,
                sources=registry["test_sources"], bootstrap_inputs=registry.get("bootstrap_inputs",False),
                new_training_started=False), indent=2, ensure_ascii=False))
            return 0
        baseline = Path(read_json(current)["run_dir"])
        proto = read_json(baseline / "PROTOCOL.json")
        print(json.dumps(dict(
            date=args.date,
            workflow=POLICY["version"],
            config=str(method_path),
            sources=[dict(
                source=str(s),
                rho=proto["public_features"][str(s)]["rho"],
                root_limit=6 if proto["public_features"][str(s)]["rho"] > 1 else policy["low_root_limit"],
                mandatory_recursion=proto["public_features"][str(s)]["rho"] > 1,
                secondhop_topk=policy["recursion_topk"],
            ) for s in registry["test_sources"]],
            selection_policy=policy,
            new_training_started=False,
        ), indent=2, ensure_ascii=False))
        return 0
    requested = args.run_dir
    if requested and (Path(requested) / "PROTOCOL.json").is_file():
        from pathveil.experiment.contracts import require, verify_files

        native = Path(requested).resolve()
        require(root / "runs" in native.parents, "Run must stay inside project runs/")
        proto = read_json(native / "PROTOCOL.json")
        require(proto["date"] == args.date and proto["sources"] == registry["test_sources"],
                "Resume date/source registry differs")
        require(proto["method"]["selection_policy"] == policy,
                "Resume parameters differ from the frozen method")
        require(not getattr(args, "fresh_run", False), "Use either --fresh-run or --run-dir")
        verify_files(proto["code_sha256"])
        if args.prepare_only or args.upstream_only:
            return 0
        if args.augmentation_only:
            from .workflow import build_all
            return build_all(native)
        run(native, args.device)
        return 0
    if getattr(args, "fresh_run", False):
        if requested:
            raise ValueError("Use either --fresh-run or --run-dir")
        requested = root / "runs" / args.date / POLICY["version"] / (
            "fresh_" + utc_now().replace(":", "").replace("-", "").replace(".", "_")
        )
    native = prepare(root, args.date, requested, getattr(args, "method_config", None))
    if args.prepare_only or args.upstream_only:
        return 0
    if args.augmentation_only:
        from .workflow import build_all
        return build_all(native)
    run(native, args.device)
    return 0
