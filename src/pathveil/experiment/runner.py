"""The registered date selects all inputs, fit scopes and rho branches."""
from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from importlib import metadata

from pathveil.data.preflight import check_date
from pathveil.data.registry import select_asset
from pathveil.prefix_tree.public_prefix_tree import PublicPrefixTopology
from pathveil.utils import project_root, read_json, sha256, utc_now, write_json
from . import assets, augmentation, preparation
from .contracts import freeze, require, verify_files


VERSION = "rho_prefix_v1"


def load_configuration(registry, override=None):
    root = project_root()
    path = Path(override or registry["workflow_config"])
    method = read_json(path if path.is_absolute() else root / path)
    require(method["workflow_id"] == VERSION, "Expected a rho_prefix_v1 workflow configuration")
    upstream = root / method["upstream_config"]
    method = copy.deepcopy(method)
    method["upstream_method"] = read_json(upstream)
    method["upstream_config_sha256"] = sha256(upstream)
    packages = {}
    for name in ("torch", "numpy", "lightgbm", "networkit", "scikit-learn"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    method["runtime"] = dict(
        python=platform.python_version(), executable=sys.executable, packages=packages
    )
    validate_configuration(method)
    return method


def validate_configuration(method):
    policy, resources = method["augmentation"], method["resources"]
    require(
        policy["rho_threshold"] == 1
        and policy["maximum_neighbor_prefix_AS_count"] == 3
        and policy["maximum_spliced_prefix_AS_count"] == 4
        and policy["maximum_full_path_AS_count"] == 20,
        "This method freezes the rho1 and prefix/full-path length definitions",
    )
    require(
        policy["quantity_cap"] is None and not policy["quality_regression_supplement"],
        "Unregistered augmentation cap or supplementary quality model",
    )
    require(
        all(
            isinstance(policy[k], (float, int))
            and not isinstance(policy[k], bool)
            and 0 < policy[k] <= 1
            for k in ("source_threshold", "neighbor_threshold")
        ),
        "Quality thresholds must be in (0,1]",
    )
    require(
        isinstance(policy["low_minimum_paths"], int)
        and not isinstance(policy["low_minimum_paths"], bool)
        and policy["low_minimum_paths"] > 0,
        "Invalid low-rho minimum path count",
    )
    require(
        method["query_generator"] == "Tree B + PG"
        and method["gmpi"]["k"] == 200
        and method["gmpi"]["workers"] == 0,
        "Unregistered query generator or candidate capacity",
    )
    require(
        method["training_objective"] == "normal_history_huber",
        "Only normal-history Huber is registered",
    )
    for group in ("high", "low"):
        require(
            method["training_generators"][group] in {"PG", "Tree B + PG"},
            "Unknown training generator",
        )
    for key, value in resources.items():
        require(
            isinstance(value, (float, int)) and not isinstance(value, bool) and value > 0,
            "Resource limits must be positive: " + key,
        )
    require(
        resources["gpu_workers"] in (1, 2)
        and isinstance(resources["cpu_workers"], int)
        and isinstance(resources["gmpi_cpu_threads"], int)
        and resources["gmpi_cpu_threads"] <= 16,
        "Invalid worker or thread limits",
    )


def public_manifest(registry):
    selected = {asset["id"]: select_asset(asset) for asset in registry["raw_inputs"]}
    fields = {
        "asrank_nodes": "asrank_asns",
        "asrank_links": "asrank_asnLinks",
        "asrel": "relationships",
        "peeringdb": "peeringdb",
    }
    require(
        all(selected.get(name) for name in fields.values()),
        "Exact-date public inputs are incomplete",
    )
    public = {key: str(selected[name].resolve()) for key, name in fields.items()}
    return dict(
        date=registry["date"],
        routing_date=registry["routing_date"],
        auxiliary_dates=registry["auxiliary_dates"],
        public_files=public,
        public_input_sha256={k: sha256(v) for k, v in public.items()},
    )


def structure(topology, sources):
    features, neighbors, origins = {}, {}, set()
    for source in sources:
        source = str(source)
        row = topology.source_features(source)
        b1, distinct = int(row["public_b1"]), int(row["public_l2_distinct_asns"])
        rho = distinct / (b1 * b1) if b1 else None
        features[source] = dict(
            source=source, public_b1=b1, public_l2_distinct_asns=distinct, rho=rho
        )
        if rho is None:
            neighbors[source] = []
        elif rho > 1:
            neighbors[source] = sorted(map(str, topology.neighbors(source)), key=int)
            origins.update(neighbors[source])
        else:
            neighbors[source] = []
            origins.add(source)
    return features, neighbors, sorted(origins, key=int)


def plan(registry, method, deep=False):
    report = check_date(registry["date"], deep=deep)
    graph = Path(registry["files_root"]) / "data/aspath/path_by_vantage" / registry["routing_date"]
    required = [
        graph / "path_without_prefix/train" / name
        for name in ("merged_paths.txt", "expand_paths.txt", "pathWithKey.pkl")
    ]
    required += [graph / "RL_Graph/vantage_dst_test.json"]
    required += [
        graph / "path_without_prefix/test" / (str(s) + ".csv") for s in registry["test_sources"]
    ]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    result = dict(
        status="INPUTS_READY" if report["raw_ready"] and not missing else "INPUTS_MISSING",
        date=registry["date"],
        mode=registry["mode"],
        sources=registry["test_sources"],
        workflow=VERSION,
        new_training_started=False,
        preflight=report,
        missing_processed_inputs=missing,
        augmentation=method["augmentation"],
        training_objective=method["training_objective"],
        resources=method["resources"],
        trained_models_are_date_local=True,
        final_labels_for_model_selection=False,
    )
    if report["raw_ready"]:
        topology = PublicPrefixTopology.from_manifest(public_manifest(registry))
        features, neighbors, origins = structure(topology, registry["test_sources"])
        result.update(public_features=features, neighbors=neighbors, score_origins=origins)
        if any(v["rho"] is None for v in features.values()):
            result["status"] = "UNKNOWN_SOURCE_STRUCTURE"
    return result


def execute(args, registry):
    from pathveil.cli import choose_run, run_identity
    from .schedule import Scheduler, Task, downstream_tasks
    from .report import publish

    method = load_configuration(registry, args.method_config)
    if args.check_only or getattr(args, "plan_only", False):
        value = plan(registry, method, args.deep_check)
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        return 0 if value["status"] == "INPUTS_READY" else 2
    require(registry["mode"] == "bgp", "Rho workflow is registered for BGP datasets only")
    root = project_root()
    requested = args.run_dir
    if getattr(args, "fresh_run", False):
        require(requested is None, "Choose either --fresh-run or --run-dir")
        requested = (
            root / "runs" / args.date / VERSION / utc_now().replace(":", "").replace(".", "_")
        )
    identity = run_identity(args.date, registry, method)
    run = choose_run(root, args.date, identity, requested)
    freeze(run / "method_config.json", method)
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME"):
        folder = run / "runtime" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    with (run / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This date/run is already active: " + str(run)) from error

        def status(stage, state="RUNNING", **extra):
            value = dict(
                status=state,
                stage=stage,
                date=args.date,
                run_dir=str(run),
                pid=os.getpid(),
                updated_utc=utc_now(),
                **extra,
            )
            write_json(run / "status.json", value)
            write_json(run / "STATUS.json", dict(value, active=[]))
            print(json.dumps(value), flush=True)

        try:
            status("data_prepare")
            data = assets.prepare_data(args.date, run)
            view = assets.downstream_view(data, run)
            if args.prepare_only:
                status("data_prepare", "COMPLETED", data_manifest=str(view))
                return 0
            status("upstream_bind_or_fit")
            assets.obtain_upstream(run, data, registry, method["upstream_method"])
            if args.upstream_only:
                status("upstream", "COMPLETED")
                return 0
            topology = PublicPrefixTopology.from_manifest(run / "data/data_manifest.json")
            features, neighbors, origins = structure(topology, registry["test_sources"])
            require(
                all(v["rho"] is not None for v in features.values()),
                "Unresolved source structure; no implicit low-rho fallback",
            )
            protocol = dict(
                version=VERSION,
                date=args.date,
                sources=list(map(str, registry["test_sources"])),
                common9=list(map(str, registry.get("common9", registry["test_sources"]))),
                method=method,
                upstream_config=method["upstream_method"],
                public_features=features,
                neighbors=neighbors,
                score_origins=origins,
                code_sha256=preparation.code_identity(),
                data_manifest_sha256=sha256(run / "data/data_manifest.json"),
                upstream_binding_sha256=sha256(run / "UPSTREAM_BOUND.json"),
                registry_sha256=sha256(registry["registry_path"]),
                final_test_labels_for_training_or_selection=False,
                unknown_relationships_preserved=True,
            )
            freeze(run / "PROTOCOL.json", protocol)
            status("score_pools")
            Scheduler(
                run, [Task("score", s) for s in origins], method["resources"], device="cpu"
            ).execute()
            status("augmentation")
            libraries = {
                s: augmentation.build_library(run, s, topology) for s in protocol["sources"]
            }
            freeze(
                run / "ALL_LIBRARIES_FROZEN.json",
                dict(
                    status="FROZEN",
                    date=args.date,
                    libraries={s: r["path_set_sha256"] for s, r in libraries.items()},
                    admissions_sha256={
                        str(run / "augmentation" / ("as" + s) / "ADMISSION.json"): sha256(
                            run / "augmentation" / ("as" + s) / "ADMISSION.json"
                        )
                        for s in libraries
                    },
                    final_labels_used=False,
                ),
            )
            if getattr(args, "augmentation_only", False):
                status("augmentation", "COMPLETED")
                return 0
            status("downstream_prepare")
            preparation.prepare_group(run, view, topology)
            publish(run)
            status("downstream")
            Scheduler(
                run, downstream_tasks(protocol["sources"]), method["resources"], args.device
            ).execute()
            result = publish(run)
            require(result["status"] == "COMPLETED", "Not every registered AS completed evaluation")
            write_json(
                run / "STATUS.json",
                dict(
                    status="COMPLETED",
                    date=args.date,
                    active=[],
                    completed_sources=len(libraries),
                    updated_utc=utc_now(),
                    result=str(run / "RESULTS.json"),
                ),
            )
            status("complete", "COMPLETED", summary=str(run / "REPORT.md"))
            return 0
        except BaseException as error:
            status("failed", "FAILED", error=repr(error))
            (run / "error.txt").write_text(traceback.format_exc())
            raise
