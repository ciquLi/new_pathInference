"""Prepare date-local PATS inputs and upstream models without running a baseline."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from pathveil.data.bgp_holdout import ensure_cohort
from pathveil.data.registry import load_registry
from pathveil.experiment import assets, preparation
from pathveil.experiment.contracts import freeze, require
from pathveil.experiment.runner import structure
from pathveil.prefix_tree.registered_topology import RegisteredPublicPrefixTopology as PublicPrefixTopology
from pathveil.utils import read_json, sha256, utc_now, write_json


def ensure_inputs(root, date):
    """Fit only missing same-date assets; publish the protocol required by PATS."""
    registry = load_registry(date, project_root=root)
    parent = root / "runs" / date / registry["baseline_workflow"]
    run = parent / "inputs"
    run.mkdir(parents=True, exist_ok=True)
    write_json(parent / "current.json", dict(date=date, run_dir=str(run), updated_utc=utc_now()))
    with (run / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def progress(stage, status="RUNNING", **details):
            state = dict(status=status, stage=stage, date=date, pid=os.getpid(),
                         run_dir=str(run), updated_utc=utc_now(), **details)
            write_json(run / "STATUS.json", state)
            (run / "PROGRESS.md").write_text(
                f"# {date} PATS 输入准备\n\n阶段：{stage}\n状态：{status}\n"
                f"更新时间（UTC）：{state['updated_utc']}\n"
                + "".join(f"{key}：{value}\n" for key, value in details.items()))
            print(state, flush=True)

        try:
            ensure_cohort(registry, progress)
            progress("data_prepare")
            data = assets.prepare_data(date, run)
            progress("independent_training_source_view")
            assets.downstream_view(data, run)
            upstream_config = read_json(root / registry["method_config"])
            progress("date_local_upstream_fit")
            assets.obtain_upstream(run, data, registry, upstream_config)
            require(read_json(run / "UPSTREAM_BOUND.json")["run"] ==
                    str((root / registry["upstream_cache"]).resolve()),
                    "Bootstrap upstream does not match date registration")
            topology = PublicPrefixTopology.from_manifest(run / "data/data_manifest.json")
            features, neighbors, origins = structure(topology, registry["test_sources"])
            require(all(v["rho"] is not None for v in features.values()),
                    "Registered source has unknown public topology")
            method = read_json(root / "configs/workflow_rho_prefix.json")
            method["upstream_method"] = upstream_config
            method["upstream_config_sha256"] = sha256(root / registry["method_config"])
            protocol = dict(
                version="pats_inputs_v1", date=date, sources=list(map(str, registry["test_sources"])),
                common9=list(map(str, registry["common9"])), method=method,
                upstream_config=upstream_config, public_features=features,
                neighbors=neighbors, score_origins=origins,
                code_sha256=preparation.code_identity(),
                data_manifest_sha256=sha256(run / "data/data_manifest.json"),
                upstream_binding_sha256=sha256(run / "UPSTREAM_BOUND.json"),
                registry_sha256=sha256(registry["registry_path"]),
                final_test_labels_for_training_or_selection=False,
                unknown_relationships_preserved=True,
                baseline_inference_executed=False,
            )
            freeze(run / "PROTOCOL.json", protocol)
            progress("inputs_ready", "COMPLETED")
            return run
        except BaseException as error:
            progress("failed", "FAILED", error=repr(error))
            raise
