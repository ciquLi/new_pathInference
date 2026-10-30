"""Bind the portable Tree B numerical API to the original PG environment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pathveil.downstream import common
from pathveil.utils import read_json, sha256, write_json
from . import public_tree_observed_pathgen as generator
from . import public_tree_pathgen as base


def run_pathgen(directory):
    from pathveil.core import activate
    from pathveil.data.prepare import resolve_read_path
    from pathveil.downstream import pathgen

    run = Path(sys.argv[sys.argv.index("--experiment_dir") + 1]).resolve()
    stage = sys.argv[sys.argv.index("--stage") + 1]
    prepared = common.verify_runtime_authorization(run)
    data = read_json(prepared["data_manifest"])
    profile = prepared["tree_B_generator"]
    common.verify_files(profile["code_sha256"])
    index = profile["observed_tail_index"]
    nominal = (
        Path(data["source_data_root"])
        / "data/aspath/path_by_vantage"
        / prepared["date"]
        / "path_without_prefix/train/pathWithKey.pkl"
    )
    common.require(
        resolve_read_path(data, nominal).resolve() == Path(index["path"]).resolve(),
        "Observed-tail index date/resolution differs",
    )
    common.verify_files({index["path"]: index["sha256"]})
    activate(data_manifest=prepared["data_manifest"])
    import environment
    import HighFrequencyVertexBasedPathGen as hf

    helpers, public, selectors = base.siblings()
    original_generate, original_load = hf.generate_kpath_nosave, environment.Env.getExistPath
    topology, records = None, {}

    def load(env, data_type):
        if data_type == "train":
            common.verify_files({index["path"]: index["sha256"]})
        result = original_load(env, data_type)
        if data_type == "train":
            common.verify_files({index["path"]: index["sha256"]})
        return result

    def generate(env, k, source, destinations):
        nonlocal topology
        source = str(source)
        item = prepared["public_tree_spec_files"][source]
        spec_file = (directory / (source + ".json")).resolve()
        common.require(
            str(spec_file) == item["path"] and sha256(spec_file) == item["sha256"],
            "Tree B specification changed",
        )
        spec = read_json(spec_file)
        common.require(spec["date"] == prepared["date"], "Tree B uses another date")
        common.verify_files(spec["data_code_input_sha256"])
        method = directory / "METHOD_FROZEN.json"
        common.require(
            sha256(method) == spec["construction_method_sha256"], "Tree B method changed"
        )
        identity = dict(
            date=prepared["date"],
            method_path=str(method),
            method_sha256=sha256(method),
            spec_path=str(spec_file),
            spec_sha256=sha256(spec_file),
        )
        trace = run / "control/tree_B" / (stage + "_" + source + "_queries.jsonl")
        trace.parent.mkdir(parents=True, exist_ok=True)
        with base.Heartbeat(source, stage) as progress:
            progress.update(layer="loading_public_graph")
            if topology is None:
                topology = public.PublicPrefixTopology.from_manifest(prepared["data_manifest"])
            common.require(
                topology.graph_identity_sha256 == spec["graph_identity_sha256"],
                "Tree B and current public topology differ",
            )
            weights = {
                tuple(r["prefix"]): float(r.get("structural_weight", 1.0))
                for r in spec["prefix_classification"]
            }
            selector = selectors.PublicHeadSelector.from_records(
                source, spec["prefix_classification"], identity, weights
            )
            added = env.augment_graph_for_external_nodes(destinations)
            library = hf.load_local_knowledge_paths(env, source)
            common.require(
                len(library) == prepared["admitted_pool"][source]["paths"],
                "PG local knowledge omitted admitted paths",
            )
            with trace.open("w") as stream:
                result, record = generator.generate_public_candidates(
                    env,
                    k,
                    source,
                    destinations,
                    spec,
                    selector,
                    topology,
                    helpers,
                    public.commercial_phases,
                    hf.networkit_sp,
                    library,
                    legacy_generator=None,
                    progress=progress,
                    query_trace=stream,
                )
        record.update(
            source_tree_sha256=sha256(spec_file),
            source_tree_path=str(spec_file),
            public_tree_method_sha256=sha256(method),
            generator_code_sha256=sha256(generator.__file__),
            base_generator_code_sha256=sha256(base.__file__),
            observed_tail_index=index,
            external_graph_edges_added=added,
            outputs_sha256={str(trace): sha256(trace)},
        )
        records[source] = record
        return result

    environment.Env.getExistPath, hf.generate_kpath_nosave = load, generate
    try:
        pathgen.main()
    finally:
        environment.Env.getExistPath, hf.generate_kpath_nosave = original_load, original_generate
    sources = sys.argv[sys.argv.index("--target_vases") + 1].split(",")
    for source in sources:
        marker = run / "control/pathgen_completed" / (stage + "_" + source + ".json")
        completed = read_json(marker)
        if source in records:
            trace = run / "control/tree_B" / (stage + "_" + source + ".json")
            write_json(trace, records[source])
            completed["experimental_prefix_generator"] = dict(
                trace=str(trace), sha256=sha256(trace), **records[source]
            )
            write_json(marker, completed)
        else:
            record = completed.get("experimental_prefix_generator", {})
            if not record:
                # A batch run may have produced the Tree B trace before the
                # process was interrupted while publishing its receipt. The
                # trace itself is the durable numerical output; rebuild only
                # the identity wrapper and keep all hash checks.
                trace = run / "control/tree_B" / (stage + "_" + source + "_queries.jsonl")
                if trace.is_file():
                    spec_file = Path(prepared["public_tree_spec_files"][source]["path"]).resolve()
                    method = directory / "METHOD_FROZEN.json"
                    record = dict(
                        trace=str(trace),
                        sha256=sha256(trace),
                        source_tree_sha256=sha256(spec_file),
                        source_tree_path=str(spec_file),
                        public_tree_method_sha256=sha256(method),
                        generator_code_sha256=sha256(generator.__file__),
                        base_generator_code_sha256=sha256(base.__file__),
                        outputs_sha256={str(trace): sha256(trace)},
                    )
                    completed["experimental_prefix_generator"] = record
                    write_json(marker, completed)
            common.require(
                record.get("generator_code_sha256") == sha256(generator.__file__),
                "Completed PATS stage lacks this Tree B identity",
            )
            common.verify_files(record["outputs_sha256"])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prefix-tree-dir", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    old = sys.argv
    sys.argv = [old[0], *forwarded]
    try:
        run = Path(forwarded[forwarded.index("--experiment_dir") + 1])
        prepared = common.verify_runtime_authorization(run)
        stage = forwarded[forwarded.index("--stage") + 1]
        if stage == "training_inputs" and prepared.get("training_generator") == "PG":
            from pathveil.downstream import pathgen

            pathgen.main()
        else:
            run_pathgen(args.prefix_tree_dir.resolve())
    finally:
        sys.argv = old
