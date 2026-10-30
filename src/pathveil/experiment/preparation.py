"""Prepare one admitted library for both labels and PG local knowledge."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from pathveil.data.prepare import resolve_read_path
from pathveil.downstream import common
from pathveil.downstream.runner import prepare_downstream
from pathveil.prefix_tree import public_tree_spec as specs
from pathveil.utils import project_root, read_json, sha256, write_json
from .contracts import freeze, require, verify_files


def code_identity():
    root = project_root()
    files = sorted((root / "src/pathveil/experiment").glob("*.py"))
    files += sorted((root / "src/pathveil/prefix_tree").glob("*.py"))
    files += sorted((root / "src/pathveil/downstream").glob("*.py"))
    files += sorted((root / "src/pathveil/core").glob("*.py"))
    return {str(p): sha256(p) for p in files}


def prepare_group(run, view_manifest, topology):
    run = Path(run)
    protocol = read_json(run / "PROTOCOL.json")
    data = read_json(view_manifest)
    group = run / "T-PG-B"
    method = dict(
        version=protocol["version"],
        date=protocol["date"],
        sources=protocol["sources"],
        public_features=protocol["public_features"],
        generator="Tree B + PG",
        normal_history_only=True,
        training_objective=protocol["method"]["training_objective"],
        protocol_sha256=sha256(run / "PROTOCOL.json"),
    )
    freeze(group / "METHOD_FROZEN.json", method)
    freeze(
        run / "EVALUATION_POLICY_AMENDMENT.json",
        dict(
            selection_scope="single_source_generator",
            training_and_selection_unchanged=True,
            final_scores_used_for_further_training=False,
        ),
    )
    upstream = read_json(run / "UPSTREAM_BOUND.json")
    release = read_json(Path(upstream["run"]) / "upstream/RELEASE.json")
    graph = {k: release[k] for k in ("selected_neighbors_csv", "recursive_anchors_csv")}
    trees = run / "query_trees"
    freeze(
        trees / "METHOD_FROZEN.json",
        dict(
            version=protocol["version"],
            date=protocol["date"],
            sources=protocol["sources"],
            configuration=specs.CONFIG,
            public_graph_identity_sha256=topology.graph_identity_sha256,
            public_inputs_sha256=topology.inputs_sha256,
            inputs_sha256={str(run / "PROTOCOL.json"): sha256(run / "PROTOCOL.json")},
            spec_code_sha256=sha256(specs.__file__),
        ),
    )
    for source in protocol["sources"]:
        prepare_source(run, source, view_manifest, topology, graph)
    return group


def prepare_source(run, source, view_manifest, topology, graph):
    run, source = Path(run), str(source)
    protocol = read_json(run / "PROTOCOL.json")
    item_file = run / "augmentation" / ("as" + source) / "ADMISSION.json"
    item = read_json(item_file)
    require(
        item["date"] == protocol["date"]
        and item["protocol_sha256"] == sha256(run / "PROTOCOL.json"),
        "Admission date/method differs",
    )
    verify_files({item["path"]: item["sha256"], item["json_path"]: item["json_sha256"]})
    child = run / "T-PG-B/pg_inputs" / ("as" + source)
    ready_file = child / "PREPARED.json"
    if ready_file.exists():
        prepared = common.verify_runtime_authorization(child)
        require(
            prepared["admitted_library_path_set_sha256"][source] == item["path_set_sha256"],
            "Existing consumer library differs",
        )
        return ready_file
    trees = run / "query_trees"
    paths = common.csv_paths(item["path"], source)
    tree = specs.build_spec(
        topology, source, [p[:3] for p in paths], {str(item_file): sha256(item_file)}
    )
    tree.update(
        admitted_path_set_sha256=item["path_set_sha256"],
        library_csv=item["path"],
        library_csv_sha256=item["sha256"],
        construction_method_sha256=sha256(trees / "METHOD_FROZEN.json"),
    )
    tree_file = trees / (source + ".json")
    freeze(tree_file, tree)
    data = copy.deepcopy(read_json(view_manifest))
    data.update(sources=[source], common9=[source])
    manifest = child / "data/data_manifest.json"
    freeze(manifest, data)
    outputs = {item["path"]: item["sha256"], item["json_path"]: item["json_sha256"]}
    outputs.update({name: sha256(name) for name in graph.values()})
    subset = child / "upstream/RELEASE.json"
    freeze(
        subset,
        dict(
            status="PASS",
            date=data["date"],
            sources=[source],
            files_by_source={source: item},
            outputs_sha256=outputs,
            execution_mode=protocol["version"],
            **graph,
        ),
    )
    prepare_downstream(child, manifest, subset)
    prepared = read_json(ready_file)
    prepared["configuration"].update(protocol["method"]["gmpi"], date=data["date"])
    rho_group = "high" if protocol["public_features"][source]["rho"] > 1 else "low"
    prepared["training_generator"] = protocol["method"]["training_generators"][rho_group]
    prepared["admitted_library_path_set_sha256"] = {source: item["path_set_sha256"]}
    # Only this endpoint inventory is passed to the test candidate compiler.
    nominal = Path(data["graph_dir"]) / "vantage_dst_test.json"
    endpoint_file = resolve_read_path(data, nominal).resolve()
    require(
        endpoint_file.is_file(), "Complete endpoint-only metadata missing: " + str(endpoint_file)
    )
    metadata = read_json(endpoint_file)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    require(
        isinstance(metadata, dict) and isinstance(metadata.get(source), list),
        "Endpoint metadata omits source",
    )
    pairs = [source + "->" + str(d) for d in metadata[source]]
    require(len(pairs) == len(set(pairs)), "Duplicate endpoint metadata pair")
    for pair in pairs:
        fields = pair.split("->")
        require(
            len(fields) == 2
            and all(
                n.isascii() and n.isdecimal() and 0 < int(n) < 2**32 and str(int(n)) == n
                for n in fields
            ),
            "Invalid endpoint metadata pair",
        )
    pairs.sort(key=lambda p: tuple(map(int, p.split("->"))))
    identity = prepared["test_identity"][source]
    digest = common.pair_identity(pairs)
    require(
        len(pairs) == identity["pairs"] and digest == identity["pair_inventory_sha256"],
        "Endpoint metadata does not cover the complete fixed test universe",
    )
    inventory = child / "control/fixed_test_pairs" / (source + ".json")
    freeze(
        inventory,
        dict(
            version="complete_endpoint_inventory_v1",
            date=data["date"],
            source=source,
            pairs=pairs,
            pair_count=len(pairs),
            pair_inventory_sha256=digest,
            inventory_source="registered_complete_endpoint_metadata",
            route_sequences_imported=False,
            endpoint_metadata=dict(path=str(endpoint_file), sha256=sha256(endpoint_file)),
        ),
    )
    prepared["fixed_test_pairs_inventory"] = {
        source: dict(
            path=str(inventory),
            sha256=sha256(inventory),
            pairs=len(pairs),
            pair_inventory_sha256=digest,
        )
    }
    nominal_index = (
        Path(data["source_data_root"])
        / "data/aspath/path_by_vantage"
        / data["date"]
        / "path_without_prefix/train/pathWithKey.pkl"
    )
    index = resolve_read_path(data, nominal_index).resolve()
    require(
        index.is_file() and "train" in index.parts and "test" not in index.parts,
        "Missing training tail index",
    )
    code = code_identity()
    prepared["tree_B_generator"] = dict(
        code_sha256=code,
        observed_tail_index=dict(path=str(index), sha256=sha256(index), bytes=index.stat().st_size),
        maximum_prefix_AS_count=3,
    )
    prepared["public_tree_spec_files"] = {
        source: dict(path=str(tree_file), sha256=sha256(tree_file))
    }
    prepared["prepared_files"].update(
        {
            str(p): sha256(p)
            for p in [inventory, tree_file, endpoint_file, run / "PROTOCOL.json", item_file]
        }
    )
    prepared["code_sha256"].update(code)
    write_json(ready_file, prepared)
    common.verify_runtime_authorization(child)
    return ready_file
