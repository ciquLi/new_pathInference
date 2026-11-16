"""Isolated generator/recursion ablations calling unchanged production APIs."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from pathInference.experiment.contracts import (
    forbid_formal_references, freeze, freeze_bytes, require, signature, verify_files,
)
from pathInference.utils import project_root, read_json, sha256, utc_now, write_json

ROOT = project_root()
NAME = "generator_recursion_v1"
CONFIG = ROOT / "research" / NAME / "CONFIG.json"
BATCH = ROOT / "runs/ablations" / NAME


def code_identity():
    return {str(p): sha256(p) for p in sorted((ROOT / "src/pathInference").rglob("*.py"))}


def baseline(date):
    return ROOT / read_json(CONFIG)["dates"][date]


def run_dir(date, variant):
    return ROOT / "runs" / date / NAME / variant


def asset(path):
    """Only reuse historical assets whose original release hashes match."""
    path = Path(path).resolve()
    inventory = read_json(ROOT / read_json(CONFIG)["baseline_manifest"])["baseline_assets_sha256"]
    key = str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)
    expected = inventory.get(key)
    if expected is None and "construction" in path.parts:
        run = path.parents[2]
        tree_file = run / "query_trees" / (path.parent.name + ".json")
        tree_key = str(tree_file.relative_to(ROOT))
        require(tree_key in inventory, "No frozen construction tree identity")
        verify_files({str(tree_file): inventory[tree_key]})
        expected = read_json(tree_file)["data_code_input_sha256"].get(str(path))
    if expected is None and path.parent.name == "independent_evaluation":
        run = path.parents[2]
        results = read_json(asset(run / "RESULTS.json"))
        source = path.stem.removeprefix("source_")
        row = next(r for r in results["per_source"] if r["source_asn"] == source)
        require(read_json(path) == row, "Independent source result differs from frozen report")
        return path
    require(expected is not None, "No frozen asset identity: " + key)
    verify_files({str(path): expected})
    return path


def setup(date, variant):
    config = read_json(CONFIG)
    require(date in config["dates"] and variant in config["variants"], "Unknown experiment")
    run, old = run_dir(date, variant), baseline(date)
    run.mkdir(parents=True, exist_ok=True)
    identity = dict(config_sha256=sha256(CONFIG), code_sha256=code_identity(),
                    baseline_protocol_sha256=sha256(asset(old / "PROTOCOL.json")),
                    date=date, variant=variant)
    freeze(run / "IDENTITY.json", identity)
    if (run / "PROTOCOL.json").exists():
        verify_files(read_json(run / "PROTOCOL.json")["code_sha256"])
        return run
    proto = copy.deepcopy(read_json(old / "PROTOCOL.json"))
    proto.update(version=NAME, code_sha256=identity["code_sha256"],
                 ablation_baseline=str(old), ablation=config["variants"][variant],
                 ablation_variant=variant, ablation_config_sha256=sha256(CONFIG))
    proto["method"]["resources"].update(cpu_workers=2, gpu_workers=1, gmpi_cpu_threads=2)
    freeze(run / "PROTOCOL.json", proto)
    for relative in ("UPSTREAM_BOUND.json", "data/data_manifest.json"):
        freeze(run / relative, read_json(asset(old / relative)))
    method = read_json(asset(old / "T-PG-B/METHOD_FROZEN.json"))
    method.update(version=NAME, protocol_sha256=sha256(run / "PROTOCOL.json"), ablation=proto["ablation"])
    freeze(run / "T-PG-B/METHOD_FROZEN.json", method)
    freeze(run / "EVALUATION_POLICY_AMENDMENT.json", dict(
        selection_scope="single_source_generator", training_and_selection_unchanged=True,
        final_scores_used_for_further_training=False))
    tree_method = read_json(asset(old / "query_trees/METHOD_FROZEN.json"))
    tree_method.update(version=NAME, inputs_sha256={str(run / "PROTOCOL.json"): sha256(run / "PROTOCOL.json")},
                       ablation=proto["ablation"])
    freeze(run / "query_trees/METHOD_FROZEN.json", tree_method)
    return run


def clone_library(run, source):
    """Rebind a verified immutable library/tree, never change path membership."""
    proto = read_json(run / "PROTOCOL.json")
    old = Path(proto["ablation_baseline"])
    directory = run / "augmentation" / ("as" + source)
    directory.mkdir(parents=True, exist_ok=True)
    original = asset(old / "augmentation" / ("as" + source) / "ADMISSION.json")
    admission = copy.deepcopy(read_json(original))
    for suffix in ("csv", "json"):
        file = Path(admission["path" if suffix == "csv" else "json_path"])
        verify_files({str(file): admission["sha256" if suffix == "csv" else "json_sha256"]})
        freeze_bytes(directory / ("paths." + suffix), file.read_bytes())
    admission.update(path=str(directory / "paths.csv"), json_path=str(directory / "paths.json"),
                     protocol_sha256=sha256(run / "PROTOCOL.json"),
                     reused_baseline_admission=dict(path=str(original), sha256=sha256(original)))
    freeze(directory / "ADMISSION.json", admission)
    tree = copy.deepcopy(read_json(asset(old / "query_trees" / (source + ".json"))))
    # Historical Python paths were renamed in the verified package migration.
    # Preserve all numerical/input provenance while explicitly binding live code.
    tree["data_code_input_sha256"] = {
        p: h for p, h in tree["data_code_input_sha256"].items() if not p.endswith(".py")}
    verify_files(tree["data_code_input_sha256"])
    tree["data_code_input_sha256"].update(proto["code_sha256"])
    tree["data_code_input_sha256"][str(directory / "ADMISSION.json")] = sha256(directory / "ADMISSION.json")
    tree.update(library_csv=admission["path"], library_csv_sha256=admission["sha256"],
                construction_method_sha256=sha256(run / "query_trees/METHOD_FROZEN.json"))
    freeze(run / "query_trees" / (source + ".json"), tree)
    from pathInference.experiment import preparation
    from pathInference.prefix_tree.registered_topology import RegisteredPublicPrefixTopology
    topology = RegisteredPublicPrefixTopology.from_manifest(run / "data/data_manifest.json")
    release = read_json(Path(read_json(run / "UPSTREAM_BOUND.json")["run"]) / "upstream/RELEASE.json")
    graph = {k: release[k] for k in ("selected_neighbors_csv", "recursive_anchors_csv")}
    previous = preparation.specs.build_spec
    preparation.specs.build_spec = lambda *args, **kwargs: copy.deepcopy(tree)
    try:
        preparation.prepare_source(run, source, Path(proto["baseline_run"]) / "data/downstream_view/data_manifest.json", topology, graph)
    finally:
        preparation.specs.build_spec = previous


def recursion_plan(source, roots, capacity, predict_next, mode, rho, width=5, threshold=300):
    """Only the recursion trigger changes; legal first-hop tails remain usable."""
    from pathInference.pats.policy import mandatory_plan
    require(mode in {"none", "all"}, "Use the unchanged production adapter for current")
    if mode == "all" and rho > 1:
        return mandatory_plan(source, roots, capacity, predict_next)
    prefixes, decisions = [], []
    for root in map(str, roots):
        prefix = (str(source), root)
        require(len(set(prefix)) == 2, "Cyclic root")
        current = capacity(prefix, threshold)
        if mode == "none" or current["valid_tail_count"]:
            prefixes.append(prefix)
        decision = dict(prefix=prefix, capacity=current)
        if mode == "none":
            decision["stop_reason"] = "ablation_no_recursion"
        else:
            children = []
            for entry in predict_next(prefix).get("candidates", []):
                node = str(entry["asn"])
                if node not in prefix and entry.get("admitted", False) and node not in children:
                    children.append(node)
                if len(children) == width:
                    break
            decision.update(expanded_neighbors=children, forced_expansion=True,
                            stop_reason=None if children else "no_supported_unvisited_next_hop")
            for child in children:
                extended = prefix + (child,)
                child_capacity = capacity(extended, threshold)
                if child_capacity["valid_tail_count"]:
                    prefixes.append(extended)
                decisions.append(dict(prefix=extended, capacity=child_capacity,
                                      stop_reason="depth_limit_with_insufficient_tail_evidence"))
        decisions.append(decision)
    return dict(source=str(source), prefixes=prefixes, decisions=decisions,
                policy=dict(valid_tail_count_min=threshold, max_prefix_AS_count=3, next_hop_topk=width),
                insufficient_leaves=[], no_task_reference_access=True, source_total_degree_not_used=True)


def prepare_source(date, variant, source):
    from pathInference.pats import construction
    run = setup(date, variant)
    proto = read_json(run / "PROTOCOL.json")
    require(source in proto["sources"], "Unregistered AS")
    ready = run / "control/research_ready" / (source + ".json")
    if ready.exists():
        verify_files(read_json(ready)["outputs_sha256"])
        return
    mode = proto["ablation"]["recursion"]
    reuse = mode == "conditional" or (mode == "all" and proto["public_features"][source]["rho"] > 1)
    root_file = asset(baseline(date) / "construction" / source / "root.json")
    frozen_root = read_json(root_file)
    roots = list(map(str, frozen_root["selected_neighbors"]))
    with frozen_reference_registration(date, source), forbid_formal_references():
        if reuse:
            clone_library(run, source)
        else:
            old_runtime, old_adapter = construction.Runtime, construction.Adapter

            class Runtime(old_runtime):
                def _infer_root(self, requested):
                    require(str(requested) == source, "Wrong frozen root")
                    return copy.deepcopy(frozen_root)

            class Adapter(old_adapter):
                def plan(self):
                    require(list(map(str, self.prediction["selected_neighbors"])) == roots,
                            "Firsthop selection changed")
                    value = recursion_plan(self.source, roots, self.capacity, self.predict_next, mode, self.rho)
                    value["prefix_edge_provenance"] = {"->".join(p): v for p, v in self.prefix_proofs.items()}
                    value["frontier_prediction_queries"] = {"->".join(p): v for p, v in self.queries.items()}
                    return value

            construction.Runtime, construction.Adapter = Runtime, Adapter
            try:
                construction.construct(run, source)
            finally:
                construction.Runtime, construction.Adapter = old_runtime, old_adapter
    child = run / "T-PG-B/pg_inputs" / ("as" + source)
    prepared = read_json(child / "PREPARED.json")
    prepared["code_sha256"].update(proto["code_sha256"])
    prepared["tree_B_generator"]["code_sha256"].update(proto["code_sha256"])
    prepared["experimental_generator"] = dict(name=proto["ablation"]["generator"],
                                               research_code_sha256=sha256(__file__), variant=variant)
    prepared["prepared_files"][str(run / "IDENTITY.json")] = sha256(run / "IDENTITY.json")
    write_json(child / "PREPARED.json", prepared)
    tree = read_json(run / "query_trees" / (source + ".json"))
    require({p["prefix"][1] for p in tree["primary_leaves"]} <= set(roots), "Extra firsthop introduced")
    audit = dict(date=date, source=source, variant=variant, selected_firsthops=roots,
                 baseline_root=dict(path=str(root_file), sha256=sha256(root_file)),
                 frozen_firsthop_selection=True, reused_library=reuse,
                 final_labels_read=False, prepared_sha256=sha256(child / "PREPARED.json"))
    freeze(run / "control/variable_audits" / (source + ".json"), audit)
    outputs = [child / "PREPARED.json", run / "query_trees" / (source + ".json"),
               run / "augmentation" / ("as" + source) / "ADMISSION.json"]
    freeze(ready, dict(status="PASS", outputs_sha256={str(p): sha256(p) for p in outputs}))


@contextmanager
def frozen_reference_registration(date, source):
    """Reuse registered endpoint/hash metadata without reopening final routes."""
    from pathInference.downstream import runner
    results = read_json(asset(baseline(date) / "RESULTS.json"))
    record = next(row for row in results["per_source"] if row["source_asn"] == source)
    identity = dict(record["fixed_identity"], reference_file=record["reference_file"],
                    observations=record["reference_observations"],
                    registration="Reused frozen endpoint count and input byte identity; no reference routes read")
    previous = runner.reference_identity

    def registered(path, requested):
        require(str(requested) == source and Path(path).resolve() == Path(identity["reference_file"]).resolve(),
                "Reference registration changed source/date/path")
        return copy.deepcopy(identity)

    runner.reference_identity = registered
    try:
        yield
    finally:
        runner.reference_identity = previous


@contextmanager
def legacy_generator(prepared):
    """Swap only the frozen candidate callback, keeping endpoint and pool guards."""
    from pathInference.core import activate
    activate(data_manifest=prepared["data_manifest"])
    import HighFrequencyVertexBasedPathGen as hf
    from pathInference.prefix_tree import public_tree_observed_pathgen as generator
    original, old = hf.generate_kpath_nosave, generator.generate_public_candidates

    def generate(env, k, source, destinations, *args, **kwargs):
        library = hf.load_local_knowledge_paths(env, source)
        expected = prepared["admitted_library_path_set_sha256"][str(source)]
        require(signature(library) == expected, "Legacy generator local library differs")
        result = original(env, k, source, destinations)
        return result, dict(generator="frozen_high_frequency_vertex", research_adapter_sha256=sha256(__file__),
                            legacy_core_sha256=sha256(hf.__file__), admitted_library_path_set_sha256=expected,
                            query_count=len(destinations), final_labels_read=False)

    generator.generate_public_candidates = generate
    try:
        yield
    finally:
        generator.generate_public_candidates = old


def pg_stage(date, variant, source, stage):
    from pathInference.experiment import worker
    run = run_dir(date, variant)
    prepared = read_json(run / "T-PG-B/pg_inputs" / ("as" + source) / "PREPARED.json")
    if variant == "legacy_generator":
        with legacy_generator(prepared):
            worker.execute(run, stage, source)
    else:
        worker.execute(run, stage, source)


def paired_assignment(pair, frozen_split, seed, fraction):
    if pair in frozen_split["validation_pairs"]:
        return "validation"
    if pair in frozen_split["train_pairs"]:
        return "train"
    score = int.from_bytes(hashlib.sha256((str(seed) + ":" + pair).encode()).digest()[:8], "big") / 2**64
    return "validation" if score < fraction else "train"


@contextmanager
def fixed_pair_split(date, source):
    from pathInference.downstream import train_evaluate_gmpi as driver
    split_file = baseline(date) / "T-PG-B/gmpi/shared_splits" / ("as" + source) / "surrogate_split/split_manifest.json"
    frozen_split = read_json(asset(split_file))
    old = driver.split_training

    def split(path, outdir, requested, seed, fraction):
        require(str(requested) == source, "Wrong paired split source")
        rows = list(driver.generated_rows(path, source))
        groups = {"train": [], "validation": []}
        inventory = {name: set(frozen_split[name + "_pairs"]) for name in groups}
        require(inventory["train"].isdisjoint(inventory["validation"]), "Frozen pair overlap")
        for row in rows:
            pair = driver.canonical_pair(row[4], source)
            name = paired_assignment(pair, inventory_to_lists(inventory), seed, fraction)
            groups[name].append(row)
        require(all(groups.values()), "Fixed pair assignment yields an empty split; record unresolved without retuning")
        outdir.mkdir()
        manifest = dict(input_csv=str(path), input_sha256=sha256(path), seed=seed,
                        split_unit="complete source/destination AS pair", validation_fraction_requested=fraction,
                        train_pairs=sorted(r[4] for r in groups["train"]),
                        validation_pairs=sorted(r[4] for r in groups["validation"]),
                        row_counts={k: len(v) for k, v in groups.items()}, test_labels_used_for_selection=False,
                        baseline_split=dict(path=str(split_file), sha256=sha256(split_file)),
                        new_pair_assignment="seeded SHA256 threshold; baseline pair sides unchanged")
        for name, values in groups.items():
            with (outdir / (name + ".csv")).open("x", newline="") as stream:
                csv.writer(stream).writerows(sorted(values, key=lambda r: r[4]))
        write_json(outdir / "split_manifest.json", manifest)
        return manifest

    driver.split_training = split
    try:
        yield
    finally:
        driver.split_training = old


def inventory_to_lists(inventory):
    # Sets are retained so membership stays O(1) on large source libraries.
    return {name + "_pairs": values for name, values in inventory.items()}


def gmpi_stage(date, variant, source, stage):
    from pathInference.experiment import worker
    with fixed_pair_split(date, source):
        worker.execute(run_dir(date, variant), stage, source, device="cuda", cpu_threads=2)


def vocabulary(prepared):
    from pathInference.data.prepare import resolve_read_path
    data = read_json(prepared["data_manifest"])
    nominal = Path(data["source_data_root"]) / "data/aspath/path_by_vantage" / prepared["date"] / "AS_frequency.json"
    path = resolve_read_path(data, nominal)
    values = read_json(path)
    values = json.loads(values) if isinstance(values, str) else values
    return ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"] + sorted(values)


def candidate_metrics(date, variant, source):
    """Evaluation only, in a fresh process after checkpoint selection."""
    from pathInference.downstream import common
    from pathInference.downstream.evaluation import read_references
    run = run_dir(date, variant)
    child = run / "T-PG-B/pg_inputs" / ("as" + source)
    prepared = common.verify_runtime_authorization(child)
    if variant != "current":
        from pathInference.experiment import train
        train.verify_selection(source, run / "T-PG-B")
    identity = prepared["test_identity"][source]
    verify_files({identity["reference_file"]: identity["reference_sha256"]})
    truth, _ = read_references(identity["reference_file"], source)
    require(common.pair_identity(truth) == identity["pair_inventory_sha256"], "Evaluation universe changed")
    itos = vocabulary(prepared)
    marker = read_json(child / "control/pathgen_completed" / ("test_inputs_" + source + ".json"))
    verify_files({marker["output"]: marker["sha256"]})
    diagnostics = run / "diagnostics" / ("as" + source)
    diagnostics.mkdir(parents=True, exist_ok=True)
    scores, counts, matches = {}, {}, {}
    with Path(marker["output"]).open(newline="") as stream:
        for row in csv.reader(stream, strict=True):
            pair = row[4]
            candidates = {tuple(itos[int(n)] for n in text.split("->")) for text in row[0].split("|")}
            require(pair in truth and pair not in scores, "Unexpected model pair")
            counts[pair] = len(candidates)
            matches[pair] = int(bool(candidates & truth[pair]))
            scores[pair] = max(len(set(p) & set(t)) / len(set(p) | set(t)) for p in candidates for t in truth[pair])
    statuses = read_json(child / "control/label_blind_test_inputs" / (source + ".json"))
    with Path(statuses["dispatch_status_file"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            pair = row["pair"]
            if row["status"] == "direct_graph_edge_candidate":
                path = tuple(pair.split("->"))
                counts[pair], matches[pair] = 1, int(path in truth[pair])
                scores[pair] = max(len(set(path) & set(t)) / len(set(path) | set(t)) for t in truth[pair])
    per_pair = diagnostics / "candidate_oracle.csv"
    with per_pair.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair", "candidate_count", "exact_available", "oracle_jaccard"])
        for pair in sorted(truth):
            writer.writerow([pair, counts.get(pair, 0), matches.get(pair, 0), scores.get(pair, 0.0)])
    library = common.csv_paths(run / "augmentation" / ("as" + source) / "paths.csv", source)
    branches = Counter(p[1] for p in library)
    tree = read_json(run / "query_trees" / (source + ".json"))
    prefixes = {tuple(r["prefix"]) for r in tree["primary_leaves"]}
    real_prefixes = {p[:3] for paths in truth.values() for p in paths if len(p) >= 3}
    # A first-hop leaf permits every legal tail through it, including its actual second hop.
    covered = sum(any(p[:len(prefix)] == prefix for prefix in prefixes) for p in real_prefixes)
    observed_branches = {p[1] for paths in truth.values() for p in paths}
    import pickle
    index_binding = prepared["tree_B_generator"]["observed_tail_index"]
    verify_files({index_binding["path"]: index_binding["sha256"]})
    with Path(index_binding["path"]).open("rb") as stream:
        index = pickle.load(stream)
    observed = index.get(source, index.get(int(source), {}))
    supported = sum("->".join(p) in observed for p in library)
    value = dict(date=date, variant=variant, source=source, queries=len(truth),
                 candidate_coverage=len(scores) / len(truth), exact_available=sum(matches.values()) / len(truth),
                 oracle_jaccard=sum(scores.values()) / len(truth),
                 real_secondhop_prefixes=len(real_prefixes), permitted_real_secondhop_prefixes=covered,
                 permitted_secondhop_coverage=covered / len(real_prefixes) if real_prefixes else None,
                 explicit_secondhop_coverage=len(real_prefixes & prefixes) / len(real_prefixes) if real_prefixes else None,
                 augmented_paths_per_observed_firsthop={n: branches[n] for n in sorted(observed_branches, key=int)},
                 full_path_train_index_support=supported / len(library), support_is_not_forwarding_truth=True,
                 outputs_sha256={str(per_pair): sha256(per_pair)}, final_labels_used_for_tuning=False,
                 prepared_sha256=sha256(child / "PREPARED.json"))
    freeze(diagnostics / "METRICS.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stage", required=True,
                        choices=("prepare", "pg_train", "pg_test", "train", "annotate", "evaluate", "rescore", "metrics"))
    args = parser.parse_args()
    started = time.monotonic()
    if args.stage == "prepare":
        prepare_source(args.date, args.variant, args.source)
    elif args.stage.startswith("pg_"):
        pg_stage(args.date, args.variant, args.source, args.stage)
    elif args.stage == "metrics":
        candidate_metrics(args.date, args.variant, args.source)
    else:
        gmpi_stage(args.date, args.variant, args.source, args.stage)
    run = run_dir(args.date, args.variant)
    if args.stage == "prepare":
        receipt = run / "control/research_ready" / (args.source + ".json")
    elif args.stage == "metrics":
        receipt = run / "diagnostics" / ("as" + args.source) / "METRICS.json"
    else:
        from pathInference.experiment.worker import receipt_path
        receipt = receipt_path(run, args.stage, args.source)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = dict(status="PASS", **vars(args), code_sha256=sha256(__file__),
                 config_sha256=sha256(CONFIG), outputs_sha256={str(receipt): sha256(receipt)},
                 elapsed_seconds=time.monotonic() - started, peak_rss_kib=usage.ru_maxrss,
                 completed_utc=utc_now())
    task_file = run / "control/research_tasks" / (args.stage + "_" + args.source + ".json")
    if not task_file.exists():
        freeze(task_file, value)
    print(json.dumps(dict(status="PASS", **vars(args))), flush=True)


if __name__ == "__main__":
    main()
