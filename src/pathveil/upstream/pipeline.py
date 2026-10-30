"""One dated upstream entry, with no LLM or cross-date model fallback."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import gc
import pickle
import sys

import lightgbm as lgb
import numpy as np

from .contracts import (
    read,
    require,
    sha,
    validate_date,
    validate_model_scope,
    verify,
    write,
    write_paths,
)
from .destination import DestinationFeatureStore
from .evidence import make_public, select_dated_sources, value
from .implementation.candidates import edge_evidence, exact_support, relation, valley_compatible
from .implementation.final_features import ActualQueryStore, FinalPathFeatures
from .implementation.frontier_adapter import SafeFrontierAdapter
from .implementation.frontier_policy import FrontierPolicy
from .implementation.joint_selector import SelectionPolicy
from .implementation.prior_spec import prediction_spec
from .method import MethodLibrary
from .neighbors import NeighborRuntime, train_neighbors
from .path_pipeline import (
    CachedPathFeatures,
    construct_case,
    fit_path_model,
    prepare_donors,
    select_paths,
)
from .replay import run_baseline
from .witnesses import WitnessStore, prepare_witnesses


def _extra_candidate(path, safe, public, prediction):
    """Original direct external-suffix candidate arithmetic and provenance."""
    p = tuple(path)
    prefix, tail = p[:2], p[1:]
    index, rel = safe["index"], public.relations
    proof = edge_evidence(index, rel, *prefix, prediction["root"]["provenance"].get(p[1], {}))
    require(proof["admitted"], "External suffix has no admitted source-firsthop boundary")
    roles = [relation(rel, a, b) for a, b in zip(p, p[1:])]
    require(valley_compatible(roles), "External suffix violates known commercial relations")
    first_known = next(
        (i for i in range(len(p) - 1) if exact_support(index, p[i:]) > 0), len(p) - 1
    )
    return dict(
        path=p,
        prefix=prefix,
        source=p[0],
        firsthop=p[1],
        first_hop=p[1],
        neighbor=p[1],
        dest=p[-1],
        destination=p[-1],
        second_hop=p[2],
        relationship=roles[0],
        second_hop_relationship=roles[1],
        segment_len=2,
        tail_len=len(tail),
        tail_path_freq=float(exact_support(index, tail)),
        tail_source=tail[0],
        generation_source="direct",
        relation_validation="fully_known" if "unknown" not in roles else "partial_compatible",
        prefix_relationship_validated=proof["relationship"] != "unknown",
        inferred_prefix_edges=int(proof["inferred_only"]),
        prefix_edge_evidence=[proof],
        unknown_prefix_extensions=first_known,
        novel=not bool(exact_support(index, p)),
        as_count=len(p),
        external_complete_origin_preserving=True,
    )


def _score_extra(source, paths, runtime, config, store, model):
    prediction = runtime.infer_root(int(source))
    selected = list(map(str, prediction["selected_neighbors"]))
    queries = ActualQueryStore(runtime)
    for n in selected:
        queries.query_frontier(int(source), int(n), [int(source), int(n)])
    adapter = SafeFrontierAdapter(
        int(source),
        prediction,
        runtime.safe["index"],
        runtime.public.relations,
        queries,
        FrontierPolicy(**config["frontier"]),
    )
    plan = adapter.plan()
    _, _, priors, _ = prediction_spec(int(source), prediction, runtime.public)
    for n in selected:
        for rank, candidate in enumerate(queries.queries[(int(source), int(n))]["candidates"], 1):
            priors["recursive"].setdefault(n, {})[str(candidate["asn"])] = dict(
                center_evidence={str(source): dict(pred_rank=rank)}
            )
    base = CachedPathFeatures(
        runtime.safe["index"],
        runtime.safe["origin_support"],
        runtime.public.relations,
        runtime.public,
        priors,
    )
    builder = FinalPathFeatures(base, plan, runtime, queries)
    candidates = [_extra_candidate(p, runtime.safe, runtime.public, prediction) for p in paths]
    X = np.empty((len(candidates), 222), np.float32)
    for i, candidate in enumerate(candidates):
        candidate["row_index"] = i
        row = builder.features(candidate)
        X[i] = [row[n] for n in builder.feature_names]
    full = np.concatenate((X, store.append(int(source), candidates)), axis=1)
    scores = model.predict(full, num_threads=2) if len(full) else np.empty(0, float)
    base._destination_aggregate.cache_clear()
    queries.clear()
    return {p: float(v) for p, v in zip(paths, scores)}


def _graph_files(runtime, sources, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected_path, anchors_path = (
        output / "selected_neighbors.csv",
        output / "recursive_anchors.csv",
    )
    with selected_path.open("w", newline="") as a, anchors_path.open("w", newline="") as b:
        selected = csv.DictWriter(
            a,
            fieldnames=[
                "date",
                "center_as",
                "neighbor_as",
                "selected",
                "asrank_link_number_paths_vu",
            ],
        )
        anchors = csv.DictWriter(
            b,
            fieldnames=[
                "date",
                "center_as",
                "first_hop",
                "rank",
                "second_hop",
                "asrank_number_paths",
            ],
        )
        selected.writeheader()
        anchors.writeheader()
        for source in sources:
            root = runtime.infer_root(int(source))
            for n in root["selected_neighbors"]:
                selected.writerow(
                    dict(
                        date=runtime.date,
                        center_as=source,
                        neighbor_as=n,
                        selected=1,
                        asrank_link_number_paths_vu=runtime.public.links.get(
                            (int(source), int(n)), {}
                        ).get("number_paths", 1.0),
                    )
                )
            queries = ActualQueryStore(runtime)
            adapter = SafeFrontierAdapter(
                int(source),
                root,
                runtime.safe["index"],
                runtime.public.relations,
                queries,
                FrontierPolicy(**runtime.config["frontier"]),
            )
            for decision in adapter.plan()["decisions"]:
                if len(decision["prefix"]) == 2:
                    first = decision["prefix"][-1]
                    for rank, second in enumerate(decision.get("expanded_neighbors", []), 1):
                        anchors.writerow(
                            dict(
                                date=runtime.date,
                                center_as=source,
                                first_hop=first,
                                rank=rank,
                                second_hop=second,
                                asrank_number_paths=runtime.public.links.get(
                                    (int(first), int(second)), {}
                                ).get("number_paths", 1.0),
                            )
                        )
    return str(selected_path), str(anchors_path)


def run_upstream(
    data_manifest,
    run_dir,
    method_config=None,
    project_root=None,
    sources=None,
    fresh=False,
    split_manifest=None,
):
    """Fit/infer the local method, or explicitly replay the verified March cache.

    Calls do execute training for an unprepared date. Project maintenance and
    dry-run callers must not invoke this function until an experiment is wanted.
    """
    library = MethodLibrary(project_root)
    data = value(data_manifest)
    config = value(method_config or library.project / "configs/method_unified255.json")
    date = validate_date(data["date"])
    require(
        config["method_id"] == "unified255_v1" and config["semantic_llm_review"] is False,
        "Unsupported upstream method or LLM review",
    )
    if date == "20260301" and not fresh:
        return run_baseline(data, run_dir, config, library.project, sources)
    run = Path(run_dir).resolve() / "upstream"
    run.mkdir(parents=True, exist_ok=True)
    explicit_split = split_manifest or data.get("split_manifest")
    requested = list(map(str, sources or data["sources"]))
    require(
        requested and len(requested) == len(set(requested)),
        "Requested original sources are empty or duplicated",
    )
    source_policy = config.get("source_selection")
    if explicit_split is None and source_policy:
        from .source_split import prepare_source_split

        require(
            source_policy.get("policy") == "all_canonical_unique_valid_paths_gt_threshold_v1"
            and source_policy.get("split_ratios") == [4, 1, 1]
            and source_policy.get("all_eligible_sources") is True,
            "Unsupported source selection policy",
        )
        explicit_split = prepare_source_split(
            data,
            run.parent / "data/source_protocol",
            min_paths=int(source_policy["min_paths_exclusive"]),
            seed=int(config["seed"]),
            formal_sources=requested,
            progress=lambda event: print(event, flush=True),
        )
    split_info = value(explicit_split or library.project / "data/upstream/20260301/split.json")
    split_info = select_dated_sources(
        data, split_info, requested, allow_date_template=explicit_split is None
    )
    split = split_info["split"]
    identity = dict(
        date=date,
        method_id=config["method_id"],
        sources=requested,
        method_config=config,
        split=split_info,
        code=library.receipt(),
        canonical_sha256=split_info["source_availability"]["canonical_sha256"],
        public_files_sha256={str(p): sha(p) for p in data["public_files"].values()},
    )
    if (run / "RELEASE.json").exists():
        previous = read(run / "RELEASE.json")
        require(
            previous.get("identity") == identity,
            "Existing upstream release uses different inputs/mode/code; choose a new run directory",
        )
        verify(previous["outputs_sha256"])
        return run / "RELEASE.json"
    write(run / "IDENTITY.json", identity)
    write(run / "split.json", split_info)
    bundle_path = train_neighbors(data, split_info, run / "neighbor_models", library, config)
    bundle = read(bundle_path)
    validate_model_scope(bundle, date, split)
    public = make_public(data, library.core())
    donor = prepare_donors(data, split_info, bundle, run / "destination_evidence", public)
    bundle.update(donor_sources=donor["donor_sources"], donor_shards=donor["donor_shards"])
    bundle["inputs_sha256"].update(donor["outputs_sha256"])
    write(run / "NEIGHBOR_BUNDLE.json", bundle)
    runtime = NeighborRuntime(bundle, data, config, method=library)
    if source_policy:
        from .heldout_evaluation import evaluate_heldout_neighbors

        evaluate_heldout_neighbors(data, split_info, runtime, run / "heldout_neighbor_evaluation")
    from .resources import runtime_resources

    resources = runtime_resources(config)
    if config.get("runtime_resources") and resources["case_workers"] > 1:
        # Release the parent's large graph and feature caches while spawned
        # workers own their independent copies.
        del runtime, public
        gc.collect()
        # The official CLI already owns this run's exclusive lock. Spawned
        # workers execute the same construct_case API with independent runtimes.
        scripts_path = str(library.project / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        from build_training_cases import prepare_configuration, schedule_cases

        case_sources, configuration = prepare_configuration(run.parent)
        code = schedule_cases(
            run.parent,
            case_sources,
            configuration,
            workers=resources["case_workers"],
            min_free_bytes=12 * 1024**3,
            worker_memory_bytes=16 * 1024**3,
            run_lock_held=True,
        )
        require(code == 0, "Parallel training case construction did not complete")
        runtime = NeighborRuntime(bundle, data, config, method=library)
    store = DestinationFeatureStore(bundle, runtime.public)
    train_cases = {}
    for source in map(str, bundle["training_sources"]):
        train_cases[source] = construct_case(
            source, runtime, config, store, run / "training_cases" / source
        )
    fitted = fit_path_model(
        data,
        split_info,
        train_cases,
        config["path_model_parameters"],
        run / "path_model",
        training_references=bundle.get("training_references"),
    )
    model = lgb.Booster(model_file=fitted["path_model"])
    require(model.feature_name() == fitted["path_feature_names"], "Trained path schema differs")
    database = prepare_witnesses(data, split_info, requested, run / "witnesses")
    witnesses = WitnessStore(database, bundle["training_sources"], bundle["source_to_fold"])
    policy = SelectionPolicy(**config["pure_selection"])
    files, diagnostics = {}, {}
    try:
        for source in requested:
            case = construct_case(source, runtime, config, store, run / "inference_cases" / source)
            with Path(case["candidates"]).open("rb") as handle:
                meta = pickle.load(handle)
            with np.load(case["features"], allow_pickle=False) as archive:
                scores = (
                    model.predict(archive["X"], num_threads=2)
                    if len(archive["X"])
                    else np.empty(0, float)
                )
            require(
                meta["feature_names"] == model.feature_name(), "Inference feature order differs"
            )
            prior = read(case["prior"])
            known = witnesses.source(source, training_only=True)
            strict = [
                bool(c["novel"]) and tuple(map(str, c["path"])) not in known
                for c in meta["candidates"]
            ]
            pure, pure_diag = select_paths(meta, scores, prior, policy, strict)
            all_external = witnesses.source(source)
            selected_roots = set(map(str, runtime.infer_root(int(source))["selected_neighbors"]))
            pure_first = {p[1] for p in pure}
            external = {
                p
                for p in all_external
                if 3 <= len(p) <= 20
                and p[1] in selected_roots & pure_first
                and valley_compatible(
                    [relation(runtime.public.relations, a, b) for a, b in zip(p, p[1:])]
                )
            }
            bypath = {
                tuple(map(str, c["path"])): float(scores[i])
                for i, c in enumerate(meta["candidates"])
            }
            ordered_pool = sorted(set(pure) | external, key=lambda p: tuple(map(int, p)))
            missing = [p for p in ordered_pool if p not in bypath]
            bypath.update(_score_extra(source, missing, runtime, config, store, model))
            candidates = [dict(path=p) for p in ordered_pool]
            original_novel = {
                tuple(map(str, c["path"])): bool(v) for c, v in zip(meta["candidates"], strict)
            }
            if len(pure) < policy.minimum_paths:
                chosen = set(pure)
                unified_diag = dict(
                    status="quantity_failure_same_budget_below_declared_minimum",
                    quantity_failure=True,
                )
            else:
                chosen_rows, unified_diag = select_paths(
                    dict(source=source, candidates=candidates),
                    [bypath[p] for p in ordered_pool],
                    prior,
                    replace(policy, budget=len(pure), minimum_novel_paths=0),
                    [original_novel.get(p, False) and p not in all_external for p in ordered_pool],
                )
                chosen = set(chosen_rows)
            ordered = [p for p in pure if p in chosen] + sorted(
                chosen - set(pure), key=lambda p: tuple(map(int, p))
            )
            target = run / "release" / (source + "_augmented_paths.csv")
            write_paths(target, ordered)
            grouped = {}
            for p in ordered:
                grouped.setdefault(source + "-" + p[-1], []).append(list(p))
            write(target.with_suffix(".json"), grouped)
            files[source] = dict(
                path=str(target),
                sha256=sha(target),
                json_path=str(target.with_suffix(".json")),
                json_sha256=sha(target.with_suffix(".json")),
                rows=len(ordered),
            )
            diagnostics[source] = dict(
                pure=pure_diag,
                unified=unified_diag,
                pure_budget=len(pure),
                eligible_external=len(external),
                selected=len(ordered),
            )
    finally:
        witnesses.close()
    selected_csv, anchors_csv = _graph_files(runtime, requested, run / "graph")
    outputs = {
        item[k]: item[h]
        for item in files.values()
        for k, h in [("path", "sha256"), ("json_path", "json_sha256")]
    }
    outputs.update({selected_csv: sha(selected_csv), anchors_csv: sha(anchors_csv)})
    receipt = dict(
        status="PASS",
        date=date,
        method_id=config["method_id"],
        execution_mode="date_local_source_disjoint_training_and_inference",
        fresh_model_training=True,
        identity=identity,
        fresh_candidate_generation=True,
        sources=requested,
        files_by_source=files,
        selected_neighbors_csv=selected_csv,
        recursive_anchors_csv=anchors_csv,
        data_root=data.get("data_root"),
        outputs_sha256=outputs,
        diagnostics=diagnostics,
        formal_reference_labels_read=False,
        training_and_PG_pool="same_R_s_hash_augmentation_only",
        source_or_query_exclusions=False,
        code=library.receipt(),
        neighbor_bundle=str(run / "NEIGHBOR_BUNDLE.json"),
        path_model=fitted,
    )
    write(run / "RELEASE.json", receipt)
    return run / "RELEASE.json"
