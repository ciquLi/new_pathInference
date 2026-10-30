"""Dated path construction, 255-feature fitting, and unified R_s selection.

No stage opens a formal source reference. Fitting opens only the split's
training origins. The two consumers downstream receive the same released pool.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import pickle
import sqlite3

import lightgbm as lgb
import numpy as np

from .contracts import read, require, sha, verify, write, write_paths
from .destination import DestinationFeatureStore
from .implementation.candidates import CandidatePolicy, collect_source, relation, valley_compatible
from .implementation.construction_budget import select_valid_candidates
from .implementation.donor_stats import nd_stats, parse
from .implementation.final_features import ActualQueryStore, FinalPathFeatures
from .implementation.frontier_adapter import SafeFrontierAdapter
from .implementation.frontier_policy import FrontierPolicy
from .implementation.joint_selector import (
    SelectionPolicy,
    length_aware_representatives,
    select_joint_budget,
)
from .implementation.path_features import PathEvidenceFeatures
from .implementation.path_labels import balanced_weights, reference
from .implementation.prior_spec import prediction_spec


class CachedPathFeatures(PathEvidenceFeatures):
    @lru_cache(maxsize=4096)
    def _destination_aggregate(self, start):
        return PathEvidenceFeatures._destination_aggregate.__wrapped__(self, start)


def prepare_donors(data, split_manifest, bundle, output, public):
    """Same unique full-path ND/NAD fractions; keep the original donor order."""
    out = Path(output)
    marker = out / "DONORS.json"
    split = split_manifest.get("split", split_manifest)
    sources = list(map(int, bundle["training_sources"]))
    require(set(sources) == set(map(int, split["train"])), "Donor/source identity mismatch")
    folds = {int(s): int(f) for s, f in bundle["source_to_fold"].items()}
    if marker.exists():
        result = read(marker)
        require(
            result["date"] == data["date"] and result["donor_sources"] == sources,
            "Donor cache belongs to another date or source order",
        )
        verify(result["inputs_sha256"])
        verify(result["outputs_sha256"])
        return result
    shards, inputs, outputs = {}, {}, {}
    for source in sources:
        references = (
            read(bundle["training_references"]) if bundle.get("training_references") else {}
        )
        path = (
            Path(references[str(source)]["path"])
            if str(source) in references
            else Path(data["train_dir"]) / (str(source) + ".csv")
        )
        require(path.is_file(), "Missing original training donor: " + str(path))
        digest = sha(path)
        routes = set()
        with path.open("rb") as handle:
            for raw in handle:
                route, _, reason = parse(raw, source)
                if reason is None:
                    routes.add(route)
        groups, direct = nd_stats(routes)
        ordered = sorted(groups)
        nd_n, nd_d, nd_total, nd_role, nad_group, nad_a, nad_count = [], [], [], [], [], [], []
        from .implementation.destination_features import public_role

        for i, (n, d) in enumerate(ordered):
            counts = groups[n, d]
            nd_n.append(n)
            nd_d.append(d)
            nd_total.append(sum(counts.values()))
            nd_role.append(public_role(public, source, n)["role_code"])
            for a in sorted(counts):
                nad_group.append(i)
                nad_a.append(a)
                nad_count.append(counts[a])
        target = out / "sources" / (str(source) + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            source=np.uint32(source),
            source_fold=np.int32(folds[source]),
            ND_N=np.asarray(nd_n, np.uint32),
            ND_D=np.asarray(nd_d, np.uint32),
            ND_total_unique_full_paths=np.asarray(nd_total, np.uint32),
            ND_role_code=np.asarray(nd_role, np.uint8),
            NAD_nd_index=np.asarray(nad_group, np.uint32),
            NAD_A=np.asarray(nad_a, np.uint32),
            NAD_unique_full_path_count=np.asarray(nad_count, np.uint32),
            direct2_N=np.asarray(sorted(direct), np.uint32),
        )
        require(sha(path) == digest, "Donor changed during materialization")
        inputs[str(path)] = digest
        outputs[str(target)] = sha(target)
        shards[str(source)] = str(target)
    result = dict(
        date=data["date"],
        donor_sources=sources,
        donor_shards=shards,
        inputs_sha256=inputs,
        outputs_sha256=outputs,
        formal_labels_read=False,
        role_scope="same_date_receiver_to_center_ASREL_then_ASRank_conflict_explicit",
    )
    write(marker, result)
    return result


def construct_case(source, runtime, config, destination_store, output):
    """The original 222+33 arithmetic over freshly inferred local contexts."""
    source = int(source)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        date=runtime.date,
        source=str(source),
        method_config=config,
        effective_predict_threads=runtime.threads,
        evidence_sha256=runtime.evidence_sha256,
        model_files_sha256=runtime.bundle["model_files_sha256"],
        donor_inputs_sha256=destination_store.expected,
        public_files_sha256={
            str(p): sha(p) for p in runtime.public.paths.values() if isinstance(p, str) and p
        },
        computational_sha256={str(p): sha(p) for p in Path(__file__).parent.rglob("*.py")},
    )
    if (out / "CASE.json").exists():
        cached = read(out / "CASE.json")
        require(cached["identity"] == identity, "Candidate cache method/model/date differs")
        verify(cached["files_sha256"])
        return cached
    root = runtime.infer_root(source)
    require(
        root["source_labels_excluded_from_predictive_heads"], "Source-isolated root heads required"
    )
    queries = ActualQueryStore(runtime)
    selected = list(map(str, root["selected_neighbors"]))
    for n in selected:
        queries.query_frontier(source, int(n), [source, int(n)])
    adapter = SafeFrontierAdapter(
        source,
        root,
        runtime.safe["index"],
        runtime.public.relations,
        queries,
        FrontierPolicy(**config["frontier"]),
    )
    plan = adapter.plan()
    _, _, priors, _ = prediction_spec(source, root, runtime.public)
    for n in selected:
        for rank, candidate in enumerate(queries.queries[(source, int(n))]["candidates"], 1):
            priors["recursive"].setdefault(n, {})[str(candidate["asn"])] = dict(
                center_evidence={str(source): dict(pred_rank=rank)}
            )
    cp = config["construction"]
    _, uncut, statistics = collect_source(
        source,
        plan["prefixes"],
        runtime.safe["index"],
        runtime.safe["origin_support"],
        runtime.public.relations,
        edge_provenance=adapter.provenance,
        policy=CandidatePolicy(cp["per_neighbor_destination_length_cap"]),
    )
    positions, budget = select_valid_candidates(
        source,
        uncut,
        budget=cp["destination_budget"],
        direct_sparse_max=cp["direct_sparse_max"],
        reserve=cp["reserve"],
    )
    candidates = [uncut[i] for i in positions]
    del uncut
    base = CachedPathFeatures(
        runtime.safe["index"],
        runtime.safe["origin_support"],
        runtime.public.relations,
        runtime.public,
        priors,
    )
    feature_builder = FinalPathFeatures(base, plan, runtime, queries)
    names = list(feature_builder.feature_names)
    matrix = np.empty((len(candidates), len(names)), np.float32)
    for i, candidate in enumerate(candidates):
        candidate["row_index"] = i
        row = feature_builder.features(candidate)
        matrix[i] = [row[n] for n in names]
    extra = destination_store.append(source, candidates)
    require(matrix.shape == (len(candidates), 222), "Original path features changed")
    full = np.concatenate((matrix, extra), axis=1)
    from .implementation.destination_features import APPENDED_NAMES

    all_names = names + list(APPENDED_NAMES)
    require(
        full.shape == (len(candidates), 255) and np.isfinite(full).all(),
        "255-feature contract failed",
    )
    np.savez_compressed(out / "features.npz", X=full)
    with (out / "candidates.pkl").open("wb") as handle:
        pickle.dump(
            dict(
                source=str(source),
                candidates=candidates,
                selected_neighbors=selected,
                feature_names=all_names,
                path_order="candidates[i] corresponds to X[i]",
            ),
            handle,
            protocol=5,
        )
    write(out / "root.json", root)
    write(out / "plan.json", plan)
    # Root distribution still controls major-branch safeguards even when KL
    # penalties are zero, so this is a fitted distribution rather than a quota.
    cand, raw, _ = runtime.builder.query(source, source)
    prior = (
        runtime.predict_prior(source, cand, raw, selected)
        if len(cand)
        else dict(
            status="unknown_empty_branch_pool",
            branch_probability={},
            length_probability={},
            joint_probability={},
        )
    )
    write(out / "prior.json", prior)
    result = dict(
        source=str(source),
        date=runtime.date,
        rows=len(candidates),
        identity=identity,
        features=str(out / "features.npz"),
        candidates=str(out / "candidates.pkl"),
        prior=str(out / "prior.json"),
        feature_names=all_names,
        statistics=statistics,
        destination_budget=budget,
        formal_labels_read=False,
        files_sha256={
            str(p): sha(p)
            for p in (
                out / "features.npz",
                out / "candidates.pkl",
                out / "root.json",
                out / "plan.json",
                out / "prior.json",
            )
        },
    )
    write(out / "CASE.json", result)
    base._destination_aggregate.cache_clear()
    queries.clear()
    return result


def fit_path_model(
    data, split_manifest, training_cases, parameters, output, training_references=None
):
    """Same membership labels, destination alternatives and source weights."""
    split = split_manifest.get("split", split_manifest)
    train = set(map(str, split["train"]))
    require(
        set(map(str, training_cases)) == train,
        "All and only registered training sources are required",
    )
    forbidden = set().union(
        *(set(map(str, split.get(role, []))) for role in ("validation", "test", "formal_test"))
    )
    require(not train & forbidden, "Held-out sources cannot fit the path model")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        date=data["date"],
        sources=sorted(train, key=int),
        parameters=parameters,
        training_cases_sha256={
            s: hashlib.sha256(json.dumps(training_cases[s], sort_keys=True).encode()).hexdigest()
            for s in sorted(train)
        },
        training_references_sha256=sha(training_references)
        if isinstance(training_references, (str, Path))
        else None,
    )
    if (out / "MODEL.json").exists():
        previous = read(out / "MODEL.json")
        require(previous["identity"] == identity, "Path model cache has different training inputs")
        require(sha(previous["path_model"]) == previous["model_sha256"], "Path model cache changed")
        return previous
    xs, ys, weights, provenance = [], [], [], []
    names = None
    for source in sorted(train, key=int):
        case = training_cases[source]
        verify(case["files_sha256"])
        with Path(case["candidates"]).open("rb") as handle:
            meta = pickle.load(handle)
        with np.load(case["features"], allow_pickle=False) as archive:
            matrix = archive["X"]
        names = meta["feature_names"] if names is None else names
        require(
            names == meta["feature_names"] and matrix.shape == (len(meta["candidates"]), 255),
            "Training matrix row/schema differs",
        )
        references = (
            read(training_references)
            if isinstance(training_references, (str, Path))
            else (training_references or {})
        )
        path = (
            Path(references[source]["path"])
            if source in references
            else Path(data["train_dir"]) / (source + ".csv")
        )
        truth, details = reference(path, source, sha(path))
        destinations = {p[-1] for p in truth}
        labels = np.asarray(
            [
                int(tuple(map(str, c["path"])) in truth)
                if str(c["path"][-1]) in destinations
                else -1
                for c in meta["candidates"]
            ],
            np.int8,
        )
        use = labels >= 0
        w = balanced_weights(meta["candidates"], labels)
        xs.append(matrix[use])
        ys.append(labels[use])
        weights.append(w[use])
        provenance.append(
            dict(
                source=source,
                reference=details,
                rows=len(labels),
                positive=int((labels == 1).sum()),
                unknown=int((labels < 0).sum()),
            )
        )
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    require(
        w.sum() > 0 and len(np.unique(y)) == 2, "Path model requires informative training labels"
    )
    w *= len(w) / w.sum()
    model = lgb.LGBMClassifier(**parameters)
    model.fit(x, y, sample_weight=w, feature_name=names)
    model.booster_.save_model(str(out / "model.txt"))
    result = dict(
        date=data["date"],
        model_fit_date=data["date"],
        identity=identity,
        path_model=str(out / "model.txt"),
        model_sha256=sha(out / "model.txt"),
        path_feature_names=names,
        parameters=parameters,
        training_sources=sorted(train, key=int),
        provenance=provenance,
        formal_labels_read=False,
        validation_labels_read=False,
        isolation="source-held upstream heads and D features; stagewise OOF, not nested end-to-end OOF",
    )
    write(out / "MODEL.json", result)
    return result


def select_paths(case, scores, prior, policy, strict_novel):
    """Common selector used for pure and unified cases, including abstentions."""
    candidates = case["candidates"]
    current = dict(
        source=case["source"],
        paths=[tuple(map(str, c["path"])) for c in candidates],
        scores=np.asarray(scores),
        novel=np.asarray(strict_novel, bool),
        structural_eligible=np.ones(len(candidates), bool),
        branch_keys=[str(c["path"][1]) for c in candidates],
    )
    if not len(candidates):
        return [], dict(status="unknown_empty_candidate_pool", quantity_failure=True)
    current["eligible"] = current["structural_eligible"] & length_aware_representatives(current)
    mask, _, diagnostics = select_joint_budget(
        current,
        prior["branch_probability"],
        prior["length_probability"],
        policy,
        predicted_joint=prior["joint_probability"],
    )
    return [current["paths"][int(i)] for i in np.flatnonzero(mask)], diagnostics
