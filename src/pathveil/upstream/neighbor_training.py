"""Date-local source-disjoint neighbor, role, and distribution model training.

Only the frozen training origins provide labels. Validation and formal-source
observations cannot affect model fitting or automatic policy selection.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
from pathlib import Path
import pickle

import numpy as np

from .contracts import read, require, sha, validate_date, verify, write
from .evidence import (
    load_policies,
    make_public,
    policy_features,
    prepare_evidence,
    public_hashes,
    source_fold_map,
    split_roles,
    value,
)
from .method import MethodLibrary
from .neighbors import STRUCTURAL_NAMES, make_root_structure
from .resources import runtime_resources


def _binding(value_):
    return hashlib.sha256(
        json.dumps(value_, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _targets(weights, candidates):
    gold = sorted(weights, key=lambda n: (-round(weights[n], 12), int(n)))
    ranks = {int(n): i + 1 for i, n in enumerate(gold)}
    mass = np.asarray([weights.get(str(n), 0.0) for n in candidates], dtype=np.float32)
    rank = np.asarray([ranks.get(n, 0) for n in candidates], dtype=np.int32)
    present = (mass > 0).astype(np.int32)
    hierarchy = np.where(
        (rank > 0) & (rank <= 3), 3, np.where((rank > 0) & (rank <= 10), 2, present)
    ).astype(np.int32)
    return present, hierarchy


def _root_features(core, public, evidence, refs, policies, date, cap):
    builder = core.retrieval.PDBRetrievalBuilder(public, evidence, cap)
    role_builder = core.rpsl.PolicyBuilder(public, evidence, policies)
    geometry = make_root_structure(public, evidence, core)
    matrices, structural, queries, labels, hierarchy = [], [], [], [], []
    role_x, role_queries = [], []
    rpsl_names = None
    for s in sorted(refs, key=int):
        source = int(s)
        require(
            builder.candidates(source, source)[0],
            "Training root has no label-blind candidates: " + s,
        )
        cand, matrix, uncapped = builder.query(source, source)
        matrices.append(matrix)
        structural.append(geometry.structural(source, cand))
        queries.append(
            dict(
                source=source,
                center=source,
                root=True,
                candidates=cand,
                size=len(cand),
                uncapped_candidates=uncapped,
            )
        )
        present, rank = _targets(refs[s]["first"], cand)
        labels.append(present)
        hierarchy.append(rank)
        role_cand, base, role_uncapped = role_builder.query(source, source)
        extra = [policy_features(policies[s], n, date, core) for n in role_cand]
        if rpsl_names is None:
            rpsl_names = sorted(extra[0])
        policy_matrix = np.asarray(
            [[row[n] for n in rpsl_names] for row in extra], dtype=np.float32
        )
        role_x.append(np.concatenate((base, policy_matrix), axis=1))
        role_queries.append(
            dict(
                source=source,
                center=source,
                root=True,
                candidates=role_cand,
                size=len(role_cand),
                uncapped_candidates=role_uncapped,
            )
        )
    require(
        builder.columns and len(builder.columns) == 467,
        "Root base feature schema differs from frozen 467 columns",
    )
    return dict(
        X=np.concatenate(matrices),
        structural=np.concatenate(structural),
        queries=queries,
        present=np.concatenate(labels),
        hierarchy=np.concatenate(hierarchy),
        schema=builder.columns,
        role_X=np.concatenate(role_x),
        role_queries=role_queries,
        role_schema=role_builder.columns + rpsl_names,
    )


def _conditional_features(core, public, evidence, refs, cap):
    builder = core.retrieval.PDBRetrievalBuilder(public, evidence, cap)
    paircols = sorted(public.pair_features(1, 2))
    matrices, queries, present, hierarchy = [], [], [], []

    @lru_cache(maxsize=30000)
    def pair(source, candidate):
        row = public.pair_features(source, candidate)
        return np.asarray([row[key] for key in paircols], dtype=np.float32)

    for s in sorted(refs, key=int):
        source = int(s)
        ref = refs[s]
        contexts = [
            n for n in core.top.top_target(ref["first"])["strict"] if str(n) in ref["anchors"]
        ]
        for center in contexts:
            if not builder.candidates(source, center)[0]:
                continue
            cand, matrix, uncapped = builder.query(source, center)
            extra = np.stack([pair(source, n) for n in cand])
            matrices.append(np.concatenate((matrix, extra), axis=1))
            queries.append(
                dict(
                    source=source,
                    center=center,
                    root=False,
                    candidates=cand,
                    size=len(cand),
                    uncapped_candidates=uncapped,
                )
            )
            p, h = _targets(ref["anchors"][str(center)], cand)
            present.append(p)
            hierarchy.append(h)
    require(matrices, "No informative same-date conditional training contexts")
    schema = builder.columns + ["origin_candidate_" + key for key in paircols]
    X, columns = core.experiment.add_relative(np.concatenate(matrices), queries, schema)
    return dict(
        X=X,
        queries=queries,
        schema=columns,
        present=np.concatenate(present),
        hierarchy=np.concatenate(hierarchy),
        pair_columns=paircols,
    )


def _fit_family(
    name, X, labels, queries, columns, spec, fit_module, folds, work, binding, threads=4
):
    family = {}
    sources = set(map(int, folds))
    for key in ["oof" + str(f) for f in range(5)] + ["full"]:
        held = {s for s in sources if key != "full" and folds[str(s)] == int(key[3:])}
        kept = sources - held
        path = work / "models" / name / (key + ".txt")
        receipt = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        model_binding = dict(
            binding=binding,
            head=name,
            key=key,
            sources=sorted(kept),
            columns=columns,
            spec=spec,
            runtime_resources={"cpu_threads": threads},
        )
        if receipt.exists():
            previous = read(receipt)
            require(
                previous["binding"] == model_binding,
                "Model cache input or training scope changed: " + name,
            )
            verify({str(path): previous["sha256"]})
        else:
            # Whole-source masks are passed into the original frozen learner.
            require(kept and not kept & held, "Invalid source-disjoint head mask")
            model = fit_module.fit_one(spec, X, labels, queries, kept, columns, threads=threads)
            model.booster_.save_model(str(path))
            write(
                receipt,
                dict(
                    binding=model_binding,
                    sha256=sha(path),
                    formal_labels_read=False,
                    validation_labels_read=False,
                    iterations=int(model.booster_.current_iteration()),
                ),
            )
        family[key] = str(path)
        write(work / "training_progress.json", dict(stage="neighbor_heads", head=name, key=key))
    return family


def _fit_role(X, queries, schema, refs, core, folds, work, binding, threads=4):
    import lightgbm as lgb

    known, truth, sources, weights = core.role_labels.role_labels(queries, refs)
    family = {}
    for key in ["oof" + str(f) for f in range(5)] + ["full"]:
        held = np.asarray([key != "full" and folds[str(s)] == int(key[3:]) for s in sources])
        mask = known & ~held
        require(mask.any(), "Role training has no observed source-neighbor examples: " + key)
        path = work / "models/role" / (key + ".txt")
        receipt = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        model_binding = dict(
            binding=binding,
            key=key,
            columns=schema,
            sources=sorted(set(map(int, sources[mask]))),
            runtime_resources={"cpu_threads": threads},
        )
        if receipt.exists():
            previous = read(receipt)
            require(previous["binding"] == model_binding, "Role cache input changed")
            verify({str(path): previous["sha256"]})
        else:
            sample_weights = weights[mask].copy()
            sample_weights /= sample_weights.mean()
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=240,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.04,
                min_child_samples=30,
                reg_lambda=1.0,
                random_state=20261021,
                n_jobs=threads,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(X[mask], truth[mask], sample_weight=sample_weights, feature_name=schema)
            model.booster_.save_model(str(path))
            write(
                receipt,
                dict(
                    binding=model_binding,
                    sha256=sha(path),
                    formal_labels_read=False,
                    unobserved_pairs_are_not_endpoint_negatives=True,
                ),
            )
        family[key] = str(path)
    return family


def _fit_distribution(X, queries, labels, core, folds, work, binding, threads=4):
    family = {}
    sources = set(map(int, folds))
    for key in ["oof" + str(f) for f in range(5)] + ["full"]:
        held = {s for s in sources if key != "full" and folds[str(s)] == int(key[3:])}
        kept = sources - held
        directory = work / "models/distribution" / key
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "HEAD_COMPLETED.json"
        model_binding = dict(
            binding=binding,
            key=key,
            sources=sorted(kept),
            runtime_resources={"cpu_threads": threads},
        )
        paths = {
            "branch": directory / "branch.txt",
            "length": directory / "length.pkl",
            "coverage": directory / "coverage.pkl",
        }
        if receipt.exists():
            previous = read(receipt)
            require(previous["binding"] == model_binding, "Distribution cache input changed")
            verify(previous["files_sha256"])
        else:
            head = core.distribution_head.DistributionHead(
                threads=threads, branch_trees=160, length_trees=96
            )
            head.fit(X, queries, labels, kept)
            head.branch_model.booster_.save_model(str(paths["branch"]))
            for name, model in [("length", head.length_model), ("coverage", head.coverage_model)]:
                with paths[name].open("wb") as stream:
                    pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
            write(
                receipt,
                dict(
                    binding=model_binding,
                    files_sha256={str(p): sha(p) for p in paths.values()},
                    fit_info=head.fit_info,
                    formal_labels_read=False,
                    validation_labels_read=False,
                    no_uniform_distribution_substitution=True,
                ),
            )
        family[key] = {name: str(path) for name, path in paths.items()}
    return family


def train_neighbors(
    data_manifest, split_manifest, work_dir, method_library=None, method_config=None
):
    """Return a completed neighbor_bundle.json after all dated heads are fitted."""
    data = value(data_manifest)
    config = value(method_config or {})
    resources = runtime_resources(config)
    threads = resources["cpu_threads"]
    manifest, split, train, excluded = split_roles(split_manifest)
    date = validate_date(data["date"])
    require(
        int(config.get("folds", 5)) == 5 and int(config.get("seed", 20261021)) == 20261021,
        "Frozen neighbor method requires the original five folds and seed",
    )
    require(
        int(config.get("root_candidate_cap", 4096)) == 4096
        and int(config.get("conditional_candidate_cap", 512)) == 512,
        "Candidate cap changed",
    )
    method = method_library or MethodLibrary()
    core = method.core()
    folds = source_fold_map(manifest, train, fallback=core.legacy.source_folds)
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    frozen = work / "neighbor_bundle.json"
    code = method.receipt()
    code["orchestration_sources_sha256"] = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("neighbors.py"),
            Path(__file__).with_name("evidence.py"),
        ]
    }
    input_hashes = public_hashes(data)
    input_hashes[str(Path(data["merged_train"]).resolve())] = sha(data["merged_train"])
    binding = _binding(
        dict(
            date=date,
            split=split,
            excluded_task_origins=sorted(excluded),
            source_to_fold=folds,
            input_hashes=input_hashes,
            config=config,
            code=code,
            runtime_resources=resources,
        )
    )
    if frozen.exists():
        previous = read(frozen)
        require(
            previous["binding"] == binding, "Completed neighbor bundle has different dated inputs"
        )
        verify(previous["inputs_sha256"])
        verify(previous["model_files_sha256"])
        return frozen
    evidence_info = prepare_evidence(data, manifest, work / "evidence", core)
    with Path(evidence_info["evidence_pickle"]).open("rb") as stream:
        evidence = pickle.load(stream)
    refs = read(evidence_info["training_references"])
    require(
        set(refs) == set(map(str, train)),
        "Training-only reference cache has unexpected source roles",
    )
    public = make_public(data, core)
    policies, policy_receipt = load_policies(data, excluded)
    policy_path = work / "parsed_policies.json"
    write(policy_path, policies)
    root = _root_features(core, public, evidence, refs, policies, date, 4096)
    cache = work / "features"
    cache.mkdir(exist_ok=True)
    np.savez_compressed(cache / "root_base.npz", X=root["X"])
    write(cache / "root_queries.json", root["queries"])
    write(cache / "root_base_schema.json", root["schema"])
    models, schemas = {}, {
        "root_base": root["schema"],
        "root_pdb": root["schema"],
        "root_struct": root["schema"] + STRUCTURAL_NAMES,
        "structural_added": STRUCTURAL_NAMES,
    }
    rank = dict(kind="rank", root_only=True, gains=[0.0, 0.1, 1.0, 2.0])
    present = dict(kind="present", root_only=True)
    pdb_x, pdb_columns = core.experiment.add_relative(root["X"], root["queries"], root["schema"])
    structural_x, structural_columns = core.experiment.add_relative(
        np.concatenate((root["X"], root["structural"]), axis=1),
        root["queries"],
        root["schema"] + STRUCTURAL_NAMES,
    )
    for variant, X, columns, learner in [
        ("root_pdb", pdb_x, pdb_columns, core.experiment),
        ("root_struct", structural_x, structural_columns, core.structural_experiment),
    ]:
        for task, labels, spec in [
            ("rank", root["hierarchy"], rank),
            ("present", root["present"], present),
        ]:
            key = variant + "_" + task
            models[key] = _fit_family(
                key,
                X,
                labels,
                root["queries"],
                columns,
                spec,
                learner,
                folds,
                work,
                binding,
                threads=threads,
            )
    role_x, role_columns = core.experiment.add_relative(
        root["role_X"], root["role_queries"], root["role_schema"]
    )
    schemas["role"] = role_columns
    models["role"] = _fit_role(
        role_x,
        root["role_queries"],
        role_columns,
        refs,
        core,
        folds,
        work,
        binding,
        threads=threads,
    )
    distribution = _fit_distribution(
        root["X"],
        root["queries"],
        read(evidence_info["training_joint_labels"]),
        core,
        folds,
        work,
        binding,
        threads=threads,
    )
    del pdb_x, structural_x, role_x, root
    conditional = _conditional_features(core, public, evidence, refs, 512)
    schemas["conditional"] = conditional["schema"]
    schemas["pair_columns"] = conditional["pair_columns"]
    for task, labels, spec in [
        (
            "rank",
            conditional["hierarchy"],
            dict(kind="rank", root_only=False, gains=[0.0, 0.1, 1.0, 2.0]),
        ),
        ("present", conditional["present"], dict(kind="present", root_only=False)),
    ]:
        key = "conditional_" + task
        models[key] = _fit_family(
            key,
            conditional["X"],
            labels,
            conditional["queries"],
            conditional["schema"],
            spec,
            core.structural_experiment,
            folds,
            work,
            binding,
            threads=threads,
        )
    input_hashes.update(evidence_info["files_sha256"])
    input_hashes[str(policy_path)] = sha(policy_path)
    if policy_receipt.get("path"):
        input_hashes[policy_receipt["path"]] = policy_receipt["sha256"]
    model_paths = [p for family in models.values() for p in family.values()]
    model_paths += [p for group in distribution.values() for p in group.values()]
    bundle = dict(
        status="PASS",
        runtime_resources=resources,
        binding=binding,
        date=date,
        model_fit_date=date,
        protocol=config.get("protocol", "date_local_source_disjoint_v1"),
        formal_labels_read=False,
        validation_labels_read=False,
        training_sources=sorted(map(str, train), key=int),
        source_to_fold=folds,
        excluded_task_origins=sorted(excluded),
        evidence_pickle=evidence_info["evidence_pickle"],
        safe_index_pickle=evidence_info["safe_index_pickle"],
        training_references=evidence_info["training_references"],
        training_joint_labels=evidence_info["training_joint_labels"],
        schemas=schemas,
        models=models,
        parsed_policy_file=str(policy_path),
        rpsl=policy_receipt,
        distribution=distribution,
        distribution_schema=schemas["root_base"],
        distribution_scope="root contexts, matching frozen trial350_root_v1",
        feature_cache=dict(
            root_base=str(cache / "root_base.npz"), root_queries=str(cache / "root_queries.json")
        ),
        inputs_sha256=input_hashes,
        model_files_sha256={p: sha(p) for p in model_paths},
        computational_receipt=code,
        source_identity="canonical actual first ASN; never per-target CSV filename",
        unknown_policy_preserved=True,
        no_uniform_distribution_substitution=True,
    )
    write(frozen, bundle)
    write(work / "training_progress.json", dict(stage="completed", bundle=str(frozen)))
    return frozen
