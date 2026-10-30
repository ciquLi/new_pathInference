"""Recompute the frozen March 1 scores and selections after code migration.

This is an explicit cached-feature replay. It proves model-score/selector/row
order equivalence and never claims a new collection, refit or blind evaluation.
Only canonical models/ and data/upstream/ inputs are used.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import shutil
import sys

import lightgbm as lgb
import numpy as np

from .contracts import (
    ReferenceBoundary,
    path_rows,
    read,
    require,
    sha,
    validate_model_scope,
    verify,
    write,
    write_paths,
)
from .implementation.joint_selector import (
    SelectionPolicy,
    length_aware_representatives,
    select_joint_budget,
)
from .method import MethodLibrary


def _pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _select(case, prior, policy):
    case = dict(case)
    case["eligible"] = np.asarray(case["structural_eligible"], bool) & length_aware_representatives(
        case
    )
    return select_joint_budget(
        case,
        prior["branch_probability"],
        prior["length_probability"],
        policy,
        predicted_joint=prior["joint_probability"],
    )


def _score(model, matrix):
    require(
        matrix.ndim == 2 and matrix.shape[1] == 255 and matrix.dtype == np.float32,
        "Exact float32 255-feature matrix required",
    )
    require(np.isfinite(matrix).all(), "Nonfinite features")
    return model.predict(matrix, num_threads=2) if len(matrix) else np.empty(0, float)


def run_baseline(data_manifest, run_dir, method_config, project_root=None, sources=None):
    """Return RELEASE.json path; inference is restricted to the March bundle."""
    library = MethodLibrary(project_root)
    project = library.project
    require(data_manifest["date"] == "20260301", "Cached March features cannot serve another date")
    config = read(method_config) if isinstance(method_config, (str, Path)) else method_config
    require(
        config["method_id"] == "unified255_v1" and config["semantic_llm_review"] is False,
        "This release supports only the frozen unified255 method without LLM review",
    )
    manifest_path = project / "data/upstream/20260301/REGRESSION.json"
    manifest = read(manifest_path)
    bundle_path = Path(manifest["bundle"])
    bundle = read(bundle_path)
    split = read(manifest["split_manifest"])["split"]
    validate_model_scope(bundle, "20260301", split)
    requested = list(map(str, sources or data_manifest.get("sources") or manifest["sources"]))
    require(
        requested
        and len(requested) == len(set(requested))
        and set(requested) <= set(manifest["sources"]),
        "Replay source identity is absent or duplicated",
    )
    # Inputs are checked before a stage output can become reusable.
    verify(manifest["expected_hashes"])
    run = Path(run_dir).resolve() / "upstream"
    release_path = run / "RELEASE.json"
    identity = dict(
        date="20260301",
        method_id=config["method_id"],
        sources=requested,
        manifest_sha256=sha(manifest_path),
        bundle_sha256=sha(bundle_path),
        method_config=config,
        code=library.receipt(),
    )
    if release_path.exists():
        receipt = read(release_path)
        require(
            receipt["identity"] == identity, "Existing release has another input/config identity"
        )
        verify(receipt["outputs_sha256"])
        return release_path
    run.mkdir(parents=True, exist_ok=True)
    write(run / "IDENTITY.json", identity)
    guard = ReferenceBoundary()
    sys.addaudithook(guard.check)
    guard.enabled = True
    try:
        model = lgb.Booster(model_file=bundle["path_model"])
        require(
            model.feature_name() == bundle["path_feature_names"]
            and len(model.feature_name()) == 255,
            "Path model feature order differs",
        )
        policy = SelectionPolicy(**config["pure_selection"])
        records, files, outputs = [], {}, {}
        for source in requested:
            casefiles = manifest["cases"][source]
            with np.load(casefiles["base222"], allow_pickle=False) as archive:
                base = archive["X"]
            with np.load(casefiles["extra33"], allow_pickle=False) as archive:
                extra = archive["X_extra"]
            require(
                base.shape[1] == 222 and extra.shape == (len(base), 33),
                "222+33 row alignment changed",
            )
            scores = _score(model, np.concatenate((base, extra), axis=1))
            original = _pickle(casefiles["original_scores"])
            require(
                np.array_equal(scores, original["scores"]),
                "255 model-score replay differs: " + source,
            )
            prior = read(casefiles["prior"])
            pure_case = dict(
                original, scores=scores, novel=np.asarray(original["strict_novel"], bool)
            )
            pure_mask, _, pure_diag = _select(pure_case, prior, policy)
            pure = [tuple(map(str, original["paths"][int(i)])) for i in np.flatnonzero(pure_mask)]
            expected_pure = path_rows(casefiles["pure_paths"], source)
            require(pure == expected_pure, "Pure selected paths/order differs: " + source)
            bypath = {
                tuple(map(str, path)): float(scores[i]) for i, path in enumerate(original["paths"])
            }
            supplemental = _pickle(casefiles["supplemental_candidates"])
            with np.load(casefiles["supplemental255"], allow_pickle=False) as archive:
                new_scores = _score(model, archive["X"])
                require(
                    np.array_equal(new_scores, archive["scores"]),
                    "Supplemental 255 replay differs: " + source,
                )
            require(len(supplemental) == len(new_scores), "Supplemental rows differ")
            bypath.update(
                {
                    tuple(map(str, candidate["path"])): float(new_scores[i])
                    for i, candidate in enumerate(supplemental)
                }
            )
            unified = _pickle(casefiles["unified_scored_pool"])
            unified_scores = np.asarray([bypath[tuple(map(str, p))] for p in unified["paths"]])
            require(
                np.array_equal(unified_scores, unified["scores"]),
                "Unified score alignment differs: " + source,
            )
            # Pure/control budgets are generated within this same date.
            common = replace(policy, minimum_novel_paths=0, budget=len(pure))
            chosen_mask, _, unified_diag = _select(
                dict(unified, scores=unified_scores), prior, common
            )
            chosen = {
                tuple(map(str, unified["paths"][int(i)])) for i in np.flatnonzero(chosen_mask)
            }
            external = set(path_rows(casefiles["external_paths"], source))
            require(chosen <= set(pure) | external, "Unified selector emitted an unsupported route")
            ordered = [p for p in pure if p in chosen]
            ordered.extend(sorted(chosen - set(pure), key=lambda p: tuple(map(int, p))))
            target = run / "release" / (source + "_augmented_paths.csv")
            write_paths(target, ordered)
            require(
                sha(target) == sha(casefiles["expected_release"]),
                "Released bytes differ from the validated baseline: " + source,
            )
            grouped = {}
            for route in ordered:
                grouped.setdefault(source + "-" + route[-1], []).append(list(route))
            json_path = target.with_suffix(".json")
            write(json_path, grouped)
            files[source] = dict(
                path=str(target),
                sha256=sha(target),
                json_path=str(json_path),
                json_sha256=sha(json_path),
                rows=len(ordered),
            )
            outputs.update({str(target): sha(target), str(json_path): sha(json_path)})
            records.append(
                dict(
                    source=source,
                    feature_rows=len(scores),
                    supplemental_rows=len(new_scores),
                    selected=len(ordered),
                    pure_selected=len(pure),
                    nominal_budget=len(pure),
                    model_scores_byte_equal=True,
                    pure_selection_order_equal=True,
                    release_bytes_equal=True,
                    minimum_quantity_met=len(ordered) >= common.minimum_paths,
                    pure_diagnostics=pure_diag,
                    unified_diagnostics=unified_diag,
                )
            )
            print(
                json.dumps(
                    dict(
                        stage="upstream_migration_replay",
                        source=source,
                        selected=len(ordered),
                        equivalent=True,
                    )
                ),
                flush=True,
            )
        graph = {}
        for key, source_path in manifest["graph"].items():
            target = run / "graph" / (key + ".csv")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            graph[key + "_csv"] = str(target)
            outputs[str(target)] = sha(target)
        receipt = dict(
            status="PASS",
            date="20260301",
            method_id=config["method_id"],
            execution_mode="cached_date_local_features_recomputed_model_scores_and_selection",
            fresh_model_training=False,
            fresh_candidate_generation=False,
            sources=requested,
            files_by_source=files,
            selected_neighbors_csv=graph["selected_neighbors_csv"],
            recursive_anchors_csv=graph["recursive_anchors_csv"],
            data_root=data_manifest.get("data_root"),
            graph=graph,
            identity=identity,
            training_and_PG_pool="same_R_s_hash_augmentation_only",
            formal_reference_labels_read=False,
            no_source_or_query_exclusions=True,
            migration_verification=dict(
                all_model_scores_byte_equal=True, all_released_CSV_bytes_equal=True, records=records
            ),
            code=library.receipt(),
            inputs_sha256={
                str(manifest_path): sha(manifest_path),
                str(bundle_path): sha(bundle_path),
            },
            outputs_sha256=outputs,
        )
        write(run / "MIGRATION_VERIFIED.json", receipt["migration_verification"])
        write(release_path, receipt)
        return release_path
    finally:
        guard.enabled = False
