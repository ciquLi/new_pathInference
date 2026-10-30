#!/usr/bin/env python3
"""Audit and separate full-source selection metadata from the model interface.

This never trains or modifies a model. The frozen model-scope validator still
runs in full, including every input and model SHA256 check, before publication.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from pathveil.upstream.contracts import read, require, sha, validate_model_scope, verify
from pathveil.upstream.evidence import public_hashes
from pathveil.upstream.method import MethodLibrary
from pathveil.upstream.neighbor_training import _binding
from pathveil.upstream.resources import runtime_resources
from pathveil.utils import object_hash, write_json

SOURCE_PROTOCOL = "date_local_all_eligible_source_disjoint_v2"
MODEL_PROTOCOL = "date_local_source_disjoint_v1"
SELECTION_POLICY = "all_canonical_unique_valid_paths_gt_threshold_v1"
FAMILIES = (
    "root_pdb_rank",
    "root_pdb_present",
    "root_struct_rank",
    "root_struct_present",
    "role",
    "conditional_rank",
    "conditional_present",
)
HEADS = {"full", "oof0", "oof1", "oof2", "oof3", "oof4"}


@contextmanager
def _run_lock(run):
    with (run / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "The experiment is still running; its bundle cannot be normalized"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _source_set(values, label):
    require(isinstance(values, list) and values, "Missing sources: " + label)
    result = set()
    for value in values:
        require(not isinstance(value, bool), "Invalid ASN: " + label)
        number = int(value)
        require(str(number) == str(value) and 0 < number < 2**32, "Invalid ASN: " + label)
        require(number not in result, "Duplicated source: " + label)
        result.add(number)
    return result


def _code_identity(project):
    return {
        str(path.relative_to(project)): sha(path)
        for path in sorted((project / "src/pathveil").rglob("*.py"))
    }


def _audit(run, project, original):
    identity, method, split_info, data = map(
        read,
        (
            run / "RUN_IDENTITY.json",
            run / "method_config.json",
            run / "upstream/split.json",
            run / "data/data_manifest.json",
        ),
    )
    require(identity["method_config"] == method, "Run method snapshot changed")
    require(method.get("protocol") == SOURCE_PROTOCOL, "This is not the full-source v2 experiment")
    selection = method.get("source_selection", {})
    require(
        selection.get("policy") == SELECTION_POLICY
        and selection.get("all_eligible_sources") is True
        and selection.get("min_paths_exclusive") == 10000
        and selection.get("split_ratios") == [4, 1, 1],
        "Unsupported full-source selection configuration",
    )
    require(original.get("protocol") == SOURCE_PROTOCOL, "Original bundle has an unknown protocol")
    require(original.get("status") == "PASS", "Neighbor training did not finish")
    require(
        identity["date"]
        == data["date"]
        == split_info["date"]
        == original["date"]
        == original["model_fit_date"],
        "Experiment/model/source dates disagree",
    )
    current_code = _code_identity(project)
    require(
        current_code and identity["code_sha256"] == current_code,
        "Current source code differs from frozen run",
    )
    registry_path = project / "configs/dates" / (data["date"] + ".json")
    require(sha(registry_path) == identity["registry_sha256"], "Date registry changed")
    registry = read(registry_path)
    configured = Path(registry.get("method_config", "configs/method_unified255.json"))
    require(
        read(configured if configured.is_absolute() else project / configured) == method,
        "Registered method configuration changed",
    )
    roles = {
        name: _source_set(split_info["split"].get(name), name)
        for name in ("train", "validation", "test", "formal_test")
    }
    assigned = set()
    for role, sources in roles.items():
        require(not assigned & sources, "Train/validation/test/formal source roles overlap")
        assigned.update(sources)
    train = roles["train"]
    heldout = roles["validation"] | roles["test"] | roles["formal_test"]
    require(
        _source_set(data["sources"], "registered formal sources") == roles["formal_test"],
        "Formal cohort differs from date data",
    )
    require(
        split_info.get("protocol") == SELECTION_POLICY,
        "Split does not declare the all-source policy",
    )
    eligibility = split_info.get("eligibility", {})
    require(
        eligibility.get("operator") == ">"
        and eligibility.get("threshold") == 10000
        and eligibility.get("path_count_field") == "unique_valid_paths"
        and eligibility.get("all_eligible_sources_assigned") is True
        and set(eligibility.get("eligible_sources", ()))
        == train | roles["validation"] | roles["test"],
        "Split does not assign every eligible original source",
    )
    require(
        set(original["excluded_task_origins"])
        == set(split_info["excluded_task_origins"])
        == assigned,
        "Full-source evidence mask differs from all task roles",
    )
    require(
        _source_set(original["training_sources"], "fitted training sources") == train,
        "Fitted source list differs from train",
    )
    folds = {str(k): v for k, v in split_info["source_to_fold"].items()}
    require(
        original["source_to_fold"] == folds
        and set(map(int, folds)) == train
        and set(folds.values()) == set(range(5)),
        "Source-fold identities differ or five folds are incomplete",
    )
    fold_policy = split_info.get("fold_assignment", {})
    require(
        fold_policy.get("policy") == "fixed_ASN_SHA256_modulo_5_v1",
        "Unexpected source-fold protocol",
    )
    salt = f"{method['seed']}:neighbor_oof_stable_source_v1:"
    require(
        fold_policy.get("salt") == salt
        and all(
            folds[str(source)]
            == int(hashlib.sha256(f"{salt}{source}".encode()).hexdigest(), 16) % 5
            for source in train
        ),
        "Registered folds differ from fixed source hash",
    )
    for name in ("formal_labels_read", "validation_labels_read"):
        require(original.get(name) is False, "Bundle declares held-out label access: " + name)
    require(original.get("test_labels_read", False) is False, "Bundle declares test-label access")
    upstream_identity = read(run / "upstream/IDENTITY.json")
    require(
        upstream_identity["method_config"] == method and upstream_identity["split"] == split_info,
        "Upstream identity config/split changed",
    )
    code = MethodLibrary(project).receipt()
    require(upstream_identity["code"] == code, "Upstream computational source identity changed")
    computational = copy.deepcopy(code)
    implementation_root = Path(__file__).resolve().parents[1] / "src/pathveil/upstream"
    computational["orchestration_sources_sha256"] = {
        str(implementation_root / name): sha(implementation_root / name)
        for name in ("neighbor_training.py", "neighbors.py", "evidence.py")
    }
    require(
        original["computational_receipt"] == computational, "Neighbor computational receipt changed"
    )
    canonical = str(Path(data["merged_train"]).resolve())
    inputs, models = original["inputs_sha256"], original["model_files_sha256"]
    expected_canonical = split_info["source_availability"]["canonical_sha256"]
    require(
        inputs.get(canonical) == expected_canonical == upstream_identity["canonical_sha256"],
        "Canonical original-byte identity differs",
    )
    model_binding_inputs = public_hashes(data)
    model_binding_inputs[canonical] = expected_canonical
    resources = runtime_resources(method)
    expected_binding = _binding(
        dict(
            date=data["date"],
            split=split_info["split"],
            excluded_task_origins=sorted(assigned),
            source_to_fold=folds,
            input_hashes=model_binding_inputs,
            config=method,
            code=computational,
            runtime_resources=resources,
        )
    )
    require(original["binding"] == expected_binding, "Neighbor input/config/code binding changed")
    require(original["runtime_resources"] == resources, "Neighbor registered CPU resources changed")
    training_references_path = Path(original["training_references"])
    references = read(training_references_path)
    require(set(map(int, references)) == train, "Training references contain non-training sources")
    require(
        str(training_references_path) in inputs, "Training references have no frozen input hash"
    )
    evidence_directory = run / "upstream/neighbor_models/evidence"
    audits, immutable = {}, {}
    for name in ("evidence_audit.json", "safe_index_audit.json"):
        path = evidence_directory / name
        require(str(path) in inputs, "Evidence audit has no frozen input hash")
        audit = read(path)
        require(
            audit["canonical_sha256"] == expected_canonical,
            "Evidence belongs to another canonical stream",
        )
        require(
            not set(map(int, audit["source_path_counts"])) & assigned,
            "Task source leaked into independent background evidence",
        )
        if "withheld_start_counts" in audit:
            require(
                not any(audit["withheld_start_counts"].values()),
                "Task source survived evidence masking",
            )
        audits[name] = audit
    raw_reference_hashes = {}
    for source, reference in references.items():
        require(int(reference["source"]) == int(source), "Reference source identity differs")
        path = Path(reference["path"])
        expected = reference["sha256"]
        require(
            sha(path)
            == expected
            == audits["evidence_audit.json"]["withheld_source_sha256"][source],
            "Training reference bytes differ from masked canonical source",
        )
        raw_reference_hashes[str(path)] = expected
    require(set(original["models"]) == set(FAMILIES), "Unsupported model family schema")
    allowed_model_paths, head_audits, head_hashes = set(), [], {}
    work = run / "upstream/neighbor_models"

    def audit_head(path, key, family, paths):
        receipt = read(path)
        binding = receipt["binding"]
        require(
            binding["binding"] == original["binding"] and binding["key"] == key,
            "Head input binding/key differs",
        )
        sources = _source_set(binding["sources"], family + "/" + key)
        allowed = (
            train
            if key == "full"
            else {source for source in train if folds[str(source)] != int(key[3:])}
        )
        require(
            sources <= allowed and not sources & heldout,
            "Model head contains held-out or own-fold training sources",
        )
        require(
            receipt.get("formal_labels_read") is False
            and receipt.get("validation_labels_read", False) is False
            and receipt.get("test_labels_read", False) is False,
            "Head declares held-out label access",
        )
        require(
            binding.get("runtime_resources") == {"cpu_threads": resources["cpu_threads"]},
            "Head CPU resource receipt differs",
        )
        if len(paths) == 1:
            require(
                receipt["sha256"] == models.get(paths[0]), "Head weight hash differs from bundle"
            )
        else:
            require(
                receipt["files_sha256"] == {p: models.get(p) for p in paths},
                "Distribution head asset hashes differ",
            )
        head_hashes[str(path)] = sha(path)
        head_audits.append(
            dict(
                family=family,
                key=key,
                declared_fit_sources=sorted(sources),
                allowed_training_sources=sorted(allowed),
                all_declared_sources_within_allowed=True,
                labels_for_every_allowed_source_assumed=False,
            )
        )
        allowed_model_paths.update(paths)

    for family in FAMILIES:
        mapping = original["models"][family]
        require(set(mapping) == HEADS, "Incomplete five-fold/full model family")
        for key, filename in mapping.items():
            path = Path(filename)
            require(
                path.resolve() == (work / "models" / family / (key + ".txt")).resolve(),
                "Model asset outside its frozen family path",
            )
            audit_head(path.with_suffix(".json"), key, family, [str(path)])
    require(set(original["distribution"]) == HEADS, "Incomplete distribution head family")
    for key, mapping in original["distribution"].items():
        require(set(mapping) == {"branch", "length", "coverage"}, "Incomplete distribution assets")
        folder = work / "models/distribution" / key
        paths = [str(folder / name) for name in ("branch.txt", "length.pkl", "coverage.pkl")]
        require(set(mapping.values()) == set(paths), "Distribution asset paths differ")
        audit_head(folder / "HEAD_COMPLETED.json", key, "distribution", paths)
    require(allowed_model_paths == set(models), "Unexpected or unbound model assets")
    normalized = copy.deepcopy(original)
    normalized["protocol"] = MODEL_PROTOCOL
    # Run the unchanged original gate, including all input and model hashes.
    validate_model_scope(normalized, data["date"], split_info["split"])
    immutable.update(
        {
            str(run / name): sha(run / name)
            for name in (
                "RUN_IDENTITY.json",
                "method_config.json",
                "upstream/IDENTITY.json",
                "upstream/split.json",
                "data/data_manifest.json",
            )
        }
    )
    return dict(
        status="PASS",
        date=data["date"],
        role_counts={k: len(v) for k, v in roles.items()},
        model_scope_protocol=MODEL_PROTOCOL,
        source_selection_protocol=SOURCE_PROTOCOL,
        neighbor_binding=original["binding"],
        model_files_sha256=models,
        inputs_sha256=inputs,
        head_receipts_sha256=head_hashes,
        head_source_audits=head_audits,
        training_reference_bytes_sha256=raw_reference_hashes,
        immutable_metadata_sha256=immutable,
        current_source_sha256=current_code,
        original_validator_passed=True,
        model_weights_modified=False,
        source_roles_folds_labels_or_features_modified=False,
    )


def normalize_run(run_dir, project_root=PROJECT):
    project, run = Path(project_root).resolve(), Path(run_dir).resolve()
    require(project / "runs" in run.parents and run.is_dir(), "Run must exist inside project runs/")
    bundle_path = run / "upstream/neighbor_models/neighbor_bundle.json"
    folder = run / "control/protocol_metadata_migration"
    original_path, receipt_path = (
        folder / "neighbor_bundle.original.json",
        folder / "migration.json",
    )
    with _run_lock(run):
        current_bytes = bundle_path.read_bytes()
        current = json.loads(current_bytes)
        migrated = current.get("protocol") == MODEL_PROTOCOL
        if migrated:
            require(
                original_path.is_file() and receipt_path.is_file(),
                "Model v1 bundle has no full-source migration evidence",
            )
            original_bytes, previous = original_path.read_bytes(), read(receipt_path)
            require(
                hashlib.sha256(original_bytes).hexdigest() == previous["original_bundle_sha256"],
                "Original bundle evidence changed",
            )
            require(
                hashlib.sha256(current_bytes).hexdigest() == previous["normalized_bundle_sha256"],
                "Normalized bundle changed after migration",
            )
            original = json.loads(original_bytes)
        else:
            require(
                current.get("protocol") == SOURCE_PROTOCOL,
                "Unknown protocol; no normalization is authorized",
            )
            require(
                "protocol_metadata_migration" not in current
                and "source_selection_protocol" not in current,
                "Unexpected preexisting migration metadata",
            )
            original_bytes, original = current_bytes, current
        audit = _audit(run, project, original)
        original_sha, audit_sha = hashlib.sha256(original_bytes).hexdigest(), object_hash(audit)
        normalized = copy.deepcopy(original)
        normalized.update(protocol=MODEL_PROTOCOL, source_selection_protocol=SOURCE_PROTOCOL)
        normalized["protocol_metadata_migration"] = dict(
            receipt_path=str(receipt_path),
            original_bundle_path=str(original_path),
            original_bundle_sha256=original_sha,
            audit_sha256=audit_sha,
        )
        validate_model_scope(
            normalized, original["date"], read(run / "upstream/split.json")["split"]
        )
        if migrated:
            require(
                current == normalized
                and previous["audit"] == audit
                and previous["audit_sha256"] == audit_sha
                and previous["status"] == "PASS",
                "Migration audit/metadata changed",
            )
            return receipt_path
        # Every check has passed. Preserve immutable original bytes before the
        # only mutable publication: the neighbor bundle's descriptive fields.
        folder.mkdir(parents=True, exist_ok=True)
        if original_path.exists():
            require(
                original_path.read_bytes() == original_bytes, "Existing original evidence differs"
            )
        else:
            with original_path.open("xb") as stream:
                stream.write(original_bytes)
        encoded = (
            json.dumps(normalized, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode()
        receipt = dict(
            status="PASS",
            migration="separate_model_scope_and_source_selection_metadata_v1",
            original_bundle_path=str(original_path),
            original_bundle_sha256=original_sha,
            normalized_bundle_path=str(bundle_path),
            normalized_bundle_sha256=hashlib.sha256(encoded).hexdigest(),
            audit_sha256=audit_sha,
            audit=audit,
            changed_fields=["protocol", "source_selection_protocol", "protocol_metadata_migration"],
            source_code_and_config_modified=False,
        )
        if receipt_path.exists():
            require(read(receipt_path) == receipt, "Existing migration receipt differs")
        else:
            write_json(receipt_path, receipt)
        temporary = bundle_path.with_name(bundle_path.name + ".normalizing." + str(os.getpid()))
        temporary.write_bytes(encoded)
        require(bundle_path.read_bytes() == original_bytes, "Neighbor bundle changed during audit")
        temporary.replace(bundle_path)
        return receipt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = normalize_run(args.run_dir)
    print(json.dumps(dict(status="PASS", receipt=str(receipt))))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
