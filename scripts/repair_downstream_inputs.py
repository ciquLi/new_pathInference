#!/usr/bin/env python3
"""Restore omitted two-AS derivatives in an isolated downstream input view.

The canonical corpus, original CSVs, upstream manifests/models and numerical
code are immutable. The original exact provenance audits must pass in full.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from pathveil.utils import read_json, sha256, utc_now, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def folder(run):
    return Path(run) / "control/derived_csv_repair"


@contextmanager
def run_lock(run):
    with (run / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Experiment is already running") from error
        yield


def verify_inputs(run):
    require(PROJECT / "runs" in run.parents, "Run must be inside project runs/")
    identity = read_json(run / "RUN_IDENTITY.json")
    code = {
        str(p.relative_to(PROJECT)): sha256(p)
        for p in sorted((PROJECT / "src/pathveil").rglob("*.py"))
    }
    require(identity["code_sha256"] == code, "Frozen numerical source changed")
    registry = PROJECT / "configs/dates" / (identity["date"] + ".json")
    require(sha256(registry) == identity["registry_sha256"], "Date registry changed")
    require(
        read_json(run / "method_config.json") == identity["method_config"],
        "Method snapshot changed",
    )
    method_path = Path(
        read_json(registry).get("method_config", "configs/method_unified255.json")
    )
    require(
        read_json(method_path if method_path.is_absolute() else PROJECT / method_path)
        == identity["method_config"],
        "Registered method changed",
    )
    data = read_json(run / "data/data_manifest.json")
    require(
        data["date"] == identity["date"] and data["mode"] == "bgp", "Wrong date or mode"
    )
    for path, digest in data["corpus_input_sha256"].items():
        # Evaluation input identities remain exactly as registered.
        require(sha256(path) == digest, "Frozen corpus input changed: " + path)
    for key, digest in data["public_input_sha256"].items():
        require(sha256(data["public_files"][key]) == digest, "Public input changed")
    release = read_json(run / "upstream/RELEASE.json")
    require(
        release["status"] == "PASS" and release["date"] == data["date"],
        "No complete dated release",
    )
    require(set(release["sources"]) == set(data["sources"]), "Formal cohort changed")
    bundle = read_json(run / "upstream/neighbor_models/neighbor_bundle.json")
    for path, digest in bundle["model_files_sha256"].items():
        require(sha256(path) == digest, "Trained neighbor weight changed")
    model = read_json(run / "upstream/path_model/MODEL.json")
    require(
        sha256(model["path_model"]) == model["model_sha256"],
        "Trained path weight changed",
    )
    return data, code


def verify_receipt(run):
    receipt = read_json(folder(run) / "REPAIR.json")
    require(receipt["status"] == "PASS", "Input repair is incomplete")
    require(
        receipt["original_data_manifest_sha256"]
        == sha256(run / "data/data_manifest.json"),
        "Original manifest changed",
    )
    for section in ("inputs_sha256", "outputs_sha256"):
        for path, digest in receipt[section].items():
            require(sha256(path) == digest, "Repair evidence changed: " + path)
    require(
        receipt["numerical_code_sha256"]
        == {
            str(p.relative_to(PROJECT)): sha256(p)
            for p in sorted((PROJECT / "src/pathveil").rglob("*.py"))
        },
        "Numerical code changed after repair",
    )
    return receipt


def derive(merged, expanded, originals, formal):
    """Read only training observations; reconstruct the full ordered suffix prefixes."""
    derived = {s: set() for s in originals}
    with Path(merged).open() as stream:
        for line in stream:
            nodes = line.strip().split("->")
            require(
                nodes[0] not in formal,
                "Formal-origin observations found in training corpus",
            )
            for index, source in enumerate(nodes[:-1]):
                if source not in derived or index == 0:
                    continue
                require(len(nodes) == len(set(nodes)), "Cyclic original training route")
                derived[source].update(
                    "->".join(nodes[index:stop])
                    for stop in range(index + 2, len(nodes) + 1)
                )
    observed = {s: set() for s in originals}
    with Path(expanded).open() as stream:
        for line in stream:
            source = line.split("->", 1)[0].strip()
            if source in observed:
                observed[source].add(line.strip())
    differences = {}
    for source, original in originals.items():
        full = derived[source]
        require(
            full == observed[source],
            "Expanded corpus differs from exact independent derivation: " + source,
        )
        require(
            original <= full,
            "CSV contains a path without independent provenance: " + source,
        )
        missing = full - original
        require(
            all(len(p.split("->")) == 2 for p in missing),
            "Difference includes longer paths: " + source,
        )
        require(all(p.startswith(source + "->") for p in full), "Mixed original source")
        differences[source] = dict(
            original_rows=len(original),
            normalized_rows=len(full),
            restored_two_AS_rows=len(missing),
        )
    require(
        any(d["restored_two_AS_rows"] for d in differences.values()),
        "No known two-AS omission to repair",
    )
    return derived, differences


def make_manifest(data, local_root, projected, original_train):
    result = copy.deepcopy(data)
    original_root = Path(data["source_data_root"])
    result["source_data_root"] = str(local_root)
    result["train_dir"] = str(projected)
    result["expanded_train"] = str(original_train / "expand_paths.txt")
    result["split_log"] = str(original_train.parent / "split_log.txt")
    # Map other inputs to the exact already registered assets. Real local CSVs
    # take precedence; preserve prepared test aggregates and graph mappings.
    for key, value in data.get("read_path_map", {}).items():
        try:
            relative = Path(key).relative_to(original_root)
        except ValueError:
            continue
        result.setdefault("read_path_map", {})[str(local_root / relative)] = value
    for key, value in data.get("path_prefix_map", {}).items():
        try:
            relative = Path(key).relative_to(original_root)
        except ValueError:
            continue
        result.setdefault("path_prefix_map", {})[str(local_root / relative)] = value
    result.setdefault("path_prefix_map", {})[str(local_root)] = str(original_root)
    # An exact mapping must not override the materialized corrected CSVs.
    for path in projected.glob("*.csv"):
        result.setdefault("read_path_map", {})[str(path)] = str(path)
    result[
        "downstream_input_view"
    ] = "exact_independent_derivatives_with_restored_two_AS_rows_v1"
    return result


def mirror_test_inputs(run, data):
    """The frozen worker uses physical paths; preserve byte-identical labels.

    Copies stay behind the original worker's test-access guard. Their paths
    and byte hashes are recorded, and no label enters construction/training.
    """
    target = (
        folder(run)
        / "files/data/aspath/path_by_vantage"
        / data["date"]
        / "path_without_prefix/test"
    )
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in data["sources"]:
        original = Path(data["test_dir"]) / (str(source) + ".csv")
        expected = data["corpus_input_sha256"][str(original)]
        path = target / original.name
        if not path.exists():
            temporary = path.with_suffix(".copying")
            shutil.copyfile(original, temporary)
            require(sha256(temporary) == expected, "Evaluation mirror copy changed")
            temporary.replace(path)
        require(sha256(path) == expected, "Evaluation mirror differs from frozen input")
        hashes[str(path)] = expected
    return hashes


def prepare_run(run_dir):
    run = Path(run_dir).resolve()
    with run_lock(run):
        data, code = verify_inputs(run)
        directory = folder(run)
        if (directory / "REPAIR.json").is_file():
            receipt = verify_receipt(run)
            mirrors = mirror_test_inputs(run, data)
            if any(receipt["outputs_sha256"].get(p) != h for p, h in mirrors.items()):
                receipt["outputs_sha256"].update(mirrors)
                receipt["byte_identical_evaluation_mirrors_sha256"] = mirrors
                write_json(directory / "REPAIR.json", receipt)
            return receipt
        require(
            not (run / "PREPARED.json").exists(),
            "Downstream already prepared; refuse input replacement",
        )
        original_train = Path(data["train_dir"])
        expanded = original_train / "expand_paths.txt"
        sources = list(map(str, data["sources"]))
        originals, inputs = {}, {}
        original_paths = {}
        for source in sources:
            path = original_train / (source + ".csv")
            if path.is_file():
                inputs[str(path)] = sha256(path)
                rows = [line.strip() for line in path.open() if line.strip()]
                require(len(rows) == len(set(rows)), "Original CSV contains duplicates")
                if rows:
                    originals[source] = set(rows)
                original_paths[source] = path
        for path in (
            run / "data/data_manifest.json",
            Path(data["merged_train"]),
            expanded,
            original_train.parent / "split_log.txt",
            run / "upstream/RELEASE.json",
        ):
            inputs[str(path)] = sha256(path)
        derived, differences = derive(
            data["merged_train"], expanded, originals, set(sources)
        )
        local_root = directory / "files"
        projected = (
            local_root
            / "data/aspath/path_by_vantage"
            / data["date"]
            / "path_without_prefix/train"
        )
        projected.mkdir(parents=True, exist_ok=True)
        for source in original_paths:
            path = projected / (source + ".csv")
            text = "".join(p + "\n" for p in sorted(derived.get(source, set())))
            if path.exists():
                require(path.read_text() == text, "Existing projected CSV changed")
            else:
                path.write_text(text)
        mirrors = mirror_test_inputs(run, data)
        manifest_path = directory / "data_manifest.json"
        normalized = make_manifest(data, local_root, projected, original_train)
        write_json(manifest_path, normalized)
        # Run both original gates without relaxing any equality condition.
        from pathveil.downstream.runner import prepare_downstream

        prepared_path = prepare_downstream(
            run, manifest_path, run / "upstream/RELEASE.json"
        )
        prepared = read_json(prepared_path)
        require(
            prepared["pg_local_knowledge_mode"]
            == prepared["train_extra_data_mode"]
            == "augmentation_only",
            "Consumer protocol changed",
        )
        for path, digest in inputs.items():
            require(
                sha256(path) == digest, "Original input changed during reconstruction"
            )
        outputs = {
            str(path): sha256(path)
            for path in [manifest_path, prepared_path, *projected.glob("*.csv")]
        }
        outputs.update(
            {path: sha256(path) for path in prepared["derived_source_audits"].values()}
        )
        outputs.update(mirrors)
        receipt = dict(
            status="PASS",
            date=data["date"],
            repaired_utc=utc_now(),
            original_data_manifest_sha256=sha256(run / "data/data_manifest.json"),
            downstream_data_manifest=str(manifest_path),
            inputs_sha256=inputs,
            outputs_sha256=outputs,
            numerical_code_sha256=code,
            source_differences=differences,
            fixed_test_pairs=prepared["fixed_test_pairs"],
            byte_identical_evaluation_mirrors_sha256=mirrors,
            original_exact_audits_passed=True,
            original_inputs_modified=False,
            upstream_weights_or_release_modified=False,
            training_labels_and_PG_library="unchanged augmentation_only admitted pool",
        )
        write_json(directory / "REPAIR.json", receipt)
        return receipt


def resume_run(run_dir, device="cuda"):
    run = Path(run_dir).resolve()
    with run_lock(run):
        verify_inputs(run)
        receipt = verify_receipt(run)
        # Match the formal CLI's run-local caches and thread configuration.
        from pathveil.upstream.resources import runtime_resources

        config = read_json(run / "method_config.json")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[key] = str(runtime_resources(config)["cpu_threads"])
        for key, name in {
            "TMPDIR": "tmp",
            "MPLCONFIGDIR": "matplotlib",
            "XDG_CACHE_HOME": "xdg",
            "TORCH_HOME": "torch",
        }.items():
            path = run / "runtime" / name
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        require(
            len(receipt.get("byte_identical_evaluation_mirrors_sha256", {}))
            == len(read_json(run / "data/data_manifest.json")["sources"]),
            "Evaluation mirrors are incomplete; prepare the repaired view first",
        )

        def status(stage, **details):
            value = dict(
                date=receipt["date"],
                pid=os.getpid(),
                run_dir=str(run),
                stage=stage,
                updated_utc=utc_now(),
                **details,
            )
            write_json(run / "status.json", value)
            print(json.dumps(value), flush=True)

        try:
            status(
                "downstream",
                status="running",
                input_repair=str(folder(run) / "REPAIR.json"),
            )
            from pathveil.downstream.runner import run_downstream

            summary = run_downstream(
                run,
                receipt["downstream_data_manifest"],
                run / "upstream/RELEASE.json",
                device=device,
            )
            status("complete", status="PASS", summary=str(summary))
            return 0
        except BaseException as error:
            status("failed", status="FAILED", error=repr(error))
            (run / "error.txt").write_text(traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_run(args.run_dir)
    print(
        json.dumps(
            dict(
                status=receipt["status"],
                receipt=str(folder(args.run_dir) / "REPAIR.json"),
            )
        )
    )


if __name__ == "__main__":
    main()
