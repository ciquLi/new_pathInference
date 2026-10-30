"""Cold GMPI controls for a new quality-admitted library.

The production network, SmoothL1 objective, optimizer and driver loops remain
unchanged.  Every source trains one full-history p0, selected only on natural-history
pair-disjoint validation. No history masking or empty-history criterion is used.  Final references open only after all choices
are frozen; evaluation annotations live in a separate ordinary input copy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

from pathveil.utils import project_root

PROJECT = project_root()
os.environ.setdefault("PATHVEIL_PROJECT_ROOT", str(PROJECT))

from . import utilities as tools
from .utilities import copy_verified
from .utilities import rho_features
from pathveil.utils import object_hash, read_json, sha256, utc_now, write_json

VERSION = "dated_normal_history_GMPI_SmoothL1_p0_v1"
INPUT_TREES = (
    "model_input_data_PG",
    "neighbor_selection",
    "pathgen_results",
    "control/pathgen_completed",
    "path_augmentation_output",
)
_GUARDS = []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze(path, value):
    path = Path(path)
    require(not path.is_symlink(), "Frozen artifact may not be a symlink")
    if path.exists():
        require(read_json(path) == value, "Frozen quality training identity differs: " + str(path))
    else:
        write_json(path, value)


def safe(path, root):
    path, root = Path(os.path.abspath(path)), Path(root).resolve()
    require(
        (path == root or root in path.parents) and path.resolve() == path,
        "Output escaped quality run: " + str(path),
    )
    for item in (path, *path.parents):
        require(not item.is_symlink(), "Ordinary input copies may not traverse symlinks")
        if item == root:
            break
    return path


def source_worker(run, source, seed):
    return Path(run) / "control/gmpi" / ("as" + source) / ("seed" + str(seed))


def install_guard(driver, root):
    from pathveil.downstream.common import StageGuard

    guard = StageGuard(root, allow_raw_test=False)
    _GUARDS.append(guard)
    driver.ACCESS_GUARD = guard
    sys.addaudithook(guard.check)
    return guard


def reused_code():
    from pathveil.downstream import train_evaluate_gmpi as driver
    from pathveil.core import activate

    core = activate()
    files = [
        Path(__file__),
        Path(tools.__file__),
        Path(inspect.getsourcefile(copy_verified)),
        Path(inspect.getsourcefile(rho_features)),
        Path(driver.__file__),
    ]
    files += [
        core / name
        for name in (
            "GMPI_pathLikelihoodEst.py",
            "myDataset.py",
            "PathInference_trainer.py",
            "environment.py",
        )
    ]
    return {str(path.resolve()): sha256(path) for path in files}


def configured_variants(info):
    method = read_json(info["root"] / "METHOD_FROZEN.json")
    require(
        method["training_objective"] == "normal_history_huber",
        "Only the frozen normal-history Huber objective is supported",
    )
    return ["p0"]


def feature_rows(path, source, require_zero=False):
    """Hash only inference features and query order; optionally prove zero labels."""
    digest, count = hashlib.sha256(), 0
    columns = {
        name: hashlib.sha256() for name in ("candidates", "history", "history_pairs", "query_order")
    }
    with Path(path).open(newline="") as stream:
        for row in csv.reader(stream, strict=True):
            require(
                len(row) == 5 and tools.canonical_pair(row[4]).split("->")[0] == source,
                "Malformed five-column source input",
            )
            labels = row[3].split("|") if row[3] else []
            require(len(labels) == len(row[0].split("|")), "Candidate/label count differs")
            if require_zero:
                require(
                    all(math.isfinite(float(value)) and float(value) == 0 for value in labels),
                    "Formal test labels must remain zero before selection",
                )
            digest.update(
                (
                    json.dumps([row[0], row[1], row[2], row[4]], separators=(",", ":")) + "\n"
                ).encode()
            )
            for name, value in zip(columns, (row[0], row[1], row[2], row[4])):
                columns[name].update((json.dumps(value) + "\n").encode())
            count += 1
    require(count > 0, "Model candidate input is empty")
    return dict(
        rows=count,
        inference_features_sha256=digest.hexdigest(),
        labels_zero=require_zero,
        feature_column_sha256={name: value.hexdigest() for name, value in columns.items()},
    )


def inputs(source, run_root):
    os.environ.setdefault("PATHVEIL_PROJECT_ROOT", str(PROJECT))
    from pathveil.downstream import common

    source = str(source)
    require(tools.canonical_pair(source + "->1").split("->")[0] == source, "Invalid source")
    root = common.validate_run(run_root)
    method_path = root / "METHOD_FROZEN.json"
    method = read_json(method_path)
    require(source in method["sources"], "Unregistered quality source")
    child = safe(root / "pg_inputs" / ("as" + source), root)
    prepared = common.verify_runtime_authorization(child)
    require(
        prepared["sources"] == [source] and prepared["date"] == method["date"],
        "Wrong source/date in new PG inputs",
    )
    require(method["date"] == prepared["date"], "Root method and PG input dates differ")
    config = prepared["configuration"]
    train = (
        child
        / "model_input_data_PG/train/meta-devoid"
        / (source + "_train_extra_data_generated_data.csv")
    )
    test = child / "model_input_data_PG/test/meta-devoid" / (source + "_generated_data.csv")
    dispatch_file = child / "control/pathgen_completed" / ("test_inputs_" + source + ".json")
    dispatch = read_json(dispatch_file)
    require(dispatch["sha256"] == sha256(test), "New PG test input/dispatch differs")
    role = dispatch.get("label_blind_compilation", {})
    require(
        role.get("label_semantics") == "unavailable_zero_placeholder"
        and role.get("reference_metrics_computed") is False,
        "Quality training requires label-blind PG test inputs",
    )
    consumer_hashes = {}
    for name in (
        "training_inputs_%s_truth.json",
        "training_inputs_%s_local.json",
        "test_inputs_%s_local.json",
    ):
        path = child / "control/consumer_audit" / (name % source)
        audit = read_json(path)
        require(
            audit["same_admitted_pool_verified"] is True
            and audit["path_set_sha256"] == prepared["admitted_library_path_set_sha256"][source],
            "Training labels and PG local knowledge consumed different admitted paths",
        )
        consumer_hashes[str(path)] = sha256(path)
    audit_path = safe(root / "control/gmpi_zero_input_audits" / (source + ".json"), root)
    state = test.stat()
    audit_binding = dict(
        test_sha256=sha256(test),
        prepared_sha256=sha256(child / "PREPARED.json"),
        code_sha256=sha256(__file__),
        stat=[state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns],
    )
    if audit_path.exists():
        audit = read_json(audit_path)
        require(audit["binding"] == audit_binding, "Zero-label audit input changed")
    else:
        audit = dict(binding=audit_binding, **feature_rows(test, source, True))
        freeze(audit_path, audit)
    rho = rho_features(source, root)
    identity = dict(
        version=VERSION,
        date=prepared["date"],
        source=source,
        root=str(root),
        method_sha256=sha256(method_path),
        input_run=str(child),
        prepared_sha256=sha256(child / "PREPARED.json"),
        train_csv_sha256=sha256(train),
        zero_test_csv_sha256=sha256(test),
        test_inference_features_sha256=audit["inference_features_sha256"],
        test_model_rows=audit["rows"],
        test_dispatch_sha256=sha256(dispatch_file),
        rs_path_set_sha256=prepared["admitted_library_path_set_sha256"][source],
        consumer_audits_sha256=consumer_hashes,
        admitted_pool=prepared["admitted_pool"][source],
        public_rho=float(rho["rho"]),
        high_rho=float(rho["rho"]) > 1,
        configuration=config,
        code_sha256=reused_code(),
        legacy_evidence_origin_boundary="legacy_uncertified; inherited graph/history origin not retrospectively certified",
        raw_formal_labels_used_for_training_or_selection=False,
        objective="unchanged production SmoothL1/Huber",
    )
    return dict(
        root=root,
        source=source,
        child=child,
        prepared=prepared,
        config=config,
        train=train,
        test=test,
        identity=identity,
    )


def copy_input_run(info, destination, role):
    root, child = info["root"], info["child"]
    destination = safe(destination, root)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for relative in (*INPUT_TREES, "PREPARED.json"):
        original = child / relative
        require(
            original.exists() and not original.is_symlink(),
            "Missing ordinary PG input: " + str(original),
        )
        entries = [original] if original.is_file() else sorted(original.rglob("*"))
        require(all(not path.is_symlink() for path in entries), "Input subtree contains a symlink")
        files = [path for path in entries if path.is_file()]
        require(files, "Empty PG input subtree: " + relative)
        for original in files:
            require(not original.is_symlink(), "PG input may not be a symlink")
            target = safe(destination / original.relative_to(child), root)
            digest = sha256(original)
            copy_verified(original, target, digest)
            require(
                target.stat().st_nlink == 1, "Copied numerical inputs must not share hard links"
            )
            records.append(dict(source=str(original), destination=str(target), sha256=digest))
    receipt = dict(
        status="PASS",
        role=role,
        input_identity=info["identity"],
        files=records,
        ordinary_independent_copies=True,
        pretrained_model_copied=False,
    )
    freeze(destination / "INPUTS_READY.json", receipt)
    return receipt


def shared_split(info):
    from pathveil.downstream import train_evaluate_gmpi as driver

    parent = safe(info["root"] / "gmpi/shared_splits", info["root"])
    parent.mkdir(parents=True, exist_ok=True)
    worker = parent / ("as" + info["source"])
    with (parent / ("as" + info["source"] + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = worker / "surrogate_split"
        if not (directory / "split_manifest.json").exists():
            require(not directory.exists(), "Incomplete new split exists; do not replace it")
            worker.mkdir(exist_ok=True)
            driver.split_training(
                info["train"],
                directory,
                info["source"],
                info["config"]["seed"],
                info["config"]["validation_fraction"],
            )
        split = tools.fixed_split(worker, info["train"])
        binding = {
            str(directory / name): sha256(directory / name)
            for name in ("train.csv", "validation.csv", "split_manifest.json")
        }
        return worker, split, binding


def variant_paths(info, variant):
    require(variant == "p0", "Only normal-history p0 training is enabled")
    run = safe(info["root"] / "gmpi" / variant / ("as" + info["source"]), info["root"])
    return run, source_worker(run, info["source"], info["config"]["seed"])


@contextmanager
def training_adapter(driver, trainer, info, variant, run, shared_worker, split_hashes):
    old_train, old_test, old_split = trainer.train, trainer.test, driver.split_training
    old_write, old_guard = driver.json_write, driver.install_write_guard
    validation_audits, mask_audits = {}, {}

    def split(path, destination, source, seed, fraction):
        require(
            source == info["source"]
            and seed == info["config"]["seed"]
            and fraction == info["config"]["validation_fraction"]
            and sha256(path) == info["identity"]["train_csv_sha256"],
            "Shared split request differs",
        )
        return tools.fixed_split(shared_worker, path, destination)

    def train(model, dataset, optimizer, env, epoch, code, num_infer=1):
        model._pathveil_metadata.update(
            quality_admission_version=VERSION,
            quality_training_variant=variant,
            initialized_from_checkpoint=False,
            admitted_library_path_set_sha256=info["identity"]["rs_path_set_sha256"],
            history_dropout_probability=0.0,
        )
        require(variant == "p0", "History-masked training is disabled")
        return old_train(model, dataset, optimizer, env, epoch, code, num_infer)

    def test(model, dataset, optimizer, env, epoch, code, num_infer=1):
        result = old_test(model, dataset, optimizer, env, epoch, code, num_infer)
        require(result[2] > 0 and math.isfinite(result[1]), "Invalid natural validation")
        receipt = dict(
            natural_jaccard=result[1] / result[2],
            criterion_value=result[1] / result[2],
            query_count=result[2],
            final_references_opened=False,
        )
        path = run / "epoch_validation" / ("epoch_%02d.json" % epoch)
        freeze(path, receipt)
        validation_audits[str(path)] = sha256(path)
        return result

    def write(path, value):
        if Path(path).name in ("configuration.json", "training_manifest.json"):
            value = dict(
                value,
                quality_admission_version=VERSION,
                training_variant=variant,
                initialized_from_checkpoint=False,
                objective="unchanged production SmoothL1/Huber",
                validation_criterion="natural J",
                history_dropout_probability=0,
                admitted_library_path_set_sha256=info["identity"]["rs_path_set_sha256"],
                quality_input_identity_sha256=object_hash(info["identity"]),
                shared_split_sha256=split_hashes,
                adapter_code_sha256=sha256(__file__),
                reused_utility_code_sha256=info["identity"]["code_sha256"],
                original_test_used_for_selection=False,
            )
            if Path(path).name == "training_manifest.json":
                value.update(
                    validation_audits_sha256=validation_audits,
                    history_mask_audits_sha256=mask_audits,
                )
        elif Path(path).name == "source_code_identity.json":
            value = dict(value, **info["identity"]["code_sha256"])
        old_write(path, value)

    trainer.train, trainer.test = train, test
    driver.split_training, driver.json_write = split, write
    driver.install_write_guard = lambda: install_guard(driver, info["root"])
    try:
        yield
    finally:
        trainer.train, trainer.test = old_train, old_test
        driver.split_training, driver.json_write, driver.install_write_guard = (
            old_split,
            old_write,
            old_guard,
        )


def invoke_driver(info, run, stage, device, cpu_threads):
    from pathveil.core import activate

    activate(data_manifest=info["prepared"]["data_manifest"])
    from pathveil.downstream import train_evaluate_gmpi as driver

    audit = info["prepared"].get("derived_source_audits", {}).get(info["source"])
    arguments = argparse.Namespace(device=device, cpu_threads=cpu_threads)
    old_argv = sys.argv
    try:
        sys.argv = tools.driver_arguments(
            run, info["source"], stage, info["config"], arguments, audit
        )
        with tools.reuse_driver_runtime():
            driver.main()
    finally:
        sys.argv = old_argv


def _verify_training(info, variant):
    run, worker = variant_paths(info, variant)
    copied = read_json(run / "INPUTS_READY.json")
    require(
        copied["role"] == variant and copied["input_identity"] == info["identity"],
        "Training input-copy identity differs",
    )
    for item in copied["files"]:
        path = safe(item["destination"], info["root"])
        require(
            path.is_file() and path.stat().st_nlink == 1 and sha256(path) == item["sha256"],
            "Training numerical input copy changed",
        )
    manifest = read_json(worker / "training_manifest.json")
    require(
        manifest["model_sha256"] == sha256(worker / "model.pkl")
        and manifest["quality_admission_version"] == VERSION
        and manifest["training_variant"] == variant
        and manifest["initialized_from_checkpoint"] is False
        and manifest["objective"] == "unchanged production SmoothL1/Huber"
        and manifest["history_dropout_probability"] == 0
        and manifest["adapter_code_sha256"] == sha256(__file__)
        and manifest["reused_utility_code_sha256"] == info["identity"]["code_sha256"]
        and manifest["training_input_sha256"] == info["identity"]["train_csv_sha256"]
        and manifest["test_input_sha256"] == info["identity"]["zero_test_csv_sha256"]
        and manifest["quality_input_identity_sha256"] == object_hash(info["identity"])
        and manifest["admitted_library_path_set_sha256"] == info["identity"]["rs_path_set_sha256"]
        and manifest["original_test_used_for_selection"] is False,
        "Cold checkpoint provenance differs from new quality inputs",
    )
    shared_worker, split, hashes = shared_split(info)
    require(manifest["shared_split_sha256"] == hashes, "Checkpoint used another surrogate split")
    local_split = {}
    for name in ("train.csv", "validation.csv", "split_manifest.json"):
        path = worker / "surrogate_split" / name
        expected = sha256(shared_worker / "surrogate_split" / name)
        require(
            not path.is_symlink() and sha256(path) == expected,
            "Variant surrogate split bytes differ",
        )
        local_split[str(path)] = expected
    configuration = read_json(worker / "configuration.json")
    arguments = configuration["arguments"]
    plan = read_json(run / "TRAINING_PLAN.json")
    require(
        plan["input_identity"] == info["identity"]
        and plan["variant"] == variant
        and plan["cold_initialization"] is True
        and arguments["device"] == plan["runtime_request"]["device"]
        and arguments["cpu_threads"] == plan["runtime_request"]["cpu_threads"],
        "Training runtime/plan identity differs",
    )
    require(
        all(
            arguments[key] == info["config"][key]
            for key in (
                "date",
                "seed",
                "k",
                "hidden_size",
                "batch_size",
                "lr",
                "epochs",
                "patience",
                "validation_fraction",
                "workers",
            )
        )
        and arguments["source_asn"] == info["source"]
        and Path(arguments["experiment_dir"]) == run,
        "Training configuration differs from the frozen production configuration",
    )
    code_identity = read_json(worker / "source_code_identity.json")
    for name, digest in code_identity.items():
        require(sha256(name) == digest, "Actual training source bytes changed")
    for name, digest in {
        **manifest["validation_audits_sha256"],
        **manifest["history_mask_audits_sha256"],
    }.items():
        require(sha256(name) == digest, "Training validation/mask audit changed")
    history = read_json(worker / "training_history.json")
    require(history and len(history) == manifest["epochs"], "Training history is incomplete")
    require(
        len(manifest["validation_audits_sha256"]) == len(history)
        and len(manifest["history_mask_audits_sha256"]) == 0,
        "Epoch validation/history-mask audits are incomplete",
    )
    for record in history:
        value = read_json(run / "epoch_validation" / ("epoch_%02d.json" % record["epoch"]))
        expected = value["natural_jaccard"]
        require(
            value["query_count"] == split["row_counts"]["validation"]
            and math.isclose(value["criterion_value"], expected, abs_tol=1e-12)
            and value["criterion_value"] == record["validation_jaccard_reward"],
            "Epoch validation criterion or complete query count differs",
        )
    best = max(history, key=lambda row: row["validation_jaccard_reward"])
    require(
        best["epoch"] == manifest["selected_epoch"]
        and best["validation_jaccard_reward"] == manifest["best_surrogate_validation_jaccard"],
        "Checkpoint did not use its frozen validation criterion",
    )
    validation = read_json(
        run / "epoch_validation" / ("epoch_%02d.json" % manifest["selected_epoch"])
    )
    require(
        validation["query_count"] == split["row_counts"]["validation"]
        and validation["criterion_value"] == manifest["best_surrogate_validation_jaccard"],
        "Selected validation view/count differs",
    )
    return dict(
        status="PASS",
        variant=variant,
        run=str(run),
        worker=str(worker),
        model=str(worker / "model.pkl"),
        model_sha256=manifest["model_sha256"],
        training_manifest=str(worker / "training_manifest.json"),
        training_manifest_sha256=sha256(worker / "training_manifest.json"),
        configuration_sha256=sha256(worker / "configuration.json"),
        source_code_identity_sha256=sha256(worker / "source_code_identity.json"),
        training_plan_sha256=sha256(run / "TRAINING_PLAN.json"),
        runtime_sha256=sha256(worker / "runtime.json"),
        input_copy_receipt_sha256=sha256(run / "INPUTS_READY.json"),
        local_split_sha256=local_split,
        selected_epoch=manifest["selected_epoch"],
        validation=validation,
        shared_split_sha256=hashes,
        input_identity_sha256=object_hash(info["identity"]),
    )


def train_variant(source, run_root, variant, device="cuda", cpu_threads=2):
    info = inputs(source, run_root)
    run, worker = variant_paths(info, variant)
    copy_input_run(info, run, variant)
    shared_worker, _, split_hashes = shared_split(info)
    freeze(
        run / "TRAINING_PLAN.json",
        dict(
            input_identity=info["identity"],
            variant=variant,
            shared_split_sha256=split_hashes,
            cold_initialization=True,
            objective="production SmoothL1/Huber",
            runtime_request=dict(device=device, cpu_threads=cpu_threads),
        ),
    )
    if not (worker / "training_manifest.json").exists():
        require(
            not (worker / "model.pkl").exists(),
            "Incomplete training checkpoint exists; do not reload or replace it",
        )
        from pathveil.core import activate

        activate(data_manifest=info["prepared"]["data_manifest"])
        import PathInference_trainer as trainer
        from pathveil.downstream import train_evaluate_gmpi as driver

        with training_adapter(driver, trainer, info, variant, run, shared_worker, split_hashes):
            invoke_driver(info, run, "train", device, cpu_threads)
    completed = _verify_training(info, variant)
    freeze(run / "TRAINING_COMPLETED.json", completed)
    return completed


def train_p0(source, run_root, device="cuda", cpu_threads=2):
    return train_variant(source, run_root, "p0", device, cpu_threads)


def verify_training(source, run_root, variant):
    info = inputs(source, run_root)
    expected = _verify_training(info, variant)
    path = Path(expected["run"]) / "TRAINING_COMPLETED.json"
    require(
        path.is_file() and read_json(path) == expected,
        "Training completion receipt is missing or changed",
    )
    return expected


def selection_value(info):
    variants = ["p0"]
    candidates = {
        variant: verify_training(info["source"], info["root"], variant) for variant in variants
    }
    chosen = "p0"
    return dict(
        status="PASS",
        version=VERSION,
        source=info["source"],
        chosen_variant=chosen,
        input_identity=info["identity"],
        candidates=candidates,
        chosen_model=candidates[chosen]["model"],
        chosen_model_sha256=candidates[chosen]["model_sha256"],
        chosen_training_manifest=candidates[chosen]["training_manifest"],
        chosen_training_manifest_sha256=candidates[chosen]["training_manifest_sha256"],
        selected_only_on_new_shared_surrogate_validation=True,
        final_test_used_for_selection=False,
        tie_policy="retain p0",
        inference="one chosen checkpoint with complete natural history",
    )


def select(source, run_root):
    info = inputs(source, run_root)
    selected = selection_value(info)
    path = info["root"] / "gmpi/selections" / ("as" + info["source"]) / "SELECTION_FROZEN.json"
    freeze(path, selected)
    return selected


def verify_selection(source, run_root):
    info = inputs(source, run_root)
    path = info["root"] / "gmpi/selections" / ("as" + info["source"]) / "SELECTION_FROZEN.json"
    require(path.is_file(), "Model selection is not frozen")
    expected = selection_value(info)
    require(read_json(path) == expected, "Frozen selection differs from validated candidates")
    return expected


def freeze_selections(run_root):
    root = Path(run_root).resolve()
    method = read_json(root / "METHOD_FROZEN.json")
    mapping = {}
    for source in method["sources"]:
        path = root / "gmpi/selections" / ("as" + source) / "SELECTION_FROZEN.json"
        require(path.is_file(), "All registered selections must freeze before evaluation")
        current = select(source, root)
        require(
            current["final_test_used_for_selection"] is False, "Selection consumed final labels"
        )
        mapping[str(path)] = sha256(path)
    receipt = dict(
        status="PASS",
        sources=method["sources"],
        selections_sha256=mapping,
        method_sha256=sha256(root / "METHOD_FROZEN.json"),
        final_labels_opened=False,
    )
    freeze(root / "ALL_MODEL_SELECTIONS_FROZEN.json", receipt)
    return receipt


def verify_barrier(info):
    root = info["root"]
    barrier_path = root / "ALL_MODEL_SELECTIONS_FROZEN.json"
    require(
        barrier_path.is_file(),
        "Freeze every registered model selection before opening final references",
    )
    barrier = read_json(barrier_path)
    method = read_json(root / "METHOD_FROZEN.json")
    require(
        barrier["status"] == "PASS"
        and barrier["sources"] == method["sources"]
        and barrier["method_sha256"] == sha256(root / "METHOD_FROZEN.json"),
        "Cohort selection barrier changed",
    )
    require(
        set(barrier["selections_sha256"])
        == {
            str(root / "gmpi/selections" / ("as" + s) / "SELECTION_FROZEN.json")
            for s in method["sources"]
        },
        "Selection barrier omitted a registered source",
    )
    for path, digest in barrier["selections_sha256"].items():
        require(sha256(path) == digest, "Frozen cohort model selection changed")
    selection = read_json(
        root / "gmpi/selections" / ("as" + info["source"]) / "SELECTION_FROZEN.json"
    )
    require(
        selection["input_identity"] == info["identity"]
        and sha256(selection["chosen_model"]) == selection["chosen_model_sha256"]
        and sha256(selection["chosen_training_manifest"])
        == selection["chosen_training_manifest_sha256"],
        "Selected model or new input identity changed",
    )
    return selection, sha256(barrier_path)


def load_vocabulary(info, training_manifest):
    data = read_json(info["prepared"]["data_manifest"])
    relative = Path("data/aspath/path_by_vantage") / info["prepared"]["date"] / "AS_frequency.json"
    candidates = [
        Path(data[key]) / relative for key in ("source_data_root", "data_root") if data.get(key)
    ]
    expected = {
        digest
        for name, digest in data.get("generated_sha256", {}).items()
        if name.endswith("/" + str(relative))
    }
    require(len(expected) == 1, "Vocabulary frequency file lacks an exact dated binding")
    frequency = next(
        (path for path in candidates if path.is_file() and sha256(path) in expected), None
    )
    require(frequency is not None, "Cannot resolve the exact dated training vocabulary")
    values = read_json(frequency)
    values = json.loads(values) if isinstance(values, str) else values
    itos = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"] + sorted(values)
    require(
        hashlib.sha256(json.dumps(itos, separators=(",", ":")).encode()).hexdigest()
        == training_manifest["vocab_sha256"],
        "Evaluation vocabulary differs from chosen model",
    )
    return itos


def annotate_test_csv(original, destination, source, itos, truth):
    """Fill only candidate labels in an evaluation-owned copy after authorization."""
    before = feature_rows(original, source, True)
    temporary = Path(destination).with_name(
        Path(destination).name + ".annotation." + str(os.getpid())
    )
    references = {pair: [set(path) for path in paths] for pair, paths in truth.items()}
    with Path(original).open(newline="") as reader, temporary.open("w", newline="") as writer:
        output = csv.writer(writer)
        for row in csv.reader(reader, strict=True):
            pair = tools.canonical_pair(row[4])
            require(
                pair in references,
                "Generated model query is absent from registered final reference",
            )
            labels = []
            for encoded in row[0].split("|"):
                tokens = encoded.split("->")
                require(
                    all(t.isascii() and t.isdecimal() and 0 <= int(t) < len(itos) for t in tokens),
                    "Invalid candidate vocabulary index",
                )
                path = tuple(itos[int(t)] for t in tokens)
                require(
                    path[0] + "->" + path[-1] == pair and all(n.isdecimal() for n in path),
                    "Candidate endpoints/vocabulary differ",
                )
                nodes = set(path)
                labels.append(
                    str(
                        round(
                            max(
                                len(nodes & actual) / len(nodes | actual)
                                for actual in references[pair]
                            ),
                            3,
                        )
                    )
                )
            row[3] = "|".join(labels)
            output.writerow(row)
    after = feature_rows(temporary, source)
    require(
        before["rows"] == after["rows"]
        and before["inference_features_sha256"] == after["inference_features_sha256"],
        "Evaluation annotations changed candidate/history/query features",
    )
    require(
        Path(original).resolve() != Path(destination).resolve(),
        "Never annotate original zero-label input",
    )
    if Path(destination).exists():
        require(
            not Path(destination).is_symlink()
            and sha256(destination) in {sha256(original), sha256(temporary)},
            "Existing evaluation copy has different features/labels",
        )
    temporary.replace(destination)
    return dict(**after, zero_csv_sha256=sha256(original), annotated_csv_sha256=sha256(destination))


def _prepare_evaluation_inputs(source, run_root):
    """CPU IO only; final labels become evaluation diagnostics after the barrier."""
    info = inputs(source, run_root)
    selection, barrier_sha = verify_barrier(info)
    out = safe(info["root"] / "evaluations" / ("as" + info["source"]), info["root"])
    run = out / "evaluation_run"
    ready_path = out / "EVALUATION_INPUTS_READY.json"
    if ready_path.exists():
        ready = read_json(ready_path)
        require(
            ready["barrier_sha256"] == barrier_sha
            and ready["chosen_model_sha256"] == selection["chosen_model_sha256"],
            "Evaluation choice changed",
        )
        for path, digest in ready["files_sha256"].items():
            require(sha256(path) == digest, "Annotated evaluation input changed")
        return info, selection, out, run, ready
    start_path = out / "INPUT_ANNOTATION_STARTED.json"
    test_relative = info["test"].relative_to(info["child"])
    dispatch_relative = Path("control/pathgen_completed") / (
        "test_inputs_" + info["source"] + ".json"
    )
    mutable = {
        run / name
        for name in (
            test_relative,
            dispatch_relative,
            Path("pathgen_results/PG-test-result.csv"),
            Path("PREPARED.json"),
        )
    }
    if start_path.exists():
        started = read_json(start_path)
        require(
            started["input_identity_sha256"] == object_hash(info["identity"])
            and started["barrier_sha256"] == barrier_sha
            and started["model_sha256"] == selection["chosen_model_sha256"]
            and started["input_copy_receipt_sha256"] == sha256(run / "INPUTS_READY.json"),
            "Interrupted annotation belongs to different frozen inputs or model",
        )
        copied = read_json(run / "INPUTS_READY.json")
        for item in copied["files"]:
            path = safe(item["destination"], info["root"])
            if path not in mutable:
                require(
                    sha256(path) == item["sha256"],
                    "Unchanged evaluation input was modified during annotation",
                )
    else:
        copy_input_run(info, run, "evaluation_zero_labels_before_annotation")
        started = dict(
            input_identity_sha256=object_hash(info["identity"]),
            barrier_sha256=barrier_sha,
            model_sha256=selection["chosen_model_sha256"],
            annotation_started_utc=utc_now(),
            input_copy_receipt_sha256=sha256(run / "INPUTS_READY.json"),
        )
    worker = source_worker(run, info["source"], info["config"]["seed"])
    worker.mkdir(parents=True, exist_ok=True)
    model = copy_verified(
        selection["chosen_model"], worker / "model.pkl", selection["chosen_model_sha256"]
    )
    manifest_path = copy_verified(
        selection["chosen_training_manifest"],
        worker / "training_manifest.json",
        selection["chosen_training_manifest_sha256"],
    )
    require(
        model.stat().st_nlink == manifest_path.stat().st_nlink == 1,
        "Evaluation checkpoint copies must be independent",
    )
    freeze(start_path, started)
    from pathveil.downstream import common, train_evaluate_gmpi as driver

    install_guard(driver, info["root"])
    for guard in _GUARDS:
        if guard.run == info["root"]:
            guard.allow_evaluation(model, selection["chosen_model_sha256"])
    data = read_json(info["prepared"]["data_manifest"])
    reference = Path(data["test_dir"]) / (info["source"] + ".csv")
    expected = info["prepared"]["test_identity"][info["source"]]
    access_started = started["annotation_started_utc"]
    require(sha256(reference) == expected["reference_sha256"], "Registered final reference changed")
    truth, _ = driver.truth_paths(reference, info["source"])
    common.validate_truth_identity(info["child"], info["source"], reference, truth)
    annotation = annotate_test_csv(
        info["test"],
        run / info["test"].relative_to(info["child"]),
        info["source"],
        load_vocabulary(info, read_json(manifest_path)),
        truth,
    )
    require(
        annotation["inference_features_sha256"]
        == info["identity"]["test_inference_features_sha256"],
        "Evaluation features differ from training-time zero-label input",
    )
    dispatch_path = run / "control/pathgen_completed" / ("test_inputs_" + info["source"] + ".json")
    dispatch = read_json(info["child"] / dispatch_relative)
    refs = {pair: [set(path) for path in paths] for pair, paths in truth.items()}
    direct_sum = math.fsum(
        driver.path_metrics(pair, tuple(pair.split("->")), refs[pair])[0]
        for pair in dispatch["direct_pairs"]
    )
    report_path = run / "pathgen_results/PG-test-result.csv"
    with (info["child"] / "pathgen_results/PG-test-result.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    selected_rows = [row for row in rows if row.get("vantage") == info["source"]]
    require(len(selected_rows) == 1, "Need one source PG report for evaluation")
    selected_rows[0]["two_hops_similarity"] = repr(direct_sum)
    temporary_report = report_path.with_name(report_path.name + ".annotation." + str(os.getpid()))
    with temporary_report.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_report.replace(report_path)
    dispatch.update(
        sha256=annotation["annotated_csv_sha256"],
        output=str(run / info["test"].relative_to(info["child"])),
        evaluation_only_annotation=dict(
            zero_input_sha256=info["identity"]["zero_test_csv_sha256"],
            barrier_sha256=barrier_sha,
            chosen_model_sha256=selection["chosen_model_sha256"],
        ),
    )
    write_json(dispatch_path, dispatch)
    lineage_path = out / "INPUT_LINEAGE.json"
    lineage = dict(
        version=VERSION,
        source=info["source"],
        annotation_started_utc=access_started,
        annotation_start_receipt_sha256=sha256(start_path),
        input_identity_sha256=object_hash(info["identity"]),
        code_sha256=info["identity"]["code_sha256"],
        admitted_library_path_set_sha256=info["identity"]["rs_path_set_sha256"],
        zero_input_run=str(info["child"]),
        zero_input_prepared_sha256=info["identity"]["prepared_sha256"],
        chosen_model_sha256=selection["chosen_model_sha256"],
        barrier_sha256=barrier_sha,
        training_manifest_sha256=selection["chosen_training_manifest_sha256"],
        training_manifest_bytes_unchanged=True,
        reference_file=str(reference),
        reference_sha256=expected["reference_sha256"],
        evaluation_annotation=annotation,
        report_sha256=sha256(report_path),
        dispatch_sha256=sha256(dispatch_path),
        direct_jaccard_diagnostic_sum=direct_sum,
        annotation_role="evaluation only after every model choice frozen",
        candidates_history_pairs_and_row_order_unchanged=True,
        labels_used_for_model_selection=False,
    )
    freeze(lineage_path, lineage)
    prepared = copy.deepcopy(info["prepared"])
    copied_paths = [
        Path(item["destination"]) for item in read_json(run / "INPUTS_READY.json")["files"]
    ]
    for path in [*copied_paths, run / "INPUTS_READY.json", model, manifest_path]:
        if path.name != "PREPARED.json":
            prepared["prepared_files"][str(path)] = sha256(path)
    prepared["code_sha256"].update(info["identity"]["code_sha256"])
    prepared["evaluation_input_lineage"] = dict(path=str(lineage_path), sha256=sha256(lineage_path))
    write_json(run / "PREPARED.json", prepared)
    require(
        sha256(manifest_path) == selection["chosen_training_manifest_sha256"],
        "Training manifest was modified by annotation",
    )
    files = {
        str(path): sha256(path)
        for path in (
            run / "PREPARED.json",
            dispatch_path,
            report_path,
            run / info["test"].relative_to(info["child"]),
            model,
            manifest_path,
            lineage_path,
        )
    }
    files[str(start_path)] = sha256(start_path)
    ready = dict(
        status="PASS",
        source=info["source"],
        date=info["prepared"]["date"],
        evaluation_run=str(run),
        input_identity_sha256=object_hash(info["identity"]),
        barrier_sha256=barrier_sha,
        chosen_model_sha256=selection["chosen_model_sha256"],
        files_sha256=files,
        zero_test_input_preserved=True,
        inference_features_unchanged=True,
    )
    freeze(ready_path, ready)
    return info, selection, out, run, ready


def verify_evaluation_inputs(source, run_root):
    info = inputs(source, run_root)
    selection, barrier_sha = verify_barrier(info)
    out = info["root"] / "evaluations" / ("as" + info["source"])
    ready = read_json(out / "EVALUATION_INPUTS_READY.json")
    require(
        ready["status"] == "PASS"
        and ready["date"] == info["prepared"]["date"]
        and ready["input_identity_sha256"] == object_hash(info["identity"])
        and ready["barrier_sha256"] == barrier_sha
        and ready["chosen_model_sha256"] == selection["chosen_model_sha256"],
        "Evaluation input/model/cohort identity changed",
    )
    for path, digest in ready["files_sha256"].items():
        require(sha256(path) == digest, "Evaluation input annotation changed")
    return ready


def prepare_evaluation_inputs(source, run_root):
    _prepare_evaluation_inputs(source, run_root)
    return verify_evaluation_inputs(source, run_root)


def evaluate(source, run_root, device="cuda", cpu_threads=2):
    info, selection, out, run, ready = _prepare_evaluation_inputs(source, run_root)
    verify_evaluation_inputs(source, run_root)
    worker = source_worker(run, info["source"], info["config"]["seed"])
    if not (worker / "evaluation_summary.json").exists():
        require(
            not (worker / "pair_predictions.csv").exists(),
            "Partial evaluation preserved; use an explicit fresh evaluation attempt",
        )
        from pathveil.downstream import train_evaluate_gmpi as driver

        old_guard = driver.install_write_guard
        driver.install_write_guard = lambda: install_guard(driver, info["root"])
        try:
            invoke_driver(info, run, "evaluate", device, cpu_threads)
        finally:
            driver.install_write_guard = old_guard
    summary = read_json(worker / "evaluation_summary.json")
    require(
        summary["model_sha256"] == selection["chosen_model_sha256"]
        and summary["prediction_csv_sha256"] == sha256(worker / "pair_predictions.csv"),
        "Final evaluation identity differs",
    )
    for name in ("pair_predictions.csv", "evaluation_summary.json"):
        copy_verified(worker / name, out / name, sha256(worker / name))
    receipt = dict(
        status="PASS",
        source=info["source"],
        model_sha256=selection["chosen_model_sha256"],
        selection_barrier_sha256=ready["barrier_sha256"],
        input_lineage_sha256=sha256(out / "INPUT_LINEAGE.json"),
        predictions_sha256=sha256(out / "pair_predictions.csv"),
        summary_sha256=sha256(out / "evaluation_summary.json"),
        raw_labels_opened_only_after_all_selections=True,
        final_test_used_for_model_selection=False,
    )
    freeze(out / "EVALUATION_COMPLETED.json", receipt)
    return receipt


def verify_evaluation(source, run_root):
    info = inputs(source, run_root)
    selection, barrier_sha = verify_barrier(info)
    out = info["root"] / "evaluations" / ("as" + info["source"])
    receipt = read_json(out / "EVALUATION_COMPLETED.json")
    require(
        receipt["status"] == "PASS"
        and receipt["model_sha256"] == selection["chosen_model_sha256"]
        and receipt["selection_barrier_sha256"] == barrier_sha
        and receipt["input_lineage_sha256"] == sha256(out / "INPUT_LINEAGE.json")
        and receipt["predictions_sha256"] == sha256(out / "pair_predictions.csv")
        and receipt["summary_sha256"] == sha256(out / "evaluation_summary.json"),
        "Completed evaluation output/model/input identity changed",
    )
    ready = read_json(out / "EVALUATION_INPUTS_READY.json")
    for path, digest in ready["files_sha256"].items():
        require(sha256(path) == digest, "Evaluation annotation or copied checkpoint changed")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--stage", choices=("train_p0", "select", "prepare_evaluation", "evaluate"), required=True
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    function = dict(
        train_p0=train_p0,
        select=select,
        prepare_evaluation=prepare_evaluation_inputs,
        evaluate=evaluate,
    )[args.stage]
    if args.stage in ("select", "prepare_evaluation"):
        function(args.source, args.run_root)
    else:
        function(args.source, args.run_root, args.device, args.cpu_threads)


if __name__ == "__main__":
    main()
