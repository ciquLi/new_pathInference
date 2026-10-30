"""Freeze each source checkpoint before opening its final reference paths."""
from contextlib import contextmanager
from pathlib import Path
from . import train
from pathveil.utils import read_json, write_json, sha256, object_hash


def require(value, message):
    if not value:
        raise ValueError(message)


def amendment(group):
    file = Path(group).resolve().parent / "EVALUATION_POLICY_AMENDMENT.json"
    policy = read_json(file)
    require(
        policy["selection_scope"] == "single_source_generator"
        and policy["training_and_selection_unchanged"] is True
        and policy["final_scores_used_for_further_training"] is False,
        "Missing authorized incremental evaluation policy",
    )
    return file


def gate_path(group, source):
    return Path(group) / "control/source_selection_barriers" / ("as" + str(source) + ".json")


def freeze_source(group, source):
    group = Path(group).resolve()
    policy = amendment(group)
    selection = train.verify_selection(source, group)
    record = dict(
        status="PASS",
        selection_scope="single_source_generator",
        source=str(source),
        date=read_json(group / "METHOD_FROZEN.json")["date"],
        group=str(group),
        method_sha256=sha256(group / "METHOD_FROZEN.json"),
        policy_amendment_sha256=sha256(policy),
        evaluation_adapter_sha256=sha256(__file__),
        selection_file=str(
            group / "gmpi/selections" / ("as" + str(source)) / "SELECTION_FROZEN.json"
        ),
        selection_sha256=sha256(
            group / "gmpi/selections" / ("as" + str(source)) / "SELECTION_FROZEN.json"
        ),
        model_sha256=selection["chosen_model_sha256"],
        training_manifest_sha256=selection["chosen_training_manifest_sha256"],
        input_identity_sha256=object_hash(selection["input_identity"]),
        final_labels_opened=False,
    )
    path = gate_path(group, source)
    if path.exists():
        require(read_json(path) == record, "Frozen source/generator choice changed")
    else:
        write_json(path, record)
    return record


def verify_source_barrier(info):
    group, source = info["root"], info["source"]
    policy = amendment(group)
    path = gate_path(group, source)
    require(path.is_file(), "Freeze this source model before evaluation")
    gate = read_json(path)
    selected = train.verify_selection(source, group)
    require(
        gate["status"] == "PASS"
        and gate["selection_scope"] == "single_source_generator"
        and gate["source"] == source
        and gate["date"] == read_json(group / "METHOD_FROZEN.json")["date"]
        and gate["method_sha256"] == sha256(group / "METHOD_FROZEN.json")
        and gate["policy_amendment_sha256"] == sha256(policy)
        and gate["evaluation_adapter_sha256"] == sha256(__file__)
        and gate["selection_file"]
        == str(group / "gmpi/selections" / ("as" + source) / "SELECTION_FROZEN.json")
        and gate["selection_sha256"] == sha256(gate["selection_file"])
        and gate["input_identity_sha256"] == object_hash(info["identity"])
        and gate["model_sha256"] == selected["chosen_model_sha256"]
        and gate["training_manifest_sha256"] == selected["chosen_training_manifest_sha256"],
        "Source evaluation gate/model/input/code identity changed",
    )
    require(
        sha256(selected["chosen_model"]) == gate["model_sha256"]
        and sha256(selected["chosen_training_manifest"]) == gate["training_manifest_sha256"],
        "Selected checkpoint or training manifest changed",
    )
    require(
        selected["input_identity"] == info["identity"], "Checkpoint belongs to different inputs"
    )
    return selected, sha256(path)


@contextmanager
def source_scope():
    old_barrier, old_freeze = train.verify_barrier, train.freeze

    def scoped_metadata(path, value):
        value = dict(value)
        if Path(path).name in {
            "INPUT_LINEAGE.json",
            "EVALUATION_INPUTS_READY.json",
            "EVALUATION_COMPLETED.json",
        }:
            value["selection_scope"] = "single_source_generator"
            value["incremental_evaluation_adapter_sha256"] = sha256(__file__)
        if Path(path).name == "INPUT_LINEAGE.json":
            value["annotation_role"] = "evaluation only after this source/generator model frozen"
        if Path(path).name == "EVALUATION_COMPLETED.json":
            value.pop("raw_labels_opened_only_after_all_selections", None)
            value["raw_labels_opened_only_after_source_selection"] = True
        return old_freeze(path, value)

    train.verify_barrier, train.freeze = verify_source_barrier, scoped_metadata
    try:
        yield
    finally:
        train.verify_barrier, train.freeze = old_barrier, old_freeze


def verify(group, source, stage):
    with source_scope():
        return (
            train.verify_evaluation_inputs(source, group)
            if stage == "prepare_evaluation"
            else train.verify_evaluation(source, group)
        )
