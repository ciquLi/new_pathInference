"""Isolated numerical stages used by the one-date resource dispatcher."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from pathveil.downstream import common
from pathveil.downstream.evaluation import score_source
from pathveil.utils import read_json, sha256
from .contracts import freeze, require, verify_files


STAGES = ("score", "pg_train", "pg_test", "train", "annotate", "evaluate", "rescore")


def receipt_path(run, stage, source):
    if stage in ("pg_train", "pg_test"):
        return (
            Path(run)
            / "T-PG-B/pg_inputs"
            / ("as" + str(source))
            / "control/workflow_tasks"
            / (stage + ".json")
        )
    return Path(run) / "T-PG-B/control/tasks" / (stage + "_" + str(source) + ".json")


def verify(run, stage, source):
    path = receipt_path(run, stage, source)
    value = read_json(path)
    require(
        value["stage"] == stage
        and value["source"] == str(source)
        and value["protocol_sha256"] == sha256(Path(run) / "PROTOCOL.json"),
        "Task cache identity differs",
    )
    verify_files(value["outputs_sha256"])
    if value.get("prepared_sha256"):
        child = Path(run) / "T-PG-B/pg_inputs" / ("as" + str(source))
        require(
            sha256(child / "PREPARED.json") == value["prepared_sha256"],
            "Prepared task inputs changed",
        )
    if stage == "score":
        scored = read_json(Path(run) / "score_pools" / ("as" + str(source)) / "SCORED.json")
        verify_files(scored["inputs_sha256"])
    return value


def execute(run, stage, source, device="cuda", cpu_threads=4):
    run, source = common.validate_run(run), str(source)
    protocol = read_json(run / "PROTOCOL.json")
    resume_root = os.environ.get("PATHVEIL_RESUME_RUN")
    resume_allowed = False
    if resume_root:
        root = Path(resume_root).resolve()
        resume_allowed = common.inside(run, root) and (root / "RESUME_PATCH.json").is_file()
    if not resume_allowed:
        verify_files(protocol["code_sha256"])
    require(stage in STAGES, "Unknown numerical stage")
    require(
        source in (protocol["score_origins"] if stage == "score" else protocol["sources"]),
        "Unregistered task origin",
    )
    if receipt_path(run, stage, source).exists():
        return verify(run, stage, source)
    group = run / "T-PG-B"
    child = group / "pg_inputs" / ("as" + source)
    outputs = []
    if stage == "score":
        from .augmentation import ScoreRuntime

        ScoreRuntime(run).score(source)
        directory = run / "score_pools" / ("as" + source)
        outputs = [directory / name for name in ("SCORED.json", "SCORED_PATHS.json")]
    elif stage in ("pg_train", "pg_test"):
        from .compile_queries import main

        mode = "training_inputs" if stage == "pg_train" else "test_inputs"
        arguments = [
            "--stage",
            mode,
            "--experiment_dir",
            str(child),
            "--date",
            protocol["date"],
            "--target_vases",
            source,
            "--meta_test_as_num",
            "1",
            "--as_type",
            "devoid",
            "--num_thread",
            "1",
            "--k",
            "200",
            "--oppNum",
            "200",
            "--pg_local_knowledge_mode",
            "augmentation_only",
            "--train-extra-data-mode",
            "augmentation_only",
            "--support_data_dir",
            str(child / "path_augmentation_output"),
            "--graph_selected_neighbors_csv",
            str(child / "neighbor_selection/reviewed_selected_neighbors.csv"),
            "--graph_recursive_anchors_csv",
            str(child / "neighbor_selection/recursive_secondhop_anchors.csv"),
            "--prefix-tree-dir",
            str(run / "query_trees"),
        ]
        main(arguments)
        marker = child / "control/pathgen_completed" / (mode + "_" + source + ".json")
        record = read_json(marker)
        require(record["sha256"] == sha256(record["output"]), "PG output changed")
        outputs = [marker, Path(record["output"])]
        for consumer in ("truth", "local") if stage == "pg_train" else ("local",):
            file = (
                child / "control/consumer_audit" / (mode + "_" + source + "_" + consumer + ".json")
            )
            audit = read_json(file)
            prepared = read_json(child / "PREPARED.json")
            require(
                audit["same_admitted_pool_verified"]
                and audit["path_set_sha256"]
                == prepared["admitted_library_path_set_sha256"][source],
                "Training/PG consumer libraries differ",
            )
            outputs.append(file)
        if stage == "pg_test":
            from .compile_queries import verify_compilation_receipt

            verify_compilation_receipt(child, source)
    elif stage == "train":
        from . import train

        info = train.inputs(source, group)
        for variant in train.configured_variants(info):
            train.train_variant(source, group, variant, device, cpu_threads)
        selected = train.select(source, group)
        outputs = [
            group / "gmpi/selections" / ("as" + source) / "SELECTION_FROZEN.json",
            Path(selected["chosen_model"]),
            Path(selected["chosen_training_manifest"]),
        ]
    elif stage in ("annotate", "evaluate"):
        from . import evaluation_gate, train

        evaluation_gate.freeze_source(group, source)
        with evaluation_gate.source_scope():
            if stage == "annotate":
                record = train.prepare_evaluation_inputs(source, group)
                outputs = [group / "evaluations" / ("as" + source) / "EVALUATION_INPUTS_READY.json"]
                outputs += [Path(p) for p in record["files_sha256"]]
            else:
                train.evaluate(source, group, device, cpu_threads)
                directory = group / "evaluations" / ("as" + source)
                outputs = [
                    directory / n
                    for n in (
                        "EVALUATION_COMPLETED.json",
                        "pair_predictions.csv",
                        "evaluation_summary.json",
                    )
                ]
    else:
        from . import evaluation_gate, train

        evaluation_gate.verify(group, source, "evaluate")
        child_prepared = read_json(child / "PREPARED.json")
        directory = group / "evaluations" / ("as" + source)
        identity = child_prepared["test_identity"][source]
        value = score_source(
            identity["reference_file"],
            directory / "pair_predictions.csv",
            source,
            identity,
            directory / "evaluation_summary.json",
        )
        admitted = read_json(run / "augmentation" / ("as" + source) / "ADMISSION.json")
        selected = train.verify_selection(source, group)
        value.update(
            rho=admitted["rho"],
            admitted_paths=admitted["rows"],
            admitted_destinations=admitted["destinations"],
            admission_strategy=admitted["strategy"],
            selected_objective=selected["chosen_variant"],
            checkpoint_sha256=selected["chosen_model_sha256"],
            date=protocol["date"],
        )
        file = group / "independent_evaluation" / ("source_" + source + ".json")
        freeze(file, value)
        outputs = [file]
    record = dict(
        status="PASS",
        stage=stage,
        source=source,
        date=protocol["date"],
        protocol_sha256=sha256(run / "PROTOCOL.json"),
        prepared_sha256=sha256(child / "PREPARED.json") if stage != "score" else None,
        outputs_sha256={str(p): sha256(p) for p in outputs},
    )
    freeze(receipt_path(run, stage, source), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    execute(args.run, args.stage, args.source, args.device, args.cpu_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
