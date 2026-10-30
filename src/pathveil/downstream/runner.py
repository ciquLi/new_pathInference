"""Prepare and execute the canonical dated unified-pool PG/GMPI workflow."""
from __future__ import annotations

from collections import Counter
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import common as c
from .evaluation import evaluate_run, reference_identity
from pathveil.utils import object_hash, project_root, read_json, sha256, utc_now, write_json


CONFIG = dict(
    seed=20261015,
    hash_seed=20261015,
    k=200,
    hidden_size=512,
    batch_size=64,
    lr=0.001,
    epochs=30,
    patience=3,
    validation_fraction=0.1,
    workers=0,
    observed_path_num=200,
)


def _copy(source, target, expected=None):
    source, target = Path(source).resolve(), Path(target).resolve()
    c.require(source.is_file(), "Required input is missing: " + str(source))
    wanted = expected or sha256(source)
    c.require(sha256(source) == wanted, "Frozen input changed: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        c.require(sha256(target) == wanted, "Existing prepared input differs: " + str(target))
    elif source != target:
        shutil.copyfile(source, target)
    return str(target), wanted


def _route(text):
    fields = tuple(text.strip().split("->"))
    c.require(
        len(fields) >= 2
        and len(fields) == len(set(fields))
        and all(n.isascii() and n.isdecimal() and 0 < int(n) < 2**32 for n in fields),
        "Invalid training-corpus route",
    )
    return fields


def audit_derived_source(source_file, source, data, heldout_sources, destination):
    """Prove a named source CSV is exactly other-origin forward subpaths.

    The canonical merged stream supplies the actual original source. Only
    training merged/expand/split inputs are read; no test path is opened.
    """
    source, source_file, destination = str(source), Path(source_file).resolve(), Path(destination)
    c.require(
        source_file.is_file() and source_file.stat().st_size,
        "A nonempty derived source CSV is required",
    )
    merged = Path(data["merged_train"]).resolve()
    expand = Path(
        data.get("expanded_train", Path(data["train_dir"]) / "expand_paths.txt")
    ).resolve()
    split_log = Path(
        data.get("split_log", Path(data["train_dir"]).parent / "split_log.txt")
    ).resolve()
    inputs = {str(p): sha256(p) for p in [source_file, merged, expand, split_log]}
    binding = object_hash(dict(inputs=inputs, source=source, heldout_sources=list(heldout_sources)))
    if destination.exists():
        old = read_json(destination)
        c.require(old.get("binding") == binding, "Derived-source audit inputs changed")
        return destination
    with source_file.open() as stream:
        rows = [line.strip() for line in stream if line.strip()]
    target = set(rows)
    c.require(len(rows) == len(target), "Derived source CSV contains duplicate paths")
    origins = Counter(_route(row)[0] for row in rows)
    c.require(origins == {source: len(rows)}, "Named source CSV contains mixed starting ASNs")
    heldout = {str(s): 0 for s in heldout_sources}
    derived, parents, parent_counts = set(), set(), Counter()
    origin_rows = containing = 0
    with merged.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            original = line.split("->", 1)[0].strip()
            if original in heldout:
                heldout[original] += 1
            if original == source:
                origin_rows += 1
                continue
            if source not in line.strip().split("->")[1:]:
                continue
            path = _route(line)
            start = path.index(source)
            if start == len(path) - 1:
                continue
            containing += 1
            parent_counts[original] += 1
            parents.add(path)
            derived.update("->".join(path[start:stop]) for stop in range(start + 2, len(path) + 1))
    expanded = set()
    expanded_rows = 0
    with expand.open() as stream:
        for line in stream:
            if line.split("->", 1)[0].strip() == source:
                route = "->".join(_route(line))
                expanded.add(route)
                expanded_rows += 1
    differences = dict(
        train_merged_source_origin_rows=origin_rows,
        source_csv_minus_train_expand_count=len(target - expanded),
        train_expand_minus_source_csv_count=len(expanded - target),
        source_csv_minus_other_source_derived_count=len(target - derived),
        other_source_derived_minus_source_csv_count=len(derived - target),
    )
    c.require(
        not any(differences.values()) and not any(heldout.values()),
        "Source-origin holdout / exact independent forward derivation is not established: "
        + str(differences),
    )
    record = dict(
        binding=binding,
        audited_at=utc_now(),
        date=data["date"],
        source_asn=source,
        decision="allow_explicit_audited_derived_source_csv",
        test_paths_read=False,
        source_csv=str(source_file),
        source_csv_sha256=inputs[str(source_file)],
        source_csv_rows=len(rows),
        source_csv_unique_paths=len(target),
        source_csv_sources=dict(origins),
        train_merged_file=str(merged),
        train_merged_sha256=inputs[str(merged)],
        train_expand_file=str(expand),
        train_expand_sha256=inputs[str(expand)],
        split_log_file=str(split_log),
        split_log_sha256=inputs[str(split_log)],
        train_expand_source_rows=expanded_rows,
        train_expand_source_unique_paths=len(expanded),
        derived_forward_contiguous_subpaths_unique=len(derived),
        heldout_source_parent_origin_counts=heldout,
        other_source_parent_rows_containing_source_as_interior=containing,
        other_source_parent_unique_paths=len(parents),
        other_source_parent_source_counts=dict(parent_counts),
        interpretation="Source CSV equals exact ordered forward subpaths of other actual original training sources, and equals its train-expand subset.",
        **differences,
    )
    write_json(destination, record)
    return destination


def prepare_downstream(run_dir, data_manifest_path, release_path):
    run = c.validate_run(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    data_manifest_path, release_path = (
        Path(data_manifest_path).resolve(),
        Path(release_path).resolve(),
    )
    data, release = read_json(data_manifest_path), read_json(release_path)
    c.require(
        release.get("status") == "PASS" and release["date"] == data["date"],
        "Upstream release is incomplete or has another date",
    )
    sources = list(map(str, release["sources"]))
    c.require(
        sources
        and len(sources) == len(set(sources))
        and set(sources) == set(map(str, data["sources"])),
        "Upstream cohort differs from the explicit dated cohort",
    )
    c.require(
        set(release["files_by_source"]) == set(sources), "Released pool omits a registered source"
    )
    c.verify_files(release.get("outputs_sha256", {}))
    prepared_path = run / "PREPARED.json"
    binding = dict(
        data_manifest_sha256=sha256(data_manifest_path), release_sha256=sha256(release_path)
    )
    if prepared_path.exists():
        previous = c.verify_runtime_authorization(run)
        c.require(previous["binding"] == binding, "Existing run was prepared with different inputs")
        return prepared_path
    files = {
        str(data_manifest_path): binding["data_manifest_sha256"],
        str(release_path): binding["release_sha256"],
    }
    admitted = {}
    for source in sources:
        item = release["files_by_source"][source]
        rows = c.csv_paths(item["path"], source)
        c.require(len(rows) == item["rows"], "Released admitted-pool row count changed")
        expected_json = {}
        for path in rows:
            expected_json.setdefault(source + "-" + path[-1], []).append(list(path))
        c.require(
            read_json(item["json_path"]) == expected_json,
            "CSV and JSON admitted route pools differ",
        )
        for src, suffix, expected in [
            (item["path"], ".csv", item["sha256"]),
            (item["json_path"], ".json", item["json_sha256"]),
        ]:
            path, digest = _copy(
                src,
                run / "path_augmentation_output/train" / (source + "_augmented_paths" + suffix),
                expected,
            )
            files[path] = digest
        admitted[source] = dict(
            paths=len(rows), csv_sha256=item["sha256"], json_sha256=item["json_sha256"]
        )
    for key, name in [
        ("selected_neighbors_csv", "reviewed_selected_neighbors.csv"),
        ("recursive_anchors_csv", "recursive_secondhop_anchors.csv"),
    ]:
        path, digest = _copy(release[key], run / "neighbor_selection" / name)
        files[path] = digest
    identity = {s: reference_identity(Path(data["test_dir"]) / (s + ".csv"), s) for s in sources}
    audits = {}
    for source in sources:
        source_file = Path(data["train_dir"]) / (source + ".csv")
        if source_file.is_file() and source_file.stat().st_size:
            path = audit_derived_source(
                source_file,
                source,
                data,
                sources,
                run / "control" / ("derived_source_" + source + "_audit.json"),
            )
            audits[source], files[str(path)] = str(path), sha256(path)
    replay = {}
    if data["date"] == "20260301" and release.get("execution_mode") == (
        "cached_date_local_features_recomputed_model_scores_and_selection"
    ):
        directory = project_root() / "data/regression/20260301/dispatch"
        for source in sources:
            original = directory / ("test_inputs_" + source + ".json")
            c.require(
                original.is_file(),
                "March equivalence run requires canonical frozen graph dispatch: " + source,
            )
            record = read_json(original)
            c.require(
                record["source"] == source and record["date"] == data["date"],
                "Frozen dispatch identity differs",
            )
            path, digest = _copy(original, run / "control/frozen_dispatch" / original.name)
            replay[source], files[path] = path, digest
    base_paths = {
        Path(data["merged_train"]),
        *(Path(p) for p in data["public_files"].values()),
        *(Path(v["reference_file"]) for v in identity.values()),
    }
    graph = Path(data["graph_dir"])
    base_paths.update(p for p in graph.iterdir() if p.is_file())
    base_paths.update(
        Path(data["train_dir"]) / (s + ".csv")
        for s in sources
        if (Path(data["train_dir"]) / (s + ".csv")).is_file()
    )
    state = {
        str(p.resolve()): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for p in base_paths
    }
    src = project_root() / "src/pathveil"
    code = sorted((src / "downstream").glob("*.py")) + sorted((src / "core").rglob("*.py"))
    prepared = dict(
        status="PREPARED",
        method="unified255",
        date=data["date"],
        sources=sources,
        common9=list(map(str, data.get("common9", sources))),
        binding=binding,
        data_manifest=str(data_manifest_path),
        configuration=dict(CONFIG, date=data["date"]),
        test_identity=identity,
        fixed_test_pairs=sum(v["pairs"] for v in identity.values()),
        prepared_files=files,
        code_sha256={str(p): sha256(p) for p in code},
        base_file_state=state,
        admitted_pool_manifest_sha256=sha256(release_path),
        admitted_pool=admitted,
        dispatch_replay=replay,
        derived_source_audits=audits,
        pg_local_knowledge_mode="augmentation_only",
        train_extra_data_mode="augmentation_only",
        no_model_training_started=True,
        endpoint_identity_registration_only=True,
    )
    write_json(prepared_path, prepared)
    c.verify_runtime_authorization(run)
    return prepared_path


def _execute(run, name, command, expected, env, prepared_sha):
    marker = run / "control/stages" / (name + ".json")
    if marker.exists():
        completed = read_json(marker)
        c.require(
            completed["prepared_sha256"] == prepared_sha and completed["command"] == command,
            "Completed stage command or input contract changed: " + name,
        )
        c.verify_files(completed["outputs_sha256"])
        return
    log = run / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        run / "progress.json",
        dict(stage=name, status="running", command=command, updated_at=utc_now()),
    )
    with log.open("ab") as stream:
        result = subprocess.run(
            command, cwd=project_root(), env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    c.require(result.returncode == 0, "Stage failed (see " + str(log) + "): " + name)
    c.require(
        all(Path(p).is_file() for p in expected),
        "Successful subprocess omitted required stage output: " + name,
    )
    write_json(
        marker,
        dict(
            status="PASS",
            command=command,
            prepared_sha256=prepared_sha,
            outputs_sha256={str(p): sha256(p) for p in expected},
            log=str(log),
            completed_at=utc_now(),
        ),
    )


def _adopt_existing(run, name, command, expected, prepared_sha):
    write_json(
        run / "control/stages" / (name + ".json"),
        dict(
            status="PASS",
            command=command,
            prepared_sha256=prepared_sha,
            outputs_sha256={str(p): sha256(p) for p in expected},
            resumed_from_verified_worker_completion=True,
            completed_at=utc_now(),
        ),
    )


def _without_stage(command):
    args = list(command)
    if "--stage" in args:
        position = args.index("--stage")
        del args[position : position + 2]
    return args


def _verify_worker_completion(worker, source, prepared, device):
    """An unrecorded finished worker must prove its configuration and inputs."""
    configuration = read_json(worker / "configuration.json")
    arguments = configuration["arguments"]
    config = prepared["configuration"]
    c.require(
        arguments["source_asn"] == source and arguments["device"] == device,
        "Unrecorded worker has another source or device",
    )
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
    ):
        c.require(arguments[key] == config[key], "Unrecorded worker configuration differs: " + key)
    run = worker.parents[3]
    files = [
        (
            run
            / "model_input_data_PG/train/meta-devoid"
            / (source + "_train_extra_data_generated_data.csv"),
            "training_csv_sha256",
        ),
        (
            run / "model_input_data_PG/test/meta-devoid" / (source + "_generated_data.csv"),
            "test_csv_sha256",
        ),
        (run / "neighbor_selection/reviewed_selected_neighbors.csv", "selected_neighbors_sha256"),
        (run / "neighbor_selection/recursive_secondhop_anchors.csv", "recursive_anchors_sha256"),
    ]
    for path, key in files:
        c.require(
            sha256(path) == configuration[key], "Unrecorded worker input changed: " + str(path)
        )


def run_downstream(run_dir, data_manifest_path, release_path, device="cuda"):
    c.require(device in {"auto", "cpu", "cuda"}, "Invalid training device")
    prepared_path = prepare_downstream(run_dir, data_manifest_path, release_path)
    run, prepared = Path(run_dir).resolve(), read_json(prepared_path)
    data = read_json(data_manifest_path)
    config, prepared_sha = prepared["configuration"], sha256(prepared_path)
    env = dict(
        os.environ,
        PATHVEIL_DATA_MANIFEST=str(Path(data_manifest_path).resolve()),
        ASINFER_FILES_ROOT=data["source_data_root"],
        PATHVEIL_GRAPH_DIR=data["graph_dir"],
        PATHVEIL_ASRANK_LINKS=data["public_files"]["asrank_links"],
        PYTHONHASHSEED=str(config["hash_seed"]),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env["PYTHONPATH"] = str(project_root() / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    for source in prepared["sources"]:
        for stage, folder, suffix in [
            ("training_inputs", "train", "_train_extra_data_generated_data.csv"),
            ("test_inputs", "test", "_generated_data.csv"),
        ]:
            command = [
                sys.executable,
                "-B",
                "-m",
                "pathveil.downstream.pathgen",
                "--stage",
                stage,
                "--experiment_dir",
                str(run),
                "--date",
                prepared["date"],
                "--target_vases",
                source,
                "--meta_test_as_num",
                str(len(prepared["sources"])),
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
                str(run / "path_augmentation_output"),
                "--graph_selected_neighbors_csv",
                str(run / "neighbor_selection/reviewed_selected_neighbors.csv"),
                "--graph_recursive_anchors_csv",
                str(run / "neighbor_selection/recursive_secondhop_anchors.csv"),
            ]
            output = run / "model_input_data_PG" / folder / "meta-devoid" / (source + suffix)
            completion = run / "control/pathgen_completed" / (stage + "_" + source + ".json")
            name = stage + "_" + source
            if completion.exists() and not (run / "control/stages" / (name + ".json")).exists():
                old = read_json(completion)
                c.require(
                    old.get("source") == source
                    and old.get("mode") == stage
                    and output.is_file()
                    and old["sha256"] == sha256(output)
                    and old["admitted_pool_manifest_sha256"]
                    == prepared["admitted_pool_manifest_sha256"],
                    "Incomplete PG stage cannot be adopted",
                )
                _adopt_existing(run, name, command, [output, completion], prepared_sha)
            _execute(run, name, command, [output, completion], env, prepared_sha)
        worker = run / "control/gmpi" / ("as" + source) / ("seed" + str(config["seed"]))
        training, evaluation, checkpoint = (
            worker / "training_manifest.json",
            worker / "evaluation_summary.json",
            worker / "model.pkl",
        )
        stage = "evaluate" if training.exists() else "all"
        command = [
            sys.executable,
            "-B",
            "-m",
            "pathveil.downstream.train_evaluate_gmpi",
            "--experiment-dir",
            str(run),
            "--source-asn",
            source,
            "--stage",
            stage,
            "--date",
            prepared["date"],
            "--device",
            device,
            "--cpu-threads",
            "4",
            "--seed",
            str(config["seed"]),
            "--batch-size",
            "64",
            "--epochs",
            "30",
            "--patience",
            "3",
            "--lr",
            "0.001",
            "--k",
            "200",
            "--hidden-size",
            "512",
            "--validation-fraction",
            "0.1",
            "--workers",
            "0",
        ]
        if source in prepared["derived_source_audits"]:
            command += ["--derived-source-audit", prepared["derived_source_audits"][source]]
        expected = [training, evaluation, checkpoint, worker / "pair_predictions.csv"]
        name = "gmpi_" + source
        marker = run / "control/stages" / (name + ".json")
        if marker.exists():
            old = read_json(marker)
            c.require(
                old["prepared_sha256"] == prepared_sha
                and _without_stage(old["command"]) == _without_stage(command),
                "Finished GMPI stage inputs, device or configuration changed",
            )
            c.verify_files(old["outputs_sha256"])
        elif evaluation.exists():
            _verify_worker_completion(worker, source, prepared, device)
            c.require(
                all(p.is_file() for p in expected), "Model evaluation completion is incomplete"
            )
            c.require(
                sha256(checkpoint) == read_json(training)["model_sha256"]
                and sha256(expected[-1]) == read_json(evaluation)["prediction_csv_sha256"],
                "Finished model artifacts changed",
            )
            _adopt_existing(run, name, command, expected, prepared_sha)
        else:
            if training.exists():
                c.require(
                    checkpoint.is_file()
                    and sha256(checkpoint) == read_json(training)["model_sha256"],
                    "Completed checkpoint changed",
                )
            _execute(run, name, command, expected, env, prepared_sha)
    independent = evaluate_run(run)
    audited = read_json(independent)
    summary = dict(
        status="PASS",
        date=prepared["date"],
        **audited["original_cohort"],
        common9=audited["common_cohort"],
        independent_evaluation=str(independent),
        independent_evaluation_sha256=sha256(independent),
        prepared_sha256=prepared_sha,
        complete_query_universe=True,
        all_sources_independently_rescored=True,
        train_extra_data_mode="augmentation_only",
        pg_local_knowledge_mode="augmentation_only",
    )
    write_json(run / "summary.json", summary)
    write_json(
        run / "COMPLETED.json",
        dict(status="PASS", summary_sha256=sha256(run / "summary.json"), completed_at=utc_now()),
    )
    write_json(
        run / "progress.json",
        dict(stage="completed", status="PASS", summary=str(run / "summary.json")),
    )
    return run / "summary.json"
