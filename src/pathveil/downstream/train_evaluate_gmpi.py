# Extracted from the frozen 2026-10-24 unified-pool run.
# Numerical training/evaluation logic preserved; date/run/core imports adapted.
#!/usr/bin/env python3
"""Isolated current-GMPI training and complete original BGP test evaluation.

The original source-AS observations stay held out. Synthetic/external training
pairs may overlap test query endpoints; their independently constructed labels
are retained. Checkpoints use only a pair-disjoint internal surrogate split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

sys.dont_write_bytecode = True
from pathveil.utils import project_root

PROJECT = project_root()
CODE = Path(__file__).resolve().parent
CONTROL = CODE
RUN = CODE
from pathveil.downstream import common
from pathveil.core import activate

CORE = activate()
# Must be set before importing torch/initializing CUDA for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def graph_edges(graph):
    return {
        tuple(sorted((str(left), str(right)))): float(data.get("weight", 1.0))
        for left, right, data in graph.edges(data=True)
    }


def graph_sha256(edges):
    triples = [[left, right, weight] for (left, right), weight in sorted(edges.items())]
    return hashlib.sha256(json.dumps(triples, separators=(",", ":")).encode()).hexdigest()


def restore_dispatch_graph(env, dispatch, source, date):
    """Restore recorded generation topology, without re-running enrichment."""
    require(
        dispatch.get("dispatch_version") == 1
        and dispatch.get("source") == source
        and dispatch.get("date") == date,
        "Wrong or unsupported pathgen graph-dispatch snapshot",
    )
    base = graph_edges(env.G)
    require(
        graph_sha256(base) == dispatch["base_graph_sha256"]
        and len(base) == dispatch["base_graph_edge_count"],
        "Base graph plus reviewed priors differs from path generation",
    )
    delta = dispatch["graph_delta_edges"]
    seen = set()
    for left, right, weight in delta:
        edge = tuple(sorted((str(left), str(right))))
        require(
            edge not in base
            and edge not in seen
            and left != right
            and math.isfinite(float(weight))
            and float(weight) > 0,
            "Invalid or duplicated recorded graph edge",
        )
        seen.add(edge)
        require(env._add_graph_edge(left, right, weight), "Recorded graph edge was not restored")
    if delta:
        env.rebuild_networkit_graph()
    restored = graph_edges(env.G)
    require(
        graph_sha256(restored) == dispatch["graph_sha256"]
        and len(restored) == dispatch["graph_edge_count"],
        "Restored source dispatch graph differs from path generation",
    )
    recorded_direct = [canonical_pair(pair, source) for pair in dispatch["direct_pairs"]]
    require(
        len(recorded_direct) == len(set(recorded_direct)) == dispatch["direct_pair_count"],
        "Recorded direct pairs are duplicated or have inconsistent counts",
    )
    return set(recorded_direct), len(delta)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(outdir, stage, **values):
    state = dict(stage=stage, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values)
    json_write(outdir / "status.json", state)
    print(json.dumps(state, sort_keys=True), flush=True)


def install_write_guard():
    """Restrict writes to this new run and close raw test labels until evaluation."""
    global ACCESS_GUARD
    ACCESS_GUARD = common.StageGuard(RUN, allow_raw_test=False)
    sys.addaudithook(ACCESS_GUARD.check)


def audit_source_training_file(path, source, audit_file=None):
    """Allow only absent/empty originals or explicitly audited external derivatives."""
    rows = []
    if path.exists():
        with path.open() as stream:
            rows = [line.strip() for line in stream if line.strip()]
    result = dict(
        source_asn=source,
        source_train_file=str(path),
        source_train_file_rows=len(rows),
        original_source_originated_rows=0,
        derived_external_rows=0,
        status="absent_or_empty_original_source_file",
    )
    if not rows:
        require(audit_file is None, "Derived-source audit supplied for an empty source file")
        return result
    require(
        audit_file is not None,
        "Nonempty source training CSV requires explicit verified --derived-source-audit: "
        + str(path),
    )
    audit_file = Path(audit_file).resolve()
    audit = json.loads(audit_file.read_text())
    require(
        audit.get("decision") == "allow_explicit_audited_derived_source_csv"
        and audit.get("source_asn") == source
        and audit.get("test_paths_read") is False,
        "Invalid derived-source audit decision/source/test isolation",
    )
    require(
        Path(audit["source_csv"]).resolve() == path.resolve()
        and digest(path) == audit["source_csv_sha256"]
        and len(rows) == audit["source_csv_rows"] == audit["source_csv_unique_paths"]
        and len(set(rows)) == len(rows),
        "Derived source CSV changed after audit",
    )
    require(
        audit["source_csv_sources"] == {source: len(rows)}, "Derived source CSV has mixed origins"
    )
    zero_fields = (
        "train_merged_source_origin_rows",
        "source_csv_minus_train_expand_count",
        "train_expand_minus_source_csv_count",
        "source_csv_minus_other_source_derived_count",
        "other_source_derived_minus_source_csv_count",
    )
    require(
        all(audit.get(field) == 0 for field in zero_fields)
        and not any(audit["heldout_source_parent_origin_counts"].values()),
        "Audit does not establish source-originated holdout / exact independent derivation",
    )
    for path_key, hash_key in (
        ("train_expand_file", "train_expand_sha256"),
        ("train_merged_file", "train_merged_sha256"),
        ("split_log_file", "split_log_sha256"),
    ):
        require(
            digest(audit[path_key]) == audit[hash_key],
            "Derived-source provenance input changed: " + path_key,
        )
    result.update(
        status="explicitly_audited_other_source_forward_subpaths",
        derived_external_rows=len(rows),
        source_train_file_sha256=digest(path),
        audit_file=str(audit_file),
        audit_sha256=digest(audit_file),
        provenance=audit,
    )
    return result


def canonical_pair(value, source):
    nodes = value.replace("->", "-").split("-")
    require(
        len(nodes) == 2
        and all(
            node.isascii() and node.isdecimal() and 0 < int(node) <= 4294967295 for node in nodes
        ),
        "Invalid query: " + value,
    )
    require(nodes[0] == str(source), "Unexpected query source: " + value)
    return "->".join(nodes)


def generated_rows(path, source):
    with Path(path).open(newline="") as stream:
        for number, row in enumerate(csv.reader(stream, strict=True), 1):
            require(len(row) == 5, "%s row %d: expected five fields" % (path, number))
            canonical_pair(row[4], source)
            yield row


def inspect_pairs(path, source):
    pairs = set()
    count = 0
    for row in generated_rows(path, source):
        pair = canonical_pair(row[4], source)
        require(pair not in pairs, "Repeated generated query in %s: %s" % (path, pair))
        pairs.add(pair)
        count += 1
    return pairs, count


def split_training(path, outdir, source, seed, fraction):
    rows_by_pair = defaultdict(list)
    for row in generated_rows(path, source):
        rows_by_pair[canonical_pair(row[4], source)].append(row)
    require(len(rows_by_pair) >= 2, "At least two synthetic training pairs are required")
    ranked = sorted(
        rows_by_pair, key=lambda pair: hashlib.sha256(("%s:%s" % (seed, pair)).encode()).digest()
    )
    validation_count = min(len(ranked) - 1, max(1, round(len(ranked) * fraction)))
    validation = set(ranked[:validation_count])
    outdir.mkdir()
    counts = Counter()
    with (outdir / "train.csv").open("x", newline="") as train, (outdir / "validation.csv").open(
        "x", newline=""
    ) as valid:
        writers = {"train": csv.writer(train), "validation": csv.writer(valid)}
        for pair in sorted(rows_by_pair):
            name = "validation" if pair in validation else "train"
            writers[name].writerows(rows_by_pair[pair])
            counts[name] += len(rows_by_pair[pair])
    manifest = dict(
        input_csv=str(path),
        input_sha256=digest(path),
        seed=seed,
        split_unit="complete source/destination AS pair",
        validation_fraction_requested=fraction,
        train_pairs=sorted(set(ranked) - validation),
        validation_pairs=sorted(validation),
        row_counts=dict(counts),
        test_labels_used_for_selection=False,
    )
    json_write(outdir / "split_manifest.json", manifest)
    return manifest


def truth_paths(path, source):
    paths = defaultdict(set)
    observations = 0
    with Path(path).open(newline="") as stream:
        for number, row in enumerate(csv.reader(stream, strict=True), 1):
            require(bool(row) and bool(row[0]), "Empty original test row %d" % number)
            nodes = tuple(row[0].strip().split("->"))
            require(
                len(nodes) >= 2
                and nodes[0] == source
                and all(
                    node.isascii() and node.isdecimal() and 0 < int(node) <= 4294967295
                    for node in nodes
                ),
                "Invalid original BGP test path row %d" % number,
            )
            paths[nodes[0] + "->" + nodes[-1]].add(nodes)
            observations += 1
    require(bool(paths), "Empty original test reference")
    return paths, observations


def path_metrics(pair, prediction, references):
    valid = bool(prediction) and prediction[0] + "->" + prediction[-1] == pair
    predicted = frozenset(prediction)
    score = (
        max(
            (len(predicted & actual) / len(predicted | actual) for actual in references),
            default=0.0,
        )
        if valid
        else 0.0
    )
    return score, round(score, 3), int(valid)


def call_model(model, fields):
    return model(
        kpathsEmbedding=fields[0],
        k_lastSecHops=fields[1],
        kcurASPair=fields[2],
        hcurASPair=fields[3],
        hispathEmbedding=fields[4],
        his_lastSecHops=fields[5],
        hisASPairsEmbedding=fields[6],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--source-asn", required=True)
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--date", default="20260301")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--selected-neighbors-csv", type=Path)
    parser.add_argument("--recursive-anchors-csv", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--derived-source-audit",
        type=Path,
        help="Explicit independent train-corpus derivation audit for a nonempty source CSV",
    )
    args = parser.parse_args()
    global RUN, CONTROL
    RUN = common.validate_run(args.experiment_dir)
    CONTROL = RUN / "control"
    common.verify_runtime_authorization(RUN)
    common.require_config(vars(args))
    require(
        (CORE / "GMPI_pathLikelihoodEst.py").is_file(), "Cannot locate preserved numerical core"
    )
    require(
        0 < args.validation_fraction < 1 and args.epochs >= 1 and args.patience >= 1,
        "Invalid training configuration",
    )
    require(args.workers == 0, "This isolated driver currently requires workers=0")
    require(args.source_asn.isdecimal() and int(args.source_asn) > 0, "Invalid source ASN")
    args.experiment_dir = args.experiment_dir.resolve()
    require(args.experiment_dir == CONTROL.parent, "Use the experiment containing this control/")
    outdir = CONTROL / "gmpi" / ("as" + args.source_asn) / ("seed" + str(args.seed))
    os.chdir(PROJECT)
    activate()
    install_write_guard()
    outdir.mkdir(parents=True, exist_ok=True)
    runtime_paths = {
        "TMPDIR": outdir / "runtime/tmp",
        "CUDA_CACHE_PATH": outdir / "runtime/cuda_cache",
        "XDG_CACHE_HOME": outdir / "runtime/xdg_cache",
        "TORCH_HOME": outdir / "runtime/torch_cache",
        "MPLCONFIGDIR": outdir / "runtime/matplotlib",
    }
    for key, directory in runtime_paths.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    require(
        os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"),
        "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8",
    )
    started = time.time()
    train_csv = (
        args.experiment_dir
        / "model_input_data_PG/train/meta-devoid"
        / (args.source_asn + "_train_extra_data_generated_data.csv")
    )
    test_csv = (
        args.experiment_dir
        / "model_input_data_PG/test/meta-devoid"
        / (args.source_asn + "_generated_data.csv")
    )
    selected = (
        args.selected_neighbors_csv
        or args.experiment_dir / "neighbor_selection/reviewed_selected_neighbors.csv"
    ).resolve()
    anchors = (
        args.recursive_anchors_csv
        or args.experiment_dir / "neighbor_selection/recursive_secondhop_anchors.csv"
    ).resolve()
    pg_result = args.experiment_dir / "pathgen_results/PG-test-result.csv"
    dispatch_file = CONTROL / "pathgen_completed" / ("test_inputs_" + args.source_asn + ".json")
    for path in (train_csv, test_csv, selected, anchors, pg_result, dispatch_file):
        require(path.is_file(), "Missing required pipeline output: " + str(path))
    dispatch = json.loads(dispatch_file.read_text())
    require(
        dispatch.get("dispatch_version") == 1 and dispatch.get("sha256") == digest(test_csv),
        "Missing pathgen dispatch or test input changed since snapshot",
    )
    config = dict(
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "stage"
        },
        training_csv_sha256=digest(train_csv),
        test_csv_sha256=digest(test_csv),
        selected_neighbors_sha256=digest(selected),
        recursive_anchors_sha256=digest(anchors),
        pathgen_test_summary_sha256=digest(pg_result),
        pathgen_dispatch_file=str(dispatch_file),
        pathgen_dispatch_sha256=digest(dispatch_file),
        training_protocol="10% surrogate validation by pair; no original test label selection",
        architecture="current GMPI v2, corrected full-history inputs",
        objective="unchanged SmoothL1 candidate Jaccard regression",
        final_metric="max AS-set Jaccard over original held-out paths; round3 per pair; fixed full query denominator; missing=0",
    )
    config_file = outdir / "configuration.json"
    if config_file.exists():
        require(
            json.loads(config_file.read_text()) == config,
            "Experiment configuration or inputs changed; use a fresh run",
        )
    else:
        json_write(config_file, config)
    source_identity = {
        str(path): digest(path)
        for path in (
            Path(__file__),
            CORE / "GMPI_pathLikelihoodEst.py",
            CORE / "myDataset.py",
            CORE / "PathInference_trainer.py",
            CORE / "environment.py",
        )
    }
    identity_path = outdir / "source_code_identity.json"
    if identity_path.exists():
        require(
            json.loads(identity_path.read_text()) == source_identity,
            "Code changed after the experiment was prepared",
        )
    else:
        json_write(identity_path, source_identity)
    emit(outdir, "loading_environment", source_asn=args.source_asn)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import environment
    import GMPI_pathLikelihoodEst as architecture
    import myDataset
    import PathInference_trainer as trainer
    from pathveil.downstream.device_runtime import configure_device

    device, device_info = configure_device(torch, args.device, args.cpu_threads)
    is_cuda = device.type == "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if is_cuda:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    json_write(
        outdir / "runtime.json",
        dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0) if is_cuda else None,
            cuda=torch.version.cuda,
            device=str(device),
            device_runtime=device_info,
            deterministic_algorithms=True,
            tf32=False,
            runtime_environment={
                key: os.environ[key] for key in ("CUBLAS_WORKSPACE_CONFIG", *runtime_paths)
            },
        ),
    )
    env = environment.Env(args.date, args.date)
    env.walksType = "_collectedPath"
    env.cur_vas = args.source_asn
    env.k = args.k
    env.epoch_test_batches = 0
    env.train_loader_params = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    env.test_loader_params = dict(
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    vocab_sha = hashlib.sha256(json.dumps(env.itos, separators=(",", ":")).encode()).hexdigest()
    data_root = Path(env.root_dir) / "data/aspath/path_by_vantage" / args.date
    original_training = data_root / "path_without_prefix/train" / (args.source_asn + ".csv")
    source_training_audit = audit_source_training_file(
        original_training, args.source_asn, args.derived_source_audit
    )
    json_write(outdir / "source_training_provenance.json", source_training_audit)
    checkpoint = outdir / "model.pkl"
    if args.stage in ("train", "all"):
        require(
            not (outdir / "training_manifest.json").exists(),
            "Training already completed; use --stage evaluate",
        )
        require(
            not checkpoint.exists(), "Incomplete prior training exists; do not silently rerun it"
        )
        training_pairs, train_rows = inspect_pairs(train_csv, args.source_asn)
        test_pairs, test_rows = inspect_pairs(test_csv, args.source_asn)
        json_write(
            outdir / "input_audit.json",
            dict(
                source_training_provenance=source_training_audit,
                synthetic_training_pairs=len(training_pairs),
                generated_test_pairs=len(test_pairs),
                training_test_endpoint_overlap=len(training_pairs & test_pairs),
                overlap_policy="Retained: synthetic/external training labels are built only from independent train evidence; shared query endpoints do not establish test-label leakage.",
                test_input_used_before_checkpoint="SHA256 and AS-pair endpoint inventory only; candidate labels are not inspected for model selection",
                source_holdout_scope="Original source-originated observations are held out. An explicitly audited source CSV may contain forward subpaths derived solely from other-source training paths. Upstream graph/embedding/augmentation provenance belongs to frozen preprocessing/pipeline.",
                original_test_path_file_read_before_checkpoint=False,
            ),
        )
        split = split_training(
            train_csv,
            outdir / "surrogate_split",
            args.source_asn,
            args.seed,
            args.validation_fraction,
        )
        del training_pairs, test_pairs
        vectors = torch.from_numpy(np.asarray(env.get_node_vectors(), dtype=np.float32))
        embedding = env.node2vecOutputDim + env.asrankFeatureDim
        model = architecture.attentionBasedScoreNetwork(
            vectors, embedding * 3, embedding, embedding * 4, args.hidden_size, 20
        ).to(device)
        model.asEmbedding.weight.requires_grad = True
        model._pathveil_metadata = dict(
            date=args.date,
            source_asn=args.source_asn,
            vocab_sha256=vocab_sha,
            k=args.k,
            max_length=20,
            seed=args.seed,
            input_sha256=digest(train_csv),
            history_argument_order="corrected",
            input_semantics_version=model.input_semantics_version,
            validation="pair-disjoint internal synthetic/external BGP-derived validation; original test unused",
        )
        require(model.input_semantics_version == 2, "Current GMPI v2 is required")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        datasets = {
            name: myDataset.IterableDatasetShuffle(
                str(outdir / "surrogate_split" / (name + ".csv")),
                env,
                20,
                args.k,
                60000,
                args.seed,
                False,
            )
            for name in ("train", "validation")
        }
        best, stale = float("-inf"), 0
        history = []
        for epoch in range(args.epochs):
            tick = time.time()
            train_loss, train_reward, train_count, _ = trainer.train(
                model, datasets["train"], optimizer, env, epoch, "train"
            )
            val_loss, val_reward, val_count, _, _, _ = trainer.test(
                model, datasets["validation"], optimizer, env, epoch, "surrogate_validation"
            )
            require(
                train_count == split["row_counts"]["train"]
                and val_count == split["row_counts"]["validation"],
                "Dataset counts changed",
            )
            score = val_reward / val_count
            require(math.isfinite(score) and math.isfinite(train_loss), "Nonfinite training state")
            improved = score > best
            if improved:
                best, stale = score, 0
                model._pathveil_metadata["selected_epoch"] = epoch
                torch.save(model, checkpoint)
            else:
                stale += 1
            record = dict(
                epoch=epoch,
                training_rows=train_count,
                validation_rows=val_count,
                training_jaccard_reward=train_reward / train_count,
                validation_jaccard_reward=score,
                training_loss_sum=train_loss,
                validation_loss_sum=val_loss,
                checkpoint_saved=improved,
                seconds=time.time() - tick,
            )
            history.append(record)
            json_write(outdir / "training_history.json", history)
            emit(outdir, "training", source_asn=args.source_asn, **record)
            if stale >= args.patience:
                break
        manifest = dict(
            model_sha256=digest(checkpoint),
            vocab_sha256=vocab_sha,
            selected_epoch=model._pathveil_metadata["selected_epoch"],
            epochs=len(history),
            best_surrogate_validation_jaccard=best,
            original_test_used_for_selection=False,
            input_semantics_version=2,
            history_argument_order="corrected",
            source_training_provenance=source_training_audit,
            runtime_environment={
                key: os.environ[key] for key in ("CUBLAS_WORKSPACE_CONFIG", *runtime_paths)
            },
            training_input_sha256=digest(train_csv),
            test_input_sha256=digest(test_csv),
            elapsed_seconds=time.time() - started,
        )
        json_write(outdir / "training_manifest.json", manifest)
        del optimizer, datasets, model, vectors
        if is_cuda:
            torch.cuda.empty_cache()
        emit(outdir, "training_complete", **manifest)
    if args.stage == "train":
        return
    require(
        (outdir / "training_manifest.json").is_file(), "Complete training before final evaluation"
    )
    require(not (outdir / "evaluation_summary.json").exists(), "Evaluation already completed")
    manifest = json.loads((outdir / "training_manifest.json").read_text())
    require(digest(checkpoint) == manifest["model_sha256"], "Frozen checkpoint changed")
    emit(outdir, "evaluating", source_asn=args.source_asn, model_sha256=manifest["model_sha256"])
    # Original held-out route sequences are first opened after freezing the checkpoint.
    reference_file = data_root / "path_without_prefix/test" / (args.source_asn + ".csv")
    ACCESS_GUARD.allow_evaluation(checkpoint, manifest["model_sha256"])
    common.json_write(
        outdir / "ORIGINAL_TEST_ACCESS_STARTED.json",
        dict(
            source_asn=args.source_asn,
            checkpoint_sha256=manifest["model_sha256"],
            original_test_used_for_selection=False,
            training_manifest_sha256=digest(outdir / "training_manifest.json"),
        ),
    )
    truth, original_observations = truth_paths(reference_file, args.source_asn)
    common.validate_truth_identity(RUN, args.source_asn, reference_file, truth)
    truth_sets = {pair: [frozenset(nodes) for nodes in paths] for pair, paths in truth.items()}
    generated_pairs, test_rows = inspect_pairs(test_csv, args.source_asn)
    require(
        generated_pairs <= set(truth),
        "Generated test query absent from original held-out reference",
    )
    env.graph_selected_neighbors_csv = str(selected)
    env.graph_recursive_anchors_csv = str(anchors)
    added_priors = env.augment_graph_from_pipeline_priors()
    recorded_direct, added_external = restore_dispatch_graph(
        env, dispatch, args.source_asn, args.date
    )
    direct = {pair for pair in truth if env.G.has_edge(*pair.split("->"))}
    require(
        direct == recorded_direct,
        "Recorded actual direct dispatch differs from original test query universe",
    )
    require(
        not generated_pairs & direct,
        "Graph reconstruction disagrees with pathgen direct/model dispatch",
    )
    with pg_result.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("vantage") == args.source_asn]
    require(len(rows) == 1, "Expected exactly one fresh PG test summary for this source")
    pg = rows[0]
    require(
        int(pg["num_dests"]) == len(truth),
        "PG query denominator differs from original held-out reference",
    )
    require(
        int(pg["two_hops"]) == len(direct), "Reconstructed direct paths differ from PG dispatch"
    )
    require(
        int(pg["total_available_APs"]) == len(generated_pairs) + len(direct),
        "PG available query count differs from generated model rows plus actual direct paths",
    )
    model = torch.load(str(checkpoint), map_location=device).eval()
    meta = model._pathveil_metadata
    require(
        meta["vocab_sha256"] == vocab_sha
        and meta["source_asn"] == args.source_asn
        and meta["history_argument_order"] == "corrected"
        and model.input_semantics_version == 2,
        "Model provenance does not match this source/environment",
    )
    dataset = myDataset.IterableDatasetShuffle(
        str(test_csv), env, 20, args.k, 60000, args.seed, False
    )
    loader = myDataset.DataLoaderX(dataset, **env.test_loader_params)
    prediction_fields = [
        "pair",
        "predicted_path",
        "status",
        "candidate_count",
        "history_count",
        "selected_candidate_index",
        "model_score",
        "cached_label",
        "jaccard",
        "jaccard_rounded3",
        "endpoint_valid",
    ]
    statuses, seen = Counter(), set()
    scores, raw_scores = [], []
    label_mismatches, direct_scores = [], []
    oracle_sum = 0.0
    last_progress = time.time()
    with (outdir / "pair_predictions.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields)
        writer.writeheader()
        with torch.no_grad():
            for item in loader:
                fields = [
                    torch.as_tensor(value).to(device, non_blocking=is_cuda) for value in item[:7]
                ]
                output = call_model(model, fields).squeeze(-1)
                require(bool(torch.isfinite(output).all()), "Nonfinite final model outputs")
                valid_counts = item[9].tolist()
                positions = torch.arange(args.k, device=device).unsqueeze(0)
                limits = torch.as_tensor(valid_counts, device=device).unsqueeze(1).clamp(max=args.k)
                masked = output.masked_fill(positions >= limits, torch.finfo(output.dtype).min)
                chosen = masked.argmax(dim=1).cpu().tolist()
                model_scores = masked.max(dim=1)[0].cpu().tolist()
                candidate_indices = item[0]
                labels = item[7]
                oracle_sum += float(labels.max(dim=1)[0].sum().item())
                for index, (slot, model_score) in enumerate(zip(chosen, model_scores)):
                    pair = canonical_pair(item[8][index], args.source_asn)
                    require(pair not in seen, "Repeated evaluation query: " + pair)
                    seen.add(pair)
                    prediction = tuple(
                        str(env.itos[token])
                        for token in candidate_indices[index, slot].tolist()
                        if token != env.stoi["<pad>"]
                    )
                    score, rounded, valid = path_metrics(pair, prediction, truth_sets[pair])
                    require(
                        valid and len(prediction) == len(set(prediction)), "Invalid selected route"
                    )
                    cached = float(labels[index, slot, 0].item())
                    if abs(cached - rounded) > 5e-7:
                        label_mismatches.append(dict(pair=pair, cached=cached, recomputed=rounded))
                    writer.writerow(
                        dict(
                            pair=pair,
                            predicted_path="->".join(prediction),
                            status="predicted",
                            candidate_count=int(item[9][index]),
                            history_count=int(item[10][index]),
                            selected_candidate_index=slot,
                            model_score=model_score,
                            cached_label=cached,
                            jaccard=score,
                            jaccard_rounded3=rounded,
                            endpoint_valid=valid,
                        )
                    )
                    scores.append(rounded)
                    raw_scores.append(score)
                    statuses["predicted"] += 1
                if time.time() - last_progress >= 20:
                    stream.flush()
                    emit(
                        outdir,
                        "evaluating",
                        processed=len(seen),
                        generated_queries=test_rows,
                        complete_test_queries=len(truth),
                        elapsed_seconds=time.time() - started,
                    )
                    last_progress = time.time()
        require(
            seen == generated_pairs, "Uncapped evaluation failed to cover every generated test row"
        )
        for pair in sorted(set(truth) - seen):
            src, dst = pair.split("->")
            if pair in direct:
                prediction, status = (src, dst), "direct_graph_edge"
            elif src == dst:
                prediction, status = (src,), "same_as"
            else:
                prediction, status = (), "no_usable_candidate"
            score, rounded, valid = path_metrics(pair, prediction, truth_sets[pair])
            writer.writerow(
                dict(
                    pair=pair,
                    predicted_path="->".join(prediction),
                    status=status,
                    candidate_count=int(bool(prediction)),
                    history_count=0,
                    jaccard=score,
                    jaccard_rounded3=rounded,
                    endpoint_valid=valid,
                )
            )
            scores.append(rounded)
            raw_scores.append(score)
            statuses[status] += 1
            if status == "direct_graph_edge":
                direct_scores.append(score)
        stream.flush()
    require(len(scores) == len(truth), "Fixed complete query count changed")
    require(
        not label_mismatches,
        "Cached labels disagree with independent original test Jaccard: "
        + json.dumps(label_mismatches[:5]),
    )
    require(
        math.isclose(
            math.fsum(direct_scores), float(pg["two_hops_similarity"]), rel_tol=0, abs_tol=1e-6
        ),
        "Actual direct-path rewards differ from PG summary",
    )
    summary = dict(
        source_asn=args.source_asn,
        date=args.date,
        seed=args.seed,
        model_sha256=manifest["model_sha256"],
        selected_epoch=manifest["selected_epoch"],
        original_test_file=str(reference_file),
        original_test_sha256=digest(reference_file),
        original_observations=original_observations,
        original_unique_paths=sum(map(len, truth.values())),
        query_count=len(truth),
        generated_model_queries=test_rows,
        statuses=dict(statuses),
        valid_predictions=statuses["predicted"]
        + statuses["direct_graph_edge"]
        + statuses["same_as"],
        jaccard_accuracy_rounded3=math.fsum(scores) / len(scores),
        jaccard_accuracy=math.fsum(raw_scores) / len(raw_scores),
        total_jaccard_reward_rounded3=math.fsum(scores),
        total_jaccard_reward=math.fsum(raw_scores),
        actual_direct_reward=math.fsum(direct_scores),
        cached_label_mismatch_count=0,
        candidate_oracle_jaccard_rounded3_on_model_queries=oracle_sum / test_rows
        if test_rows
        else None,
        graph_prior_edges_added=added_priors,
        external_node_edges_added=added_external,
        pathgen_dispatch_sha256=digest(dispatch_file),
        pathgen_dispatch_graph_sha256=dispatch["graph_sha256"],
        selected_neighbors_sha256=digest(selected),
        recursive_anchors_sha256=digest(anchors),
        original_test_used_for_selection=False,
        uncapped_complete_original_test=True,
        prediction_csv_sha256=digest(outdir / "pair_predictions.csv"),
        device=str(device),
        cpu_threads=args.cpu_threads,
        elapsed_seconds=time.time() - started,
    )
    json_write(outdir / "evaluation_summary.json", summary)
    emit(outdir, "complete", **summary)


if __name__ == "__main__":
    main()
