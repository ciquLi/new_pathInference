"""Controlled, resumable confidence research over already generated candidates."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

from pathInference.experiment.contracts import forbid_formal_references, freeze, require, verify_files
from pathInference.research import candidate_scorer as previous
from pathInference.utils import project_root, read_json, sha256, utc_now, write_json

ROOT = project_root()
CONFIG = ROOT / "research/candidate_confidence_v2/CONFIG.json"
RUN = ROOT / "runs/20260201/candidate_confidence_v2"


def config():
    return read_json(CONFIG)


def source_dir(source):
    require(source in config()["sources"], "Unregistered confidence source")
    return RUN / ("as" + source)


def emit(source, stage, **values):
    record = dict(source=source, stage=stage, updated_utc=utc_now(), **values)
    write_json(source_dir(source) / "STATUS.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def grouped_softmax(values, groups, temperature=1.0):
    values = np.asarray(values, dtype=np.float64)
    sizes = np.asarray(groups, dtype=np.int64)
    require(values.ndim == 1 and sizes.ndim == 1 and len(sizes) > 0
            and np.all(sizes > 0) and sizes.sum() == len(values), "Invalid query groups")
    require(np.isfinite(values).all() and np.isfinite(temperature) and temperature > 0,
            "Invalid scores or temperature")
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    shifted = (values - np.repeat(np.maximum.reduceat(values, starts), sizes)) / temperature
    exp = np.exp(shifted)
    return exp / np.repeat(np.add.reduceat(exp, starts), sizes)


def listwise_derivatives(scores, labels, groups, temperature):
    """Exact gradient and diagonal Hessian of sum of query cross entropies."""
    probabilities = grouped_softmax(scores, groups)
    targets = grouped_softmax(labels, groups, temperature)
    return probabilities - targets, probabilities * (1.0 - probabilities)


def audited_matrix(source, name, binding):
    """Verify matrices against original candidate row order and continuous labels."""
    original = previous.RUN / ("as" + source)
    receipt = read_json(original / (name + "_features.json"))
    verify_files(receipt["input_sha256"])
    verify_files(receipt["outputs_sha256"])
    with np.load(original / (name + ".npz"), allow_pickle=False) as data:
        X, y, groups = data["X"], data["y"], data["groups"]
    require(X.shape == (len(y), 255) and np.isfinite(X).all()
            and np.isfinite(y).all() and np.all((y >= 0) & (y <= 1))
            and np.all(groups > 0) and groups.sum() == len(y), "Malformed frozen matrix")
    vocabulary = read_json(original / "VOCABULARY.json")
    pairs, offset = [], 0
    with (Path(binding["split"]) / (name + ".csv")).open(newline="") as stream:
        for i, row in enumerate(csv.reader(stream)):
            paths = previous.decode_candidates(row, vocabulary, source)
            targets = np.asarray(list(map(float, row[3].split("|"))), np.float32)
            require(i < len(groups) and len(paths) == groups[i] == len(targets), "Group alignment differs")
            require(np.array_equal(y[offset:offset + len(paths)], targets), "Weak labels differ")
            offset += len(paths)
            pairs.append(row[4])
    split = read_json(Path(binding["split"]) / "split_manifest.json")
    require(len(pairs) == len(groups) == len(set(pairs))
            and set(pairs) == set(split[name + "_pairs"]) and offset == len(y), "Pair inventory differs")
    return X, y, groups, receipt["schema"]


def snapshot():
    cfg = config()
    require(ROOT / cfg["reference_run"] == previous.RUN, "Reference binding differs")
    require(read_json(previous.RUN / "COMPLETION_AUDIT.json")["status"] == "PASS", "v1 audit failed")
    files = {str(CONFIG): sha256(CONFIG), str(Path(__file__)): sha256(__file__),
             str(ROOT / "research/candidate_confidence_v2/run.py"):
                 sha256(ROOT / "research/candidate_confidence_v2/run.py")}
    identity = read_json(previous.RUN / "IDENTITY.json")
    verify_files(identity["code_sha256"])
    require(identity["config_sha256"] == sha256(previous.CONFIG), "Reference config changed")
    files.update(identity["code_sha256"])
    for file in (previous.CONFIG, previous.RUN / "IDENTITY.json", previous.RUN / "COMPLETION_AUDIT.json"):
        files[str(file)] = sha256(file)
    for file in (previous.RUN / "PROTOCOL.json", previous.RUN / "UPSTREAM_BOUND.json",
                 previous.RUN / "data/data_manifest.json"):
        files[str(file)] = sha256(file)
    upstream = read_json(previous.RUN / "UPSTREAM_BOUND.json")
    verify_files({upstream["bundle"]: upstream["bundle_sha256"]})
    files[upstream["bundle"]] = upstream["bundle_sha256"]
    bundle = read_json(upstream["bundle"])
    runtime_expected = dict(bundle.get("inputs_sha256", {}), **bundle.get("model_files_sha256", {}))
    # Runtime verifies these explicit public/source-isolated dependencies as it loads them.
    # Do not traverse the data manifest's final-test corpus paths.
    freeze(RUN / "PROTOCOL.json", dict(configuration=cfg, code_branch="research/generator-recursion-ablation-v1",
           boundary="after candidate generation", final_labels_used_for_training=False,
           historical_test_results_already_known=True, independent_untouched_test=False,
           gmpi_current_objective="smooth_l1_loss", changed_factor=cfg["changed_factor"]))
    sources = {}
    with forbid_formal_references():
        for source in cfg["sources"]:
            emit(source, "auditing_frozen_inputs")
            original = previous.RUN / ("as" + source)
            binding = read_json(original / "REUSE.json")
            verify_files(binding["inputs_sha256"])
            files.update(binding["inputs_sha256"])
            child = previous.RUN / "T-PG-B/pg_inputs" / ("as" + source)
            marker_file = child / "control/pathgen_completed" / ("test_inputs_" + source + ".json")
            marker = read_json(marker_file)
            compiled = marker["label_blind_compilation"]
            prepared = read_json(child / "PREPARED.json")
            inventory = prepared["fixed_test_pairs_inventory"][source]
            dependencies = {marker["output"]: marker["sha256"],
                            compiled["dispatch_status_file"]: compiled["dispatch_status_sha256"],
                            inventory["path"]: inventory["sha256"]}
            verify_files(dependencies)
            files.update(dependencies)
            model = read_json(original / "MODEL_FROZEN.json")
            require(model["parameters"]["objective"] == "regression", "Unexpected reference objective")
            files[model["model"]] = model["model_sha256"]
            verify_files({model["model"]: model["model_sha256"]})
            for file in (marker_file, child / "PREPARED.json", original / "VOCABULARY.json",
                         original / "MODEL_FROZEN.json", original / "REUSE.json",
                         previous.RUN / "query_trees" / (source + ".json"),
                         previous.baseline() / "construction" / source / "plan.json",
                         previous.baseline() / "construction" / source / "root.json",
                         previous.baseline() / "construction" / source / "CASE.json"):
                files[str(file)] = sha256(file)
            split = read_json(Path(binding["split"]) / "split_manifest.json")
            require(set(split["train_pairs"]).isdisjoint(split["validation_pairs"]), "Pair overlap")
            counts = {}
            for name in ("train", "validation"):
                X, y, groups, schema = audited_matrix(source, name, binding)
                require(schema == model["feature_schema"], "Feature schema differs")
                counts[name] = dict(queries=len(groups), candidates=len(y),
                                    singleton_queries=int(np.sum(groups == 1)))
                for file in (original / (name + ".npz"), original / (name + "_features.json")):
                    files[str(file)] = sha256(file)
                del X, y, groups
            sources[source] = dict(binding=binding, reference_model=model,
                marker=str(marker_file), candidate=marker["output"], candidate_sha256=marker["sha256"],
                dispatch=compiled["dispatch_status_file"], inventory=inventory["path"], counts=counts)
            emit(source, "inputs_audited", **counts)
    freeze(RUN / "SNAPSHOT.json", dict(files_sha256=files, sources=sources,
           runtime_expected_sha256=runtime_expected,
           final_route_labels_read=False, candidates_regenerated=False,
           training_feature_matrices_reused=True, all_sources_audited=True))


def frozen_source(source):
    snap = read_json(RUN / "SNAPSHOT.json")
    # Avoid rereading all ten large candidate files on every worker stage.
    relevant = {p: h for p, h in snap["files_sha256"].items()
                if p.endswith(".py") or p == str(CONFIG) or p == str(previous.CONFIG)
                or "/as" + source + "/" in p or "/" + source + "/" in p
                or p.endswith("/" + source + ".json")
                or p in {str(previous.RUN / "PROTOCOL.json"), str(previous.RUN / "UPSTREAM_BOUND.json"),
                         str(previous.RUN / "data/data_manifest.json")}}
    verify_files(relevant)
    return snap["sources"][source]


def train(source):
    import lightgbm as lgb
    cfg = config()
    bound = frozen_source(source)
    original = bound["reference_model"]
    directory = source_dir(source)
    if (directory / "MODEL_FROZEN.json").is_file():
        record = read_json(directory / "MODEL_FROZEN.json")
        require(record["snapshot_sha256"] == sha256(RUN / "SNAPSHOT.json"), "Model snapshot differs")
        verify_files({record["model"]: record["model_sha256"]})
        return
    with forbid_formal_references():
        X, y, groups, schema = audited_matrix(source, "train", bound["binding"])
        V, vy, vg, vschema = audited_matrix(source, "validation", bound["binding"])
        require(schema == vschema, "Validation feature schema differs")
        dataset = lgb.Dataset(X, label=y, weight=np.repeat(1.0 / groups, groups), feature_name=schema)
        val = lgb.Dataset(V, label=vy, feature_name=schema, reference=dataset)
        targets = grouped_softmax(y, groups, cfg["label_temperature"])
        def objective(scores, data):
            require(len(scores) == len(y), "Training group offsets differ")
            p = grouped_softmax(scores, groups)
            return p - targets, p * (1.0 - p)
        def metric(scores, data):
            return "query_top1_jaccard", previous.query_top1(data.get_label(), scores, vg), True
        parameters = dict(original["parameters"], objective=objective)
        history = {}
        emit(source, "training", train_queries=len(groups), train_candidates=len(y))
        start, cpu = time.monotonic(), time.process_time()
        model = lgb.train(parameters, dataset, num_boost_round=cfg["max_rounds"],
            valid_sets=[val], valid_names=["validation"], feval=metric,
            callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"]),
                       lgb.log_evaluation(20), lgb.record_evaluation(history)])
        fit_wall, fit_cpu = time.monotonic() - start, time.process_time() - cpu
        path = directory / "model.txt"
        model.save_model(str(path), num_iteration=model.best_iteration)
        reference = lgb.Booster(model_file=original["model"])
        old_scores = reference.predict(V, num_threads=cfg["threads"])
        reference_reward = previous.query_top1(vy, old_scores, vg)
        require(abs(reference_reward - original["validation_query_jaccard"]) < 1e-6,
                "Reference validation result not reproduced")
        offsets = np.r_[0, np.cumsum(vg)[:-1]]
        diagnostics = dict(validation_oracle=float(np.mean(np.maximum.reduceat(vy, offsets))),
            validation_equal_label_queries=int(np.sum(np.maximum.reduceat(vy, offsets)
                                                      == np.minimum.reduceat(vy, offsets))),
            validation_queries=len(vg), reference_validation_jaccard=reference_reward)
    freeze(directory / "MODEL_FROZEN.json", dict(model=str(path), model_sha256=sha256(path),
        snapshot_sha256=sha256(RUN / "SNAPSHOT.json"),
        best_iteration=model.best_iteration, validation_jaccard=model.best_score["validation"]["query_top1_jaccard"],
        reference_model_sha256=original["model_sha256"], parameters=dict(original["parameters"],
        objective="query_soft_label_cross_entropy"), label_temperature=cfg["label_temperature"],
        feature_schema=schema, history=history, diagnostics=diagnostics,
        fit_wall_seconds=fit_wall, fit_cpu_seconds=fit_cpu, final_labels_read=False))
    emit(source, "model_frozen", best_iteration=model.best_iteration,
         validation_jaccard=model.best_score["validation"]["query_top1_jaccard"], **diagnostics)


def open_journal(path, identity):
    database = sqlite3.connect(str(path))
    database.execute("PRAGMA journal_mode=DELETE")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    database.execute("CREATE TABLE IF NOT EXISTS predictions (pair TEXT PRIMARY KEY, path TEXT NOT NULL, status TEXT NOT NULL)")
    encoded = json.dumps(identity, sort_keys=True)
    old = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if old:
        if old[0] != encoded:
            database.close()
            raise ValueError("Prediction journal identity differs")
    else:
        database.execute("INSERT INTO metadata VALUES ('identity', ?)", (encoded,))
        database.execute("INSERT INTO metadata VALUES ('offset', '0')")
        database.commit()
    return database


def commit_batch(database, rows, offset):
    """Rows and candidate byte offset commit together; a failed batch replays safely."""
    with database:
        database.executemany("INSERT INTO predictions VALUES (?, ?, ?)", rows)
        database.execute("UPDATE metadata SET value=? WHERE key='offset'", (str(offset),))


def predict(source):
    import lightgbm as lgb
    require((RUN / "ALL_MODELS_FROZEN.json").is_file(), "Freeze all models before prediction")
    cfg, bound, directory = config(), frozen_source(source), source_dir(source)
    model_record = read_json(directory / "MODEL_FROZEN.json")
    verify_files({model_record["model"]: model_record["model_sha256"],
                  bound["candidate"]: bound["candidate_sha256"]})
    if (directory / "PREDICTIONS_FROZEN.json").is_file():
        record = read_json(directory / "PREDICTIONS_FROZEN.json")
        require(record["model_sha256"] == model_record["model_sha256"]
                and record["candidate_sha256"] == bound["candidate_sha256"], "Frozen prediction identity differs")
        verify_files({record["predictions"]: record["predictions_sha256"]})
        return
    identity = dict(snapshot_sha256=sha256(RUN / "SNAPSHOT.json"),
                    model_sha256=model_record["model_sha256"], candidate_sha256=bound["candidate_sha256"])
    database = open_journal(directory / "prediction.sqlite", identity)
    try:
        offset = int(database.execute("SELECT value FROM metadata WHERE key='offset'").fetchone()[0])
        with forbid_formal_references():
            emit(source, "loading_prediction_features", committed_queries=database.execute(
                "SELECT COUNT(*) FROM predictions").fetchone()[0])
            extractor = previous.Features(source)
            require(extractor.names == model_record["feature_schema"], "Prediction feature schema differs")
            vocabulary = read_json(previous.RUN / ("as" + source) / "VOCABULARY.json")
            model = lgb.Booster(model_file=model_record["model"])
            total = database.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            def flush(batch, position):
                nonlocal total
                flat = [p for _, paths in batch for p in paths]
                scores = model.predict(extractor.matrix(flat), num_threads=cfg["threads"])
                rows, start = [], 0
                for pair, paths in batch:
                    best = int(np.argmax(scores[start:start + len(paths)]))
                    rows.append((pair, "->".join(paths[best]), "predicted"))
                    start += len(paths)
                commit_batch(database, rows, position)
                total += len(batch)
                emit(source, "predicting", committed_queries=total, candidate_byte_offset=position,
                     candidate_bytes=Path(bound["candidate"]).stat().st_size)
            # Frozen generator emits one CSV record per line, ASCII ASNs only.
            with Path(bound["candidate"]).open("rb") as stream:
                stream.seek(offset)
                batch, size = [], 0
                for raw in iter(stream.readline, b""):
                    row = next(csv.reader([raw.decode("utf-8")], strict=True))
                    require(len(row) == 5 and set(row[3].split("|")) <= {"0.0"}, "Test labels not closed")
                    paths = previous.decode_candidates(row, vocabulary, source)
                    batch.append((row[4], paths)); size += len(paths)
                    if len(batch) >= cfg["batch_queries"] or size >= cfg["max_batch_candidates"]:
                        flush(batch, stream.tell()); batch, size = [], 0
                if batch:
                    flush(batch, stream.tell())
            inventory = read_json(bound["inventory"])["pairs"]
            with Path(bound["dispatch"]).open() as stream:
                statuses = {item["pair"]: item["status"] for item in map(json.loads, stream)}
            require(set(statuses) == set(inventory) and len(inventory) == len(set(inventory)), "Query inventory differs")
            with database:
                for pair in inventory:
                    status = statuses[pair]
                    old = database.execute("SELECT path,status FROM predictions WHERE pair=?", (pair,)).fetchone()
                    if status == "model_candidates":
                        require(old is not None and old[1] == "predicted", "Missing model prediction")
                    else:
                        require(status in ("direct_graph_edge_candidate", "same_as_query", "no_usable_candidate"),
                                "Unexpected dispatch status")
                        expected = (pair if status == "direct_graph_edge_candidate" else "",
                                    "direct_graph_edge" if status == "direct_graph_edge_candidate" else status)
                        require(old is None or tuple(old) == expected, "Dispatch recovery differs")
                        database.execute("INSERT OR IGNORE INTO predictions VALUES (?,?,?)", (pair, *expected))
            require(database.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == len(inventory),
                    "Prediction denominator differs")
            temporary = directory / "ranked_predictions.tmp.csv"
            with temporary.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["pair", "predicted_path", "status"])
                for pair in inventory:
                    writer.writerow((pair, *database.execute(
                        "SELECT path,status FROM predictions WHERE pair=?", (pair,)).fetchone()))
            output = directory / "ranked_predictions.csv"
            temporary.replace(output)
        # Publishing hashes occurs after the closed-label guard has ended.
        freeze(directory / "PREDICTIONS_FROZEN.json", dict(predictions=str(output),
            predictions_sha256=sha256(output), model_sha256=model_record["model_sha256"],
            candidate_sha256=bound["candidate_sha256"], queries=len(inventory),
            feature_diagnostics=dict(extractor.counts), final_labels_read=False))
        emit(source, "predictions_frozen", committed_queries=len(inventory))
    finally:
        database.close()


def evaluate(source):
    from pathInference.downstream.evaluation import score_source
    from pathInference.research.ablation import asset
    require((RUN / "ALL_PREDICTIONS_FROZEN.json").is_file(), "Freeze all predictions before evaluation")
    directory, bound = source_dir(source), frozen_source(source)
    frozen = read_json(directory / "PREDICTIONS_FROZEN.json")
    model = read_json(directory / "MODEL_FROZEN.json")
    verify_files({frozen["predictions"]: frozen["predictions_sha256"], model["model"]: model["model_sha256"],
                  bound["candidate"]: frozen["candidate_sha256"]})
    original = read_json(asset(previous.baseline() / "RESULTS.json"))
    comparison = next(r for r in original["per_source"] if r["source_asn"] == source)
    metrics = score_source(comparison["reference_file"], frozen["predictions"], source, comparison["fixed_identity"])
    reference = read_json(previous.RUN / ("as" + source) / "RESULTS.json")
    metrics.update(source=source, rho=comparison["rho"], gmpi_jaccard=comparison["jaccard_accuracy"],
        gmpi_exact=comparison["ordered_exact_accuracy"], regression_jaccard=reference["jaccard_accuracy"],
        regression_exact=reference["ordered_exact_accuracy"], oracle_jaccard=reference["oracle_jaccard"],
        same_candidate_bytes_as_v1=True, candidate_sha256=bound["candidate_sha256"],
        objective_change_jaccard=metrics["jaccard_accuracy"]-reference["jaccard_accuracy"],
        model_sha256=model["model_sha256"], final_labels_used_for_tuning=False)
    require(metrics["query_count"] == reference["query_count"], "Evaluation query denominator differs")
    freeze(directory / "RESULTS.json", metrics)
    emit(source, "evaluated", jaccard=metrics["jaccard_accuracy"],
         delta_vs_regression=metrics["objective_change_jaccard"])
