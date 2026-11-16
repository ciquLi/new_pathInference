"""Separate weakly supervised final scorer over frozen PATS candidates."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from pathInference.experiment.contracts import (
    forbid_formal_references, freeze, freeze_bytes, require, signature, verify_files,
)
from pathInference.utils import project_root, read_json, sha256, write_json, utc_now

ROOT = project_root()
CONFIG = ROOT / "research/candidate_scorer_v1/CONFIG.json"
RUN = ROOT / "runs/20260201/candidate_scorer_v1"


def config():
    return read_json(CONFIG)


def baseline():
    return ROOT / config()["baseline"]


def emit(source, stage, **values):
    value = dict(source=source, stage=stage, updated_utc=utc_now(), **values)
    write_json(RUN / ("as" + source) / "STATUS.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def code_identity():
    paths = list((ROOT / "src/pathInference").rglob("*.py"))
    paths += list((ROOT / "research/candidate_scorer_v1").glob("*.py"))
    return {str(p): sha256(p) for p in sorted(paths)}


def prepare(source):
    """Bind frozen assets with the existing, label-blind preparation adapter."""
    from pathInference.research import ablation
    from pathInference.experiment import preparation

    old = baseline()
    cfg = config()
    require(source in cfg["sources"], "Unregistered pilot source")
    RUN.mkdir(parents=True, exist_ok=True)
    identity = dict(config_sha256=sha256(CONFIG), code_sha256=code_identity(),
                    baseline_protocol_sha256=sha256(ablation.asset(old / "PROTOCOL.json")))
    freeze(RUN / "IDENTITY.json", identity)
    (RUN / ("as" + source)).mkdir(exist_ok=True)
    with forbid_formal_references():
        proto = copy.deepcopy(read_json(old / "PROTOCOL.json"))
        proto.update(version=cfg["name"], code_sha256=identity["code_sha256"],
                     ablation_baseline=str(old), candidate_scorer=cfg)
        freeze(RUN / "PROTOCOL.json", proto)
        for relative in ("UPSTREAM_BOUND.json", "data/data_manifest.json"):
            freeze(RUN / relative, read_json(ablation.asset(old / relative)))
        for relative in ("T-PG-B/METHOD_FROZEN.json", "query_trees/METHOD_FROZEN.json"):
            method = copy.deepcopy(read_json(ablation.asset(old / relative)))
            method.update(version=cfg["name"], protocol_sha256=sha256(RUN / "PROTOCOL.json"))
            freeze(RUN / relative, method)
        # This only rebinds metadata and copies small admitted libraries.
        # The temporary reference-registration callback imports no route labels.
        with ablation.frozen_reference_registration(cfg["date"], source):
            ablation.clone_library(RUN, source)
        child = RUN / "T-PG-B/pg_inputs" / ("as" + source)
        prepared = read_json(child / "PREPARED.json")
        vocabulary = ablation.vocabulary(prepared)
        freeze(RUN / ("as" + source) / "VOCABULARY.json", vocabulary)
        frozen_split = old / "T-PG-B/gmpi/shared_splits" / ("as" + source) / "surrogate_split"
        split = read_json(ablation.asset(frozen_split / "split_manifest.json"))
        require(set(split["train_pairs"]).isdisjoint(split["validation_pairs"]), "Pair overlap")
        inputs = {}
        for name in ("train.csv", "validation.csv", "split_manifest.json"):
            file = ablation.asset(frozen_split / name)
            inputs[str(file)] = sha256(file)
        admission = read_json(RUN / "augmentation" / ("as" + source) / "ADMISSION.json")
        library = preparation.common.csv_paths(admission["path"], source)
        require(signature(library) == admission["path_set_sha256"]
                == prepared["admitted_library_path_set_sha256"][source], "Library binding differs")
        manifest_path = old / "T-PG-B/gmpi/p0" / ("as" + source) / "control/gmpi" / ("as" + source) / "seed20261015/training_manifest.json"
        trained = read_json(manifest_path)
        require(trained["admitted_library_path_set_sha256"] == admission["path_set_sha256"],
                "Historical training consumed another library")
        inputs[str(manifest_path)] = sha256(manifest_path)
        inputs[str(child / "PREPARED.json")] = sha256(child / "PREPARED.json")
        inputs[admission["path"]] = admission["sha256"]
        freeze(RUN / ("as" + source) / "REUSE.json", dict(
            inputs_sha256=inputs, split=str(frozen_split), prepared=str(child / "PREPARED.json"),
            admitted_library_path_set_sha256=admission["path_set_sha256"],
            test_routes_read=False, augmentation_rebuilt=False))
    emit(source, "prepared", reused_training_rows=split["row_counts"])


def decode_candidates(row, vocabulary, source):
    require(len(row) == 5 and row[4].split("->")[0] == source, "Malformed source row")
    paths = [tuple(vocabulary[int(v)] for v in p.split("->")) for p in row[0].split("|")]
    require(paths and len(paths) == len(set(paths)), "Empty or duplicated candidate set")
    for path in paths:
        require(len(path) >= 2 and path[0] + "->" + path[-1] == row[4]
                and len(set(path)) == len(path) and all(v.isdecimal() for v in path), "Bad decoded candidate")
    return paths


def query_top1(labels, scores, groups):
    require(sum(groups) == len(labels) == len(scores) and all(n > 0 for n in groups), "Invalid query offsets")
    start = 0
    rewards = []
    for size in groups:
        rewards.append(float(labels[start + int(np.argmax(scores[start:start + size]))]))
        start += size
    return float(np.mean(rewards))


class Features:
    """Unchanged frozen 222+33 computations, new candidate task/model."""
    def __init__(self, source):
        from pathInference.upstream.neighbors import NeighborRuntime
        from pathInference.upstream.destination import DestinationFeatureStore
        from pathInference.upstream.path_pipeline import CachedPathFeatures
        from pathInference.upstream.implementation.final_features import FinalPathFeatures, ActualQueryStore
        from pathInference.upstream.implementation.prior_spec import prediction_spec
        from pathInference.upstream.implementation.destination_features import APPENDED_NAMES
        self.source = source
        old = baseline()
        proto = read_json(RUN / "PROTOCOL.json")
        upstream = read_json(RUN / "UPSTREAM_BOUND.json")
        verify_files({upstream["bundle"]: upstream["bundle_sha256"]})
        bundle = read_json(upstream["bundle"])
        self.runtime = NeighborRuntime(bundle, RUN / "data/data_manifest.json", proto["upstream_config"], threads=2)
        self.store = DestinationFeatureStore(bundle, self.runtime.public)
        self.plan = read_json(old / "construction" / source / "plan.json")
        root = read_json(old / "construction" / source / "root.json")
        _, _, priors, _ = prediction_spec(int(source), root, self.runtime.public)
        queries = ActualQueryStore(self.runtime)
        for text, record in self.plan["frontier_prediction_queries"].items():
            prefix = tuple(map(int, text.split("->")))
            queries.queries[prefix] = record
            if len(prefix) == 2:
                queries.lookup[prefix] = {str(c["asn"]): dict(
                    missing=False, probability=c["local_observed_support_probability"], rank=rank,
                    probability_calibrated=False, phase_status=c["phase_status"])
                    for rank, c in enumerate(record["candidates"], 1)}
                for rank, c in enumerate(record["candidates"], 1):
                    priors["recursive"].setdefault(str(prefix[1]), {})[str(c["asn"])] = dict(
                        center_evidence={source: dict(pred_rank=rank)})
        base = CachedPathFeatures(self.runtime.safe["index"], self.runtime.safe["origin_support"],
                                  self.runtime.public.relations, self.runtime.public, priors)
        self.builder = FinalPathFeatures(base, self.plan, self.runtime, queries)
        self.names = list(self.builder.feature_names) + list(APPENDED_NAMES)
        expected = read_json(old / "construction" / source / "CASE.json")["feature_names"]
        require(self.names == expected and len(self.names) == 255, "Frozen255 schema differs")
        tree = read_json(RUN / "query_trees" / (source + ".json"))
        self.prefixes = {tuple(p["prefix"]) for p in tree["primary_leaves"] + tree["fallback_leaves"]}
        self.counts = Counter()

    def matrix(self, paths):
        from pathInference.upstream.implementation.candidates import edge_evidence, exact_support
        rows = []
        for path in paths:
            prefix = next((path[:n] for n in (3, 2) if path[:n] in self.prefixes), path[:2])
            self.counts["outside_tree_context"] += int(prefix not in self.prefixes)
            proofs = [edge_evidence(self.runtime.safe["index"], self.runtime.public.relations, a, b,
                       self.plan["prefix_edge_provenance"].get("->".join(prefix[:i + 2]), {}))
                      for i, (a, b) in enumerate(zip(prefix, prefix[1:]))]
            tail = path[len(prefix) - 1:]
            frequency = exact_support(self.runtime.safe["index"], tail)
            rows.append(dict(path=path, prefix=prefix, tail_len=len(tail), row_index=len(rows),
                             tail_path_freq=frequency if frequency else None, prefix_edge_evidence=proofs))
        if not rows:
            return np.empty((0, 255), np.float32)
        core = np.asarray([[row[n] for n in self.builder.feature_names]
                           for row in map(self.builder.features, rows)], np.float32)
        result = np.concatenate((core, self.store.append(int(self.source), rows)), axis=1)
        require(result.shape == (len(paths), 255) and np.isfinite(result).all(), "Invalid feature matrix")
        return result


def training_matrix(source, name, extractor, vocabulary, reuse):
    directory = RUN / ("as" + source)
    cache = directory / (name + ".npz")
    receipt = directory / (name + "_features.json")
    if receipt.exists():
        verify_files(read_json(receipt)["outputs_sha256"])
        with np.load(cache, allow_pickle=False) as data:
            return data["X"], data["y"], data["groups"]
    file = Path(reuse["split"]) / (name + ".csv")
    xs, ys, groups, paths, labels = [], [], [], [], []
    manifest = read_json(Path(reuse["split"]) / "split_manifest.json")
    pairs = set()
    with file.open(newline="") as stream:
        for row in csv.reader(stream):
            candidates = decode_candidates(row, vocabulary, source)
            targets = list(map(float, row[3].split("|")))
            require(len(targets) == len(candidates) and all(0 <= y <= 1 for y in targets), "Label alignment differs")
            require(row[4] not in pairs, "Duplicated training query")
            pairs.add(row[4])
            groups.append(len(candidates))
            paths.extend(candidates)
            labels.extend(targets)
            if len(paths) >= 4096:
                xs.append(extractor.matrix(paths)); ys.extend(labels)
                paths, labels = [], []
                emit(source, "features_" + name, queries=len(groups), candidates=len(ys))
    require(pairs == set(manifest[name + "_pairs"]), "Changed split inventory")
    if paths:
        xs.append(extractor.matrix(paths)); ys.extend(labels)
    X, y, groups = np.concatenate(xs), np.asarray(ys, np.float32), np.asarray(groups, np.int32)
    np.savez_compressed(cache, X=X, y=y, groups=groups)
    freeze(receipt, dict(input_sha256={str(file): sha256(file)}, schema=extractor.names,
                         outputs_sha256={str(cache): sha256(cache)}, queries=len(groups), candidates=len(y)))
    return X, y, groups


def train(source):
    import lightgbm as lgb
    cfg = config()
    directory = RUN / ("as" + source)
    reuse = read_json(directory / "REUSE.json")
    verify_files(reuse["inputs_sha256"])
    with forbid_formal_references():
        emit(source, "loading_feature_runtime")
        extractor = Features(source)
        vocabulary = read_json(directory / "VOCABULARY.json")
        X, y, groups = training_matrix(source, "train", extractor, vocabulary, reuse)
        V, vy, vg = training_matrix(source, "validation", extractor, vocabulary, reuse)
        weights = np.repeat(1.0 / groups, groups)
        ds = lgb.Dataset(X, label=y, weight=weights, feature_name=extractor.names)
        val = lgb.Dataset(V, label=vy, feature_name=extractor.names, reference=ds)
        params = dict(objective="regression", metric="None", learning_rate=cfg["learning_rate"],
                      num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
                      num_threads=cfg["threads"], seed=cfg["seed"], deterministic=True,
                      force_col_wise=True, verbosity=-1)
        history = {}
        def metric(scores, data):
            return "query_top1_jaccard", query_top1(data.get_label(), scores, vg), True
        emit(source, "training", train_queries=len(groups), validation_queries=len(vg), train_candidates=len(y))
        model = lgb.train(params, ds, num_boost_round=cfg["max_rounds"], valid_sets=[val], valid_names=["validation"],
                          feval=metric, callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=True),
                          lgb.log_evaluation(20), lgb.record_evaluation(history)])
        target = directory / "model.txt"
        model.save_model(str(target), num_iteration=model.best_iteration)
        freeze(directory / "MODEL_FROZEN.json", dict(
            model=str(target), model_sha256=sha256(target), best_iteration=model.best_iteration,
            validation_query_jaccard=model.best_score["validation"]["query_top1_jaccard"],
            history=history, parameters=params, feature_schema=extractor.names,
            feature_schema_sha256=__import__("pathInference.utils", fromlist=["object_hash"]).object_hash(extractor.names),
            inputs_sha256=reuse["inputs_sha256"], feature_diagnostics=dict(extractor.counts),
            label_semantics=cfg["target"], pair_disjoint=True, final_labels_read=False))
    emit(source, "model_frozen", best_iteration=model.best_iteration)


def generate(source):
    from pathInference.experiment import worker
    emit(source, "regenerating_test_candidates", training_candidates_reused=True)
    os.environ["PATHVEIL_PROJECT_ROOT"] = str(ROOT)
    os.environ["PATHVEIL_STAGE_GUARD_ROOT"] = str(RUN)
    worker.execute(RUN, "pg_test", source, device="cpu", cpu_threads=2)


def predict(source):
    import lightgbm as lgb
    cfg = config()
    directory = RUN / ("as" + source)
    frozen = read_json(directory / "MODEL_FROZEN.json")
    verify_files({frozen["model"]: frozen["model_sha256"]})
    child = RUN / "T-PG-B/pg_inputs" / ("as" + source)
    marker = read_json(child / "control/pathgen_completed" / ("test_inputs_" + source + ".json"))
    compiled = marker["label_blind_compilation"]
    verify_files({marker["output"]: marker["sha256"], compiled["dispatch_status_file"]: compiled["dispatch_status_sha256"]})
    output = directory / "pair_predictions.csv"
    vocabulary = read_json(directory / "VOCABULARY.json")
    predictions = {}
    with forbid_formal_references():
        extractor = Features(source)
        require(frozen["feature_schema"] == extractor.names, "Training/prediction feature drift")
        model = lgb.Booster(model_file=frozen["model"])
        def flush(batch):
            flat = [p for _, paths in batch for p in paths]
            scores = model.predict(extractor.matrix(flat), num_threads=2)
            offset = 0
            for pair, paths in batch:
                best = int(np.argmax(scores[offset:offset + len(paths)]))
                predictions[pair] = (paths[best], "predicted")
                offset += len(paths)
        batch = []
        with Path(marker["output"]).open(newline="") as stream:
            for row in csv.reader(stream):
                require(set(row[3].split("|")) <= {"0.0"}, "Test candidate labels are not closed")
                batch.append((row[4], decode_candidates(row, vocabulary, source)))
                if len(batch) >= cfg["batch_queries"]:
                    flush(batch); batch = []
                    emit(source, "predicting", completed_model_queries=len(predictions))
        if batch:
            flush(batch)
        inventory = read_json(read_json(child / "PREPARED.json")["fixed_test_pairs_inventory"][source]["path"])
        statuses = {}
        with Path(compiled["dispatch_status_file"]).open() as stream:
            for line in stream:
                item = json.loads(line)
                pair = item["pair"]
                require(pair not in statuses, "Duplicated dispatch query")
                statuses[pair] = item["status"]
                if item["status"] == "direct_graph_edge_candidate":
                    predictions[pair] = (tuple(pair.split("->")), "direct_graph_edge")
                elif item["status"] in ("same_as_query", "no_usable_candidate"):
                    predictions[pair] = ((), item["status"])
                else:
                    require(item["status"] == "model_candidates" and pair in predictions, "Missing model query")
        require(set(statuses) == set(inventory["pairs"]) == set(predictions), "Incomplete endpoint coverage")
        temporary = directory / "predictions.tmp.csv"
        with temporary.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["pair", "predicted_path", "status"])
            for pair in inventory["pairs"]:
                path, status = predictions[pair]
                writer.writerow([pair, "->".join(path), status])
        temporary.replace(output)
        freeze(directory / "PREDICTIONS_FROZEN.json", dict(
            model_sha256=frozen["model_sha256"], predictions=str(output), predictions_sha256=sha256(output),
            candidate_sha256=marker["sha256"], queries=len(predictions),
            feature_diagnostics=dict(extractor.counts), final_labels_read=False, frozen_utc=utc_now()))
    emit(source, "predictions_frozen", queries=len(predictions))


def evaluate(source):
    """Only this stage opens final labels; model and predictions already fixed."""
    from pathInference.downstream.evaluation import score_source, read_references
    from pathInference.research.ablation import asset
    directory = RUN / ("as" + source)
    frozen = read_json(directory / "PREDICTIONS_FROZEN.json")
    model = read_json(directory / "MODEL_FROZEN.json")
    verify_files({model["model"]: frozen["model_sha256"], frozen["predictions"]: frozen["predictions_sha256"]})
    original = read_json(asset(baseline() / "RESULTS.json"))
    comparison = next(r for r in original["per_source"] if r["source_asn"] == source)
    metrics = score_source(comparison["reference_file"], frozen["predictions"], source, comparison["fixed_identity"])
    truth, _ = read_references(comparison["reference_file"], source)
    vocabulary = read_json(directory / "VOCABULARY.json")
    child = RUN / "T-PG-B/pg_inputs" / ("as" + source)
    marker = read_json(child / "control/pathgen_completed" / ("test_inputs_" + source + ".json"))
    oracle = {}
    def score(pair, paths):
        refs = truth[pair]
        return len(paths), int(bool(set(paths) & refs)), max(
            (len(set(p) & set(t)) / len(set(p) | set(t)) for p in paths for t in refs), default=0.)
    with Path(marker["output"]).open(newline="") as stream:
        for row in csv.reader(stream):
            oracle[row[4]] = score(row[4], decode_candidates(row, vocabulary, source))
    dispatch = marker["label_blind_compilation"]["dispatch_status_file"]
    with Path(dispatch).open() as stream:
        for line in stream:
            item = json.loads(line)
            if item["status"] == "direct_graph_edge_candidate":
                oracle[item["pair"]] = score(item["pair"], [tuple(item["pair"].split("->"))])
    previous = ROOT / "runs/20260201/generator_recursion_v1/current/diagnostics" / ("as" + source) / "candidate_oracle.csv"
    mismatches = None
    if previous.is_file():
        mismatches = 0
        seen = set()
        with previous.open(newline="") as stream:
            for row in csv.DictReader(stream):
                pair = row["pair"]; seen.add(pair)
                actual = oracle.get(pair, (0, 0, 0.))
                mismatches += int(actual[:2] != (int(row["candidate_count"]), int(row["exact_available"]))
                                  or abs(actual[2] - float(row["oracle_jaccard"])) > 1e-10)
        require(seen == set(truth), "Historical oracle has a different query universe")
    count = len(truth)
    metrics.update(date=config()["date"], source=source, baseline_gmpi_jaccard=comparison["jaccard_accuracy"],
                   baseline_gmpi_exact=comparison["ordered_exact_accuracy"],
                   jaccard_change=metrics["jaccard_accuracy"] - comparison["jaccard_accuracy"],
                   oracle_jaccard=sum(v[2] for v in oracle.values()) / count,
                   candidate_exact_available=sum(v[1] for v in oracle.values()) / count,
                   candidate_coverage=len(oracle) / count, candidate_diagnostic_mismatches=mismatches,
                   diagnostic_comparison_passed=mismatches == 0,
                   candidate_identity_limitation="Counts/exact/oracle checked; historical full candidate cache was cleaned",
                   final_labels_used_for_tuning=False, model_sha256=model["model_sha256"],
                   rho=comparison["rho"])
    freeze(directory / "RESULTS.json", metrics)
    emit(source, "completed", jaccard=metrics["jaccard_accuracy"], gmpi_jaccard=comparison["jaccard_accuracy"],
         candidate_diagnostic_mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "generate", "predict", "evaluate"), required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, set(config()["cpu_affinity"]))
    if args.stage != "prepare":
        identity = read_json(RUN / "IDENTITY.json")
        require(identity["config_sha256"] == sha256(CONFIG), "Experiment configuration changed")
        verify_files(identity["code_sha256"])
    globals()[args.stage](args.source)


if __name__ == "__main__":
    main()
