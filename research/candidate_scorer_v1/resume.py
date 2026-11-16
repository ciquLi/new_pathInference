#!/usr/bin/env python3
"""Audited publication repair; preserve frozen code, models and candidates."""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pathInference.research import candidate_scorer as experiment
from pathInference.experiment.contracts import freeze, require, verify_files, forbid_formal_references
from pathInference.utils import read_json, sha256, utc_now, write_json


def binding():
    """Keep the original frozen source inventory and bind the explicit repair."""
    identity = read_json(experiment.RUN / "IDENTITY.json")
    verify_files(identity["code_sha256"])
    require(identity["config_sha256"] == sha256(experiment.CONFIG), "Configuration changed")
    freeze(experiment.RUN / "PUBLICATION_REPAIR.json", dict(
        repair_file=str(Path(__file__).resolve()), repair_sha256=sha256(__file__),
        original_code_sha256=identity["code_sha256"],
        change="Move publication receipt after closed-label feature/prediction context; numerical work unchanged",
        recovery="Validate completed predictions against frozen candidate sets, dispatch statuses and full endpoints",
        model_retrained=False, candidates_regenerated=False, final_labels_used_for_repair=False))


def patched_predict():
    """Move exactly one publication statement; leave numerical code unchanged."""
    tree = ast.parse(inspect.getsource(experiment.predict))
    function = tree.body[0]
    scopes = [node for node in function.body if isinstance(node, ast.With)]
    require(len(scopes) == 1, "Unexpected prediction guard layout")
    scope = scopes[0]
    publication = scope.body[-1]
    require(isinstance(publication, ast.Expr) and isinstance(publication.value, ast.Call)
            and isinstance(publication.value.func, ast.Name) and publication.value.func.id == "freeze",
            "Unexpected publication layout")
    scope.body.pop()
    function.body.insert(function.body.index(scope) + 1, publication)
    ast.fix_missing_locations(tree)
    namespace = dict(vars(experiment))
    exec(compile(tree, str(Path(__file__).resolve()) + ":publication_repair", "exec"), namespace)
    return namespace["predict"]


def recover(source):
    """Original failure occurred after atomic CSV publication; validate and retain."""
    directory = experiment.RUN / ("as" + source)
    output = directory / "pair_predictions.csv"
    receipt = directory / "PREDICTIONS_FROZEN.json"
    if receipt.is_file():
        record = read_json(receipt)
        verify_files({str(output): record["predictions_sha256"]})
        return True
    if not output.is_file():
        return False
    frozen = read_json(directory / "MODEL_FROZEN.json")
    verify_files({frozen["model"]: frozen["model_sha256"]})
    child = experiment.RUN / "T-PG-B/pg_inputs" / ("as" + source)
    marker = read_json(child / "control/pathgen_completed" / ("test_inputs_" + source + ".json"))
    compiled = marker["label_blind_compilation"]
    verify_files({marker["output"]: marker["sha256"],
                  compiled["dispatch_status_file"]: compiled["dispatch_status_sha256"]})
    # Only this repair can read its own published output. Final labels are never opened.
    predictions = {}
    with output.open(newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["pair"] not in predictions, "Duplicate prediction")
            predictions[row["pair"]] = row
    inventory = read_json(read_json(child / "PREPARED.json")["fixed_test_pairs_inventory"][source]["path"])
    require(set(predictions) == set(inventory["pairs"]), "Published output incomplete")
    vocabulary = read_json(directory / "VOCABULARY.json")
    model_pairs = set()
    with forbid_formal_references():
        with Path(marker["output"]).open(newline="") as stream:
            for row in csv.reader(stream):
                pair = row[4]
                require(pair not in model_pairs and pair in predictions, "Candidate/prediction mismatch")
                model_pairs.add(pair)
                candidates = experiment.decode_candidates(row, vocabulary, source)
                predicted = tuple(predictions[pair]["predicted_path"].split("->"))
                require(predicted in candidates and predictions[pair]["status"] == "predicted",
                        "Prediction is outside frozen candidate set")
        statuses = set()
        with Path(compiled["dispatch_status_file"]).open() as stream:
            for line in stream:
                item = json.loads(line); pair = item["pair"]
                require(pair not in statuses, "Duplicated dispatch pair")
                statuses.add(pair)
                predicted = predictions[pair]
                if item["status"] == "model_candidates":
                    require(pair in model_pairs, "Missing model prediction")
                elif item["status"] == "direct_graph_edge_candidate":
                    require(predicted["predicted_path"] == pair and predicted["status"] == "direct_graph_edge",
                            "Direct prediction mismatch")
                else:
                    require(item["status"] in ("same_as_query", "no_usable_candidate")
                            and not predicted["predicted_path"] and predicted["status"] == item["status"],
                            "Missing prediction status mismatch")
        require(statuses == set(predictions), "Incomplete status inventory")
    freeze(receipt, dict(model_sha256=frozen["model_sha256"], predictions=str(output),
                        predictions_sha256=sha256(output), candidate_sha256=marker["sha256"],
                        queries=len(predictions), final_labels_read=False, frozen_utc=utc_now(),
                        recovered_existing_complete_csv=True, repair_sha256=sha256(__file__)))
    experiment.emit(source, "predictions_frozen", queries=len(predictions), reused_completed_predictions=True)
    return True


def stage(source, name):
    binding()
    os.sched_setaffinity(0, set(experiment.config()["cpu_affinity"]))
    # The repair launcher is separately bound, outside the original frozen
    # source inventory. Later AS preparation must retain that original identity.
    experiment.code_identity = lambda: read_json(experiment.RUN / "IDENTITY.json")["code_sha256"]
    if name == "predict":
        if not recover(source):
            patched_predict()(source)
    else:
        getattr(experiment, name)(source)


def dispatch():
    import fcntl
    spec = importlib.util.spec_from_file_location("scorer_dispatch", Path(__file__).with_name("dispatch.py"))
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    run = experiment.RUN
    lock = (run / ".dispatch.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding()
    os.sched_setaffinity(0, set(experiment.config()["cpu_affinity"]))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PATHVEIL_PROJECT_ROOT=str(ROOT))
    for source in experiment.config()["sources"]:
        for name in driver.STAGES:
            receipt = run / "tasks" / (source + "_" + name + ".json")
            if receipt.is_file():
                value = read_json(receipt)
                require(value["config_sha256"] == sha256(experiment.CONFIG), "Cached configuration changed")
                verify_files(value["outputs_sha256"])
                continue
            driver.report((source, name))
            log = run / "logs" / (source + "_" + name + ".log")
            started = time.monotonic()
            with log.open("a") as stream:
                child = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                          "--source", source, "--stage", name], cwd=ROOT, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT)
                write_json(run / "ACTIVE.json", dict(source=source, stage=name, worker_pid=child.pid,
                                                      log=str(log), started_utc=utc_now()))
                while child.poll() is None:
                    driver.report((source, name)); time.sleep(10)
            if child.returncode:
                error = f"AS{source} {name} exit={child.returncode}; 日志：{log}"
                driver.report(error=error)
                raise SystemExit(error)
            freeze(receipt, dict(status="PASS", source=source, stage=name,
                                 config_sha256=sha256(experiment.CONFIG), outputs_sha256=driver.outputs(source, name),
                                 elapsed_seconds=time.monotonic() - started, completed_utc=utc_now(),
                                 repair_sha256=sha256(__file__)))
            driver.report()
    driver.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--stage", choices=("prepare", "train", "generate", "predict", "evaluate"))
    args = parser.parse_args()
    if args.stage:
        stage(args.source, args.stage)
    else:
        dispatch()
