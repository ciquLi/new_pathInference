"""Run all three PathRadar learning rounds and evaluate the final model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from performance.pair_details import persist as persist_pair_details, test_pairs


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
FILES = ROOT.parent / "files"
LEGACY = ROOT.parent / "pathRadar"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_identity(directory):
    digest = hashlib.sha256()
    files = sorted(directory.glob("*.csv"))
    for path in files:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return {"files": len(files), "sha256": digest.hexdigest()}


def run_stage(run_dir, label, command, env):
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    preparation_marker = None
    if label in ("build", "combine"):
        preparation_marker = run_dir / "control" / ("completed_" + label + ".json")
        preparation_identity = {
            "identity_sha256": sha256(run_dir / "IDENTITY.json"),
            "command": list(map(str, command)),
            "script_sha256": sha256(Path(command[2])),
        }
        if preparation_marker.is_file():
            previous = json.loads(preparation_marker.read_text())
            if previous.get("identity") != preparation_identity:
                raise ValueError("Preparation cache identity changed: " + str(preparation_marker))
            print(run_dir.name, label, "reused completed preparation", flush=True)
            return
    with (log_dir / (label + ".log")).open("a") as log:
        log.write("COMMAND: " + " ".join(map(str, command)) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if preparation_marker is not None:
        preparation_marker.parent.mkdir(parents=True, exist_ok=True)
        preparation_marker.write_text(json.dumps({"identity": preparation_identity,
                                                  "status": "PASS"}, indent=2) + "\n")
    print(run_dir.name, label, "complete", flush=True)


def db_counts(db_path):
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        return dict(connection.execute(
            "SELECT dataset, COUNT(*) FROM grouped_paths GROUP BY dataset"))
    finally:
        connection.close()


def model_paths(run_dir, round_number):
    models = run_dir / "trained/models"
    return [models / ("xgboost_%d_%s%s.model" % (round_number, kind, index))
            for kind in ("core", "shell") for index in (1, 2, 3)]


def prediction_count(folder):
    return sum(1 for _ in folder.glob("*.txt")) if folder.is_dir() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config_file = ROOT / "configs/dates" / (args.date + ".json")
    config = json.loads(config_file.read_text())
    if config["mode"] != "bgp":
        parser.error("PathRadar BGP runner cannot evaluate traceroute")
    date_data = FILES / "data/aspath/path_by_vantage" / args.date
    source_data = date_data / "path_without_prefix"
    train = source_data / "train"
    test = source_data / "test"
    if not train.is_dir() or not test.is_dir():
        parser.error("Missing dated train/test split")
    test_files = {path.stem for path in test.glob("*.csv")}
    if test_files != set(config["test_sources"]):
        parser.error("Test CSV sources do not match registered date")

    run_dir = ROOT / "runs/baselines/pathradar" / args.date
    run_dir.mkdir(parents=True, exist_ok=True)
    asrel = FILES / "data/aspath" / (args.date + ".as-rel.txt")

    legacy_run = LEGACY / (args.date + "_devoid")
    use_cache = args.date == "20260201"
    if use_cache:
        cache = legacy_run / "intermediate"
        required = ("asrel_rib.txt", "core_shell.txt", "joinfreq_data.txt",
                    "joinfreq_train.txt", "linkfreq_data.txt", "linkfreq_train.txt",
                    "linkorder_data.txt", "linkorder_train.txt", "pathradar_store.sqlite")
        for name in required:
            if not (cache / name).is_file():
                raise FileNotFoundError(cache / name)
        legacy_train = legacy_run / "source_data/path_without_prefix/train"
        legacy_test = legacy_run / "source_data/path_without_prefix/test"
        if csv_identity(train) != csv_identity(legacy_train):
            raise ValueError("Legacy PathRadar train staging differs from date input")
        if csv_identity(test) != csv_identity(legacy_test):
            raise ValueError("Legacy PathRadar test staging differs from date input")

    code_files = sorted(path for path in SOURCE.glob("*.py")
                        if path.name not in ("run_baseline.py", "run_after_knownpath.py"))
    identity = {
        "date": args.date,
        "config_sha256": sha256(config_file),
        "train_csv": csv_identity(train),
        "test_csv": csv_identity(test),
        "asrel_sha256": sha256(asrel),
        "code_sha256": {path.name: sha256(path) for path in code_files},
        "cached_intermediate": use_cache,
    }
    identity_file = run_dir / "IDENTITY.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("Run identity changed: " + str(identity_file))
    identity_file.write_text(json.dumps(identity, indent=2) + "\n")
    if args.prepare_only:
        print(args.date, "inputs and identity verified", flush=True)
        return

    env = os.environ.copy()
    env.update({"PATHRADAR_RUN_DIR": str(run_dir), "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1"})
    py = sys.executable
    db = (cache if use_cache else run_dir / "intermediate") / "pathradar_store.sqlite"
    if use_cache:
        env["PATHRADAR_READONLY_DB"] = "1"
        env["PATHRADAR_INTERMEDIATE_DIR"] = str(cache)
    elif not (run_dir / "PREPARED.json").is_file():
        run_stage(run_dir, "build", [py, "-u", str(SOURCE / "tools/build_from_split_csv.py"),
                                     "--date", args.date, "--run-dir", str(run_dir),
                                     "--train-dir", str(train), "--asrel-file", str(asrel)], env)
        run_stage(run_dir, "combine", [py, "-u", str(SOURCE / "combine_path.py")], env)
        run_stage(run_dir, "shortest", [str(SOURCE / "bin/pathradar_shortest_cpp"),
                                        "-m", "10", "--skip-existing", "1",
                                        "--workers", os.environ.get("PATHRADAR_SHORTEST_WORKERS", "10")], env)
        run_stage(run_dir, "migrate", [py, "-u", str(SOURCE / "migrate_intermediate_to_sqlite.py")], env)
    counts = db_counts(db)
    if not counts.get("train") or not counts.get("validation") or not counts.get("shortest"):
        raise ValueError("Incomplete PathRadar intermediate DB: " + str(counts))
    prepared_file = run_dir / "PREPARED.json"
    prepared = {"db_sha256": sha256(db), "dataset_counts": counts}
    if prepared_file.is_file() and json.loads(prepared_file.read_text()) != prepared:
        raise ValueError("PathRadar prepared database identity changed")
    prepared_file.write_text(json.dumps(prepared, indent=2) + "\n")
    if not use_cache:
        raw = run_dir / "intermediate"
        for name in ("data", "train", "validation", "test", "rib", "origintest", "shortest"):
            folder = raw / name
            if folder.is_dir():
                shutil.rmtree(folder)
        for name in ("done_rib.txt", "rib_data.txt", "rib_train.txt",
                     "rib_validation.txt", "rib_test.txt", "rib_origintest.txt"):
            (raw / name).unlink(missing_ok=True)

    rib_count = counts["rib"]
    for round_number in range(3):
        expected_models = model_paths(run_dir, round_number)
        if not all(path.is_file() for path in expected_models):
            run_stage(run_dir, "train_%d" % round_number,
                      [py, "-u", str(SOURCE / "pathradar_train.py"),
                       "-r", str(round_number), "-q", "1"], env)
        if not all(path.is_file() for path in expected_models):
            raise ValueError("PathRadar round %d lacks models" % round_number)
        if round_number == 2:
            break
        folder = run_dir / "testing" / ("result_round%d" % round_number)
        marker = run_dir / ("PREDICT_TRAIN_ROUND%d.json" % round_number)
        if marker.is_file():
            record = json.loads(marker.read_text())
            if record.get("rib_count") != rib_count or prediction_count(folder) != rib_count:
                raise ValueError("Training prediction cache changed for round %d" % round_number)
        else:
            run_stage(run_dir, "predict_train_%d" % round_number,
                      [py, "-u", str(SOURCE / "pathradar_prediction.py"),
                       "-r", str(round_number), "-t", "0.5", "-m", "train",
                       "-q", "0", "-w", "8"], env)
            if prediction_count(folder) != rib_count:
                raise ValueError("Incomplete training predictions for round %d" % round_number)
            marker.write_text(json.dumps({"round": round_number,
                                          "rib_count": rib_count}, indent=2) + "\n")

    output = run_dir / "testing/top1_eval_round2_full.json"
    if not output.is_file() or not json.loads(output.read_text()).get("include_pair_details"):
        run_stage(run_dir, "evaluate", [py, "-u", str(SOURCE / "evaluate_top1_pairs.py"),
                                        "--rounds", "2", "--threshold", "0.7",
                                        "--date", args.date, "--test-dir", str(test),
                                        "--sample-size", "0", "--workers", "1",
                                        "--include-pair-details", "1",
                                        "--pair-batch-size", "100",
                                        "--output", str(output)], env)
    result = json.loads(output.read_text())
    requested = result.get("requested_pairs")
    if requested != result.get("pairs_with_ground_truth"):
        raise ValueError("PathRadar evaluation query count mismatch")
    details = persist_pair_details(output, result, test_pairs(test))
    (run_dir / "COMPLETE.json").write_text(json.dumps(
        {"date": args.date, "requested_pairs": requested,
         "round": 2, "jaccard": result.get("mean_jaccard_over_requested"),
         "pair_details": details}, indent=2) + "\n")
    if not use_cache:
        shutil.rmtree(run_dir / "intermediate")
    for round_number in (0, 1):
        shutil.rmtree(run_dir / "testing" / ("result_round%d" % round_number))
    print(args.date, "complete", requested, flush=True)


if __name__ == "__main__":
    main()
