"""Run the isolated legacy KnownPath baseline on one BGP date."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
FILES = ROOT.parent / "files"
SOURCE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_run(run_root, date_root, date, sources):
    with (date_root / "RL_Graph/vantage_dst_test.json").open() as stream:
        queries = json.loads(json.load(stream))
    expected = sum(len(set(queries[source])) for source in sources)
    destinations = set()
    scored = 0
    result_dir = run_root / "results" / date
    for path in sorted(result_dir.glob("*-knownpath-result.csv")):
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 6:
                    raise ValueError("Malformed result row in " + str(path))
                if row[0] in destinations:
                    raise ValueError("Repeated destination: " + row[0])
                destinations.add(row[0])
                scored += int(row[1])
    if scored != expected:
        raise ValueError("Incomplete query coverage: %d/%d" % (scored, expected))
    return {"date": date, "destinations": len(destinations), "scored_pairs": scored,
            "expected_pairs": expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume-two-to-four", action="store_true")
    parser.add_argument("--resume-four-to-eight", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume_two_to_four and args.workers != 4:
        parser.error("--resume-two-to-four requires --workers 4")
    if args.resume_four_to_eight and args.workers != 8:
        parser.error("--resume-four-to-eight requires --workers 8")

    config_file = ROOT / "configs/dates" / (args.date + ".json")
    config = json.loads(config_file.read_text())
    if config["mode"] != "bgp":
        parser.error("This runner needs BGP test CSVs; traceroute needs a separate adapter")
    sources = config["test_sources"]
    date_root = FILES / "data/aspath/path_by_vantage" / args.date
    local_root = ROOT / "runs/baselines/knownpath/indexes" / args.date
    local_db = local_root / "knownpath.db"
    local_record = local_root / "INDEX_COMPLETE.json"
    if local_db.is_file() and not local_record.is_file():
        parser.error("Local train index is unfinished: " + str(local_db))
    if local_record.is_file():
        record = json.loads(local_record.read_text())
        if record["date"] != args.date or record["db_sha256"] != digest(local_db):
            parser.error("Local train index identity mismatch")
    train_db = local_db if local_record.is_file() else date_root / "path_without_prefix/train/knownpath.db"
    if not train_db.is_file():
        parser.error("Missing train-only KnownPath index: " + str(train_db))
    test_dir = date_root / "path_without_prefix/test"
    missing = [source for source in sources if not (test_dir / (source + ".csv")).is_file()]
    if missing:
        parser.error("Missing test CSVs for: " + ", ".join(missing))

    run_root = ROOT / "runs/baselines/knownpath" / args.date
    run_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "date": args.date,
        "sources": sources,
        "workers": args.workers,
        "config_sha256": digest(config_file),
        "train_db_sha256": digest(train_db),
        "code_sha256": {name: digest(SOURCE / name) for name in (
            "knownpath.py", "environment.py", "commonFunction.py",
            "get_evaluate_data.py", "cpp_engine.py", "libknownpath.so")},
    }
    identity_file = run_root / "IDENTITY.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        previous = json.loads(identity_file.read_text())
        if (args.resume_two_to_four or args.resume_four_to_eight) and previous.get("dispatch_history"):
            identity["dispatch_history"] = previous["dispatch_history"]
        if previous != identity:
            transition = (args.resume_two_to_four and previous.get("workers") == 2
                          or args.resume_four_to_eight and previous.get("workers") == 4)
            if not (transition
                    and previous["date"] == identity["date"]
                    and previous["sources"] == identity["sources"]
                    and previous["config_sha256"] == identity["config_sha256"]
                    and previous["train_db_sha256"] == identity["train_db_sha256"]):
                parser.error("Existing run identity differs: " + str(identity_file))
            snapshot = "IDENTITY_INITIAL.json" if args.resume_two_to_four else "IDENTITY_FOUR.json"
            (run_root / snapshot).write_text(json.dumps(previous, indent=2) + "\n")
            identity["dispatch_history"] = previous.get("dispatch_history", ["two workers"])
            identity["dispatch_history"].append("%d workers with preserved partitions" % args.workers)
    identity_file.write_text(json.dumps(identity, indent=2) + "\n")

    env = os.environ.copy()
    env.update({
        "KNOWNPATH_FILES_ROOT": str(FILES),
        "KNOWNPATH_RUN_ROOT": str(run_root),
        "KNOWNPATH_SOURCES": ",".join(sources),
        "KNOWNPATH_TRAIN_DB": str(train_db),
    })
    processes = []
    logs = []
    for index in range(args.workers):
        log = (run_root / ("worker_%d.log" % index)).open("a")
        logs.append(log)
        resumed = args.resume_two_to_four or args.resume_four_to_eight
        base_index = index % 2 if resumed else index
        command = [sys.executable, "-u", str(SOURCE / "knownpath.py"),
                   "--date", args.date, "--num_vas", str(len(sources)),
                   "--split_num", str(2 if resumed else args.workers),
                   "--cur_idx", str(base_index),
                   "--cur_node_idx", str(index)]
        if args.resume_two_to_four:
            half = 17847 if args.date == "20260201" else None
            if half is None:
                parser.error("Resume partition is only verified for 20260201")
            command += (["--range_start", "0", "--range_end", str(half)]
                        if index < 2 else ["--range_start", str(half)])
        if args.resume_four_to_eight:
            if args.date != "20260201":
                parser.error("Resume partition is only verified for 20260201")
            quarter = 17847 // 2
            if index < 2:
                start, end = 0, quarter
            elif index < 4:
                start, end = 17847, 17847 + quarter
            elif index < 6:
                start, end = quarter, 17847
            else:
                start, end = 17847 + quarter, None
            command += ["--range_start", str(start)]
            if end is not None:
                command += ["--range_end", str(end)]
        processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
    print("KnownPath", args.date, "workers", [p.pid for p in processes], flush=True)
    try:
        while any(p.poll() is None for p in processes):
            time.sleep(15)
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                break
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for process in processes:
            process.wait()
        for log in logs:
            log.close()
    codes = [process.returncode for process in processes]
    print("KnownPath", args.date, "exit codes", codes, flush=True)
    if not all(code == 0 for code in codes):
        return 1
    summary = verify_run(run_root, date_root, args.date, sources)
    (run_root / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("KnownPath", args.date, summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
