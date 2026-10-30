"""Dependency-light contracts shared by isolated traceroute workers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from pathveil.utils import project_root, read_json, sha256 as digest, write_json as json_write


def project_imports(data_manifest):
    sys.dont_write_bytecode = True
    from pathveil.core import activate

    activate(data_manifest=data_manifest)


def vocab_digest(env):
    return hashlib.sha256(json.dumps(env.itos, separators=(",", ":")).encode()).hexdigest()


def path_signature(paths):
    return hashlib.sha256(
        ("\n".join(sorted("->".join(map(str, path)) for path in paths)) + "\n").encode()
    ).hexdigest()


def load_contract(data_manifest_path, release_path, source):
    data, release = read_json(data_manifest_path), read_json(release_path)
    source = str(source)
    if release.get("date") != data.get("date") or source not in release.get("sources", []):
        raise ValueError("Release source/date differs from traceroute data manifest")
    if data.get("sources") and source not in data["sources"]:
        raise ValueError("Trace source is absent from the registered data sources")
    if release.get("formal_reference_labels_read") is not False:
        raise ValueError("Release lacks the no-formal-reference declaration")
    if release.get("training_and_PG_pool") != "same_R_s_hash_augmentation_only":
        raise ValueError("Trace training and generation require one admitted pool")
    entry = release["files_by_source"][source]
    path = Path(entry["path"]).resolve()
    if digest(path) != entry["sha256"]:
        raise ValueError("Released admitted path CSV changed")
    from pathveil.upstream.contracts import path_rows

    paths = path_rows(path, source)
    if len(paths) != entry["rows"] or not paths:
        raise ValueError("Released admitted pool count mismatch or empty")
    graph = {}
    for key in ["selected_neighbors_csv", "recursive_anchors_csv"]:
        filename = release[key]
        if digest(filename) != release["outputs_sha256"].get(filename):
            raise ValueError("Released topology CSV hash mismatch: " + key)
        graph[key] = str(Path(filename).resolve())
    return {
        "date": data["date"],
        "source": source,
        "path": str(path),
        "rows": len(paths),
        "sha256": entry["sha256"],
        "path_set_sha256": path_signature(paths),
        "release_path": str(Path(release_path).resolve()),
        "release_sha256": digest(release_path),
        "data_manifest_path": str(Path(data_manifest_path).resolve()),
        "data_manifest_sha256": digest(data_manifest_path),
        "pool_mode": "augmentation_only",
        **graph,
    }


def assert_source_held_out(env, source):
    path = (
        Path(env.root_dir)
        / "data/aspath/path_by_vantage"
        / env.parent
        / "path_without_prefix/train"
        / (str(source) + ".csv")
    )
    if path.exists():
        with path.open() as handle:
            if any(line.strip() for line in handle):
                raise ValueError(
                    "Trace source training file must be absent/empty or independently audited: "
                    + str(path)
                )
    return {
        "source_asn": str(source),
        "source_train_file": str(path),
        "source_train_paths": 0,
        "scope": "No original source-AS observation file; upstream graph/embedding provenance is separately bound by the data manifest.",
    }


def install_label_guard(outdir):
    from pathveil.downstream.common import StageGuard

    guard = StageGuard(os.environ.get("PATHVEIL_TRACE_RUN_DIR", outdir), allow_raw_test=False)

    def close_measurements(event, values):
        if event != "open" or not values or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        mode = values[1] or "r"
        if isinstance(mode, str) and ("r" in mode or "+" in mode):
            path = Path(os.fsdecode(values[0])).resolve()
            if "measurement" in path.parts:
                raise PermissionError("Traceroute measurements are evaluation-only: " + str(path))

    sys.addaudithook(close_measurements)
    sys.addaudithook(guard.check)
    return guard


def audit_pool_consumers(env, source, contract):
    import getData
    import HighFrequencyVertexBasedPathGen as generator

    if (
        env.train_extra_data_mode != "augmentation_only"
        or env.pg_local_knowledge_mode != "augmentation_only"
    ):
        raise ValueError("Both PG and training labels must use augmentation_only")
    truth = getData.load_train_extra_truth_paths(env, source)
    truth_paths = [path for rows in truth.values() for path in rows]
    local = generator.load_local_knowledge_paths(env, source)
    result = {}
    for consumer, paths in [("training_labels", truth_paths), ("pg_local", local)]:
        actual = path_signature(paths)
        if actual != contract["path_set_sha256"] or len(paths) != contract["rows"]:
            raise ValueError(consumer + " did not consume exactly the admitted R_s")
        result[consumer] = {
            "path_set_sha256": actual,
            "rows": len(paths),
            "mode": "augmentation_only",
        }
    return result


def stable_bgp_split(input_csv, outdir, validation_fraction=0.1, seed=20261015):
    """Keep complete BGP-derived AS pairs on one side of the split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    outdir = Path(outdir)
    rows_by_pair = {}
    with Path(input_csv).open(newline="") as handle:
        for rownum, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            if len(row) != 5 or "->" not in row[4]:
                raise ValueError("Bad generated BGP row %d" % rownum)
            rows_by_pair.setdefault(row[4], []).append(row)
    if len(rows_by_pair) < 2:
        raise ValueError("Need at least two BGP-derived AS pairs for validation")
    ordered = sorted(
        rows_by_pair, key=lambda pair: hashlib.sha256(("%s:%s" % (seed, pair)).encode()).digest()
    )
    count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    validation = set(ordered[:count])
    counts = {"train": 0, "validation": 0}
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "train.csv").open("x", newline="") as train, (outdir / "validation.csv").open(
        "x", newline=""
    ) as valid:
        writers = {"train": csv.writer(train), "validation": csv.writer(valid)}
        for pair in sorted(rows_by_pair):
            split = "validation" if pair in validation else "train"
            writers[split].writerows(rows_by_pair[pair])
            counts[split] += len(rows_by_pair[pair])
    manifest = {
        "input_csv": str(Path(input_csv).resolve()),
        "input_sha256": digest(input_csv),
        "split_unit": "source-destination AS pair",
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "row_counts": counts,
        "validation_pairs": sorted(validation),
        "train_pairs": sorted(set(ordered) - validation),
        "validation_role": "BGP-derived surrogate labels only; no traceroute measurements",
    }
    json_write(outdir / "split_manifest.json", manifest)
    return manifest
