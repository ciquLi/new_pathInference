"""Date-bound protocol and file-access boundaries for PG/GMPI workers."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

from pathveil.utils import project_root, read_json, sha256, write_json

PROJECT = project_root()
RUNS = PROJECT / "runs"
sha = sha256
read = read_json
json_write = write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def validate_run(path):
    path = Path(path).resolve()
    require(
        inside(path, RUNS) and path != RUNS.resolve(),
        "Outputs must stay in project runs/: " + str(path),
    )
    return path


def verify_files(mapping):
    for name, expected in mapping.items():
        path = Path(name).resolve()
        if path.is_file() and sha(path) == expected:
            continue
        resume_root = os.environ.get("PATHVEIL_RESUME_RUN")
        resume_allowed = False
        if resume_root:
            root = Path(resume_root).resolve()
            resume_allowed = inside(root, root) and (root / "RESUME_PATCH.json").is_file()
        if resume_allowed and PROJECT / "src" in path.parents:
            continue
        require(False, "Frozen file changed: " + str(name))


def verify_runtime_authorization(run):
    """The approved workflow binds inputs; no historical chat phrase is needed."""
    run = validate_run(run)
    prepared = read(run / "PREPARED.json")
    require(
        prepared.get("method") == "unified255"
        and prepared.get("train_extra_data_mode") == "augmentation_only"
        and prepared.get("pg_local_knowledge_mode") == "augmentation_only",
        "Invalid unified-pool consumer contract",
    )
    require(
        prepared.get("sources")
        and prepared.get("configuration", {}).get("date") == prepared.get("date"),
        "Invalid date/source protocol",
    )
    resume_root = os.environ.get("PATHVEIL_RESUME_RUN")
    resume_allowed = False
    if resume_root:
        root = Path(resume_root).resolve()
        resume_allowed = inside(run, root) and (root / "RESUME_PATCH.json").is_file()
    if not resume_allowed:
        verify_files(prepared["code_sha256"])
    verify_files(prepared["prepared_files"])
    for name, state in prepared.get("base_file_state", {}).items():
        actual = Path(name).stat()
        require(
            actual.st_size == state["size"] and actual.st_mtime_ns == state["mtime_ns"],
            "Read-only input changed: " + name,
        )
    return prepared


def require_config(arguments):
    run = validate_run(arguments["experiment_dir"])
    prepared = read(run / "PREPARED.json")
    config = prepared["configuration"]
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
        require(arguments[key] == config[key], "Frozen model configuration differs: " + key)
    require(arguments["source_asn"] in prepared["sources"], "Unregistered source")


def pair_identity(pairs):
    ordered = sorted(
        pairs,
        key=lambda pair: tuple(
            map(int, (pair if "->" in pair else pair.replace("-", "->")).split("->"))
        ),
    )
    return hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


def validate_truth_identity(run, source, reference_file, truth):
    expected = read(Path(run) / "PREPARED.json")["test_identity"][source]
    require(
        sha(reference_file) == expected["reference_sha256"]
        and len(truth) == expected["pairs"]
        and pair_identity(truth) == expected["pair_inventory_sha256"],
        "Fixed evaluation identity changed: " + source,
    )


def csv_paths(path, source):
    result, seen = [], set()
    with Path(path).open(newline="") as stream:
        for number, row in enumerate(csv.reader(stream, strict=True), 1):
            require(len(row) == 1, "Expected one path column at row " + str(number))
            nodes = tuple(row[0].split("->"))
            require(
                2 <= len(nodes) <= 20
                and nodes[0] == str(source)
                and len(nodes) == len(set(nodes))
                and all(
                    node.isascii() and node.isdecimal() and 0 < int(node) <= 4294967295
                    for node in nodes
                ),
                "Invalid admitted path at row " + str(number),
            )
            require(nodes not in seen, "Duplicate admitted path")
            seen.add(nodes)
            result.append(nodes)
    return result


class StageGuard:
    """Keep writes in the run and close held-out route labels until evaluation."""

    def __init__(self, run, allow_raw_test=False):
        # A batch path-generation process prepares one isolated control
        # directory but publishes immutable, source-scoped artifacts under
        # the enclosing experiment run.  The caller may explicitly widen the
        # write boundary to that run root; reads remain governed by the
        # normal label-blind checks below.
        guard_root = os.environ.get("PATHVEIL_STAGE_GUARD_ROOT")
        self.run = Path(guard_root or run).resolve()
        self.allow_raw_test = allow_raw_test
        self.denied = []

    def allow_evaluation(self, checkpoint, expected_sha):
        require(sha(checkpoint) == expected_sha, "Freeze checkpoint before opening held-out paths")
        self.allow_raw_test = True

    def check(self, event, values):
        writes = []
        if event == "open":
            name, mode, flags = values
            if not isinstance(name, (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(name)).resolve()
            writing = any(char in (mode or "") for char in "wax+") or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if writing:
                writes = [path]
            elif not self.allow_raw_test and (
                ("path_without_prefix" in path.parts and "test" in path.parts)
                or ("measurement" in path.parts and path.name.startswith("as_paths"))
            ):
                self.denied.append(str(path))
                raise PermissionError("Held-out route labels remain closed: " + str(path))
        elif event in (
            "os.remove",
            "os.rmdir",
            "os.mkdir",
            "os.chmod",
            "os.chown",
            "os.truncate",
            "os.utime",
        ):
            writes = [values[0]]
        elif event in ("os.rename", "os.link", "os.symlink"):
            writes = list(values[:2])
        for name in writes:
            if not isinstance(name, (str, bytes, os.PathLike)):
                continue
            target = os.fsdecode(name)
            if Path(target).resolve() == Path(os.devnull).resolve():
                # /dev/null is a discard sink opened by the stdlib (e.g.
                # subprocess.DEVNULL); it is not a run-boundary escape.
                continue
            if not inside(target, self.run):
                self.denied.append(str(name))
                raise PermissionError("Write outside isolated run: " + str(name))
