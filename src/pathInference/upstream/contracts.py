"""Small, dependency-free contracts shared by the dated upstream stages."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


class UpstreamError(RuntimeError):
    """A missing stage or inconsistent research input; never a fallback signal."""


def validate_date(value):
    value = str(value)
    if len(value) != 8 or not value.isascii() or not value.isdecimal():
        raise UpstreamError("Date must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise UpstreamError("Invalid calendar date: " + value) from exc
    return value


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise UpstreamError(message)


def verify(mapping):
    for path, expected in mapping.items():
        require(
            Path(path).is_file() and sha(path) == expected,
            "Frozen input changed or missing: " + str(path),
        )


def path_rows(path, source=None):
    result = []
    seen = set()
    with Path(path).open(newline="") as handle:
        for number, row in enumerate(csv.reader(handle, strict=True), 1):
            require(len(row) == 1, f"Expected one path column: {path}:{number}")
            nodes = tuple(row[0].split("->"))
            require(
                2 <= len(nodes) <= 20
                and len(nodes) == len(set(nodes))
                and all(n.isascii() and n.isdecimal() and 0 < int(n) < 2**32 for n in nodes),
                f"Malformed path: {path}:{number}",
            )
            require(source is None or nodes[0] == str(source), "Path has wrong original source")
            require(nodes not in seen, "Duplicate route in released pool")
            result.append(nodes)
            seen.add(nodes)
    return result


def write_paths(path, paths):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows([["->".join(map(str, p))] for p in paths])


def validate_model_scope(bundle, date, split):
    """A later model or another month's evidence cannot become a local model."""
    date = validate_date(date)
    require(
        bundle.get("date") == date, "Model/evidence bundle date differs from the requested date"
    )
    require(
        bundle.get("model_fit_date") == date,
        "Date-local protocol requires models fitted on this same date",
    )
    require(
        bundle.get("protocol") == "date_local_source_disjoint_v1",
        "Only explicit date-local source-disjoint models are accepted",
    )
    require(
        bundle.get("formal_labels_read") is False,
        "Model bundle must declare no formal reference access",
    )
    train = set(map(str, split["train"]))
    formal = set(map(str, split["formal_test"]))
    validation = set(map(str, split["validation"]))
    require(not train & (formal | validation), "Source roles overlap")
    require(
        set(map(str, bundle["training_sources"])) == train,
        "Fitted model source identities differ from the frozen split",
    )
    folds = bundle["source_to_fold"]
    require(
        set(map(str, folds)) == train and set(folds.values()) == set(range(5)),
        "Exactly five source-disjoint training folds are required",
    )
    verify(bundle.get("inputs_sha256", {}))
    verify(bundle.get("model_files_sha256", {}))


class ReferenceBoundary:
    """CPython open-event guard around label-free inference, not syscall isolation."""

    def __init__(self, allowed_training=()):
        self.allowed = {str(Path(p).resolve()) for p in allowed_training}
        self.enabled = False
        self.events = []

    def check(self, event, args):
        if (
            not self.enabled
            or event != "open"
            or not args
            or not isinstance(args[0], (str, bytes, os.PathLike))
        ):
            return
        mode = args[1]
        reading = (
            ("r" in mode or "+" in mode)
            if isinstance(mode, str)
            else (int(args[2]) & os.O_ACCMODE) != os.O_WRONLY
        )
        if not reading:
            return
        path = Path(os.fsdecode(args[0])).resolve()
        reference = (
            "path_without_prefix" in path.parts
            or path.name.startswith("references_")
            or path.name == "labels_summary.json"
        )
        if reference:
            allowed = str(path) in self.allowed
            self.events.append(dict(path=str(path), allowed=allowed))
            require(allowed, "Inference cannot read source reference: " + str(path))
