"""Frozen score normalization, inclusive admission and immutable artifacts."""
from collections import defaultdict
from contextlib import contextmanager
import csv
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import sys
import numpy as np
from pathveil.utils import read_json, sha256, write_json

DUPLICATE_POLICY = "maximum_score_preserve_first_position"


class AdmissionStopped(ValueError):
    """A frozen non-PASS source result cannot be consumed as an admitted library."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(mapping):
    for name, expected in mapping.items():
        require(
            Path(name).is_file() and sha256(name) == expected,
            "Frozen admission input changed: " + str(name),
        )


@contextmanager
def forbid_formal_references():
    """Scope the audit hook to this call; never poison a caller's later stages."""
    forbidden, enabled = set(), [True]

    def guard(event, arguments):
        if not enabled[0] or event != "open" or not arguments:
            return
        name = arguments[0]
        if not isinstance(name, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(name)).resolve()
        if (
            str(path) in forbidden
            or ("path_without_prefix" in path.parts and "test" in path.parts)
            or path.name in {"pair_predictions.csv", "evaluation_summary.json"}
            or ("measurement" in path.parts and path.name.startswith("as_paths"))
        ):
            raise PermissionError(
                "Formal reference/prediction access forbidden in admission: " + str(path)
            )

    sys.addaudithook(guard)
    try:
        yield forbidden
    finally:
        enabled[0] = False


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def path_bytes(paths):
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows([["->".join(path)] for path in paths])
    grouped = defaultdict(list)
    for path in paths:
        grouped[path[0] + "-" + path[-1]].append(list(path))
    return stream.getvalue().encode("utf-8"), json_bytes(dict(grouped))


def freeze_bytes(path, value):
    """Resume accepts identical ordinary bytes, including after a partial publish."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        require(
            not path.is_symlink() and path.is_file() and path.stat().st_nlink == 1,
            "Admission artifacts may not be links: " + str(path),
        )
        require(path.read_bytes() == value, "Frozen admission artifact differs: " + str(path))
        return
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(value)
    temporary.replace(path)


def quality_selection(paths, raw_scores, threshold, minimum_paths=None):
    relative = source_relative_scores(paths, raw_scores)
    require(len(relative) == len(paths), "Source-relative score alignment differs")
    original_count = sum(value >= threshold for value in relative)
    effective = threshold
    if minimum_paths is not None:
        require(
            isinstance(minimum_paths, int)
            and not isinstance(minimum_paths, bool)
            and minimum_paths > 0,
            "Minimum path count must be a positive integer",
        )
        if len(paths) < minimum_paths:
            raise AdmissionStopped(
                "INSUFFICIENT_LEGAL_POOL: fewer paths than the requested minimum"
            )
        cutoff = float(sorted(relative, reverse=True)[minimum_paths - 1])
        if cutoff <= 0:
            raise AdmissionStopped(
                "INSUFFICIENT_SCORED_PATHS: minimum would require zero-score paths"
            )
        effective = min(float(threshold), cutoff)
    chosen = [path for path, value in zip(paths, relative) if value >= effective]
    if not chosen:
        raise AdmissionStopped(
            "UNKNOWN_EMPTY_QUALITY_GATE: no quantity floor or automatic fallback"
        )
    return chosen, dict(
        calibrated_threshold=float(threshold),
        effective_threshold=float(effective),
        requested_minimum=minimum_paths,
        original_quality_paths=int(original_count),
        completion_paths=len(chosen) - int(original_count),
        cutoff_ties="retain all",
        final_quality_paths=len(chosen),
        threshold_relaxed=effective < threshold,
    )


class UnknownSourceScale(ValueError):
    """An empty or zero-maximum pool supplies no usable relative evidence."""


def source_relative(scores):
    require(
        all(not isinstance(v, bool) and isinstance(v, Real) for v in scores),
        "Invalid raw source scores",
    )
    values = list(map(float, scores))
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid raw source scores")
    maximum = max(values, default=0.0)
    if maximum <= 0:
        raise UnknownSourceScale("UNKNOWN_ZERO_MAX: refuse empty/zero source pool; no epsilon")
    return [v / maximum for v in values]


def source_relative_scores(paths, scores):
    """Normalize an already legal, deduplicated complete source pool in order."""
    require(len(paths) == len(scores), "Candidate/score alignment differs")
    return source_relative(scores)


def canonical_source(source):
    source = str(source)
    require(
        source.isascii()
        and source.isdecimal()
        and source == str(int(source))
        and 0 < int(source) < 2**32,
        "Invalid canonical source ASN",
    )
    return source


def valid_path(path, source):
    if not isinstance(path, (list, tuple, np.ndarray)):
        return False
    nodes = tuple(map(str, path))
    return (
        3 <= len(nodes) <= 20
        and nodes[0] == source
        and len(set(nodes)) == len(nodes)
        and all(
            n.isascii() and n.isdecimal() and 0 < int(n) < 2**32 and n == str(int(n))
            for n in nodes
        )
    )


def normalize_pool(source, raw_paths, raw_scores):
    """Keep legal CASE paths in first-row order; duplicates receive max score."""
    source = canonical_source(source)
    scores = np.asarray(raw_scores, dtype=np.float64)
    require(scores.shape == (len(raw_paths),), "CASE paths and scores are not row-aligned")
    require(
        np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
        "Unknown or invalid raw model scores may not enter a quality pool",
    )
    paths, values, positions = [], [], {}
    invalid = too_short = duplicates = 0
    for path, score in zip(raw_paths, scores):
        if isinstance(path, (list, tuple, np.ndarray)) and len(path):
            first = str(path[0])
            if first != source and valid_path(path, first):
                raise ValueError("A complete CASE pool may not mix legal source identities")
        if not valid_path(path, source):
            invalid += 1
            too_short += isinstance(path, (list, tuple, np.ndarray)) and len(path) < 3
            continue
        path = tuple(map(str, path))
        if path in positions:
            duplicates += 1
            index = positions[path]
            values[index] = max(values[index], float(score))
        else:
            positions[path] = len(paths)
            paths.append(path)
            values.append(float(score))
    values = np.asarray(values, dtype=np.float64)
    require(
        len(paths) and float(values.max()) > 0,
        "Complete legal model pool has no positive source maximum",
    )
    return (
        paths,
        values,
        dict(
            raw_rows=len(raw_paths),
            invalid_candidate_paths=invalid,
            too_short_candidate_paths=int(too_short),
            duplicate_candidate_rows=duplicates,
            known_model_score_paths=len(paths),
            unknown_model_score_paths=0,
            source_max=float(values.max()),
            duplicate_policy=DUPLICATE_POLICY,
            raw_key_order="original_CASE_row_order",
            pool_key_order="first_legal_CASE_occurrence",
        ),
    )


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    freeze_bytes(path, json_bytes(value))


def signature(paths):
    from pathveil.prefix_tree.prefix_tree_pathgen import path_set_signature

    return path_set_signature(paths)
