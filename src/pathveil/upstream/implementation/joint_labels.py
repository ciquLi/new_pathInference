"""Unified255 joint labels computations."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from pathlib import Path


VERSION = "destination_equal_joint_length_train160_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def parse_reference_line(raw, source):
    """Same acceptance semantics as propagation_scoring.train_support.reference."""
    try:
        path = tuple((str(int(v)) for v in raw.decode().strip().split("->")))
    except (UnicodeDecodeError, ValueError):
        return None
    if (
        len(path) < 2
        or path[0] != str(source)
        or len(path) != len(set(path))
        or any((not 0 < int(v) < 2**32 for v in path))
    ):
        return None
    return path


def read_training_reference(path, source, expected_sha):
    truth = set()
    excluded = total = 0
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in stream:
            h.update(raw)
            total += 1
            route = parse_reference_line(raw, source)
            if route is None:
                excluded += 1
            else:
                truth.add(route)
    if h.hexdigest() != expected_sha:
        raise ValueError(f"Original training bytes changed: {source}")
    return (
        truth,
        dict(
            source=int(source),
            input=str(path),
            sha256=expected_sha,
            raw_rows=total,
            excluded_rows=excluded,
            unique_paths=len(truth),
        ),
    )


def length_bin(length):
    return str(length) if length <= 20 else "21+"


def summarize_context(paths, next_index, minimum_length=3):
    """Each destination contributes one unit, split among its unique routes.

    Length is the TOTAL source-origin AS count. Suffix length is also retained
    so callers evaluating fixed recursive prefixes need not call them collapse.
    Empty contexts remain explicit; no uniform distribution is invented.
    """
    by_dest = defaultdict(list)
    for path in paths:
        if len(path) >= minimum_length and len(path) > next_index:
            by_dest[path[-1]].append(path)
    branch_mass = Counter()
    hist = Counter()
    joint = defaultdict(Counter)
    suffix_hist = Counter()
    mean = 0.0
    for routes in by_dest.values():
        weight = 1.0 / (len(by_dest) * len(routes))
        for path in routes:
            branch = path[next_index]
            length = length_bin(len(path))
            branch_mass[branch] += weight
            hist[length] += weight
            joint[branch][length] += weight
            suffix_hist[length_bin(len(path) - next_index + 1)] += weight
            mean += weight * len(path)
    counts = Counter((p[next_index] for routes in by_dest.values() for p in routes))
    destinations = Counter()
    for routes in by_dest.values():
        destinations.update({p[next_index] for p in routes})
    return dict(
        destinations=len(by_dest),
        paths=sum(map(len, by_dest.values())),
        branch_mass=dict(branch_mass),
        length_mass=dict(hist),
        branch_length_mass={b: dict(v) for (b, v) in joint.items()},
        suffix_length_mass=dict(suffix_hist),
        mean_length=mean if by_dest else None,
        branch_path_counts=dict(counts),
        branch_destination_counts=dict(destinations),
        target_minimum_total_length=minimum_length,
        next_index=next_index,
        unknown=not bool(by_dest),
    )


def source_labels(truth):
    truth = sorted(truth)
    groups = defaultdict(list)
    for path in truth:
        groups[path[1]].append(path)
    return dict(
        root=summarize_context(truth, 1),
        root_including_two_as=summarize_context(truth, 1, 2),
        conditional={
            n: summarize_context(paths, 2)
            for (n, paths) in groups.items()
            if any((len(p) >= 3 for p in paths))
        },
    )
