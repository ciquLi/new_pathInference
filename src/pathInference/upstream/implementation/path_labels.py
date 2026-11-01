"""Unified255 path labels computations."""
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np


def reference(path, source, expected_sha):
    h = hashlib.sha256()
    truth = set()
    excluded = 0
    with Path(path).open("rb") as stream:
        for raw in stream:
            h.update(raw)
            try:
                p = tuple((str(int(v)) for v in raw.decode().strip().split("->")))
            except ValueError:
                excluded += 1
                continue
            if (
                len(p) < 2
                or p[0] != str(source)
                or len(p) != len(set(p))
                or any((not 0 < int(v) < 2**32 for v in p))
            ):
                excluded += 1
                continue
            truth.add(p)
    assert h.hexdigest() == expected_sha, (source, "original reference bytes changed")
    return (
        truth,
        dict(path=str(path), sha256=h.hexdigest(), unique_paths=len(truth), excluded_rows=excluded),
    )


def balanced_weights(candidates, y):
    pairs = [
        (str(c["path"][1]), str(c["path"][-1])) for (c, label) in zip(candidates, y) if label >= 0
    ]
    sizes = Counter(pairs)
    destinations = Counter((n for (n, d) in sizes))
    return np.array(
        [
            1.0
            / (
                len(destinations)
                * destinations[str(c["path"][1])]
                * sizes[str(c["path"][1]), str(c["path"][-1])]
            )
            if label >= 0
            else 0.0
            for (c, label) in zip(candidates, y)
        ],
        dtype=np.float64,
    )
