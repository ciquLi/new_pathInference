"""Unified255 role labels computations."""
import numpy as np


def role_labels(queries, refs):
    known = []
    transit = []
    sources = []
    weights = []
    for q in queries:
        ref = refs[str(q["source"])]
        first = ref["first"]
        anchors = ref["anchors"]
        mask = [str(n) in first for n in q["candidates"]]
        total = max(1, sum(mask))
        known.extend(mask)
        transit.extend((int(str(n) in anchors) for n in q["candidates"]))
        sources.extend([q["source"]] * q["size"])
        weights.extend([1 / total] * q["size"])
    return (
        np.asarray(known, dtype=bool),
        np.asarray(transit, dtype=np.int32),
        np.asarray(sources),
        np.asarray(weights),
    )
