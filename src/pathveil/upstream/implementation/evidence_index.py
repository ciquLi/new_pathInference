"""Unified255 evidence index computations."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from .neighbor_features import parse_path


def build_index(stream, targets, allowed_sources, progress=None):
    """All observed forward contiguous segments, raw multiplicity and origin count."""
    targets = set(map(int, targets))
    allowed_sources = set(map(int, allowed_sources))
    index = defaultdict(Counter)
    support = defaultdict(Counter)
    rawsha = hashlib.sha256()
    withheld = Counter()
    withheld_sha = {}
    source_counts = Counter()
    excluded = Counter()
    nlines = retained = 0
    last = None
    closed = set()
    local = set()

    def flush():
        for left, text in local:
            support[left][text] += 1

    for raw in stream:
        nlines += 1
        rawsha.update(raw)
        try:
            source = int(raw.split(b"->", 1)[0])
        except ValueError:
            excluded["malformed_source"] += 1
            continue
        if source in targets:
            withheld[source] += 1
            withheld_sha.setdefault(source, hashlib.sha256()).update(raw)
            continue
        if source not in allowed_sources:
            raise AssertionError("Unknown canonical origin " + str(source))
        if source != last:
            if last is not None:
                flush()
                closed.add(last)
            if source in closed:
                raise AssertionError("Canonical stream must be source-grouped")
            last = source
            local = set()
        nodes = parse_path(raw.decode().strip())
        if nodes is None:
            excluded[str(source)] += 1
            continue
        retained += 1
        source_counts[source] += 1
        for start in range(len(nodes) - 1):
            left = str(nodes[start])
            parts = []
            for stop in range(start, len(nodes)):
                parts.append(str(nodes[stop]))
                if stop == start:
                    continue
                text = "->".join(parts)
                index[left][text] += 1
                local.add((left, text))
        if progress and nlines % 1000000 == 0:
            progress(nlines, retained, len(index), sum(map(len, index.values())))
    if last is not None:
        flush()
    return (
        dict(index=dict(index), origin_support=dict(support)),
        dict(
            canonical_sha256=rawsha.hexdigest(),
            total_lines=nlines,
            total_paths=retained,
            source_path_counts=dict(source_counts),
            withheld_lines=dict(withheld),
            withheld_source_sha256={str(s): h.hexdigest() for (s, h) in withheld_sha.items()},
            excluded=dict(excluded),
            index_sources=len(index),
            unique_segments=sum(map(len, index.values())),
            segment_raw_observations=sum((sum(c.values()) for c in index.values())),
            no_reverse_or_synthetic_segments=True,
            origin_support_definition="Distinct actual original canonical source ASes witnessing exact ordered segment; one vote per origin",
        ),
    )
