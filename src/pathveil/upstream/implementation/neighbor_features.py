"""Unified255 neighbor features computations."""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
import numpy as np


SEED = 20261021


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path, source=None):
    path = Path(path)
    stat = path.stat()
    result = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if source is not None:
        result["source"] = int(source)
    return result


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def parse_path(line):
    try:
        nodes = tuple((int(part) for part in line.strip().split("->")))
    except ValueError:
        return None
    if len(nodes) < 2 or any((n <= 0 or n >= 2**32 for n in nodes)):
        return None
    if len(set(nodes)) != len(nodes):
        return None
    return nodes


def read_reference(path, source):
    by_dest = defaultdict(Counter)
    seen = set()
    counts = Counter()
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for raw in handle:
            sha.update(raw)
            nodes = parse_path(raw.decode().strip())
            if nodes is None or nodes[0] != source:
                counts["excluded"] += 1
                continue
            if nodes in seen:
                counts["duplicates"] += 1
                continue
            seen.add(nodes)
            by_dest[nodes[-1]][nodes[1], nodes[2] if len(nodes) >= 3 else 0] += 1
            counts["paths"] += 1
    (first, joint) = (defaultdict(float), defaultdict(float))
    d3 = 0
    for counter in by_dest.values():
        total = sum(counter.values())
        for (n, a), count in counter.items():
            first[n] += count / total
        eligible = {(n, a): count for ((n, a), count) in counter.items() if a}
        if not eligible:
            continue
        d3 += 1
        total3 = sum(eligible.values())
        for (n, a), count in eligible.items():
            joint[n, a] += count / total3
    anchor = defaultdict(dict)
    first_joint_mass = Counter()
    for (n, a), weight in joint.items():
        first_joint_mass[n] += weight
    for (n, a), weight in joint.items():
        anchor[n][str(a)] = weight / first_joint_mass[n]
    destinations = len(by_dest)
    return dict(
        source=source,
        path=str(path),
        sha256=sha.hexdigest(),
        counts=dict(counts),
        destinations=destinations,
        destinations_ge3=d3,
        first={str(n): weight / destinations for (n, weight) in first.items()},
        joint={f"{n}:{a}": weight / d3 for ((n, a), weight) in joint.items()},
        anchors={str(n): weights for (n, weights) in anchor.items()},
    )


def build_evidence(manifest, out):
    targets = {r["source"] for r in manifest["tasks"]}
    (edges, edge_sources, node_occurrences, starts, triples) = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    adjacency = defaultdict(set)
    canonical = manifest["canonical_evidence_input"]["path"]
    (sha, source_shas) = (hashlib.sha256(), {})
    (source_lines, excluded, withheld_lines) = (Counter(), Counter(), Counter())
    allowed_sources = {item["source"] for item in manifest["evidence"]}
    (total, line_number) = (0, 0)
    (last_source, local_edges, closed_sources) = (None, set(), set())
    with Path(canonical).open("rb") as handle:
        for raw in handle:
            line_number += 1
            sha.update(raw)
            try:
                source = int(raw.split(b"->", 1)[0])
            except ValueError:
                excluded["malformed_source"] += 1
                continue
            if source in targets:
                withheld_lines[source] += 1
                if source not in source_shas:
                    source_shas[source] = hashlib.sha256()
                source_shas[source].update(raw)
                continue
            assert source in allowed_sources, f"Unknown canonical source {source}"
            if last_source != source:
                if last_source is not None:
                    edge_sources.update(local_edges)
                    closed_sources.add(last_source)
                assert (
                    source not in closed_sources
                ), "Canonical stream is not source-grouped; distinct-source counts require a different accumulator"
                local_edges = set()
                last_source = source
            nodes = parse_path(raw.decode().strip())
            if nodes is None:
                excluded[source] += 1
                continue
            source_lines[source] += 1
            total += 1
            starts[source] += 1
            node_occurrences.update(nodes)
            for u, v in zip(nodes, nodes[1:]):
                edges[u, v] += 1
                local_edges.add((u, v))
                adjacency[u].add(v)
                adjacency[v].add(u)
            for s, n, a in zip(nodes, nodes[1:], nodes[2:]):
                if s in targets or a in targets:
                    triples[s, n, a] += 1
            if line_number % 1000000 == 0:
                log(f"evidence canonical {line_number:,} lines; {total:,} retained paths")
                dump(
                    out / "progress.json",
                    dict(stage="evidence", canonical_lines=line_number, retained_paths=total),
                )
    edge_sources.update(local_edges)
    result = dict(
        edges=edges,
        edge_sources=edge_sources,
        node_occurrences=node_occurrences,
        starts=starts,
        triples=triples,
        adjacency=dict(adjacency),
    )
    with (out / "evidence.pkl").open("wb") as handle:
        pickle.dump(result, handle, protocol=5)
    dump(
        out / "evidence_audit.json",
        dict(
            canonical_path=canonical,
            canonical_sha256=sha.hexdigest(),
            total_lines=line_number,
            source_path_counts=dict(source_lines),
            excluded=dict(excluded),
            withheld_lines=dict(withheld_lines),
            withheld_source_sha256={str(s): h.hexdigest() for (s, h) in source_shas.items()},
            total_paths=total,
            directed_edges=len(edges),
            withheld_start_counts={str(s): starts[s] for s in sorted(targets)},
        ),
    )
    return result


class FeatureBuilder:
    def __init__(self, public, evidence, cap=256):
        (self.public, self.evidence, self.cap) = (public, evidence, cap)
        self.columns = None
        (self.outcounts, self.incounts) = (Counter(), Counter())
        for (a, b), count in evidence["edges"].items():
            self.outcounts[a] += count
            self.incounts[b] += count
        self.pair = lru_cache(maxsize=30000)(public.pair_features)

    def candidates(self, source, center):
        residual = self.evidence["adjacency"].get(center, set())
        external = self.public.adjacency.get(center, set())
        candidates = (residual | external) - {source, center}

        def priority(v):
            return (
                -self.evidence["edges"][center, v],
                -self.public.links.get((center, v), {}).get("number_paths", 0),
                -self.evidence["edges"][v, center],
                v,
            )

        return (sorted(candidates, key=priority)[: self.cap], len(candidates))

    def query(self, source, center):
        (candidates, uncapped) = self.candidates(source, center)
        rows = []
        ev = self.evidence
        context_pair = self.pair(source, center) if source != center else {}
        for v in candidates:
            edges = ev["edges"]
            features = dict(self.pair(center, v))
            for key, value in context_pair.items():
                bare = key.split("_", 1)[1]
                namespace = key.split("_", 1)[0]
                features[f"{namespace}_prefix_{bare}"] = value
            if source == center:
                for key in self.pair(source, v):
                    bare = key.split("_", 1)[1]
                    features[f"{key.split('_', 1)[0]}_prefix_{bare}"] = 0.0
            features.update(
                {
                    "route_recursive": float(source != center),
                    "route_source_visible_degree": math.log1p(len(ev["adjacency"].get(source, ()))),
                    "route_center_visible_degree": math.log1p(len(ev["adjacency"].get(center, ()))),
                    "route_candidate_visible_degree": math.log1p(len(ev["adjacency"].get(v, ()))),
                    "route_candidate_occurrences": math.log1p(ev["node_occurrences"][v]),
                    "route_candidate_start_paths": math.log1p(ev["starts"][v]),
                    "route_center_occurrences": math.log1p(ev["node_occurrences"][center]),
                    "route_edge_forward": math.log1p(edges[center, v]),
                    "route_edge_reverse": math.log1p(edges[v, center]),
                    "route_edge_forward_share": edges[center, v] / max(1, self.outcounts[center]),
                    "route_edge_reverse_share": edges[v, center] / max(1, self.incounts[center]),
                    "route_edge_evidence_sources": math.log1p(ev["edge_sources"][center, v]),
                    "route_reverse_evidence_sources": math.log1p(ev["edge_sources"][v, center]),
                    "route_prefix_forward": math.log1p(ev["triples"][source, center, v])
                    if source != center
                    else 0.0,
                    "route_prefix_reverse": math.log1p(ev["triples"][v, center, source])
                    if source != center
                    else 0.0,
                    "route_prefix_edge_forward": math.log1p(edges[source, center])
                    if source != center
                    else 0.0,
                    "route_prefix_edge_reverse": math.log1p(edges[center, source])
                    if source != center
                    else 0.0,
                    "route_observed_graph_edge": float(v in ev["adjacency"].get(center, ())),
                    "route_candidates_uncapped": math.log1p(uncapped),
                }
            )
            rows.append(features)
        if self.columns is None and rows:
            self.columns = sorted(rows[0])
        for row in rows:
            assert set(row) == set(
                self.columns
            ), "Feature schema drift between root and recursive query"
        matrix = np.asarray([[row[key] for key in self.columns] for row in rows], dtype=np.float32)
        if not rows:
            matrix = np.zeros((0, len(self.columns)), dtype=np.float32)
        return (candidates, matrix, uncapped)
