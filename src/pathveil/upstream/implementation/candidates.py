"""Unified255 candidates computations."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math


KNOWN = {"c2p", "p2p", "p2c", "s2s"}


def relation(relations, left, right):
    (left, right) = (str(left), str(right))
    value = relations.get((int(left), int(right)), relations.get((left, right)))
    if value is None:
        value = relations.get(left, {}).get(right)
    if isinstance(value, int):
        value = {1: "c2p", 0: "p2p", -1: "p2c", 2: "s2s"}.get(value)
    return value if value in KNOWN else "unknown"


def valley_compatible(roles):
    """True iff at least one assignment of missing roles is valley-free."""
    states = {"up"}
    for role in roles:
        possibilities = KNOWN if role == "unknown" else (role,)
        following = set()
        for state in states:
            for current in possibilities:
                if current == "s2s":
                    following.add(state)
                elif state == "up":
                    following.add({"c2p": "up", "p2p": "post_peer", "p2c": "down"}[current])
                elif current == "p2c":
                    following.add("down")
        states = following
        if not states:
            return False
    return True


def exact_support(index, path):
    return index.get(str(path[0]), {}).get("->".join(map(str, path)), 0)


def edge_evidence(index, relations, left, right, provenance=None):
    """Keep independent facts; never convert a co-location into adjacency."""
    provenance = provenance or {}
    known_relation = relation(relations, left, right)
    forward = exact_support(index, (left, right))
    reverse = exact_support(index, (right, left))
    probability = provenance.get("observed_support_probability")
    if probability is not None:
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Neighbor support probability must be in [0,1]")
    public_edge = bool(provenance.get("public_edge", False))
    colocated = bool(
        provenance.get("pdb_shared_ixp", 0) or provenance.get("pdb_shared_facility", 0)
    )
    observed_edge = bool(forward or reverse or public_edge or (known_relation != "unknown"))
    admitted = observed_edge or (colocated and probability is not None)
    return dict(
        left=str(left),
        right=str(right),
        relationship=known_relation,
        forward_observations=float(forward),
        reverse_observations=float(reverse),
        public_edge=public_edge,
        pdb_shared_ixp=provenance.get("pdb_shared_ixp", 0),
        pdb_shared_facility=provenance.get("pdb_shared_facility", 0),
        inferred_only=not observed_edge,
        observed_support_probability=probability,
        admitted=admitted,
    )


def candidate_from_tail(prefix, tail, frequency, index, relations, edge_provenance=None):
    (prefix, tail) = (tuple(map(str, prefix)), tuple(map(str, tail)))
    if len(prefix) < 2 or len(tail) < 2 or prefix[-1] != tail[0]:
        return (None, "bad_join")
    path = prefix + tail[1:]
    if not 3 <= len(path) <= 20:
        return (None, "length")
    if len(path) != len(set(path)):
        return (None, "loop")
    if any((not n.isdigit() or not 0 < int(n) < 2**32 for n in path)):
        return (None, "asn")
    if exact_support(index, tail) <= 0:
        return (None, "tail_not_observed_forward")
    edge_provenance = edge_provenance or {}
    proofs = [
        edge_evidence(index, relations, a, b, edge_provenance.get((a, b), {}))
        for (a, b) in zip(prefix, prefix[1:])
    ]
    if any((not proof["admitted"] for proof in proofs)):
        return (None, "unsupported_prefix_edge")
    roles = [relation(relations, a, b) for (a, b) in zip(path, path[1:])]
    if not valley_compatible(roles):
        return (None, "known_commercial_violation")
    first_known = next(
        (i for i in range(len(path) - 1) if exact_support(index, path[i:]) > 0), len(path) - 1
    )
    return (
        dict(
            path=path,
            prefix=prefix,
            source=path[0],
            firsthop=path[1],
            first_hop=path[1],
            neighbor=path[1],
            dest=path[-1],
            destination=path[-1],
            second_hop=path[2],
            relationship=roles[0],
            second_hop_relationship=roles[1],
            segment_len=len(prefix),
            tail_len=len(tail),
            tail_path_freq=float(frequency),
            tail_source=tail[0],
            generation_source="direct" if len(prefix) == 2 else "recursive",
            relation_validation="fully_known" if "unknown" not in roles else "partial_compatible",
            prefix_relationship_validated=all((p["relationship"] != "unknown" for p in proofs)),
            inferred_prefix_edges=sum((p["inferred_only"] for p in proofs)),
            prefix_edge_evidence=proofs,
            unknown_prefix_extensions=first_known,
            novel=not bool(exact_support(index, path)),
            as_count=len(path),
        ),
        None,
    )


@dataclass(frozen=True)
class CandidatePolicy:
    per_neighbor_destination_length_cap: int = 3
    max_length: int = 20


def evidence_priority(candidate, index, origins):
    (path, prefix) = (candidate["path"], candidate["prefix"])
    suffix = "->".join(path[1:])
    tail = "->".join(path[len(prefix) - 1 :])
    return (
        candidate["unknown_prefix_extensions"],
        len(prefix),
        -int(origins.get(path[1], {}).get(suffix, 0)),
        -int(origins.get(prefix[-1], {}).get(tail, 0)),
        -candidate["tail_path_freq"],
        tuple(map(int, path)),
    )


def collect_source(
    source,
    prefixes,
    index,
    origin_support,
    relations,
    feature_builder=None,
    edge_provenance=None,
    policy=None,
):
    """Collect full ordered paths, capping within (N,A,D,AS-count).

    Prefixes are tuples S,N or S,N,A (longer explicit chains also supported).
    No task references, global disk fallback, tail reversal, global quantile,
    shortest-per-destination selection, or labels are used.
    """
    policy = policy or CandidatePolicy()
    source = str(source)
    stats = Counter()
    unique = {}
    for prefix in sorted(
        set((tuple(map(str, p)) for p in prefixes)), key=lambda p: tuple(map(int, p))
    ):
        if prefix[0] != source:
            raise ValueError("Prefix source mismatch")
        stats["prefixes"] += 1
        for text, frequency in index.get(prefix[-1], {}).items():
            stats["forward_tail_visits"] += 1
            tail = tuple(text.split("->"))
            proofs = edge_provenance(prefix) if callable(edge_provenance) else edge_provenance
            (candidate, rejected) = candidate_from_tail(
                prefix, tail, frequency, index, relations, proofs
            )
            if rejected:
                stats["rejected_" + rejected] += 1
                continue
            stats["valid_tail_visits"] += 1
            path = candidate["path"]
            old = unique.get(path)
            if old is None or evidence_priority(
                candidate, index, origin_support
            ) < evidence_priority(old, index, origin_support):
                unique[path] = candidate
    groups = defaultdict(list)
    for candidate in unique.values():
        candidate["conditional_key"] = candidate["neighbor"] + ">" + candidate["path"][2]
        groups[
            candidate["neighbor"], candidate["path"][2], candidate["dest"], len(candidate["path"])
        ].append(candidate)
    retained = []
    for key, rows in sorted(
        groups.items(),
        key=lambda item: (int(item[0][0]), int(item[0][1]), int(item[0][2]), item[0][3]),
    ):
        rows.sort(key=lambda c: evidence_priority(c, index, origin_support))
        retained.extend(rows[: policy.per_neighbor_destination_length_cap])
    for i, candidate in enumerate(retained):
        candidate["row_index"] = i
    stats.update(
        unique_before_cap=len(unique),
        retained_candidates=len(retained),
        removed_by_cap=len(unique) - len(retained),
        length_groups=len(groups),
        inferred_prefix_candidates=sum((bool(c["inferred_prefix_edges"]) for c in retained)),
        novel_candidates=sum((c["novel"] for c in retained)),
    )
    if feature_builder is None:
        return (None, retained, dict(stats))
    import numpy as np

    names = tuple(feature_builder.feature_names)
    matrix = np.empty((len(retained), len(names)), dtype=np.float32)
    for i, candidate in enumerate(retained):
        values = feature_builder.features(candidate)
        matrix[i] = [values[name] for name in names]
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite path features")
    return (matrix, retained, dict(stats))
