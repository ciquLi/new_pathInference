"""Unified255 construction budget computations."""
from collections import defaultdict
from .destination_budget import destination_order


def select_valid_candidates(source, candidates, budget=5000, direct_sparse_max=1000, reserve=50):
    contexts = defaultdict(set)
    for c in candidates:
        contexts[tuple(c["prefix"])].add(c["dest"])
    available = set().union(*contexts.values()) if contexts else set()
    selected = (
        set(sorted(available, key=lambda d: destination_order(source, d))[:budget])
        if budget
        else available.copy()
    )
    reservoirs = {}
    for prefix, destinations in contexts.items():
        preserve_all = len(prefix) == 2 and len(destinations) <= direct_sparse_max
        chosen = (
            destinations.copy()
            if preserve_all
            else set(sorted(destinations, key=lambda d: destination_order(source, d))[:reserve])
        )
        reservoirs["->".join(prefix)] = dict(
            kind="direct" if len(prefix) == 2 else "explicit_recursive",
            valid_destination_capacity=len(destinations),
            preserve_all=preserve_all,
            selected=sorted(chosen, key=int),
        )
        selected.update(chosen)
    indices = [i for (i, c) in enumerate(candidates) if c["dest"] in selected]
    return (
        indices,
        dict(
            policy="Source hashD5000 + independently protected direct construction prefixes<=1000D +50D per explicit planned prefix; natural tail next hops are not construction contexts",
            source=str(source),
            budget=budget,
            direct_sparse_destination_max=direct_sparse_max,
            reserve=reserve,
            construction_contexts=reservoirs,
            source_valid_destination_capacity=len(available),
            selected_destination_ids=sorted(selected, key=int),
            selected_destination_count=len(selected),
            candidate_count_before_budget=len(candidates),
            candidate_count_after_budget=len(indices),
            no_reference_destinations_used=True,
        ),
    )
