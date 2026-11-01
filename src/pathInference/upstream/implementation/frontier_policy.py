"""Unified255 frontier policy computations."""
from collections import deque
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FrontierPolicy:
    valid_tail_count_min: int = 300
    max_prefix_AS_count: int = 4
    next_hop_topk: int = 10


def plan_frontiers(source, roots, capacity, predict_next, policy=None):
    """Return emitted prefixes plus every stop/expansion decision.

    capacity(prefix,threshold) -> {valid_tail_count:int, count_is_lower_bound:bool}
    predict_next(prefix) -> {candidates:[{asn,admitted, ...}],...}
    candidates must already be in source-isolated predicted rank order. Caller
    remains responsible for evidence checks on every newly joined full path.
    """
    policy = policy or FrontierPolicy()
    source = str(source)
    queue = deque(((source, str(n)) for n in roots))
    visited = set()
    emitted = []
    decisions = []
    while queue:
        prefix = queue.popleft()
        if prefix in visited:
            continue
        visited.add(prefix)
        if len(prefix) != len(set(prefix)):
            raise ValueError("Cyclic prefix")
        current = capacity(prefix, policy.valid_tail_count_min)
        count = int(current["valid_tail_count"])
        if count < 0:
            raise ValueError("Negative safe-tail capacity")
        decision = dict(prefix=prefix, capacity=current)
        if count:
            emitted.append(prefix)
        if count >= policy.valid_tail_count_min:
            decision["stop_reason"] = "adequate_valid_forward_tails"
        elif len(prefix) >= policy.max_prefix_AS_count:
            decision["stop_reason"] = "depth_limit_with_insufficient_tail_evidence"
        else:
            prediction = predict_next(prefix)
            ranked = []
            discarded = []
            for entry in prediction.get("candidates", []):
                n = str(entry["asn"])
                if n in prefix:
                    discarded.append(dict(asn=n, reason="already_visited"))
                    continue
                if not entry.get("admitted", False):
                    discarded.append(dict(asn=n, reason="unsupported_or_incompatible_edge"))
                    continue
                if n not in ranked:
                    ranked.append(n)
            children = ranked[: policy.next_hop_topk]
            queue.extend((prefix + (n,) for n in children))
            decision.update(
                expanded_neighbors=children,
                discarded=discarded,
                prediction_scope=prediction.get("scope", "local_next_hop_proxy"),
                stop_reason=None if children else "no_supported_unvisited_next_hop",
            )
        decisions.append(decision)
    return dict(
        source=source,
        policy=asdict(policy),
        prefixes=emitted,
        decisions=decisions,
        source_total_degree_not_used=True,
        no_task_reference_access=True,
        insufficient_leaves=[
            d["prefix"]
            for d in decisions
            if d.get("stop_reason")
            in ("depth_limit_with_insufficient_tail_evidence", "no_supported_unvisited_next_hop")
        ],
    )
