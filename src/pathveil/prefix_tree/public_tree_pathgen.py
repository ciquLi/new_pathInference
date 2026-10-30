"""Portable numerical API extracted from the verified Tree B implementation."""
from __future__ import annotations


import argparse


from collections import Counter, defaultdict


import csv


from functools import lru_cache


import hashlib


import importlib


import json


import os


from pathlib import Path


import resource


import sys


import threading


import time


from types import ModuleType, SimpleNamespace


from pathveil.utils import read_json, write_json, sha256, object_hash, utc_now


from pathveil.downstream import common


VERSION = "public_tree_HIGH_then_UNKNOWN_full_phase_PG_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def memory_record():
    rss = 0
    try:
        with Path("/proc/self/status").open() as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
    except OSError:
        pass
    return dict(
        rss_bytes=rss,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )


class Heartbeat:
    def __init__(self, source, stage, interval=30):
        self.source, self.stage, self.interval = source, stage, interval
        self.started = time.monotonic()
        self.state = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def emit(self):
        with self.lock:
            state = dict(self.state)
        print(
            json.dumps(
                dict(
                    event="public_tree_pathgen_progress",
                    source=self.source,
                    stage=self.stage,
                    elapsed_seconds=time.monotonic() - self.started,
                    utc=utc_now(),
                    **memory_record(),
                    **state,
                )
            ),
            flush=True,
        )

    def __enter__(self):
        def watch():
            while not self.stop.wait(self.interval):
                self.emit()

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        self.emit()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=1)
        self.emit()


def checked_leaves(spec, source, helpers, phases):
    require(
        str(spec["source"]) == source
        and spec["maximum_prefix_AS_count"] == 3
        and spec.get("no_formal_reference_paths") is True,
        "Wrong public spec source/depth/evidence boundary",
    )
    leaves, states = [], set()
    for layer in ("primary_leaves", "fallback_leaves"):
        for raw in spec[layer]:
            leaf = dict(raw)
            prefix = tuple(map(str, leaf["prefix"]))
            require(
                len(prefix) in (2, 3) and helpers.valid_path(prefix, source),
                "Invalid public prefix",
            )
            trace = tuple(leaf["phase_trace"])
            state = leaf["phase_state"]
            require(
                len(trace) == len(prefix)
                and trace[0] == "UP"
                and trace[-1] == state
                and all(x in {"UP", "DOWN"} for x in trace),
                "Invalid public prefix phase trace",
            )
            roles = leaf["public_roles"]
            require(
                len(roles) == len(prefix) - 1 and len(leaf["public_edge_evidence"]) == len(roles),
                "Missing public prefix roles",
            )
            for role, before, after in zip(roles, trace, trace[1:]):
                require(
                    after in phases([role], initial=(before,))["possible_phases"],
                    "Prefix phase transition is impossible",
                )
            key = (prefix, trace)
            require(key not in states, "Duplicate complete prefix/phase state")
            states.add(key)
            leaf.update(prefix=prefix, phase_trace=trace)
            require(
                leaf["tier"]
                in {
                    "HIGH",
                    "fully_public_observed_commercial_compatible",
                    "UNKNOWN",
                    "public_unknown_or_role_conflict",
                },
                "Unsupported leaf evidence tier",
            )
            require(
                layer != "fallback_leaves"
                or leaf["tier"] in {"UNKNOWN", "public_unknown_or_role_conflict"},
                "Fallback leaf cannot displace the public HIGH layer",
            )
            leaves.append(leaf)
    require(leaves, "Public spec has no usable prefix states")
    return leaves


def generate_public_candidates(
    env,
    k,
    source,
    destinations,
    spec,
    selector,
    topology,
    helpers,
    phases,
    shortest_paths,
    library,
    legacy_generator=None,
    progress=None,
    query_trace=None,
):
    """Injectable, streaming per-anchor generator; returns PG dictionary + audit."""
    source = str(source)
    destinations = list(dict.fromkeys(map(str, destinations)))
    require(k > 0 and k <= 200, "Use the registered per-query candidate capacity")
    library = [tuple(map(str, p)) for p in library]
    require(
        len(library) == len(set(library)) and all(helpers.valid_path(p, source) for p in library),
        "Invalid or duplicated admitted library",
    )
    require(
        helpers.path_set_signature(library) == spec["admitted_path_set_sha256"],
        "Public spec and actual PG R_s differ",
    )
    leaves = checked_leaves(spec, source, helpers, phases)
    classifications = {tuple(map(str, r["prefix"])) for r in spec["prefix_classification"]}
    require(
        all(leaf["prefix"] in classifications for leaf in leaves),
        "Leaf has no physical-prefix classification",
    )
    counts = Counter()
    banks = {}
    high_destinations = set()
    started = time.monotonic()
    attempted_prefix_states = Counter()
    accepted_prefix_states = Counter()

    def update(**values):
        if progress is not None:
            progress.update(**values)

    def encodable(path):
        return all(node in env.stoi and env.stoi[node] != 0 for node in path)

    @lru_cache(maxsize=131072)
    def role(left, right):
        value = topology.public_role(left, right)
        require(
            value in {"c2p", "p2c", "p2p", "s2s", "unknown", "conflict"},
            "Unexpected public relationship role",
        )
        return value

    # Revalidate head proofs against the same full public graph used below.
    for leaf in leaves:
        for (left, right), evidence in zip(
            zip(leaf["prefix"], leaf["prefix"][1:]), leaf["public_edge_evidence"]
        ):
            actual = topology.edge_evidence(left, right)
            require(
                str(evidence["left"]) == left
                and str(evidence["right"]) == right
                and evidence["public_role"] == actual["public_role"]
                and evidence["public_edge_observed"] == actual["public_edge_observed"],
                "Spec public proof differs from current graph",
            )

    def add(destination, raw, origin, leaf=None):
        counts["candidate_attempts"] += 1
        path = tuple(map(str, raw))
        if not helpers.valid_path(path, source, destination):
            counts["rejected_loop_endpoint_ASN_or_length"] += 1
            return
        if not encodable(path):
            counts["rejected_unencodable_candidate"] += 1
            return
        if not any(path[:n] in classifications for n in (2, 3) if len(path) >= n):
            counts["rejected_outside_explicit_prefix_tree"] += 1
            return
        if leaf is not None and path[: len(leaf["prefix"])] != leaf["prefix"]:
            counts["rejected_prefix_mismatch"] += 1
            return
        roles = [role(a, b) for a, b in zip(path, path[1:])]
        policy = phases(roles)
        if policy["status"] == "incompatible":
            counts["rejected_full_path_commercial_incompatible"] += 1
            return
        if leaf is not None:
            continuation = phases(roles[len(leaf["prefix"]) - 1 :], initial=(leaf["phase_state"],))
            if continuation["status"] == "incompatible":
                counts["rejected_leaf_phase_continuation"] += 1
                return
        tier = selector.classify_path(path)
        if tier == "INCOMPATIBLE":
            counts["rejected_head_commercial_incompatible"] += 1
            return
        pair = source + "-" + destination
        bank = banks.get(pair)
        if bank is None:
            bank = banks[pair] = selector.new_query(k)
        bank.add(path)
        if bank.has_high:
            high_destinations.add(destination)
        counts["accepted_" + tier] += 1
        counts["accepted_origin_" + origin] += 1
        counts["accepted_full_phase_" + policy["status"]] += 1
        if any(r in {"unknown", "conflict"} for r in roles):
            counts["accepted_with_unknown_or_conflicting_relationship"] += 1
        if "unknown" in roles:
            counts["accepted_with_unknown_relationship"] += 1
        if "conflict" in roles:
            counts["accepted_with_conflicting_relationship"] += 1
        if leaf is not None:
            accepted_prefix_states[
                leaf.get("state_key", "->".join(leaf["prefix"]) + "|" + leaf["phase_state"])
            ] += 1

    def generate_layer(layer, selected):
        by_anchor = defaultdict(list)
        for leaf in selected:
            if not encodable(leaf["prefix"]):
                counts["rejected_unencodable_prefix_state"] += 1
                continue
            by_anchor[leaf["prefix"][-1]].append(leaf)
        update(
            layer=layer,
            anchors_total=len(by_anchor),
            anchors_completed=0,
            prefix_states_total=len(selected),
            high_queries=len(high_destinations),
        )
        for anchor_index, (anchor, anchor_leaves) in enumerate(by_anchor.items()):
            eligible = [d for d in destinations if layer == "HIGH" or d not in high_destinations]
            if not eligible:
                break
            update(
                layer=layer,
                anchor=anchor,
                anchor_index=anchor_index + 1,
                anchors_completed=anchor_index,
                destinations_for_anchor=len(eligible),
            )
            if anchor in env.nk_asn2id:
                shortest = shortest_paths(env, anchor, eligible)
                counts["shortest_path_anchor_calls_" + layer] += 1
            else:
                shortest = {}
                counts["missing_topology_anchor_states"] += len(anchor_leaves)
            # Prefix/phase states remain separate even when A is the same.
            for leaf in anchor_leaves:
                key = leaf.get("state_key", "->".join(leaf["prefix"]) + "|" + leaf["phase_state"])
                attempted_prefix_states[key] += 1
                update(
                    prefix=list(leaf["prefix"]),
                    phase_state=leaf["phase_state"],
                    prefix_states_completed=sum(attempted_prefix_states.values()),
                )
                for index, destination in enumerate(eligible):
                    if layer != "HIGH" and destination in high_destinations:
                        continue
                    prefix = leaf["prefix"]
                    if anchor == destination:
                        add(destination, prefix, "prefix_is_destination", leaf)
                        continue
                    for tail in env.existPath.get(anchor + "-" + destination, ()):
                        tail = tuple(map(str, tail))
                        if tail and tail[0] == anchor:
                            add(destination, prefix + tail[1:], "independent_observed_tail", leaf)
                    tail = shortest.get(destination)
                    if tail and str(tail[0]) == anchor:
                        add(
                            destination,
                            prefix + tuple(map(str, tail[1:])),
                            "topology_tail_proposal",
                            leaf,
                        )
                    if index % 128 == 0:
                        update(
                            destinations_processed_in_prefix=index + 1,
                            high_queries=len(high_destinations),
                            candidate_attempts=counts["candidate_attempts"],
                        )
            # Never retain the previous anchor's complete destination paths.
            del shortest
            update(anchors_completed=anchor_index + 1, high_queries=len(high_destinations))

    # A fallback state keeps its declared layer even if a different shorter
    # public prefix could classify the completed path as HIGH.
    high = [
        leaf
        for leaf in leaves
        if leaf["tier"] in {"HIGH", "fully_public_observed_commercial_compatible"}
    ]
    unknown = [
        leaf
        for leaf in leaves
        if leaf["tier"] not in {"HIGH", "fully_public_observed_commercial_compatible"}
    ]
    generate_layer("HIGH", high)
    remaining_before_fallback = len(destinations) - len(high_destinations)
    if legacy_generator is not None:
        remaining = [d for d in destinations if d not in high_destinations]
        remaining_set = set(remaining)
        update(layer="frozen_HF_fallback", destinations_for_legacy_fallback=len(remaining))
        previous = legacy_generator(env, k, source, remaining) if remaining else {}
        for pair in list(previous):
            paths = previous.pop(pair)
            destination = pair.split("-")[-1]
            require(
                destination in remaining_set, "Frozen HF fallback returned an unrequested query"
            )
            for path in paths:
                add(destination, path, "frozen_HF_fallback")
    generate_layer("UNKNOWN", unknown)
    result = {}
    tiers = Counter()
    fallback_queries = 0
    for pair in sorted(banks):
        bank = banks.pop(pair)
        selected = bank.result()
        require(
            len(selected.paths) <= k and not selected.coverage_fallback,
            "Invalid query tier/capacity selection",
        )
        require(
            not selected.paths
            or all(selector.classify_path(p) == selected.tier for p in selected.paths),
            "Lower tier filled higher-tier candidate slots",
        )
        if not selected.paths:
            continue
        result[pair] = selected.paths
        tiers[selected.tier] += 1
        fallback_queries += selected.tier != "HIGH"
        if query_trace is not None:
            query_trace.write(
                json.dumps(
                    dict(
                        pair=pair,
                        selection=selected.receipt,
                        selected_path_set_sha256=helpers.path_set_signature(selected.paths),
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    info = dict(
        version=VERSION,
        source=source,
        admitted_library_paths=len(library),
        library_path_set_sha256=spec["admitted_path_set_sha256"],
        query_slots=k,
        query_count=len(destinations),
        queries_with_candidates=len(result),
        queries_without_candidates=len(destinations) - len(result),
        queries_with_unencodable_destination=sum(not encodable((source, d)) for d in destinations),
        queries_without_HIGH_before_fallback=remaining_before_fallback,
        selected_queries_by_tier=dict(tiers),
        unknown_fallback_queries=fallback_queries,
        candidate_counts=dict(counts),
        attempted_prefix_phase_states=len(attempted_prefix_states),
        accepted_prefix_phase_states=len(accepted_prefix_states),
        prefix_phase_state_attempt_counts=dict(attempted_prefix_states),
        prefix_phase_state_accept_counts=dict(accepted_prefix_states),
        maximum_predicted_prefix_AS_count=3,
        candidate_prefixes_verified=True,
        full_path_commercial_phase_audited=True,
        unknown_relationships_preserved=True,
        lower_tiers_fill_unused_slots=False,
        shortest_path_cache_policy="one anchor at a time, release before next anchor; full prefix/phase states remain distinct",
        role_cache_max_entries=131072,
        role_cache_hits=role.cache_info().hits,
        role_cache_misses=role.cache_info().misses,
        augmentation_quantity_cap=None,
        prefix_depth_is_not_full_path_length=True,
        full_path_maximum_AS_count=20,
        public_priority_is_not_forwarding_probability=True,
        graph_tail_is_not_observed_route=True,
        elapsed_seconds=time.monotonic() - started,
        **memory_record(),
    )
    update(
        layer="completed",
        queries_with_candidates=len(result),
        candidate_attempts=counts["candidate_attempts"],
    )
    return result, info


def siblings():
    from . import prefix_tree_pathgen, public_prefix_tree, public_head_selector

    return prefix_tree_pathgen, public_prefix_tree, public_head_selector
