"""Portable numerical API extracted from the verified Tree B implementation."""
from __future__ import annotations


import argparse


from collections import Counter


from collections.abc import Mapping


import copy


import importlib.util


import json


from pathlib import Path


import pickletools


import sys


import time


from types import SimpleNamespace


VERSION = "public_tree_observed_arrow_tail_adapter_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_asn(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    node = str(value)
    if (
        not node.isascii()
        or not node.isdecimal()
        or not 0 < int(node) < 2**32
        or str(int(node)) != node
    ):
        return None
    return node


def normalize_tail(raw, anchor, destination, stoi):
    """Return (canonical tuple, reason); malformed rows never become proposals."""
    if isinstance(raw, str):
        values = raw.split("->")
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        return None, "unsupported_tail_type"
    if not 2 <= len(values) <= 20:
        return None, "length"
    nodes = tuple(canonical_asn(value) for value in values)
    if any(node is None for node in nodes):
        return None, "noncanonical_ASN"
    if nodes[0] != anchor or nodes[-1] != destination:
        return None, "endpoint"
    if len(set(nodes)) != len(nodes):
        return None, "loop"
    if any(node not in stoi or stoi[node] == 0 for node in nodes):
        return None, "unencodable_ASN"
    return nodes, "valid"


class NormalizedTailIndex(Mapping):
    """No index copy/cache/mutation; every lookup returns a repeatable tuple."""

    def __init__(self, original, stoi):
        require(isinstance(original, Mapping), "Observed-tail index must be a mapping")
        self.original, self.stoi, self.counts = original, stoi, Counter()

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)

    def __getitem__(self, key):
        raw = self.original[key]
        self.counts["lookups"] += 1
        fields = key.split("-") if isinstance(key, str) else []
        if len(fields) != 2 or any(canonical_asn(value) != value for value in fields):
            self.counts["rejected_key"] += 1
            return ()
        if isinstance(raw, str):
            rows = (raw,)
        elif isinstance(raw, (list, tuple)):
            rows = raw
        elif raw is None:
            rows = ()
        else:
            self.counts["rejected_bucket_type"] += 1
            return ()
        result = []
        for tail in rows:
            self.counts["raw_tail_type_" + type(tail).__name__] += 1
            nodes, reason = normalize_tail(tail, fields[0], fields[1], self.stoi)
            self.counts[reason] += 1
            if nodes is not None:
                result.append(nodes)
        return tuple(result)

    def audit(self):
        return dict(
            counts=dict(self.counts),
            index_policy="read-only per-key normalization, no whole-index copy or persistent tail cache",
            values_policy="arrow STR split on ->; canonical list/tuple accepted; endpoints/loop/2..20/vocabulary checked",
            original_history_index_unchanged=True,
            forwarding_reality_not_certified=True,
        )


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
    base = base_module()
    view = NormalizedTailIndex(env.existPath, env.stoi)
    adapted = copy.copy(env)
    adapted.existPath = view
    # Optional legacy HF remains on its original index contract.
    legacy = (
        (lambda ignored, *a, **kw: legacy_generator(env, *a, **kw))
        if legacy_generator is not None
        else None
    )
    result, audit = base.generate_public_candidates(
        adapted,
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
        legacy_generator=legacy,
        progress=progress,
        query_trace=query_trace,
    )
    audit.update(
        version=VERSION,
        base_generator_version=base.VERSION,
        base_generator_code_sha256=BASE_SHA256,
        observed_tail_adapter_code_sha256=base.sha256(__file__),
        observed_tail_normalization=view.audit(),
    )
    return result, audit


def base_module():
    from . import public_tree_pathgen

    return public_tree_pathgen


from pathveil.utils import sha256

BASE_SHA256 = sha256(base_module().__file__)
