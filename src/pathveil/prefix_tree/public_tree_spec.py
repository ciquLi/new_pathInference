"""Portable numerical API extracted from the verified Tree B implementation."""
from __future__ import annotations


import argparse


import csv


import hashlib


import json


from pathlib import Path


from . import public_prefix_tree, public_head_admission


from .public_prefix_tree import PublicPrefixTopology, commercial_phases


from .public_head_admission import HIGH, UNKNOWN, classify_prefix


VERSION = "public_two_edge_tree_spec_v1"


CONFIG = dict(
    maximum_prefix_AS_count=3,
    all_public_N_if_A_unique_over_b1_squared_gt=1,
    expand_N_only_if_source_ratio_gt1_and_compatible_B2_gt_b1=True,
    source_ratio_le1_means_stop_SN_for_every_original_R_s_root=True,
    maximum_path_count=None,
    fallback_layer="Uncovered original R_s prefixes, always UNKNOWN, below public primary",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def signature(paths):
    return hashlib.sha256(
        ("\n".join(sorted("->".join(p) for p in set(paths))) + "\n").encode()
    ).hexdigest()


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError("Frozen input changed: " + str(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read(path) != value:
        raise ValueError("Frozen tree artifact differs: " + str(path))
    if not path.exists():
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def asn(value):
    text = str(value)
    if (
        not text.isascii()
        or not text.isdecimal()
        or not 0 < int(text) < 2**32
        or text != str(int(text))
    ):
        raise ValueError("Invalid canonical ASN")
    return text


def phase_traces(roles):
    traces = {("UP",)}
    for role in roles:
        possibilities = {"c2p", "p2p", "p2c"} if role in {"unknown", "conflict"} else {role}
        following = set()
        for trace in traces:
            phase = trace[-1]
            for candidate in possibilities:
                if candidate == "s2s":
                    next_phase = phase
                elif candidate == "c2p" and phase == "UP":
                    next_phase = "UP"
                elif candidate in {"p2p", "p2c"} and phase == "UP":
                    next_phase = "DOWN"
                elif candidate == "p2c" and phase == "DOWN":
                    next_phase = "DOWN"
                else:
                    continue
                following.add(trace + (next_phase,))
        traces = following
    return sorted(traces)


def _leaf_records(topology, prefix, layer):
    evidence = [topology.edge_evidence(left, right) for left, right in zip(prefix, prefix[1:])]
    classification = classify_prefix(prefix, evidence)
    traces = phase_traces(classification["public_roles"])
    if layer == "public_primary" and not traces:
        raise ValueError("Public compatible child has no valid commercial phase")
    fallback = layer == "R_s_unknown_fallback"
    tier = UNKNOWN if fallback else classification["tier"]
    if tier not in {HIGH, UNKNOWN}:
        raise ValueError("Primary public prefix is commercially incompatible")
    if not traces:
        traces = [tuple(["UP"] + ["UNKNOWN"] * (len(prefix) - 1))]
    records = []
    for trace in traces:
        records.append(
            dict(
                classification,
                prefix=list(prefix),
                tier=tier,
                layer=layer,
                state_key=layer + "|" + "->".join(prefix) + "|" + ">".join(trace),
                phase_state=trace[-1],
                phase_trace=list(trace),
                public_edge_evidence=evidence,
                structural_weight=1.0,
                weight_is_probability=False,
                forced_unknown_fallback=fallback,
                public_layer_priority=1 if fallback else 2,
                fallback_cannot_displace_public=True,
            )
        )
    return records


def build_spec(topology, source, original_R_s_prefix_fallbacks, input_bindings=None):
    """Return an immutable-input-bound public tree; no destination labels read.

    Prefix inputs contain two or three ASes, never complete route references.
    Phase history stays part of each leaf identity; shared terminal ASes and
    different UP/DOWN states are not collapsed into one tree vertex.
    """
    source = asn(source)
    original = []
    for raw in original_R_s_prefix_fallbacks:
        prefix = tuple(map(asn, raw))
        if not 2 <= len(prefix) <= 3 or prefix[0] != source or len(prefix) != len(set(prefix)):
            raise ValueError("Invalid R_s fallback prefix")
        original.append(prefix)
    original = sorted(set(original))
    shape = dict(topology.source_features(source))
    public_N = sorted({asn(n) for n in topology.neighbors(source)}, key=int)
    b1 = int(shape["public_b1"])
    A_unique = int(shape["public_l2_distinct_asns"])
    if b1 < 0 or A_unique < 0 or b1 != len(public_N):
        raise ValueError("Public source shape/adjacency differs")
    ratio = A_unique / (b1 * b1) if b1 else None
    shape.update(
        A_unique_over_b1_squared=ratio,
        ratio_is_routing_probability=False,
        ratio_threshold_is_engineering_ablation=True,
        ratio_has_theoretical_accuracy_guarantee=False,
    )
    use_all = ratio is not None and ratio > 1
    selected = public_N if use_all else sorted({p[1] for p in original}, key=int)
    primary = []
    decisions = []
    for neighbor in selected:
        summary = topology.second_hop_summary(source, neighbor, include_asns=True)
        children = sorted({asn(a) for a in summary["all_asns"]}, key=int)
        b2 = int(summary["total_count"])
        if b2 != len(children) or source in children or neighbor in children:
            raise ValueError("Invalid compatible public second-hop set")
        expand = use_all and b2 > b1 and b2 > 0
        decisions.append(
            dict(
                prefix=[source, neighbor],
                public_b1=b1,
                compatible_public_b2=b2,
                action="expand_to_all_public_A" if expand else "stop_at_N",
                public_edge_observed=neighbor in public_N,
                source_gate_allows_expansion=use_all,
                original_R_s_root=any(p[1] == neighbor for p in original),
                second_layer_public=summary,
            )
        )
        for prefix in [(source, neighbor, a) for a in children] if expand else [(source, neighbor)]:
            primary.extend(_leaf_records(topology, prefix, "public_primary"))
    primary_prefixes = {tuple(leaf["prefix"]) for leaf in primary}
    uncovered = [
        prefix for prefix in original if not any(prefix[: len(p)] == p for p in primary_prefixes)
    ]
    fallback = []
    for prefix in uncovered:
        fallback.extend(_leaf_records(topology, prefix, "R_s_unknown_fallback"))
    # This physical-prefix table is only the evidence lookup for a selector;
    # every state/history occurrence remains separate in the two leaf lists.
    classifications = []
    for layer, leaves in (("public_primary", primary), ("R_s_unknown_fallback", fallback)):
        for prefix in sorted({tuple(leaf["prefix"]) for leaf in leaves}):
            variants = [leaf for leaf in leaves if tuple(leaf["prefix"]) == prefix]
            first = variants[0]
            row = {
                key: first[key]
                for key in (
                    "prefix",
                    "tier",
                    "fully_public_observed",
                    "commercial_compatibility",
                    "public_roles",
                    "public_role_conflict",
                    "unknown_is_not_absence",
                    "commercial_incompatible_is_not_route_nonexistence",
                )
            }
            row.update(
                layer=layer,
                phase_states=sorted({leaf["phase_state"] for leaf in variants}),
                phase_traces=[leaf["phase_trace"] for leaf in variants],
                forced_unknown_fallback=layer != "public_primary",
                structural_weight=1.0,
                weight_is_probability=False,
            )
            classifications.append(row)
    if len({tuple(row["prefix"]) for row in classifications}) != len(classifications):
        raise ValueError("Physical prefix has conflicting layers")
    files = dict(topology.inputs_sha256)
    files.update(
        {
            str(Path(__file__).resolve()): sha(__file__),
            str(Path(public_prefix_tree.__file__).resolve()): sha(public_prefix_tree.__file__),
            str(Path(public_head_admission.__file__).resolve()): sha(
                public_head_admission.__file__
            ),
        }
    )
    files.update(input_bindings or {})
    verify(files)
    return dict(
        version=VERSION,
        date=topology.date,
        source=source,
        maximum_prefix_AS_count=3,
        status="UNVALIDATED_ENGINEERING_STRUCTURE_ABLATION",
        configuration=CONFIG,
        configuration_sha256=content_sha(CONFIG),
        source_shape=shape,
        firsthop_policy="all_public_N" if use_all else "original_R_s_model_selected_N_only",
        firsthop_policy_reason="A_unique/(b1*b1)>1" if use_all else "ratio<=1_or_undefined",
        selected_firsthops=selected,
        all_public_firsthops=public_N,
        firsthop_decisions=decisions,
        primary_leaves=primary,
        fallback_leaves=fallback,
        prefix_classification=classifications,
        primary_unique_prefix_count=len(primary_prefixes),
        primary_phase_leaf_count=len(primary),
        fallback_unique_prefix_count=len(uncovered),
        fallback_phase_leaf_count=len(fallback),
        original_R_s_prefix_count=len(original),
        original_R_s_prefix_set_sha256=signature(original),
        admitted_path_set_sha256=None,
        library_csv_sha256=None,
        construction_method_sha256=None,
        graph_identity_sha256=topology.graph_identity_sha256,
        data_code_input_sha256=files,
        phase_state_and_parent_context_are_not_merged=True,
        unknown_is_not_absence=True,
        ordinal_public_layer_priority_is_not_probability=True,
        no_formal_reference_paths=True,
        no_model_training_or_PG_started=True,
    )
