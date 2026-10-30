"""Frozen source-rho PATS policy with one recursion width."""
from __future__ import annotations

import copy
import math
import os


VERSION = "pats_rho_v1"
LOW_ROOT_LIMIT = int(os.environ.get("PATHVEIL_PATS_LOW_ROOT_LIMIT", "21"))
RECURSION_TOPK = int(os.environ.get("PATHVEIL_PATS_RECURSION_TOPK", "5"))
LOW_TAIL_MIN = int(os.environ.get("PATHVEIL_PATS_LOW_TAIL_MIN", "300"))
SOURCE_THRESHOLD = float(os.environ.get("PATHVEIL_PATS_SOURCE_THRESHOLD", "0.75"))
MINIMUM_PATHS = int(os.environ.get("PATHVEIL_PATS_MINIMUM_PATHS", "1000"))
QUERY_CAPACITY = int(os.environ.get("PATHVEIL_PATS_QUERY_CAPACITY", "200"))
PATHGEN_TRAIN_THREADS = int(os.environ.get("PATHVEIL_PATS_PATHGEN_TRAIN_THREADS", "4"))
PATHGEN_TEST_THREADS = int(os.environ.get("PATHVEIL_PATS_PATHGEN_TEST_THREADS", "4"))

if LOW_ROOT_LIMIT <= 0 or RECURSION_TOPK <= 0 or LOW_TAIL_MIN <= 0 or MINIMUM_PATHS <= 0:
    raise ValueError("PATS count parameters must be positive")
if not 0 < SOURCE_THRESHOLD <= 1 or QUERY_CAPACITY <= 0 or not 1 <= PATHGEN_TRAIN_THREADS <= 4 or not 1 <= PATHGEN_TEST_THREADS <= 4:
    raise ValueError("PATS thresholds and query capacity are invalid")

POLICY = dict(
    version=VERSION,
    algorithm_name="prefix_anchor_tail_stitching",
    source_rho_only=True,
    rho_threshold=1.0,
    low_root_limit=LOW_ROOT_LIMIT,
    low_root_limit_provenance=f"User-designated per-date low root limit = {LOW_ROOT_LIMIT}",
    low_root_selection="dual Top3 protection, original combined score and mean-support threshold",
    high_root_selection="dual raw-ranker Top3 union only; no score-based additions",
    low_recursion="original valid-tail gate; recurse with the same fixed top-k",
    high_recursion="expand every selected root independent of tail capacity",
    recursion_topk=RECURSION_TOPK,
    low_valid_tail_count_min=LOW_TAIL_MIN,
    maximum_prefix_AS_count=3,
    maximum_full_path_AS_count=20,
    augmentation="source tree plus original TRAIN donor tails; existing 255-column scorer",
    quality=dict(
        source_threshold=SOURCE_THRESHOLD,
        minimum_paths=MINIMUM_PATHS,
        semantics="Source candidate pool relative quality threshold; no neighbor-first splice protocol",
    ),
    query_capacity=QUERY_CAPACITY,
    pathgen_train_threads=PATHGEN_TRAIN_THREADS,
    pathgen_test_threads=PATHGEN_TEST_THREADS,
    query_tree="same complete selected source tree for training and queries",
    training_generators={"low": "Tree B + PATS", "high": "Tree B + PATS"},
    path_constructor="PATS",
    final_model="unchanged normal-history Huber GMPI; source pair-disjoint validation",
    new_mass_prediction=False,
    quantity_cap=None,
    unknown_is_not_absence=True,
)


def runtime_policy(method):
    """Merge shell-controlled parameters into the registered method defaults."""
    value = copy.deepcopy(method.get("selection_policy", {}))
    value.update(POLICY)
    value["quality"] = dict(POLICY["quality"])
    return value


def select_root(original, rho):
    """Select source roots using only the source AS rho branch."""
    if rho is None or not math.isfinite(rho) or rho < 0:
        raise ValueError("Unknown/invalid source shape cannot select a default branch")
    value = copy.deepcopy(original)
    if rho <= 1:
        return value
    raw = value["root"]
    core = {str(n) for group in value["top3_union_sources"].values() for n in group}
    order = list(map(str, raw["rankings"][value["method"]]))
    selected = [n for n in order if n in core]
    if not selected or len(selected) > 6 or set(selected) != core:
        raise ValueError("Invalid dual raw-ranker Top3 union")
    chosen = list(map(int, selected))
    raw["selected_neighbors"] = chosen
    value["selected_neighbors"] = chosen
    for n, evidence in raw.get("provenance", {}).items():
        evidence["selected"] = n in core
    value["raw_support"] = {n: raw["support_probabilities"][n] for n in selected}
    uncertain = sum(value["raw_support"].values()) / len(selected) < 0.7
    value.update(
        selection_uncertain=uncertain,
        protected_core_below_raw_support=uncertain,
        rho_top3_policy=POLICY["high_root_selection"],
        source_rho=rho,
    )
    return value


def mandatory_plan(source, roots, capacity, predict_next):
    """Force every high-rho source root to recurse with fixed top-k."""
    source = str(source)
    prefixes, decisions = [], []
    for root in map(str, roots):
        prefix = (source, root)
        if root == source:
            raise ValueError("Cyclic root")
        current = capacity(prefix, LOW_TAIL_MIN)
        prediction = predict_next(prefix)
        children, discarded = [], []
        for candidate in prediction.get("candidates", []):
            node = str(candidate["asn"])
            if node in prefix or not candidate.get("admitted", False):
                discarded.append(node)
                continue
            if node not in children:
                children.append(node)
            if len(children) >= RECURSION_TOPK:
                break
        prefixes.extend(prefix + (node,) for node in children)
        decisions.append(
            dict(
                prefix=prefix,
                capacity=current,
                expanded_neighbors=children,
                discarded=discarded,
                forced_expansion=True,
                fixed_recursion_topk=RECURSION_TOPK,
                tail_capacity_cannot_stop_expansion=True,
                stop_reason=None if children else "no_supported_unvisited_next_hop",
            )
        )
    return dict(
        source=source,
        policy=dict(
            valid_tail_count_min=LOW_TAIL_MIN,
            max_prefix_AS_count=3,
            next_hop_topk=RECURSION_TOPK,
        ),
        prefixes=prefixes,
        decisions=decisions,
        source_total_degree_not_used=True,
        no_task_reference_access=True,
        insufficient_leaves=[],
        high_rho_forced_expansion=True,
        no_broad_root_tail_bypass=True,
    )
