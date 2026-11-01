"""User-specified selection counts and the original model score semantics."""
from __future__ import annotations

import copy
import math


LOW_ROOT_LIMIT = 21
POLICY = dict(version="rho_adaptive_v1",
    low_root_limit=LOW_ROOT_LIMIT,
    low_root_limit_provenance="User-designated 0201 original TRAIN K80 P90=21; fixed engineering count, not per-target route selection",
    low_root_selection="unchanged dual Top3 protection, scores and mean-support threshold; max_k=21",
    high_root_selection="dual raw-ranker Top3 union only; no score-based additions",
    low_recursion="unchanged valid_tail_count_min=300, max_prefix_AS_count=3, next_hop_topk=10",
    high_recursion="expand every selected root independent of tail capacity; only S-N-A leaves",
    high_secondhop_formula="3 + floor(log2(rho)/2 + 0.5)",
    high_secondhop_limit="min(formula, evidence-admissible unvisited ranked candidates)",
    augmentation="source's own unified tree plus original TRAIN donor tails; existing 255-column scorer",
    quality=dict(low_threshold=.75,low_minimum_paths=1000,high_threshold=.75,high_minimum_paths=1000,
                 semantics="All candidates are source pools: use original source q0.75/min1000. Neighbor q0.90 applies only to the removed neighbor-first splice protocol."),
    query_tree="high: full selected source tree also used for augmentation; low: original S-N Tree B+PG",
    training_generators=dict(low="PG",high="Tree B + PG"),
    final_model="unchanged normal-history Huber GMPI; source pair-disjoint validation",
    new_mass_prediction=False,maximum_prefix_AS_count=3,maximum_full_path_AS_count=20,
    quantity_cap=None,unknown_is_not_absence=True)


def high_secondhop_count(rho):
    if not math.isfinite(rho) or rho<1:
        raise ValueError("Secondhop formula requires finite rho>=1")
    return 3+math.floor(math.log2(rho)/2+.5)


def select_root(original, rho):
    """The caller computes low-rho records using the original slate with max_k=21."""
    if rho is None or not math.isfinite(rho) or rho<0:
        raise ValueError("Unknown/invalid source shape cannot select a default branch")
    value=copy.deepcopy(original)
    if rho<=1:
        return value
    raw=value["root"]
    core={str(n) for group in value["top3_union_sources"].values() for n in group}
    order=list(map(str,raw["rankings"][value["method"]]))
    selected=[n for n in order if n in core]
    if not selected or len(selected)>6 or set(selected)!=core:
        raise ValueError("Invalid dual raw-ranker Top3 union")
    chosen=list(map(int,selected))
    raw["selected_neighbors"]=chosen
    value["selected_neighbors"]=chosen
    for n,evidence in raw.get("provenance",{}).items():
        evidence["selected"]=n in core
    value["raw_support"]={n:raw["support_probabilities"][n] for n in selected}
    uncertain=sum(value["raw_support"].values())/len(selected)<.7
    value.update(selection_uncertain=uncertain,protected_core_below_raw_support=uncertain,
                 rho_top3_policy=POLICY["high_root_selection"],source_rho=rho)
    return value


def mandatory_plan(source,roots,capacity,predict_next,k):
    """Expand every high-rho root; do not emit a broad S-N tail join as a shortcut."""
    source=str(source);prefixes=[];decisions=[]
    for root in map(str,roots):
        prefix=(source,root)
        if root==source:
            raise ValueError("Cyclic root")
        current=capacity(prefix,300)
        prediction=predict_next(prefix)
        children=[];discarded=[]
        for candidate in prediction.get("candidates",[]):
            node=str(candidate["asn"])
            if node in prefix or not candidate.get("admitted",False):
                discarded.append(node);continue
            if node not in children:
                children.append(node)
            if len(children)>=k:
                break
        prefixes.extend(prefix+(node,) for node in children)
        decisions.append(dict(prefix=prefix,capacity=current,expanded_neighbors=children,
            discarded=discarded,forced_expansion=True,tail_capacity_cannot_stop_expansion=True,
            stop_reason=None if children else "no_supported_unvisited_next_hop"))
    return dict(source=source,policy=dict(valid_tail_count_min=300,max_prefix_AS_count=3,next_hop_topk=k),
                prefixes=prefixes,decisions=decisions,source_total_degree_not_used=True,
                no_task_reference_access=True,insufficient_leaves=[],high_rho_forced_expansion=True,
                no_broad_root_tail_bypass=True)
