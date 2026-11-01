"""Unified255 root computations."""
import numpy as np


METHOD = "neighbor_ensemble350_core_v1"
RELATIONS = ["c2p", "p2p", "p2c", "unknown"]


def zscore(v):
    return (v - float(np.mean(v))) / max(float(np.std(v)), 1e-08)


def slate(candidates, old_rank, new_rank, presence, policy):
    cand = np.asarray(candidates, dtype=np.int64)
    rank0 = np.lexsort((cand, -old_rank))
    rank1 = np.lexsort((cand, -new_rank))
    weight = policy["weight_v1"]
    if policy["method"] == "zscore":
        base = weight * zscore(old_rank) + (1 - weight) * zscore(new_rank)
    else:
        ranks = []
        for order in [rank0, rank1]:
            r = np.empty(len(cand), dtype=np.int32)
            r[order] = np.arange(1, len(cand) + 1)
            ranks.append(r)
        base = zscore(weight / (60 + ranks[0]) + (1 - weight) / (60 + ranks[1]))
    p = np.clip(presence, 1e-06, 1 - 1e-06)
    values = base + policy["alpha"] * np.log(p / (1 - p))
    order = np.lexsort((cand, -values))
    core = set(map(int, rank0[:3])) | set(map(int, rank1[:3])) if policy["top3_union"] else set()
    head = [int(i) for i in order if int(i) in core]
    tail = [int(i) for i in order if int(i) not in core]
    order = np.asarray(head + tail, dtype=np.int64)
    prefix = order[: policy["max_k"]]
    minimum = len(head)
    mean = np.cumsum(presence[prefix]) / np.arange(1, len(prefix) + 1)
    eligible = np.flatnonzero(
        (mean >= policy["threshold"] - 1e-12) & (np.arange(1, len(prefix) + 1) >= minimum)
    )
    count = int(eligible[-1]) + 1 if len(eligible) else minimum
    selected = order[:count]
    return dict(
        order=order,
        selected=selected,
        values=values,
        core_v1=[int(v) for v in cand[rank0[:3]]],
        core_struct=[int(v) for v in cand[rank1[:3]]],
        uncertain=bool(count and float(np.mean(presence[selected])) < policy["threshold"] - 1e-12),
    )


def record(
    source,
    candidates,
    uncapped,
    visible_degree,
    old_rank,
    new_rank,
    presence,
    policy,
    public,
    evidence,
    head,
    evidence_sha256,
):
    choice = slate(candidates, old_rank, new_rank, presence, policy)
    cand = np.asarray(candidates)
    order = choice["order"]
    selected = [int(v) for v in cand[choice["selected"]]]
    provenance = {}
    for i in order[:64]:
        n = int(cand[i])
        a = n in evidence["adjacency"].get(source, set())
        b = n in public.adjacency.get(source, set())
        provenance[str(n)] = dict(
            residual_edge=a,
            public_edge=b,
            inferred_only=not (a or b),
            selected=n in selected,
            protected_core_v1=n in choice["core_v1"],
            protected_core_structural=n in choice["core_struct"],
            relationship=public.relations.get((source, n), "unknown"),
            pdb_shared_ixp=len(public._ixps.get(source, set()) & public._ixps.get(n, set())),
            pdb_shared_facility=len(
                public._facilities.get(source, set()) & public._facilities.get(n, set())
            ),
            observed_support_probability=float(presence[i]),
            probability_calibrated=False,
            probability_semantics="arithmetic mean of two source-disjoint finite-observation presence heads",
        )
    root = dict(
        source=source,
        center=source,
        candidates=list(map(int, candidates)),
        uncapped_candidates=uncapped,
        rankings={METHOD: [int(v) for v in cand[order]]},
        scores={METHOD: {str(int(v)): float(choice["values"][i]) for (i, v) in enumerate(cand)}},
        ranking_semantics="protected raw-ranker Top3 union first, then descending combined raw score; scores alone do not represent core protection",
        support_probabilities={str(int(v)): float(presence[i]) for (i, v) in enumerate(cand)},
        provenance=provenance,
        provenance_scope="top64 in protected-core ordering",
        selected_neighbors=selected,
        top3_union_sources={"v1": choice["core_v1"], "struct": choice["core_struct"]},
    )
    return dict(
        root=root,
        recursive={},
        visible_degree=int(visible_degree),
        method=METHOD,
        selected_neighbors=selected,
        top3_union_sources=root["top3_union_sources"],
        protected_core_below_raw_support=choice["uncertain"],
        selection_uncertain=choice["uncertain"],
        raw_support={
            str(int(v)): float(presence[i]) for (i, v) in enumerate(cand) if int(v) in selected
        },
        calibrated_support=None,
        support_calibration_status="raw mean presence; no calibration guarantee",
        original_source_model_head=head,
        source_labels_excluded_from_head=True,
        source_labels_excluded_from_predictive_heads=True,
        evidence_sha256=evidence_sha256,
        recursive_status="dynamic frontier requires independent valid-tail adequacy check",
        frontier_interface="frontier_predictor350_v2.FrontierPredictor.query_frontier",
        no_source_specific_policy=True,
    )


from collections import defaultdict
import math


class RootStructuralFeatures:
    def structural(self, source, candidates):
        p = self.public
        a = self.graph.get(source, set())
        result = []
        for n in candidates:
            b = self.graph.get(n, set())
            common = a & b
            si = p._ixps.get(source, set())
            ci = p._ixps.get(n, set())
            sf = p._facilities.get(source, set())
            cf = p._facilities.get(n, set())
            sharedix = si & ci
            sharedfac = sf & cf
            bridge = defaultdict(float)
            for w in common:
                bridge[
                    p.relations.get((source, w), "unknown"), p.relations.get((w, n), "unknown")
                ] += self.ra.get(w, 0.0)
            result.append(
                [
                    math.log1p(len(common)),
                    len(common) / max(1, len(a | b)),
                    len(common) / max(1, min(len(a), len(b))),
                    sum((self.aa.get(w, 0.0) for w in common)),
                    sum((self.ra.get(w, 0.0) for w in common)),
                    len(common) / max(1, len(a)),
                    len(common) / max(1, len(b)),
                    sum((1 / math.log(max(2, self.ixsize[w])) for w in sharedix)),
                    sum((1 / self.ixsize[w] for w in sharedix)),
                    sum((1 / math.log(max(2, self.facsize[w])) for w in sharedfac)),
                    sum((1 / self.facsize[w] for w in sharedfac)),
                    math.log1p(sum((self.ixsize[w] for w in si)) / max(1, len(si))),
                    math.log1p(sum((self.ixsize[w] for w in ci)) / max(1, len(ci))),
                    math.log1p(sum((self.facsize[w] for w in sf)) / max(1, len(sf))),
                    math.log1p(sum((self.facsize[w] for w in cf)) / max(1, len(cf))),
                ]
                + [bridge[r1, r2] for r1 in RELATIONS for r2 in RELATIONS]
            )
        return np.asarray(result, dtype=np.float32)
