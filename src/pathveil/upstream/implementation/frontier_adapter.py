"""Unified255 frontier adapter computations."""
from collections import Counter
from .frontier_policy import FrontierPolicy, plan_frontiers
from .candidates import candidate_from_tail, edge_evidence


class SafeFrontierAdapter:
    def __init__(self, source, prediction, index, relations, predictor, policy=None):
        self.source = str(source)
        self.prediction = prediction
        self.index = index
        self.relations = relations
        self.predictor = predictor
        self.policy = policy or FrontierPolicy()
        self.prefix_proofs = {}
        self.capacity_cache = {}
        self.queries = {}
        for neighbor in prediction["selected_neighbors"]:
            n = str(neighbor)
            self.prefix_proofs[self.source, n] = prediction["root"].get("provenance", {}).get(n, {})

    def provenance(self, prefix):
        prefix = tuple(map(str, prefix))
        return {
            (prefix[i - 1], prefix[i]): self.prefix_proofs.get(prefix[: i + 1], {})
            for i in range(1, len(prefix))
        }

    def capacity(self, prefix, threshold):
        prefix = tuple(map(str, prefix))
        key = (prefix, int(threshold))
        if key in self.capacity_cache:
            return self.capacity_cache[key]
        provenance = self.provenance(prefix)
        scanned = 0
        valid = 0
        rejected = Counter()
        lengths = Counter()
        tails = self.index.get(prefix[-1], {})
        for text, frequency in tails.items():
            scanned += 1
            (candidate, reason) = candidate_from_tail(
                prefix, tuple(text.split("->")), frequency, self.index, self.relations, provenance
            )
            if reason:
                rejected[reason] += 1
                continue
            valid += 1
            lengths[len(candidate["path"])] += 1
            if valid >= threshold:
                break
        result = dict(
            valid_tail_count=valid,
            count_is_lower_bound=scanned < len(tails),
            safe_tail_capacity_before_checks=len(tails),
            scanned_tails=scanned,
            rejection_counts=dict(rejected),
            measured_valid_AS_count_counts=dict(lengths),
            reference_paths_used=False,
        )
        result["transit_availability_status"] = (
            "observed_forward_tail"
            if tails
            else "endpoint_observed_transit_unknown"
            if self.index.get(prefix[0], {}).get("->".join(prefix), 0) > 0
            else "unknown_transit_availability"
        )
        self.capacity_cache[key] = result
        return result

    def predict_next(self, prefix):
        prefix = tuple(map(str, prefix))
        raw = self.predictor.query_frontier(
            int(self.source), int(prefix[-1]), list(map(int, prefix))
        )
        proposed = []
        reasons = Counter()
        for candidate in raw["candidates"]:
            n = str(candidate["asn"])
            proof = dict(
                candidate,
                observed_support_probability=candidate["local_observed_support_probability"],
            )
            checked = edge_evidence(self.index, self.relations, prefix[-1], n, proof)
            if candidate["phase_status"] == "conflict":
                reasons["known_commercial_conflict"] += 1
                continue
            if not checked["admitted"]:
                reasons["insufficient_prefix_edge_evidence"] += 1
                continue
            self.prefix_proofs[prefix + (n,)] = proof
            proposed.append(
                dict(
                    candidate,
                    admitted=True,
                    observed_support_probability=candidate["local_observed_support_probability"],
                )
            )
            if len(proposed) >= self.policy.next_hop_topk:
                break
        result = dict(
            scope=raw["probability_semantics"],
            candidates=proposed,
            raw_candidate_count=len(raw["candidates"]),
            excluded_before_topk=dict(reasons),
            head=raw["head"],
            possible_commercial_phases=raw["possible_commercial_phases"],
            deeper_prefix_assumption=raw["deeper_prefix_assumption"],
        )
        self.queries[prefix] = result
        return result

    def plan(self):
        result = plan_frontiers(
            self.source,
            self.prediction["selected_neighbors"],
            self.capacity,
            self.predict_next,
            self.policy,
        )
        result["prefix_edge_provenance"] = {
            "->".join(prefix): proof for (prefix, proof) in self.prefix_proofs.items()
        }
        result["frontier_prediction_queries"] = {
            "->".join(prefix): query for (prefix, query) in self.queries.items()
        }
        return result
