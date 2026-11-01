"""Unified255 final features computations."""

from . import prefix_features, next_features


class FinalPathFeatures:
    def __init__(self, base, plan, role_predictor, conditional_store):
        self.base = base
        self.plan = plan
        self.roles = role_predictor
        self.conditional = conditional_store
        self.role_cache = {}
        self.feature_names = (
            tuple(base.feature_names) + prefix_features.FEATURE_NAMES + next_features.FEATURE_NAMES
        )
        if len(self.feature_names) != 222 or len(set(self.feature_names)) != 222:
            raise ValueError("Final350 feature schema must be222 unique columns")

    def features(self, candidate):
        enriched = prefix_features.enrich_prefix_proofs(candidate, self.plan)
        key = tuple(candidate["path"][:2])
        if key not in self.role_cache:
            self.role_cache[key] = self.roles.predict_firsthop(*map(int, key))
        out = self.base.features(enriched)
        out.update(prefix_features.features(enriched, self.role_cache[key]))
        out.update(next_features.features(enriched, self.conditional))
        if set(out) != set(self.feature_names):
            raise ValueError("Final222 row schema mismatch")
        return out


class ActualQueryStore:
    """Store actual label-blind local queries, independently of recursion."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.queries = {}
        self.lookup = {}

    def query_frontier(self, source, center, prefix, commercial_phase=None):
        key = tuple(map(int, prefix))
        if key not in self.queries:
            self.queries[key] = self.predictor.query_frontier(
                source, center, list(prefix), commercial_phase
            )
            if len(key) == 2:
                self.lookup[key] = {
                    str(c["asn"]): dict(
                        missing=False,
                        probability=c["local_observed_support_probability"],
                        rank=rank,
                        probability_calibrated=False,
                        phase_status=c["phase_status"],
                    )
                    for (rank, c) in enumerate(self.queries[key]["candidates"], 1)
                }
        return self.queries[key]

    def next(self, source, neighbor, actual_next):
        raw = self.lookup.get((int(source), int(neighbor)))
        if raw is None:
            return dict(missing=True, reason="selected_firsthop_query_unavailable")
        if str(actual_next) in raw:
            return raw[str(actual_next)]
        return dict(
            missing=True, probability=None, rank=None, reason="actual_next_AS_not_retrieved"
        )

    def clear(self):
        self.queries.clear()
        self.lookup.clear()
