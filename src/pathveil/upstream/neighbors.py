"""Public dated-neighbor entry points and shared structural feature factory."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np

from .contracts import read, require, sha, verify
from .evidence import load_policies, policy_features, unknown_policy, value
from .method import MethodLibrary
from .resources import positive_int, runtime_resources

from .evidence import make_public


STRUCTURAL_NAMES = [
    "struct_common_log1p",
    "struct_jaccard",
    "struct_overlap_coefficient",
    "struct_adamic_adar",
    "struct_resource_allocation",
    "struct_common_fraction_source",
    "struct_common_fraction_candidate",
    "struct_pdb_ix_adamic_adar",
    "struct_pdb_ix_resource_allocation",
    "struct_pdb_fac_adamic_adar",
    "struct_pdb_fac_resource_allocation",
    "struct_pdb_source_ix_mean_population",
    "struct_pdb_candidate_ix_mean_population",
    "struct_pdb_source_fac_mean_population",
    "struct_pdb_candidate_fac_mean_population",
] + [
    "struct_bridge_" + a + "_" + b
    for a in ("c2p", "p2p", "p2c", "unknown")
    for b in ("c2p", "p2p", "p2c", "unknown")
]


def make_root_structure(public, evidence, core):
    obj = core.root.RootStructuralFeatures()
    obj.public = public
    nodes = set(public.adjacency) | set(evidence["adjacency"])
    obj.graph = {
        n: public.adjacency.get(n, set()) | evidence["adjacency"].get(n, set()) for n in nodes
    }
    obj.aa = {n: 1 / math.log(max(2, len(v))) for n, v in obj.graph.items()}
    obj.ra = {n: 1 / max(1, len(v)) for n, v in obj.graph.items()}
    obj.ixsize = {k: len(v) for k, v in core.retrieval.reverse_presence(public._ixps).items()}
    obj.facsize = {
        k: len(v) for k, v in core.retrieval.reverse_presence(public._facilities).items()
    }
    return obj


def train_neighbors(
    data_manifest, split_manifest, work_dir, method_library=None, method_config=None
):
    """Fit same-date source-disjoint heads; importing this module never trains."""
    from .neighbor_training import train_neighbors as fit

    return fit(data_manifest, split_manifest, work_dir, method_library, method_config)


class NeighborRuntime:
    """Label-blind inference using explicitly supplied local model/evidence files."""

    def __init__(self, bundle, data_manifest, method_config, method=None, threads=None):
        import lightgbm as lgb

        self.bundle, self.data, self.config = (
            value(bundle),
            value(data_manifest),
            value(method_config),
        )
        self.date = self.data["date"]
        require(self.bundle["date"] == self.data["date"], "Model and public data dates differ")
        require(
            self.bundle["model_fit_date"] == self.data["date"], "Model was fitted on another date"
        )
        require(
            self.bundle.get("formal_labels_read") is False,
            "Model bundle lacks formal-label exclusion",
        )
        verify(self.bundle.get("model_files_sha256", {}))
        verify(self.bundle.get("inputs_sha256", {}))
        self.core = (method or MethodLibrary()).core()
        self.runtime_resources = runtime_resources(self.config)
        self.threads = positive_int(
            self.runtime_resources["case_predict_threads"] if threads is None else threads,
            "case_predict_threads",
        )
        self.runtime_resources["case_predict_threads"] = self.threads
        self.public = make_public(self.data, self.core)
        with Path(self.bundle["evidence_pickle"]).open("rb") as stream:
            self.evidence = pickle.load(stream)
        with Path(self.bundle["safe_index_pickle"]).open("rb") as stream:
            self.safe = pickle.load(stream)
        self.folds = self.bundle["source_to_fold"]
        self.builder = self.core.retrieval.PDBRetrievalBuilder(self.public, self.evidence, 4096)
        self.builder.columns = list(
            self.bundle["schemas"].get("root_base", self.bundle["schemas"]["root_pdb"])
        )[:467]
        self.geometry = make_root_structure(self.public, self.evidence, self.core)
        self.conditional_builder = self.core.retrieval.PDBRetrievalBuilder(
            self.public, self.evidence, 512
        )
        self.conditional_builder.columns = self.builder.columns
        self.paircols = sorted(self.public.pair_features(1, 2))
        policy_data = dict(self.data, parsed_policy_file=self.bundle["parsed_policy_file"])
        sources = set(self.folds) | set(
            map(str, self.bundle.get("sources", self.data.get("sources", [])))
        )
        raw_policies = read(self.bundle["parsed_policy_file"])
        sources.update(raw_policies.get("policies", raw_policies))
        self.policies, _ = load_policies(policy_data, sources)
        self.role_builder = self.core.rpsl.PolicyBuilder(self.public, self.evidence, self.policies)
        self.role_builder.columns = self.builder.columns
        self.heads = {}
        self.distribution_heads = {}
        self._booster = lgb.Booster
        self.evidence_sha256 = sha(self.bundle["evidence_pickle"])
        self.infer_root = lru_cache(maxsize=64)(self._infer_root)
        self.query_source_role = lru_cache(maxsize=32)(self._query_source_role)
        self._frontier_cached = lru_cache(maxsize=64)(self._frontier)
        self._origin_pair = lru_cache(maxsize=30000)(self._origin_pair_uncached)

    def _head_key(self, source):
        return "oof" + str(self.folds[str(source)]) if str(source) in self.folds else "full"

    def _model(self, family, source):
        key = (family, self._head_key(source))
        if key not in self.heads:
            self.heads[key] = self._booster(model_file=self.bundle["models"][family][key[1]])
        return self.heads[key]

    def _infer_root(self, source):
        source = int(source)
        if not self.builder.candidates(source, source)[0]:
            return dict(
                source=source,
                method=self.core.root.METHOD,
                selected_neighbors=[],
                status="unknown_no_public_candidates",
                selection_uncertain=True,
                root=dict(
                    source=source,
                    center=source,
                    candidates=[],
                    selected_neighbors=[],
                    rankings={self.core.root.METHOD: []},
                    scores={self.core.root.METHOD: {}},
                    support_probabilities={},
                    provenance={},
                ),
                recursive={},
                visible_degree=0,
                original_source_model_head=self._head_key(source),
                source_labels_excluded_from_head=True,
                source_labels_excluded_from_predictive_heads=True,
                evidence_sha256=self.evidence_sha256,
                calibrated_support=None,
                support_calibration_status="unknown; no public candidates",
            )
        cand, X, uncapped = self.builder.query(source, source)
        old, _ = self.core.experiment.add_relative(X, [{"size": len(cand)}], self.builder.columns)
        struct_base = np.concatenate((X, self.geometry.structural(source, cand)), axis=1)
        new, _ = self.core.experiment.add_relative(
            struct_base, [{"size": len(cand)}], self.builder.columns + STRUCTURAL_NAMES
        )
        a = self._model("root_pdb_rank", source).predict(old, num_threads=self.threads)
        b = self._model("root_struct_rank", source).predict(new, num_threads=self.threads)
        p = (
            self._model("root_pdb_present", source).predict(old, num_threads=self.threads)
            + self._model("root_struct_present", source).predict(new, num_threads=self.threads)
        ) / 2
        key = self._head_key(source)
        return self.core.root.record(
            source,
            cand,
            uncapped,
            len(self.evidence["adjacency"].get(source, set())),
            a,
            b,
            p,
            self.config["root_selection"],
            self.public,
            self.evidence,
            key[3:] if key.startswith("oof") else key,
            self.evidence_sha256,
        )

    def _origin_pair_uncached(self, source, candidate):
        f = self.public.pair_features(source, candidate)
        return np.asarray([f[k] for k in self.paircols], dtype=np.float32)

    def query_frontier(self, source, center, prefix, commercial_phase=None):
        source, center, prefix = int(source), int(center), tuple(map(int, prefix))
        require(
            len(prefix) >= 2 and prefix[0] == source and prefix[-1] == center,
            "Frontier prefix must start at original source and end at current center",
        )
        require(len(prefix) == len(set(prefix)), "Frontier prefix contains a loop")
        provided = (
            ()
            if commercial_phase is None
            else (commercial_phase,)
            if isinstance(commercial_phase, str)
            else tuple(sorted(set(commercial_phase)))
        )
        require(set(provided) <= self.core.frontier.PHASES, "Unknown commercial phase")
        return self._frontier_cached(source, center, prefix, provided)

    def _frontier(self, source, center, prefix, provided):
        previous = prefix[-2]
        if self.conditional_builder.candidates(previous, center)[0]:
            cand, X, uncapped = self.conditional_builder.query(previous, center)
            extra = np.stack([self._origin_pair(previous, n) for n in cand])
            matrix, columns = self.core.experiment.add_relative(
                np.concatenate((X, extra), axis=1),
                [{"size": len(cand)}],
                self.builder.columns + ["origin_candidate_" + k for k in self.paircols],
            )
            require(columns == self.bundle["schemas"]["conditional"], "Conditional schema mismatch")
            support = self._model("conditional_present", source).predict(
                matrix, num_threads=self.threads
            )
            ranks = self._model("conditional_rank", source).predict(
                matrix, num_threads=self.threads
            )
        else:
            cand, uncapped, support, ranks = [], 0, [], []
        relations = [
            self.public.relations.get((a, b), "unknown") for a, b in zip(prefix, prefix[1:])
        ]
        states = {"UP"}
        for relation in relations:
            states = self.core.frontier.advance_phases(states, relation)
        derived = set(states)
        if provided:
            states &= set(provided)
        candidates = []
        for i, n in enumerate(cand):
            if n in prefix:
                continue
            rel = self.public.relations.get((center, n), "unknown")
            after = self.core.frontier.advance_phases(states, rel)
            residual = n in self.evidence["adjacency"].get(center, set())
            public = n in self.public.adjacency.get(center, set())
            candidates.append(
                dict(
                    asn=n,
                    local_transition_rank_score=float(ranks[i]),
                    local_observed_support_probability=float(support[i]),
                    probability_calibrated=False,
                    relationship=rel,
                    possible_commercial_phases_after=sorted(after),
                    phase_status="conflict"
                    if not after
                    else "ambiguous"
                    if rel == "unknown" or len(after) > 1
                    else "compatible",
                    residual_edge=residual,
                    public_edge=public,
                    inferred_only=not (residual or public),
                    pdb_shared_ixp=len(
                        self.public._ixps.get(center, set()) & self.public._ixps.get(n, set())
                    ),
                    pdb_shared_facility=len(
                        self.public._facilities.get(center, set())
                        & self.public._facilities.get(n, set())
                    ),
                )
            )
        candidates.sort(key=lambda r: (-r["local_transition_rank_score"], r["asn"]))
        key = self._head_key(source)
        return dict(
            original_source=source,
            current=center,
            previous=previous,
            prefix=list(prefix),
            semantics_version="350-frontier-v2-sibling-neutral",
            prefix_relationships=relations,
            derived_commercial_phases=sorted(derived),
            possible_commercial_phases=sorted(states),
            prefix_relation_conflict=not bool(states),
            prefix_unknown_relationships=sum(r == "unknown" for r in relations),
            candidates=candidates,
            uncapped_candidate_count=uncapped,
            head=key[3:] if key.startswith("oof") else key,
            source_labels_excluded_from_head=True,
            query_is_label_blind=True,
            probability_semantics="local previous/current/next finite-observation propensity; not full original-source path probability",
            full_prefix_is_checked_for_loops_and_known_commercial_phase=True,
            deeper_prefix_assumption="local Markov transfer; empirical deeper-path calibration not claimed",
            adequacy_decision="caller must compute independent valid-tail capacity/diversity; no fixed source-degree trigger here",
        )

    def _query_source_role(self, source):
        source = int(source)
        self.policies.setdefault(str(source), unknown_policy())
        policy = self.policies[str(source)]
        if not self.role_builder.candidates(source, source)[0]:
            return dict(candidate_role_p={}, rpsl_policy_unknown=not policy["valid"])
        cand, X, _ = self.role_builder.query(source, source)
        names = sorted(policy_features(policy, 0, self.data["date"], self.core))
        rows = [policy_features(policy, n, self.data["date"], self.core) for n in cand]
        extra = np.asarray([[r[n] for n in names] for r in rows], dtype=np.float32)
        matrix, columns = self.core.experiment.add_relative(
            np.concatenate((X, extra), axis=1), [{"size": len(cand)}], self.builder.columns + names
        )
        require(columns == self.bundle["schemas"]["role"], "First-hop role schema mismatch")
        scores = self._model("role", source).predict(matrix, num_threads=self.threads)
        return dict(
            candidate_role_p={str(n): float(scores[i]) for i, n in enumerate(cand)},
            rpsl_policy_unknown=not policy["valid"],
        )

    def predict_firsthop(self, source, neighbor):
        source, neighbor = int(source), int(neighbor)
        record = self.query_source_role(source)
        p = record["candidate_role_p"].get(str(neighbor))
        return dict(
            source=source,
            neighbor=neighbor,
            role_p=p,
            role_missing=p is None,
            role_scope="P(finite >=3-AS source path via firsthop N | source-neighbor observed); not physical transit capability",
            role_probability_calibrated=False,
            role_source_labels_excluded=True,
            rpsl_policy_unknown=record["rpsl_policy_unknown"],
            no_network_lookup=True,
            permitted_use="soft full-path score feature only; low p never deletes a neighbor or proves transit impossible",
        )

    def predict_prior(self, source, candidates, X, selected=None):
        key = self._head_key(source)
        if key not in self.distribution_heads:
            paths = self.bundle["distribution"][key]
            head = self.core.distribution_head.DistributionHead(threads=self.threads)
            head.branch_model = self._booster(model_file=paths["branch"])
            for name in ("length", "coverage"):
                with Path(paths[name]).open("rb") as stream:
                    setattr(head, name + "_model", pickle.load(stream))
            self.distribution_heads[key] = head
        head = self.distribution_heads[key]
        X = np.asarray(X, dtype=np.float32)
        require(
            X.shape == (len(candidates), len(self.builder.columns)),
            "Prior feature matrix schema mismatch",
        )
        require(
            len(candidates) > 0,
            "Empty branch pool has unknown prior; uniform substitution is forbidden",
        )
        query = dict(
            source=int(source), center=int(source), root=True, candidates=list(map(int, candidates))
        )
        mass, lengths = head.predict_arrays(X, [query])
        coverage = float(head.predict_candidate_coverage(X, [query])[0])
        result = self.core.prior_interface.as_prior(query, mass, lengths, coverage, selected)
        result.update(
            model_scope=(
                "Source5fold OOF; own source labels excluded"
                if key != "full"
                else "Full160 training-source model; no predicted-source reference used"
            ),
            feature_scope=self.bundle.get(
                "distribution_feature_scope", "Same-date original-source-masked root contexts"
            ),
            fixed_inference_temperature=head.temperature,
        )
        return result
