"""Portable numerical API extracted from the verified Tree B implementation."""
from __future__ import annotations


import argparse


from collections import Counter, defaultdict


from datetime import datetime


from functools import lru_cache


import hashlib


import json


import math


from pathlib import Path


VERSION = "public_prefix_tree_v1"


MAX_PREFIX_AS_COUNT = 3


ROLES = ("c2p", "p2c", "p2p", "s2s", "unknown")


BITS = {role: 1 << index for index, role in enumerate(ROLES)}


INVERSE = {"c2p": "p2c", "p2c": "c2p", "p2p": "p2p", "s2s": "s2s", "unknown": "unknown"}


ROLE_DEGREE = {
    "c2p": "provider",
    "p2c": "customer",
    "p2p": "peer",
    "s2s": "sibling",
    "unknown": "unknown",
    "conflict": "conflict",
}


NODE_NUMERIC_FIELDS = (
    "public_total_degree",
    "public_provider_degree",
    "public_peer_degree",
    "public_customer_degree",
    "public_sibling_degree",
    "public_unknown_degree",
    "public_conflict_degree",
    "asrank_degree_provider",
    "asrank_degree_peer",
    "asrank_degree_customer",
    "asrank_degree_total",
    "asrank_degree_transit",
    "asrank_degree_sibling",
    "asrank_cone_asns",
    "asrank_cone_prefixes",
    "asrank_cone_addresses",
    "asrank_rank",
    "asrank_node_missing",
    "asrank_node_conflict",
    "asrel_node_missing",
    "asrank_links_node_missing",
)


SHAPE_FIELDS = (
    "public_b1",
    "public_l2_compatible_prefixes",
    "public_l2_known_compatible_prefixes",
    "public_l2_unknown_prefixes",
    "public_l2_distinct_asns",
    "public_l2_shared_asns",
    "public_l2_shared_prefixes",
    "public_l2_max_parent_count",
    "public_l2_merge_fraction",
    "public_l2_mean_children",
    "public_l2_max_children",
    "public_l2_zero_child_firsthops",
    "public_l2_known_valley_rejected",
    "public_l2_loop_rejected",
    "public_first_branch_depth",
    "public_provider_l1_branches",
    "public_provider_l2_prefixes",
    "public_provider_l2_unique_asns",
    "public_provider_l2_shared_asns",
    "public_provider_l2_max_parent_count",
    "public_provider_l2_zero_child_branches",
    "public_provider_l1_with_provider_expansion",
    "public_provider_l2_branching_fraction",
    "public_provider_first_branch_depth",
)


SOURCE_FEATURE_NAMES = NODE_NUMERIC_FIELDS + SHAPE_FIELDS


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def _asn(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid ASN: " + repr(value))
    if isinstance(value, bool) or not 0 < number < 2**32:
        raise ValueError("Invalid ASN: " + repr(value))
    return number


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _roles(mask):
    return {role for role in ROLES if mask & BITS[role]}


def _reverse_mask(mask):
    return sum(BITS[INVERSE[role]] for role in _roles(mask))


def commercial_phases(roles, initial=("UP",)):
    """Three-state evidence judgment under the valley-free abstraction.

    UP allows c2p, one peer transition, or descent. DOWN allows only descent.
    An unknown/conflicting edge admits possible phases but is never certified.
    """
    phases = set(initial)
    uncertain = False
    for role in roles:
        alternatives = {role}
        if role in {"unknown", "conflict"}:
            alternatives = {"c2p", "p2p", "p2c"}
            uncertain = True
        following = set()
        for phase in phases:
            for possibility in alternatives:
                if possibility == "s2s":
                    following.add(phase)
                elif possibility == "c2p" and phase == "UP":
                    following.add("UP")
                elif possibility in {"p2p", "p2c"} and phase == "UP":
                    following.add("DOWN")
                elif possibility == "p2c" and phase == "DOWN":
                    following.add("DOWN")
        phases = following
        if not phases:
            return dict(status="incompatible", possible_phases=[], unknown_evidence=uncertain)
    return dict(
        status="unknown" if uncertain else "compatible",
        possible_phases=sorted(phases),
        unknown_evidence=uncertain,
    )


class PublicPrefixTopology:
    """Shared public graph API for candidate construction and model context."""

    def __init__(self, date, asrel_path, asrank_nodes_path, asrank_links_path):
        self.date = str(date)
        datetime.strptime(self.date, "%Y%m%d")
        if len(self.date) != 8:
            raise ValueError("Date must be YYYYMMDD")
        self.paths = {
            "asrel": str(Path(asrel_path).resolve()),
            "asrank_nodes": str(Path(asrank_nodes_path).resolve()),
            "asrank_links": str(Path(asrank_links_path).resolve()),
        }
        self.inputs_sha256 = {}
        self.statistics = {}
        self._intern = {}
        self._nodes = {}
        self._node_conflicts = set()
        self._adjacency = defaultdict(set)
        self._source_nodes = {"asrel": set(), "asrank_links": set()}
        # Undirected canonical pair -> source masks/counts and raw link counts.
        self._edges = {}
        self._load_nodes()
        self._load_edges("asrel")
        self._load_edges("asrank_links")
        self._role_adjacency = defaultdict(lambda: defaultdict(set))
        for (left, right), record in self._edges.items():
            role = self._resolved_role(record[0] | record[1])
            self._role_adjacency[left][role].add(right)
            self._role_adjacency[right][INVERSE.get(role, role)].add(left)
        identity = dict(date=self.date, inputs_sha256=self.inputs_sha256)
        self.graph_identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.statistics["graph"] = dict(
            nodes_with_public_edges=len(self._adjacency),
            undirected_public_edges=len(self._edges),
            conflicting_role_edges=sum(
                self._resolved_role(r[0] | r[1]) == "conflict" for r in self._edges.values()
            ),
            unknown_role_edges=sum(
                self._resolved_role(r[0] | r[1]) == "unknown" for r in self._edges.values()
            ),
        )

    @classmethod
    def from_manifest(cls, manifest):
        data = _read(manifest) if isinstance(manifest, (str, Path)) else manifest
        files = data["public_files"]
        return cls(data["date"], files["asrel"], files["asrank_nodes"], files["asrank_links"])

    def _canonical_asn(self, value):
        number = _asn(value)
        return self._intern.setdefault(number, number)

    def _rows(self, kind):
        digest = hashlib.sha256()
        with Path(self.paths[kind]).open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                yield raw
        self.inputs_sha256[self.paths[kind]] = digest.hexdigest()

    def _check_date(self, row, counts):
        value = row.get("date")
        if not value:
            counts["undated_rows"] += 1
            return
        actual = str(value)[:10].replace("-", "")
        if actual != self.date:
            raise ValueError(
                "Public input row date differs; no nearby snapshot fallback: " + str(value)
            )

    def _load_nodes(self):
        counts = Counter()
        fields = Counter()
        clique = Counter()
        for raw in self._rows("asrank_nodes"):
            if not raw.strip():
                continue
            row = json.loads(raw)
            counts["rows"] += 1
            self._check_date(row, counts)
            fields.update(row.keys())
            clique[str(row.get("cliqueMember"))] += 1
            try:
                asn = self._canonical_asn(row.get("asn"))
            except ValueError:
                counts["invalid_asn"] += 1
                continue
            node = {
                key: row.get(key)
                for key in (
                    "date",
                    "asnName",
                    "rank",
                    "cone",
                    "asnDegree",
                    "organization",
                    "country",
                    "seen",
                    "announcing",
                )
            }
            if asn in self._nodes:
                counts["duplicate_asn_rows"] += 1
                if self._nodes[asn] != node:
                    self._node_conflicts.add(asn)
            else:
                self._nodes[asn] = node
        self.statistics["asrank_nodes"] = dict(
            counts,
            unique_asns=len(self._nodes),
            conflicting_duplicate_asns=len(self._node_conflicts),
            field_occurrences=dict(fields),
            cliqueMember_raw_distribution=dict(clique),
            cliqueMember_used_for_classification=False,
        )

    def _load_edges(self, kind):
        counts = Counter()
        relation_counts = Counter()
        for raw in self._rows(kind):
            if not raw.strip() or raw.startswith(b"#"):
                continue
            counts["rows"] += 1
            if kind == "asrel":
                columns = raw.decode().strip().split("|")
                if len(columns) < 3:
                    counts["malformed_rows"] += 1
                    continue
                a, b = columns[:2]
                relation = {"-1": "p2c", "0": "p2p", "1": "c2p"}.get(columns[2], "unknown")
                number_paths = None
            else:
                row = json.loads(raw)
                self._check_date(row, counts)
                a, b = (row.get("asn0") or {}).get("asn"), (row.get("asn1") or {}).get("asn")
                relation = {
                    "provider": "c2p",
                    "customer": "p2c",
                    "peer": "p2p",
                    "sibling": "s2s",
                }.get(row.get("relationship"), "unknown")
                number_paths = _number(row.get("numberPaths"))
            relation_counts[relation] += 1
            try:
                a, b = self._canonical_asn(a), self._canonical_asn(b)
            except ValueError:
                counts["invalid_asn"] += 1
                continue
            if a == b:
                counts["self_loop_rows"] += 1
                continue
            left, right = min(a, b), max(a, b)
            normalized = relation if a == left else INVERSE[relation]
            record = self._edges.setdefault((left, right), [0, 0, 0, 0, None, None, 0.0])
            index = 0 if kind == "asrel" else 1
            record[index] |= BITS[normalized]
            record[2 + index] += 1
            if kind == "asrank_links" and number_paths is not None:
                record[4] = number_paths if record[4] is None else min(record[4], number_paths)
                record[5] = number_paths if record[5] is None else max(record[5], number_paths)
                record[6] += number_paths
            self._adjacency[a].add(b)
            self._adjacency[b].add(a)
            self._source_nodes[kind].update((a, b))
        self.statistics[kind] = dict(counts, relation_rows=dict(relation_counts))

    @staticmethod
    def _resolved_role(mask):
        known = _roles(mask) - {"unknown"}
        return next(iter(known)) if len(known) == 1 else "conflict" if len(known) > 1 else "unknown"

    def neighbors(self, asn):
        return tuple(map(str, sorted(self._adjacency.get(_asn(asn), ()))))

    def public_role(self, left, right):
        a, b = _asn(left), _asn(right)
        record = self._edges.get((min(a, b), max(a, b)))
        if record is None:
            return "unknown"
        role = self._resolved_role(record[0] | record[1])
        return role if a <= b else INVERSE.get(role, role)

    def edge_evidence(self, left, right):
        a, b = _asn(left), _asn(right)
        record = self._edges.get((min(a, b), max(a, b)))
        if record is None:
            return dict(
                left=str(a),
                right=str(b),
                public_role="unknown",
                public_edge_observed=False,
                roles_by_source={},
                data_coverage_unknown=True,
            )
        masks = record[:2] if a < b else list(map(_reverse_mask, record[:2]))
        by_source = {
            name: sorted(_roles(mask))
            for name, mask in zip(("asrel", "asrank_links"), masks)
            if mask
        }
        role = self.public_role(a, b)
        return dict(
            left=str(a),
            right=str(b),
            public_role=role,
            public_edge_observed=True,
            roles_by_source=by_source,
            role_conflict=role == "conflict",
            unknown_evidence_present=any("unknown" in values for values in by_source.values()),
            asrel_record_count=record[2],
            asrank_link_record_count=record[3],
            asrank_numberPaths_raw=dict(min=record[4], max=record[5], sum_over_records=record[6]),
            numberPaths_is_routing_probability=False,
            public_role_is_inferred=True,
            data_coverage_unknown=True,
        )

    @lru_cache(maxsize=131072)
    def node_attributes(self, asn):
        node = _asn(asn)
        raw = self._nodes.get(node)
        conflict = node in self._node_conflicts
        roles = self._role_adjacency.get(node, {})
        degree = {label: len(roles.get(role, ())) for role, label in ROLE_DEGREE.items()}
        degree["total"] = len(self._adjacency.get(node, ()))
        numeric = {"public_" + key + "_degree": value for key, value in degree.items()}
        reported = (raw or {}).get("asnDegree") or {}
        cone = (raw or {}).get("cone") or {}
        for key in ("provider", "peer", "customer", "total", "transit", "sibling"):
            numeric["asrank_degree_" + key] = None if conflict else _number(reported.get(key))
        for suffix, key in (
            ("asns", "numberAsns"),
            ("prefixes", "numberPrefixes"),
            ("addresses", "numberAddresses"),
        ):
            numeric["asrank_cone_" + suffix] = None if conflict else _number(cone.get(key))
        numeric.update(
            asrank_rank=None if conflict else _number((raw or {}).get("rank")),
            asrank_node_missing=int(raw is None),
            asrank_node_conflict=int(conflict),
            asrel_node_missing=int(node not in self._source_nodes["asrel"]),
            asrank_links_node_missing=int(node not in self._source_nodes["asrank_links"]),
        )
        if set(numeric) != set(NODE_NUMERIC_FIELDS):
            raise RuntimeError("Public node numeric schema differs")
        return dict(
            asn=str(node),
            asrank=raw,
            graph_role_degree=degree,
            numeric=numeric,
            coverage=dict(
                asrank_node_present=raw is not None,
                asrank_node_conflict=conflict,
                complete_topology_claim=False,
                unobserved_edge_means="unknown, not absent",
            ),
            cliqueMember_used_for_classification=False,
        )

    def _children(self, source, neighbor):
        """Use compact role sets; do not materialize candidate node objects."""
        source, neighbor = _asn(source), _asn(neighbor)
        first_role = self.public_role(source, neighbor)
        phase = commercial_phases([first_role])
        all_neighbors = self._adjacency.get(neighbor, set())
        roles = self._role_adjacency.get(neighbor, {})
        if "UP" in phase["possible_phases"]:
            allowed = all_neighbors - {source, neighbor}
        else:
            allowed = set().union(
                *(roles.get(role, ()) for role in ("p2c", "s2s", "unknown", "conflict"))
            ) - {source, neighbor}
        uncertain = (
            allowed
            if phase["unknown_evidence"]
            else allowed & (roles.get("unknown", set()) | roles.get("conflict", set()))
        )
        return (
            allowed,
            uncertain,
            dict(
                loop_rejected=int(source in all_neighbors) + int(neighbor in all_neighbors),
                known_valley_rejected=len(all_neighbors - {source, neighbor}) - len(allowed),
            ),
        )

    def second_hop_summary(self, source, neighbor, include_asns=False):
        """All public commercial candidates after S,N; no learned pruning."""
        source, neighbor = _asn(source), _asn(neighbor)
        children, unknown, rejected = self._children(source, neighbor)
        role = self.public_role(source, neighbor)
        result = dict(
            source=str(source),
            neighbor=str(neighbor),
            firsthop_role=role,
            firsthop_edge_observed=neighbor in self._adjacency.get(source, set()),
            possible_phases=commercial_phases([role])["possible_phases"],
            total_count=len(children),
            known_count=len(children) - len(unknown),
            unknown_count=len(unknown),
            data_coverage_unknown=True,
            TopK_pruning=False,
            **rejected,
        )
        if include_asns:
            result.update(
                all_asns=list(map(str, sorted(children))),
                known_asns=list(map(str, sorted(children - unknown))),
                unknown_asns=list(map(str, sorted(unknown))),
            )
        return result

    @lru_cache(maxsize=2048)
    def source_features(self, source):
        source = _asn(source)
        features = dict(self.node_attributes(source)["numeric"])
        first = self._adjacency.get(source, set())
        parents, sizes = Counter(), []
        unknown = loops = valleys = 0
        for neighbor in first:
            children, uncertain, rejected = self._children(source, neighbor)
            parents.update(children)
            sizes.append(len(children))
            unknown += len(uncertain)
            loops += rejected["loop_rejected"]
            valleys += rejected["known_valley_rejected"]
        leaves = sum(sizes)
        shared = [count for count in parents.values() if count > 1]
        providers = self._role_adjacency.get(source, {}).get("c2p", set())
        provider_parents, provider_sizes = Counter(), []
        for neighbor in providers:
            children = self._role_adjacency.get(neighbor, {}).get("c2p", set()) - {
                source,
                neighbor,
            }
            provider_parents.update(children)
            provider_sizes.append(len(children))
        features.update(
            public_b1=len(first),
            public_l2_compatible_prefixes=leaves,
            public_l2_known_compatible_prefixes=leaves - unknown,
            public_l2_unknown_prefixes=unknown,
            public_l2_distinct_asns=len(parents),
            public_l2_shared_asns=len(shared),
            public_l2_shared_prefixes=sum(shared),
            public_l2_max_parent_count=max(parents.values(), default=0),
            public_l2_merge_fraction=sum(shared) / leaves if leaves else 0.0,
            public_l2_mean_children=leaves / len(first) if first else 0.0,
            public_l2_max_children=max(sizes, default=0),
            public_l2_zero_child_firsthops=sum(size == 0 for size in sizes),
            public_l2_known_valley_rejected=valleys,
            public_l2_loop_rejected=loops,
            public_first_branch_depth=1
            if len(first) > 1
            else 2
            if len(first) == 1 and sizes[0] > 1
            else 0,
            public_provider_l1_branches=len(providers),
            public_provider_l2_prefixes=sum(provider_sizes),
            public_provider_l2_unique_asns=len(provider_parents),
            public_provider_l2_shared_asns=sum(count > 1 for count in provider_parents.values()),
            public_provider_l2_max_parent_count=max(provider_parents.values(), default=0),
            public_provider_l2_zero_child_branches=sum(size == 0 for size in provider_sizes),
            public_provider_l1_with_provider_expansion=sum(size > 0 for size in provider_sizes),
            public_provider_l2_branching_fraction=sum(size > 1 for size in provider_sizes)
            / len(providers)
            if providers
            else 0.0,
            public_provider_first_branch_depth=1
            if len(providers) > 1
            else 2
            if len(providers) == 1 and provider_sizes[0] > 1
            else 0,
        )
        if set(features) != set(SOURCE_FEATURE_NAMES):
            raise RuntimeError("Public source feature schema differs")
        return features

    def _prefix_node(self, prefix, decision):
        prefix = tuple(map(str, prefix))
        evidence = [self.edge_evidence(a, b) for a, b in zip(prefix, prefix[1:])]
        roles = [edge["public_role"] for edge in evidence]
        policy = commercial_phases(roles)
        return dict(
            fullprefix=list(prefix),
            asn=prefix[-1],
            AS_count=len(prefix),
            edge_evidence=evidence,
            publicrole=roles[-1] if roles else "source",
            public_roles=roles,
            commercial_compatibility=policy,
            asrankattrs=self.node_attributes(prefix[-1])["asrank"],
            node_attributes=self.node_attributes(prefix[-1]),
            decision=decision,
            tree_role="public_candidate_only_not_model_admitted",
            graph_identity_sha256=self.graph_identity_sha256,
            data_coverage_unknown=True,
        )

    def build_source_tree(self, source, full=True):
        source = _asn(source)
        first = sorted(self._adjacency.get(source, ()))
        features = self.source_features(source)
        provider_rows = []
        for neighbor in sorted(self._role_adjacency.get(source, {}).get("c2p", ())):
            providers = sorted(
                self._role_adjacency.get(neighbor, {}).get("c2p", set()) - {source, neighbor}
            )
            provider_rows.append(
                dict(
                    provider=str(neighbor),
                    source_edge=self.edge_evidence(source, neighbor),
                    asrankattrs=self.node_attributes(neighbor)["asrank"],
                    second_hop_providers=[
                        dict(
                            asn=str(a),
                            edge_evidence=self.edge_evidence(neighbor, a),
                            asrankattrs=self.node_attributes(a)["asrank"],
                        )
                        for a in providers
                    ],
                    upstream_coverage_unknown=True,
                )
            )
        nodes = []
        merges = defaultdict(list)
        if full:
            nodes.append(
                self._prefix_node(
                    [source],
                    dict(
                        action="expand" if first else "stop",
                        reason="public_neighbors_observed"
                        if first
                        else "no_public_neighbors_observed",
                        child_count=len(first),
                    ),
                )
            )
            for neighbor in first:
                children, uncertain, rejected = self._children(source, neighbor)
                decision = dict(
                    action="expand" if children else "stop",
                    reason="public_commercial_candidates"
                    if children
                    else "no_public_commercial_extension_observed",
                    child_count=len(children),
                    unknown_children=len(uncertain),
                    **rejected,
                )
                nodes.append(self._prefix_node([source, neighbor], decision))
                for child in sorted(children):
                    merges[child].append(str(neighbor))
                    nodes.append(
                        self._prefix_node(
                            [source, neighbor, child],
                            dict(
                                action="stop",
                                reason="max_prefix_AS_count_3",
                                max_prefix_AS_count=3,
                                stop_is_observed_topology_leaf=False,
                                further_public_degree=self.node_attributes(child)["numeric"][
                                    "public_total_degree"
                                ],
                            ),
                        )
                    )
        return dict(
            version=VERSION,
            date=self.date,
            source=str(source),
            max_prefix_AS_count=3,
            tree_role="public_candidate_only_not_model_admitted",
            numerical_model_scores_read=False,
            route_reference_files_read=False,
            TopK_pruning=False,
            source_features=features,
            source_attributes=self.node_attributes(source),
            provider_structure=provider_rows,
            provider_attachment="no_provider_observed"
            if not provider_rows
            else "single_provider_observed"
            if len(provider_rows) == 1
            else "multiple_providers_observed",
            public_customer_role="has_public_customers"
            if features["public_customer_degree"]
            else "no_public_customers_observed",
            nodes=nodes if full else None,
            complete_candidate_tree_written=bool(full),
            shared_A=[
                dict(
                    asn=str(a),
                    firsthop_parents=parents,
                    physical_AS_copies=len(parents),
                )
                for a, parents in sorted(merges.items())
                if len(parents) > 1
            ]
            if full
            else None,
            graph_inputs_sha256=self.inputs_sha256,
            graph_identity_sha256=self.graph_identity_sha256,
            coverage=dict(
                complete_topology_claim=False,
                missing_edges_are_unknown=True,
                ASRank_is_routing_probability=False,
                ASRank_cliqueMember_used=False,
            ),
            projection="Path-prefix unfolding preserves every retained S,N,A path and duplicates shared A under each N; no arbitrary spanning-tree parent is selected",
            policy_scope="Commercial compatibility uses the valley-free abstraction; it is not proof of actual forwarding or export policies",
        )

    def receipt(self):
        return dict(
            version=VERSION,
            date=self.date,
            graph_inputs_sha256=self.inputs_sha256,
            graph_identity_sha256=self.graph_identity_sha256,
            statistics=self.statistics,
            node_numeric_fields=list(NODE_NUMERIC_FIELDS),
            source_feature_names=list(SOURCE_FEATURE_NAMES),
            public_only=True,
            route_files_read=False,
            model_scores_read=False,
            unknown_policy="unobserved edges unknown; conflicting roles retained, never overwritten",
            data_scope="ASRank and ASREL are public inference snapshots, not forwarding truth or traffic probabilities",
        )
