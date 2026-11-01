"""Unified255 public computations."""
from __future__ import annotations
from collections import defaultdict
import json
import math
from pathlib import Path
import re


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _asn(value):
    try:
        result = int(value)
        return result if 0 < result < 2**32 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _day(value):
    match = re.search("(20\\d{2})[-_]?(\\d{2})[-_]?(\\d{2})", str(value))
    return "".join(match.groups()) if match else ""


def _past_file(directories, pattern, date):
    candidates = []
    for directory in directories:
        for path in Path(directory).glob(pattern):
            day = _day(path.name)
            if day and day <= date and path.is_file() and path.stat().st_size:
                candidates.append((day, str(path.resolve()), path))
    return max(candidates, default=(None, None, None))[-1]


def _reverse_role(role):
    return {"provider": "customer", "customer": "provider", "peer": "peer"}.get(role, "unknown")


def _role_relation(role):
    return {"provider": "c2p", "customer": "p2c", "peer": "p2p"}.get(role, "unknown")


def _token(value):
    return re.sub("[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


class PublicKnowledge:
    """Read-only, past-or-current public knowledge with a fixed feature schema."""

    NODE_NUMBERS = (
        "cone_asns",
        "cone_prefixes",
        "cone_addresses",
        "degree_provider",
        "degree_peer",
        "degree_customer",
        "degree_total",
        "degree_transit",
        "announcing_prefixes",
        "announcing_addresses",
    )
    TYPES = (
        "nsp",
        "content",
        "cable_dsl_isp",
        "enterprise",
        "educational_research",
        "non_profit",
        "route_server",
        "network_services",
        "not_disclosed",
        "other",
    )
    SCOPES = (
        "global",
        "regional",
        "north_america",
        "south_america",
        "europe",
        "asia_pacific",
        "africa",
        "australia",
        "not_disclosed",
        "other",
    )
    POLICIES = ("open", "selective", "restrictive", "no", "not_disclosed", "other")

    def __init__(self, repo_root, files_root, date):
        (self.repo_root, self.files_root) = (Path(repo_root), Path(files_root))
        self.date = _day(date)
        if not self.date:
            raise ValueError("Expected a YYYYMMDD date")
        self.paths = {}
        self.adjacency = defaultdict(set)
        (self.links, self.nodes, self.relations) = ({}, {}, {})
        self.asrel_relations = {}
        self.skipped_future_rows = {"asrank_links": 0, "asrank_nodes": 0}
        self.history_enabled = False
        self.history_available = False
        self.paths["history"] = []
        (self._nets, self._ixps, self._facilities, self._countries) = ({}, {}, {}, {})
        (self._link_totals, self._link_ranks) = ({}, {})
        self._load_asrel()
        self._load_asrank()
        self._load_peeringdb()
        self.asrank_links_available = bool(self.paths.get("asrank_links"))
        self.asrank_nodes_available = bool(self.paths.get("asrank_nodes"))
        self.asrank_available = self.asrank_links_available or self.asrank_nodes_available
        self.asrel_available = bool(self.paths.get("asrel"))
        by_source = defaultdict(list)
        for (left, right), edge in self.links.items():
            by_source[left].append((right, edge["number_paths"]))
        for source, neighbors in by_source.items():
            self._link_totals[source] = sum((count for (_, count) in neighbors))
            self._link_ranks[source] = {
                neighbor: rank
                for (rank, (neighbor, _)) in enumerate(
                    sorted(neighbors, key=lambda item: (-item[1], item[0])), 1
                )
            }
        self.adjacency = dict(self.adjacency)

    def _load_asrel(self):
        path = _past_file([self.files_root / "data/aspath"], "*.as-rel.txt", self.date)
        self.paths["asrel"] = str(path.resolve()) if path else None
        if path is None:
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                columns = line.strip().split("|")
                if len(columns) < 3:
                    continue
                (left, right) = (_asn(columns[0]), _asn(columns[1]))
                if left is None or right is None or left == right:
                    continue
                relation = {"-1": "p2c", "0": "p2p", "1": "c2p"}.get(columns[2], "unknown")
                inverse = {"p2c": "c2p", "c2p": "p2c", "p2p": "p2p"}.get(relation, "unknown")
                self.asrel_relations[left, right] = relation
                self.asrel_relations[right, left] = inverse
                self.relations[left, right] = relation
                self.relations[right, left] = inverse
                self.adjacency[left].add(right)
                self.adjacency[right].add(left)

    def _rows(self, path, kind):
        if path is None:
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_date = _day(row.get("date", ""))
                if record_date and record_date > self.date:
                    self.skipped_future_rows[kind] += 1
                    continue
                yield row

    def _load_asrank(self):
        directories = [self.repo_root / "asrank_data", self.files_root / "data/ASRank"]
        links_path = _past_file(directories, "*links.jsonl", self.date)
        nodes_path = _past_file(directories, "asns*.jsonl", self.date)
        self.paths["asrank_links"] = str(links_path.resolve()) if links_path else None
        self.paths["asrank_nodes"] = str(nodes_path.resolve()) if nodes_path else None
        for row in self._rows(links_path, "asrank_links"):
            left = _asn((row.get("asn0") or {}).get("asn"))
            right = _asn((row.get("asn1") or {}).get("asn"))
            if left is None or right is None or left == right:
                continue
            role = str(row.get("relationship") or "unknown").lower()
            role = role if role in {"provider", "customer", "peer"} else "unknown"
            count = _number(row.get("numberPaths"))
            for source, neighbor, neighbor_role in (
                (left, right, role),
                (right, left, _reverse_role(role)),
            ):
                self.links[source, neighbor] = {"number_paths": count, "role": neighbor_role}
                if self.relations.get((source, neighbor), "unknown") == "unknown":
                    self.relations[source, neighbor] = _role_relation(neighbor_role)
                self.adjacency[source].add(neighbor)
        for row in self._rows(nodes_path, "asrank_nodes"):
            asn = _asn(row.get("asn"))
            if asn is None:
                continue
            (cone, degree) = (row.get("cone") or {}, row.get("asnDegree") or {})
            announcing = row.get("announcing") or {}
            country = row.get("country") or {}
            organization = row.get("organization") or {}
            self.nodes[asn] = {
                "rank": _number(row.get("rank")),
                "cone_asns": _number(cone.get("numberAsns")),
                "cone_prefixes": _number(cone.get("numberPrefixes")),
                "cone_addresses": _number(cone.get("numberAddresses")),
                **{
                    "degree_" + key: _number(degree.get(key))
                    for key in ("provider", "peer", "customer", "total", "transit")
                },
                "announcing_prefixes": _number(announcing.get("numberPrefixes")),
                "announcing_addresses": _number(announcing.get("numberAddresses")),
                "country": str(country.get("iso") or "")
                if isinstance(country, dict)
                else str(country),
                "organization": str(organization.get("orgId") or "")
                if isinstance(organization, dict)
                else str(organization),
                "clique_member": float(bool(row.get("cliqueMember"))),
            }

    def _load_peeringdb(self):
        path = _past_file([self.repo_root / "peeringdb_data"], "peeringdb_2_dump_*.json", self.date)
        self.paths["peeringdb"] = str(path.resolve()) if path else None
        self.peeringdb_available = path is not None
        self.pdb_available = self.peeringdb_available
        self.pdb_counts = {"net": 0, "netixlan": 0, "netfac": 0, "fac": 0, "ix": 0}
        if path is None:
            return
        with path.open(encoding="utf-8") as handle:
            dump = json.load(handle)

        def active(section):
            value = dump.get(section) or {}
            records = value.get("data", []) if isinstance(value, dict) else value
            records = [row for row in records if str(row.get("status", "ok")) == "ok"]
            if section in self.pdb_counts:
                self.pdb_counts[section] = len(records)
            return records

        nets = active("net")
        self._nets = {_asn(row.get("asn")): row for row in nets if _asn(row.get("asn")) is not None}
        network_asn = {row.get("id"): _asn(row.get("asn")) for row in nets}
        fac_country = {row.get("id"): str(row.get("country") or "") for row in active("fac")}
        ix_country = {row.get("id"): str(row.get("country") or "") for row in active("ix")}
        lan_ix = {row.get("id"): row.get("ix_id") for row in active("ixlan")}
        (ixps, facilities, countries) = (defaultdict(set), defaultdict(set), defaultdict(set))
        for row in active("netixlan"):
            asn = _asn(row.get("asn")) or network_asn.get(row.get("net_id"))
            ix = row.get("ix_id") or lan_ix.get(row.get("ixlan_id"))
            if asn is not None and ix is not None:
                ixps[asn].add(ix)
                if ix_country.get(ix):
                    countries[asn].add(ix_country[ix])
        for row in active("netfac"):
            asn = (
                _asn(row.get("local_asn"))
                or _asn(row.get("asn"))
                or network_asn.get(row.get("net_id"))
            )
            fac = row.get("fac_id")
            if asn is not None and fac is not None:
                facilities[asn].add(fac)
                if fac_country.get(fac):
                    countries[asn].add(fac_country[fac])
        (self._ixps, self._facilities, self._countries) = (
            dict(ixps),
            dict(facilities),
            dict(countries),
        )
        self.pdb_counts["unique_asns"] = len(self._nets)
        self.pdb_counts["asns_with_ixp"] = len(self._ixps)
        self.pdb_counts["asns_with_facility"] = len(self._facilities)
        self.pdb_counts["asns_with_country_presence"] = len(self._countries)

    def _node_features(self, asn, side):
        (node, net) = (self.nodes.get(asn, {}), self._nets.get(asn, {}))
        (ext, pb) = ("ext_" + side + "_", "pb_" + side + "_")
        rank = node.get("rank", 0.0)
        values = {
            ext + "node_missing": float(not node),
            ext + "rank_missing": float(rank <= 0),
            ext + "rank_inverse": 1.0 / (1.0 + rank) if rank > 0 else 0.0,
            ext + "country_missing": float(not node.get("country")),
            ext + "organization_missing": float(not node.get("organization")),
            ext + "clique_member": node.get("clique_member", 0.0),
            ext + "public_degree_log1p": math.log1p(len(self.adjacency.get(asn, ()))),
            pb + "net_missing": float(not net),
            pb + "ixp_presence_missing": float(not self._ixps.get(asn)),
            pb + "facility_presence_missing": float(not self._facilities.get(asn)),
            pb + "country_presence_missing": float(not self._countries.get(asn)),
        }
        for field in self.NODE_NUMBERS:
            values[ext + field + "_log1p"] = math.log1p(node.get(field, 0.0))
        for field in ("info_prefixes4", "info_prefixes6"):
            values[pb + field + "_log1p"] = math.log1p(_number(net.get(field)))
            values[pb + field + "_missing"] = float(net.get(field) is None)
        for field, vocabulary in (
            ("info_type", self.TYPES),
            ("info_scope", self.SCOPES),
            ("policy_general", self.POLICIES),
        ):
            category = _token(net.get(field))
            values[pb + field + "_missing"] = float(not category or category == "not_disclosed")
            category = category if category in vocabulary else "other"
            for name in vocabulary:
                values[pb + field + "_" + name] = float(bool(net) and category == name)
        for name, collection in (
            ("ixp", self._ixps),
            ("facility", self._facilities),
            ("country", self._countries),
        ):
            values[pb + name + "_count_log1p"] = math.log1p(len(collection.get(asn, ())))
        for field in ("policy_ratio", "policy_contracts"):
            value = net.get(field)
            values[pb + field + "_missing"] = float(value is None)
            values[pb + field + "_yes"] = float(value is True)
        return values

    def pair_features(self, left: int, right: int) -> dict[str, float]:
        (left, right) = (int(left), int(right))
        values = self._node_features(left, "source")
        values.update(self._node_features(right, "candidate"))
        relation = self.relations.get((left, right), "unknown")
        explicit = self.asrel_relations.get((left, right), "unknown")
        edge = self.links.get((left, right), {})
        role = edge.get("role", "unknown")
        count = edge.get("number_paths", 0.0)
        rank = self._link_ranks.get(left, {}).get(right, 0)
        total = self._link_totals.get(left, 0.0)
        values.update(
            {
                "ext_asrel_pair_missing": float((left, right) not in self.asrel_relations),
                "ext_asrank_pair_missing": float(not edge),
                "ext_relation_missing": float(relation == "unknown"),
                "ext_link_number_paths_log1p": math.log1p(count),
                "ext_link_local_share": count / total if total > 0 else 0.0,
                "ext_link_local_rank_inverse": 1.0 / rank if rank > 0 else 0.0,
                "ext_role_agreement": float(
                    explicit != "unknown" and _role_relation(role) == explicit
                ),
                "ext_role_disagreement": float(
                    explicit != "unknown"
                    and role != "unknown"
                    and (_role_relation(role) != explicit)
                ),
                "ext_history_missing": 1.0,
                "ext_history_available": 0.0,
                "pb_snapshot_missing": float(not self.peeringdb_available),
            }
        )
        for name in ("c2p", "p2p", "p2c", "unknown"):
            values["ext_relation_" + name] = float(relation == name)
        for name in ("provider", "peer", "customer", "unknown"):
            values["ext_asrank_role_" + name] = float(role == name)
        (source_node, candidate_node) = (self.nodes.get(left, {}), self.nodes.get(right, {}))
        for field in self.NODE_NUMBERS:
            (source, candidate) = (source_node.get(field, 0.0), candidate_node.get(field, 0.0))
            values["ext_pair_" + field + "_log_difference"] = math.log1p(candidate) - math.log1p(
                source
            )
            values["ext_pair_" + field + "_candidate_fraction"] = (
                candidate / (source + candidate) if source + candidate > 0 else 0.0
            )
        for field in ("country", "organization"):
            (source, candidate) = (source_node.get(field, ""), candidate_node.get(field, ""))
            values["ext_pair_same_" + field] = float(
                bool(source and candidate) and source == candidate
            )
            values["ext_pair_" + field + "_missing"] = float(not (source and candidate))
        (source_net, candidate_net) = (self._nets.get(left, {}), self._nets.get(right, {}))
        for field in ("info_type", "info_scope", "policy_general"):
            (source, candidate) = (source_net.get(field), candidate_net.get(field))
            values["pb_pair_same_" + field] = float(
                bool(source and candidate) and source == candidate
            )
            values["pb_pair_" + field + "_missing"] = float(not (source and candidate))
        for name, collection in (
            ("ixp", self._ixps),
            ("facility", self._facilities),
            ("country", self._countries),
        ):
            (source, candidate) = (collection.get(left, set()), collection.get(right, set()))
            (shared, union) = (len(source & candidate), len(source | candidate))
            values["pb_pair_shared_" + name + "_log1p"] = math.log1p(shared)
            values["pb_pair_" + name + "_jaccard"] = shared / union if union else 0.0
            values["pb_pair_" + name + "_source_fraction"] = shared / len(source) if source else 0.0
            values["pb_pair_" + name + "_candidate_fraction"] = (
                shared / len(candidate) if candidate else 0.0
            )
            values["pb_pair_" + name + "_either_presence_missing"] = float(
                not source or not candidate
            )
        shared_location = float(
            bool(
                self._ixps.get(left, set()) & self._ixps.get(right, set())
                or self._facilities.get(left, set()) & self._facilities.get(right, set())
            )
        )
        values["pb_pair_shared_location"] = shared_location
        for rel in ("c2p", "p2p", "p2c"):
            flag = float(relation == rel)
            values["ext_interaction_" + rel + "_candidate_transit_log1p"] = flag * math.log1p(
                candidate_node.get("degree_transit", 0.0)
            )
            values["pb_interaction_" + rel + "_shared_location"] = flag * shared_location
            for kind in ("nsp", "content", "cable_dsl_isp", "enterprise", "route_server"):
                values["pb_interaction_" + rel + "_candidate_" + kind] = flag * float(
                    _token(candidate_net.get("info_type")) == kind
                )
        for source_kind in ("nsp", "content", "cable_dsl_isp", "enterprise"):
            for candidate_kind in ("nsp", "content", "cable_dsl_isp", "enterprise"):
                values["pb_pair_type_" + source_kind + "_to_" + candidate_kind] = float(
                    _token(source_net.get("info_type")) == source_kind
                    and _token(candidate_net.get("info_type")) == candidate_kind
                )
        return {key: float(value) for (key, value) in values.items()}
