"""Unified255 path features computations."""
from functools import lru_cache
import math


SCHEMA_VERSION = "path_evidence_v1"
RELATIONS = ("c2p", "p2p", "p2c", "s2s", "unknown")
SUPPORT_FIELDS = ("raw_log1p", "origin_support_log1p", "known", "origin_support_missing")
DEST_FIELDS = (
    "context_missing",
    "start_missing",
    "destination_missing",
    "via_missing",
    "start_route_mass_log1p",
    "destination_route_mass_log1p",
    "destination_path_count_log1p",
    "destination_fraction_of_start_mass",
    "via_route_mass_log1p",
    "via_path_count_log1p",
    "via_fraction_of_destination_mass",
    "via_fraction_of_destination_paths",
    "destination_max_route_origin_support_log1p",
    "via_max_route_origin_support_log1p",
    "destination_route_origin_support_sum_log1p",
    "via_route_origin_support_sum_log1p",
    "origin_support_missing",
)
PUBLIC_FIELDS = (
    "available",
    "source_network_missing",
    "candidate_network_missing",
    "ixp_jaccard",
    "facility_jaccard",
    "same_country",
    "shared_location",
    "source_nsp",
    "source_content",
    "source_isp",
    "candidate_nsp",
    "candidate_content",
    "candidate_isp",
)
BASIC_FIELDS = (
    "path_length",
    "prefix_length",
    "is_recursive",
    "tail_length",
    "candidate_tail_frequency_log1p",
    "candidate_tail_frequency_missing",
    "tail_length_missing",
    "max_known_suffix_length",
    "known_suffix_fraction",
    "neighbor_suffix_known_full_path_unobserved",
    "relation_unknown_edges",
    "relation_peer_edges",
    "relation_up_edges",
    "relation_down_edges",
    "relation_sibling_edges",
    "relation_unknown_fraction",
    "relation_edge_count",
    "evidence_unknown_edges",
    "evidence_unknown_edge_fraction",
    "prefix_evidence_unknown_edges",
    "prefix_relation_unknown_edges",
)
PRIOR_FIELDS = (
    "source_visible_degree_log1p",
    "source_visible_degree_missing",
    "root_rank_log1p",
    "root_rank_inverse",
    "root_rank_missing",
    "anchor_rank_log1p",
    "anchor_rank_inverse",
    "anchor_rank_missing",
)


def _schema():
    keys = list(BASIC_FIELDS)
    for context in (
        "full_path",
        "suffix_neighbor_to_destination",
        "suffix_anchor_to_destination",
        "prefix_s_n",
        "prefix_s_n_a",
        "splice_n_a_next",
        "reverse_s_n_a",
        "reverse_n_a_next",
    ):
        keys.extend((context + "_" + field for field in SUPPORT_FIELDS))
    for position in ("first", "second"):
        keys.extend(("relation_" + position + "_" + r for r in RELATIONS))
    for direction in ("forward", "reverse"):
        for statistic in ("minimum", "maximum", "mean", "sum"):
            keys.extend(
                (
                    "edge_" + direction + "_" + statistic + "_" + quantity + "_log1p"
                    for quantity in ("raw", "origin_support")
                )
            )
        keys.extend(
            (
                "edge_" + direction + "_known_fraction",
                "edge_" + direction + "_origin_support_missing_fraction",
            )
        )
    for context in ("s_n_d", "n_a_d", "a_next_d"):
        keys.extend(("destination_" + context + "_" + field for field in DEST_FIELDS))
    keys.extend(("prior_" + field for field in PRIOR_FIELDS))
    for pair in ("s_n", "n_a", "s_a"):
        keys.extend(("public_" + pair + "_" + field for field in PUBLIC_FIELDS))
    return tuple(sorted(keys))


FEATURE_NAMES = _schema()


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def _nodes(value):
    if value is None:
        return ()
    values = value.split("->") if isinstance(value, str) else value
    result = tuple((str(int(v)) for v in values))
    if any((int(v) <= 0 or int(v) >= 2**32 for v in result)):
        raise ValueError("Invalid ASN in candidate")
    return result


def _lookup(mapping, key, default=None):
    if not mapping:
        return default
    if key in mapping:
        return mapping[key]
    try:
        other = int(key)
    except (TypeError, ValueError):
        return default
    return mapping.get(other, default)


class PathEvidenceFeatures:
    """Stable numeric feature dictionaries from already-loaded safe evidence.

    priors may contain gateway[S][N] with pred_rank/priority and
    vas_graph_neighbor_count; recursive[N][A].center_evidence[S].pred_rank.
    Optional normalized source_visible_degree[S], root_rank[(S,N)] and
    recursive_rank[(S,N,A)] mappings are also accepted. Missing stays explicit.
    Inputs must remain unchanged for the lifetime of this cached extractor.
    """

    feature_names = FEATURE_NAMES
    schema_version = SCHEMA_VERSION

    def __init__(self, index, origin_support, relations, public=None, priors=None):
        self.index = index or {}
        self.origin_support = origin_support or {}
        self.relations = relations or {}
        self.public = public
        self.priors = priors or {}
        self.origin_information_available = origin_support is not None

    def _support(self, nodes):
        if len(nodes) < 2:
            return (0.0, 0.0, True)
        text = "->".join(nodes)
        raw = _number(_lookup(self.index, nodes[0], {}).get(text, 0))
        origin_counter = _lookup(self.origin_support, nodes[0], {})
        missing = not self.origin_information_available or text not in origin_counter
        return (raw, _number(origin_counter.get(text, 0)), missing)

    def _write_support(self, out, prefix, nodes):
        (raw, origins, missing) = self._support(nodes)
        out.update(
            {
                prefix + "_raw_log1p": math.log1p(raw),
                prefix + "_origin_support_log1p": math.log1p(origins),
                prefix + "_known": float(raw > 0),
                prefix + "_origin_support_missing": float(missing),
            }
        )

    @lru_cache(maxsize=100000)
    def _relation(self, left, right):
        value = self.relations.get((left, right), self.relations.get((int(left), int(right))))
        if value is None:
            value = _lookup(_lookup(self.relations, left, {}), right)
        if isinstance(value, str):
            value = value.lower().strip()
            return value if value in RELATIONS else "unknown"
        return {1: "c2p", 0: "p2p", -1: "p2c", 2: "s2s"}.get(value, "unknown")

    @lru_cache(maxsize=256)
    def _destination_aggregate(self, start):
        """Build route aggregates once per start, without destination/label access."""
        routes = _lookup(self.index, start, {})
        destinations = {}
        total = 0.0
        totalpaths = 0
        origin_routes = _lookup(self.origin_support, start, {})
        for text, frequency in routes.items():
            raw = _number(frequency)
            if not raw:
                continue
            nodes = text.split("->")
            if len(nodes) < 2 or nodes[0] != start:
                continue
            total += raw
            totalpaths += 1
            origins = _number(origin_routes.get(text, 0))
            missing = text not in origin_routes or not self.origin_information_available
            record = destinations.setdefault(nodes[-1], [[0.0, 0, 0.0, 0.0, 0], {}])
            for stats in (record[0], record[1].setdefault(nodes[1], [0.0, 0, 0.0, 0.0, 0])):
                stats[0] += raw
                stats[1] += 1
                stats[2] = max(stats[2], origins)
                stats[3] += origins
                stats[4] += int(missing)
        return (total, totalpaths, destinations)

    def _write_destination(self, out, name, start, via, destination):
        prefix = "destination_" + name + "_"
        values = {key: 0.0 for key in DEST_FIELDS}
        values["context_missing"] = float(start is None or via is None)
        if start is None or via is None:
            values.update(
                start_missing=1.0,
                destination_missing=1.0,
                via_missing=1.0,
                origin_support_missing=1.0,
            )
        else:
            (total, paths, destinations) = self._destination_aggregate(start)
            record = destinations.get(destination)
            overall = record[0] if record else [0.0, 0, 0.0, 0.0, 0]
            branch = record[1].get(via, [0.0, 0, 0.0, 0.0, 0]) if record else [0.0, 0, 0.0, 0.0, 0]
            values.update(
                start_missing=float(not paths),
                destination_missing=float(not overall[1]),
                via_missing=float(not branch[1]),
                start_route_mass_log1p=math.log1p(total),
                destination_route_mass_log1p=math.log1p(overall[0]),
                destination_path_count_log1p=math.log1p(overall[1]),
                destination_fraction_of_start_mass=overall[0] / total if total else 0.0,
                via_route_mass_log1p=math.log1p(branch[0]),
                via_path_count_log1p=math.log1p(branch[1]),
                via_fraction_of_destination_mass=branch[0] / overall[0] if overall[0] else 0.0,
                via_fraction_of_destination_paths=branch[1] / overall[1] if overall[1] else 0.0,
                destination_max_route_origin_support_log1p=math.log1p(overall[2]),
                via_max_route_origin_support_log1p=math.log1p(branch[2]),
                destination_route_origin_support_sum_log1p=math.log1p(overall[3]),
                via_route_origin_support_sum_log1p=math.log1p(branch[3]),
                origin_support_missing=float(
                    not self.origin_information_available or not overall[1] or overall[4] > 0
                ),
            )
        out.update({prefix + k: v for (k, v) in values.items()})

    def _prior_values(self, source, neighbor, anchor):
        gateway = _lookup(self.priors.get("gateway", self.priors), source, {})
        entry = _lookup(gateway, neighbor, {}) or {}
        root = _lookup(self.priors.get("root_rank", {}), (source, neighbor), None)
        if root is None:
            root = entry.get("pred_rank", entry.get("priority"))
        degree = _lookup(self.priors.get("source_visible_degree", {}), source, None)
        if degree is None:
            degree = entry.get("vas_graph_neighbor_count", entry.get("visible_degree"))
        arecord = (
            _lookup(_lookup(self.priors.get("recursive", {}), neighbor, {}), anchor, {})
            if anchor is not None
            else {}
        )
        aevidence = _lookup((arecord or {}).get("center_evidence", {}), source, {}) or {}
        arank = self.priors.get("recursive_rank", {}).get((source, neighbor, anchor))
        if arank is None:
            arank = aevidence.get("pred_rank", aevidence.get("priority"))
        return dict(
            source_visible_degree_log1p=math.log1p(_number(degree)),
            source_visible_degree_missing=float(degree is None),
            root_rank_log1p=math.log1p(_number(root)),
            root_rank_inverse=1 / _number(root) if _number(root) else 0.0,
            root_rank_missing=float(not _number(root)),
            anchor_rank_log1p=math.log1p(_number(arank)),
            anchor_rank_inverse=1 / _number(arank) if _number(arank) else 0.0,
            anchor_rank_missing=float(not _number(arank)),
        )

    @lru_cache(maxsize=20000)
    def _public_pair(self, left, right):
        values = {field: 0.0 for field in PUBLIC_FIELDS}
        values.update(source_network_missing=1.0, candidate_network_missing=1.0)
        if self.public is None or left is None or right is None:
            return values
        values["available"] = 1.0
        nets = getattr(self.public, "_nets", {})
        a = _lookup(nets, left, {}) or {}
        b = _lookup(nets, right, {}) or {}
        values["source_network_missing"] = float(not a)
        values["candidate_network_missing"] = float(not b)
        for field, collection_name in (("ixp", "_ixps"), ("facility", "_facilities")):
            collection = getattr(self.public, collection_name, {})
            aset = set(_lookup(collection, left, set()) or ())
            bset = set(_lookup(collection, right, set()) or ())
            union = len(aset | bset)
            overlap = len(aset & bset)
            values[field + "_jaccard"] = overlap / union if union else 0.0
            values["shared_location"] = max(values["shared_location"], float(overlap > 0))
        countries = getattr(self.public, "_countries", {})
        aset = set(_lookup(countries, left, set()) or ())
        bset = set(_lookup(countries, right, set()) or ())
        values["same_country"] = float(bool(aset & bset))
        for prefix, net in (("source", a), ("candidate", b)):
            types = net.get("info_types") or [net.get("info_type", "")]
            if isinstance(types, str):
                types = [types]
            types = {str(t).strip().lower().replace("/", "_").replace(" ", "_") for t in types}
            values[prefix + "_nsp"] = float("nsp" in types)
            values[prefix + "_content"] = float("content" in types)
            values[prefix + "_isp"] = float(
                bool(types & {"cable_dsl_isp", "cable_dsl__isp", "access", "isp"})
            )
        return values

    def features(self, candidate):
        path = _nodes(candidate.get("path"))
        if len(path) < 2 or len(set(path)) != len(path):
            raise ValueError("Path must be loopfree with source and neighbor")
        prefix = _nodes(candidate.get("prefix"))
        if not prefix:
            prefix = path[: max(2, min(len(path), int(candidate.get("segment_len", 2) or 2)))]
        if len(prefix) < 2 or path[: len(prefix)] != prefix:
            raise ValueError(
                "Candidate prefix must contain source/neighbor and be an exact initial segment"
            )
        (source, neighbor) = path[:2]
        anchor = path[2] if len(path) > 2 else None
        nxt = path[3] if len(path) > 3 else None
        dest = path[-1]
        tail_len = candidate.get("tail_len")
        tail_frequency = candidate.get("tail_path_freq")
        out = dict(
            path_length=float(len(path)),
            prefix_length=float(len(prefix)),
            is_recursive=float(len(prefix) > 2),
            tail_length=_number(tail_len)
            if tail_len is not None
            else float(len(path) - len(prefix) + 1),
            candidate_tail_frequency_log1p=math.log1p(_number(tail_frequency)),
            candidate_tail_frequency_missing=float(tail_frequency is None),
            tail_length_missing=float(tail_len is None),
        )
        suffixmax = 0
        for start in range(len(path) - 1):
            if self._support(path[start:])[0] > 0:
                suffixmax = len(path) - start
                break
        out["max_known_suffix_length"] = float(suffixmax)
        out["known_suffix_fraction"] = suffixmax / len(path)
        out["neighbor_suffix_known_full_path_unobserved"] = float(
            self._support(path[1:])[0] > 0 and (not self._support(path)[0])
        )
        for name, nodes in (
            ("full_path", path),
            ("suffix_neighbor_to_destination", path[1:]),
            ("suffix_anchor_to_destination", path[2:]),
            ("prefix_s_n", path[:2]),
            ("prefix_s_n_a", path[:3] if len(path) > 2 else ()),
            ("splice_n_a_next", path[1:4] if len(path) > 3 else ()),
            ("reverse_s_n_a", tuple(reversed(path[:3])) if len(path) > 2 else ()),
            ("reverse_n_a_next", tuple(reversed(path[1:4])) if len(path) > 3 else ()),
        ):
            self._write_support(out, name, nodes)
        rel = [self._relation(a, b) for (a, b) in zip(path, path[1:])]
        for pos, i in (("first", 0), ("second", 1)):
            value = rel[i] if i < len(rel) else "unknown"
            for r in RELATIONS:
                out["relation_" + pos + "_" + r] = float(value == r)
        for name, r in (
            ("unknown", "unknown"),
            ("peer", "p2p"),
            ("up", "c2p"),
            ("down", "p2c"),
            ("sibling", "s2s"),
        ):
            out["relation_" + name + "_edges"] = float(rel.count(r))
        out["relation_edge_count"] = float(len(rel))
        out["relation_unknown_fraction"] = rel.count("unknown") / len(rel)
        out["prefix_relation_unknown_edges"] = float(rel[: len(prefix) - 1].count("unknown"))
        for direction, edges in (
            ("forward", list(zip(path, path[1:]))),
            ("reverse", [(b, a) for (a, b) in zip(path, path[1:])]),
        ):
            observed = [self._support(edge) for edge in edges]
            if direction == "forward":
                out["evidence_unknown_edges"] = float(sum((r[0] == 0 for r in observed)))
                out["evidence_unknown_edge_fraction"] = sum((r[0] == 0 for r in observed)) / len(
                    observed
                )
                out["prefix_evidence_unknown_edges"] = float(
                    sum((r[0] == 0 for r in observed[: len(prefix) - 1]))
                )
            for quantity, position in (("raw", 0), ("origin_support", 1)):
                values = [r[position] for r in observed]
                for stat, value in (
                    ("minimum", min(values)),
                    ("maximum", max(values)),
                    ("mean", sum(values) / len(values)),
                    ("sum", sum(values)),
                ):
                    out["edge_" + direction + "_" + stat + "_" + quantity + "_log1p"] = math.log1p(
                        value
                    )
            out["edge_" + direction + "_known_fraction"] = sum((r[0] > 0 for r in observed)) / len(
                observed
            )
            out["edge_" + direction + "_origin_support_missing_fraction"] = sum(
                (r[2] for r in observed)
            ) / len(observed)
        for name, start, via in (
            ("s_n_d", source, neighbor),
            ("n_a_d", neighbor, anchor),
            ("a_next_d", anchor, nxt),
        ):
            self._write_destination(out, name, start, via, dest)
        out.update(
            {"prior_" + k: v for (k, v) in self._prior_values(source, neighbor, anchor).items()}
        )
        for name, left, right in (
            ("s_n", source, neighbor),
            ("n_a", neighbor, anchor),
            ("s_a", source, anchor),
        ):
            out.update(
                {"public_" + name + "_" + k: v for (k, v) in self._public_pair(left, right).items()}
            )
        if set(out) != set(self.feature_names):
            raise AssertionError("Path feature schema drift")
        return {key: float(out[key]) for key in self.feature_names}
