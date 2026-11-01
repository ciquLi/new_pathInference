"""Portable numerical API extracted from the verified Tree B implementation."""
from __future__ import annotations


from collections import Counter


from dataclasses import dataclass


import hashlib


import json


import math


from pathlib import Path


import re


from . import prefix_tree_pathgen as frozen_pathgen


HIGH = "HIGH"


UNKNOWN = "UNKNOWN"


INCOMPATIBLE = "INCOMPATIBLE"


TIERS = (HIGH, UNKNOWN, INCOMPATIBLE)


_TIER_NAMES = {
    HIGH: HIGH,
    UNKNOWN: UNKNOWN,
    INCOMPATIBLE: INCOMPATIBLE,
    "fully_public_observed_commercial_compatible": HIGH,
    "public_unknown_or_role_conflict": UNKNOWN,
    "public_commercial_abstraction_incompatible": INCOMPATIBLE,
}


VERSION = "query_local_public_head_tier_selection_v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("An explicit SHA256 is required: " + name)
    return value


def _path(raw, source):
    if isinstance(raw, (str, bytes)):
        raise ValueError("A path must be a sequence of ASNs")
    path = tuple(map(str, raw))
    if not frozen_pathgen.valid_path(path, source) or any(n != str(int(n)) for n in path):
        raise ValueError("Invalid query candidate path")
    return path


@dataclass(frozen=True)
class QuerySelection:
    paths: list
    tier: str | None
    coverage_fallback: bool
    receipt: dict


class PublicHeadSelector:
    """One immutable source's classifications, bound to upstream identities.

    ``from_records`` requires explicit upstream method/spec SHA256 values. Optional
    ``method_path``/``spec_path`` entries revalidate their bytes; callers without
    paths must themselves have verified those upstream identities. The exact
    classification records additionally receive their own content hash."""

    def __init__(
        self,
        admission_dir,
        source,
        *,
        expected_method_sha256,
        expected_statistics_sha256,
        prefix_weights=None,
    ):
        directory = Path(admission_dir).resolve()
        method_path = directory / "METHOD_FROZEN.json"
        statistics_path = directory / "statistics" / (str(source) + ".json")
        method_sha = _digest(expected_method_sha256, "method_sha256")
        statistics_sha = _digest(expected_statistics_sha256, "statistics_sha256")
        if _sha(method_path) != method_sha or _sha(statistics_path) != statistics_sha:
            raise ValueError("Frozen public admission method/statistics changed")
        method = json.loads(method_path.read_text())
        statistics = json.loads(statistics_path.read_text())
        if (
            str(statistics["source"]) != str(source)
            or str(source) not in method["experimental_sources"]
            or method.get("no_formal_reference_paths") is not True
            or method.get("public_evidence_is_not_route_probability") is not True
            or method.get("quantity_cap") is not None
            or statistics.get("formal_reference_paths_opened") is not False
            or statistics.get("final_path_count_cap") is not None
            or statistics["prefix_count"] != len(statistics["prefix_classification"])
        ):
            raise ValueError("Public admission scope/source contract differs")
        balanced_path = str(Path(frozen_pathgen.__file__).resolve())
        expected_balanced = method["inputs_sha256"].get(balanced_path)
        if expected_balanced is None or _sha(balanced_path) != expected_balanced:
            raise ValueError("Frozen balanced candidate implementation differs")
        identity = dict(
            method_sha256=method_sha,
            spec_sha256=statistics_sha,
            method_path=str(method_path),
            spec_path=str(statistics_path),
            statistics_sha256=statistics_sha,
            public_graph_identity_sha256=method["public_graph_identity_sha256"],
            balanced_code_sha256=expected_balanced,
            date=method["date"],
            upstream_kind="public_head_admission_statistics",
        )
        self._initialize(source, statistics["prefix_classification"], identity, prefix_weights)

    @classmethod
    def from_records(cls, source, prefix_classifications, identity, prefix_weights=None):
        """Accept a frozen public-tree spec, including primary heads absent in R_s.

        Records use the public_head_admission.classify_prefix schema. Extra fields
        (layer, phase_states, provenance) are retained in the records' content hash.
        Physical prefixes are full (S,N) or (S,N,A) tuples, never an A-only key."""
        instance = cls.__new__(cls)
        instance._initialize(source, prefix_classifications, identity, prefix_weights)
        return instance

    def _initialize(self, source, records, identity, prefix_weights):
        self.source = str(source)
        if (
            not self.source.isascii()
            or not self.source.isdecimal()
            or not 0 < int(self.source) < 2**32
            or self.source != str(int(self.source))
        ):
            raise ValueError("Invalid source ASN")
        if not isinstance(identity, dict):
            raise ValueError("Explicit frozen upstream identity is required")
        identity = dict(identity)
        for name in ("method_sha256", "spec_sha256"):
            _digest(identity.get(name), name)
        verified_files = {}
        for name in ("method", "spec"):
            if name + "_path" in identity:
                path = str(Path(identity[name + "_path"]).resolve())
                if _sha(path) != identity[name + "_sha256"]:
                    raise ValueError("Frozen upstream " + name + " bytes changed")
                identity[name + "_path"] = path
                verified_files[path] = identity[name + "_sha256"]
        records = json.loads(json.dumps(list(records), allow_nan=False))
        record_sha = _object_sha(records)
        if identity.get("prefix_classifications_sha256", record_sha) != record_sha:
            raise ValueError("Classification records content differs")
        if "spec_path" in identity:
            spec = json.loads(Path(identity["spec_path"]).read_text())
            if (
                str(spec.get("source")) != self.source
                or _object_sha(spec.get("prefix_classification")) != record_sha
                or spec.get("construction_method_sha256", identity["method_sha256"])
                != identity["method_sha256"]
            ):
                raise ValueError("Classification records differ from their frozen upstream spec")
        classifications = {}
        for record in records:
            prefix = _path(record["prefix"], self.source)
            if len(prefix) not in (2, 3):
                raise ValueError("Predicted public prefix must contain at most three ASes")
            tier = _TIER_NAMES.get(record.get("tier"))
            if tier is None:
                raise ValueError("Unknown public evidence tier")
            if prefix in classifications:
                raise ValueError("Duplicate physical public prefix classification")
            if tier == HIGH and (
                record.get("fully_public_observed") is not True
                or record.get("commercial_compatibility", {}).get("status") != "compatible"
            ):
                raise ValueError("HIGH requires fully public compatible evidence")
            if (
                tier == INCOMPATIBLE
                and record.get("commercial_compatibility", {}).get("status") != "incompatible"
            ):
                raise ValueError("INCOMPATIBLE requires explicit incompatibility evidence")
            if (
                record.get("forced_unknown_fallback") is True
                or "fallback" in str(record.get("layer", "")).lower()
            ) and tier != UNKNOWN:
                raise ValueError("An UNKNOWN fallback layer cannot become a HIGH prior")
            classifications[prefix] = tier
        self._classifications = classifications
        weights = {}
        for raw, value in (prefix_weights or {}).items():
            prefix = _path(raw, self.source)
            value = float(value)
            if len(prefix) not in (2, 3) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid fixed structural prefix weight")
            weights[prefix] = value
        self._weights = weights
        balanced_sha = _sha(frozen_pathgen.__file__)
        if identity.get("balanced_code_sha256", balanced_sha) != balanced_sha:
            raise ValueError("Balanced candidate code changed")
        self._metadata = dict(
            identity,
            version=VERSION,
            source=self.source,
            prefix_classifications_sha256=record_sha,
            verified_upstream_files_sha256=verified_files,
            balanced_code_sha256=balanced_sha,
            selector_code_sha256=_sha(__file__),
            fixed_prefix_weights_sha256=_object_sha(
                [[list(p), w] for p, w in sorted(weights.items())]
            ),
            maximum_predicted_prefix_AS_count=3,
            augmentation_quantity_cap=None,
            tier_is_route_probability=False,
        )
        self._identity_sha = _object_sha(self._metadata)

    @property
    def metadata(self):
        return json.loads(json.dumps(self._metadata))

    def classify_path(self, path):
        path = _path(path, self.source)
        matches = {
            self._classifications[path[:n]]
            for n in (2, 3)
            if len(path) >= n and path[:n] in self._classifications
        }
        # A short UNKNOWN fallback cannot conceal a concrete public S,N,A head.
        return next((tier for tier in TIERS if tier in matches), UNKNOWN)

    def select_query(self, pool, k):
        return self.new_query(k).extend(pool).result()

    def new_query(self, k):
        return QueryTierBank(self, k)


class QueryTierBank:
    """Exact k-path summaries, with a small append buffer, for one (S,D) query.

    The frozen selector has fixed root/prefix order and FIFO paths inside each
    prefix. Adding paths cannot move a previously excluded path into its first k
    outputs: it only inserts round-robin events or extends existing queues. Its
    first k outputs therefore form a mergeable summary, independently per tier.
        Appends compact at 2*k rows; result/merge expose at most k per tier. This
        avoids running the frozen selector on every candidate arrival. Keeping
        tiers separate is essential: later HIGH arrivals must evict the entire
    UNKNOWN selection, even when HIGH supplies fewer than k paths."""

    def __init__(self, selector, k):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("Query candidate slot count must be a positive integer")
        self.selector, self.k = selector, k
        self._banks = {tier: [] for tier in TIERS}
        self._arrivals = Counter()
        self._query = None

    def _check_query(self, path):
        query = (path[0], path[-1])
        if self._query is not None and self._query != query:
            raise ValueError("One tier bank cannot mix source/destination queries")
        self._query = query

    def _summarize(self, tier, additions):
        self._banks[tier] = frozen_pathgen.balanced_candidates(
            self._banks[tier] + list(additions), self.k, self.selector._weights
        )

    @property
    def has_high(self):
        return bool(self._banks[HIGH])

    def has_tier(self, tier):
        normalized = _TIER_NAMES.get(tier)
        if normalized is None:
            raise ValueError("Unknown public evidence tier")
        return bool(self._banks[normalized])

    def add(self, raw):
        path = _path(raw, self.selector.source)
        self._check_query(path)
        tier = self.selector.classify_path(path)
        self._arrivals[tier] += 1
        self._banks[tier].append(path)
        if len(self._banks[tier]) >= 2 * self.k:
            self._summarize(tier, [])
        return self

    def extend(self, pool):
        for path in pool:
            self.add(path)
        return self

    def merge(self, later):
        """Append a later chunk's bounded summary, preserving FIFO semantics."""
        if (
            not isinstance(later, QueryTierBank)
            or self.k != later.k
            or self.selector._identity_sha != later.selector._identity_sha
        ):
            raise ValueError("Cannot merge tier banks with different frozen identities")
        if self._query is not None and later._query is not None and self._query != later._query:
            raise ValueError("Cannot merge different queries")
        self._query = self._query or later._query
        for tier in TIERS:
            other_summary = frozen_pathgen.balanced_candidates(
                later._banks[tier], self.k, self.selector._weights
            )
            self._summarize(tier, other_summary)
        self._arrivals.update(later._arrivals)
        return self

    def result(self):
        for possible in TIERS:
            self._summarize(possible, [])
        tier = next((t for t in TIERS if self._banks[t]), None)
        paths = [list(p) for p in self._banks[tier]] if tier else []
        fallback = tier == INCOMPATIBLE
        reason = (
            "public_HIGH_available"
            if tier == HIGH
            else "no_HIGH_use_UNKNOWN"
            if tier == UNKNOWN
            else "only_INCOMPATIBLE_explicit_coverage_fallback"
            if fallback
            else "empty_valid_pool"
        )
        receipt = dict(
            version=VERSION,
            source=self.selector.source,
            query=list(self._query) if self._query else None,
            selected_tier=tier,
            reason=reason,
            coverage_fallback=fallback,
            query_slots=self.k,
            selected_paths=len(paths),
            candidate_arrivals_by_tier=dict(self._arrivals),
            arrival_counts_include_duplicates=True,
            retained_summary_rows_by_tier={t: len(self._banks[t]) for t in TIERS},
            maximum_retained_summary_rows_per_tier=self.k,
            maximum_temporary_append_rows_per_tier=2 * self.k,
            lower_tiers_fill_unused_slots=False,
            unknown_is_not_false=True,
            tier_is_route_probability=False,
            maximum_predicted_prefix_AS_count=3,
            augmentation_quantity_cap=None,
            selector_identity_sha256=self.selector._identity_sha,
            bindings=self.selector.metadata,
        )
        return QuerySelection(paths, tier, fallback, receipt)
