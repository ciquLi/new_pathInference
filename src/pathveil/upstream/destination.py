"""Dated donor-conditioned 33-column features with source-fold isolation.

Per-donor complete-path fractions are summed in the frozen donor order before
the query source's entire fold is subtracted. Numeric feature assembly is the
unchanged unified255 feature_matrix implementation. Receiver identities choose
only a public relationship role; they never choose the donor exclusion scope.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .contracts import require, sha
from .implementation.destination_features import (
    APPENDED_NAMES,
    KEY,
    candidate_queries,
    feature_matrix,
    keys,
    missing_boundary_features,
    public_role,
)


def _matches(index, query):
    positions = np.searchsorted(index, query)
    yes = positions < len(index)
    if len(index):
        yes &= index[np.minimum(positions, len(index) - 1)] == query
    return positions, yes


class DestinationFeatureStore:
    def __init__(self, bundle, public):
        self.public = public
        self.sources = list(map(int, bundle["donor_sources"]))
        self.shards = {int(s): Path(p) for s, p in bundle["donor_shards"].items()}
        self.folds = {int(s): int(f) for s, f in bundle["source_to_fold"].items()}
        self.expected = bundle.get("inputs_sha256", {})
        require(
            len(self.sources) == len(set(self.sources)) and set(self.sources) == set(self.shards),
            "Donor identities must match their explicit shards",
        )
        require(set(self.sources) == set(self.folds), "Donor and OOF source roles differ")
        self.verified = set()

    @lru_cache(maxsize=100000)
    def role(self, receiver, center):
        return public_role(self.public, receiver, center)["role_code"]

    def batch_query(self, source, center, next_as, terminal, receiver):
        require(
            isinstance(source, (int, np.integer)) and not isinstance(source, (bool, np.bool_)),
            "Original query source must be an integer AS identity",
        )
        source = int(source)
        query = keys(np.asarray(center), np.asarray(next_as), np.asarray(terminal))
        receiver = np.asarray(receiver)
        if receiver.ndim == 0:
            receiver = np.full(len(query), receiver, dtype=receiver.dtype)
        require(
            receiver.shape == query.shape
            and receiver.dtype.kind in "iu"
            and np.all((receiver > 0) & (receiver < 2**32)),
            "Aligned positive uint32 receivers are required",
        )
        role = np.asarray(
            [self.role(int(r), int(n)) for r, n in zip(receiver, query["N"])], np.uint8
        )
        counts = np.zeros((len(query), 4), np.int32)
        fold_counts = np.zeros_like(counts)
        mass = np.zeros((len(query), 2), np.float64)
        fold_mass = np.zeros_like(mass)
        held = self.folds.get(source)
        ndquery = (query["N"].astype(np.uint64) << np.uint64(32)) | query["D"].astype(np.uint64)
        for donor in self.sources:
            path = self.shards[donor]
            if donor not in self.verified:
                require(
                    str(path) in self.expected and sha(path) == self.expected[str(path)],
                    "Unbound or changed donor shard: " + str(path),
                )
                self.verified.add(donor)
            with np.load(path, allow_pickle=False) as archive:
                nd = (archive["ND_N"].astype(np.uint64) << np.uint64(32)) | archive["ND_D"].astype(
                    np.uint64
                )
                groups = archive["NAD_nd_index"]
                nad = keys(archive["ND_N"][groups], archive["NAD_A"], archive["ND_D"][groups])
                ndpos, has_nd = _matches(nd, ndquery)
                napos, has_na = _matches(nad, query)
                selected = np.flatnonzero(has_nd)
                same = np.zeros(len(query), bool)
                same[selected] = (role[selected] < 4) & (
                    archive["ND_role_code"][ndpos[selected]] == role[selected]
                )
                c = np.column_stack((has_nd, has_na, same, same & has_na)).astype(np.int32)
                weights = np.zeros((len(query), 2), np.float64)
                hit = np.flatnonzero(has_na)
                numerator = archive["NAD_unique_full_path_count"][napos[hit]].astype(np.float64)
                denominator = archive["ND_total_unique_full_paths"][groups[napos[hit]]].astype(
                    np.float64
                )
                require(np.all(denominator > 0), "Zero donor path-mass denominator")
                weights[hit, 0] = numerator / denominator
                weights[:, 1] = weights[:, 0] * same
                counts += c
                mass += weights
                if held is not None and self.folds[donor] == held:
                    fold_counts += c
                    fold_mass += weights
        counts -= fold_counts
        mass = np.maximum(mass - fold_mass, 0.0)
        stats = dict(
            ND=counts[:, 0],
            NAD=counts[:, 1],
            NroleD=counts[:, 2],
            NroleAD=counts[:, 3],
            role_code=role,
            mass_sum=mass[:, 0],
            role_mass_sum=mass[:, 1],
        )
        return feature_matrix(stats)

    def append(self, source, candidates):
        n = len(candidates)
        if not n:
            return np.empty((0, 33), np.float32)
        first, boundary, valid = [], [], []
        for i, candidate in enumerate(candidates):
            f, b, v, _ = candidate_queries(candidate, int(source), i)
            first.append(f)
            boundary.append(b)
            valid.append(v)
        f = np.asarray(first, np.uint32)
        b = np.asarray(boundary, np.uint32)
        valid = np.asarray(valid, bool)
        one = self.batch_query(source, f[:, 0], f[:, 1], f[:, 2], f[:, 3])
        two = missing_boundary_features(n)
        if valid.any():
            two[valid] = self.batch_query(
                source, b[valid, 0], b[valid, 1], b[valid, 2], b[valid, 3]
            )
        result = np.column_stack((one, two, (~valid).astype(np.float32)))
        require(
            result.shape == (n, len(APPENDED_NAMES)) and np.isfinite(result).all(),
            "Dated donor features must have exactly 33 finite columns",
        )
        return result
