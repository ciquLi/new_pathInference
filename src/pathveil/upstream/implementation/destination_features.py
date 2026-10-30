"""Unified255 destination features computations."""
import numpy as np


KEY = np.dtype([("N", "<u4"), ("D", "<u4"), ("A", "<u4")])
LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)
FEATURE_NAMES = (
    "D_ND_sources_log1p",
    "D_NAD_sources_log1p",
    "D_NAD_source_fraction_given_ND",
    "D_A_source_equal_mass_given_ND",
    "D_ND_unseen",
    "D_NAD_absent_with_ND",
    "D_role_ND_sources_log1p",
    "D_role_NAD_sources_log1p",
    "D_role_NAD_source_fraction_given_ND",
    "D_role_A_source_equal_mass_given_ND",
    "D_role_ND_unseen",
    "D_role_NAD_absent_with_ND",
    "D_public_role_unknown",
    "D_public_role_conflict",
    "D_public_role_identified",
    "D_role_stats_missing",
)
APPENDED_NAMES = (
    tuple(("firsthop_" + n for n in FEATURE_NAMES))
    + tuple(("construction_boundary_" + n for n in FEATURE_NAMES))
    + ("construction_boundary_no_next",)
)
ROLES = ("c2p", "p2p", "p2c", "s2s", "unknown", "conflict")


def public_role(public, receiver, center):
    """D-prepare's exact directional public role: receiver -> current center."""
    pair = (int(receiver), int(center))
    actual = public.relations.get(pair, "unknown")
    explicit = public.asrel_relations.get(pair, "unknown")
    ar = public.links.get(pair, {}).get("role", "unknown")
    ar_rel = {"customer": "p2c", "provider": "c2p", "peer": "p2p"}.get(ar, "unknown")
    conflict = explicit != "unknown" and ar_rel != "unknown" and (explicit != ar_rel)
    code = 5 if conflict else ROLES.index(actual) if actual in ROLES[:4] else 4
    return dict(
        public_receiver_to_center_role=actual,
        role_code=code,
        role_conflict=conflict,
        ASREL=explicit,
        ASRank_relation=ar_rel,
        public_role_source="ASREL"
        if explicit != "unknown"
        else "ASRank_fallback"
        if actual != "unknown"
        else "unknown",
    )


def candidate_queries(candidate, source, row_index):
    path = tuple(map(int, candidate["path"]))
    prefix = tuple(map(int, candidate["prefix"]))
    if len(path) < 3 or path[0] != source or candidate["row_index"] != row_index:
        raise ValueError("Original row/source alignment invalid")
    if len(prefix) < 2 or prefix != path[: len(prefix)]:
        raise ValueError("Construction prefix must equal ordered path prefix with >=2 ASes")
    if any((int(candidate[k]) != path[-1] for k in ("dest", "destination") if k in candidate)):
        raise ValueError("Generated D differs from complete path[-1]")
    first = (path[1], path[2], path[-1], path[0])
    valid = len(prefix) < len(path)
    boundary = (prefix[-1], path[len(prefix)] if valid else 0, path[-1], prefix[-2])
    return (first, boundary, valid, len(prefix))


def missing_boundary_features(count):
    x = np.zeros((count, 16), np.float32)
    x[:, 4] = 1
    x[:, 15] = 1
    return x


def keys(n, a, d):
    n = np.asarray(n)
    a = np.asarray(a)
    d = np.asarray(d)
    if (
        n.ndim != 1
        or n.shape != a.shape
        or n.shape != d.shape
        or any((v.dtype.kind not in "iu" or np.any((v <= 0) | (v >= 2**32)) for v in (n, a, d)))
    ):
        raise ValueError("Aligned positive uint32N,A,D vectors required")
    out = np.empty(len(n), KEY)
    out["N"] = n
    out["A"] = a
    out["D"] = d
    return out


def popcount(mask):
    return (
        LUT[np.ascontiguousarray(mask).view(np.uint8).reshape(len(mask), 24)]
        .sum(1)
        .astype(np.uint16)
    )


def gather(index_keys, index_values, query):
    ix = np.searchsorted(index_keys, query)
    yes = ix < len(index_keys)
    clip = np.minimum(ix, len(index_keys) - 1)
    yes &= index_keys[clip] == query
    out = np.zeros((len(query), 3), np.uint64)
    out[yes] = index_values[ix[yes]]
    return out


def feature_matrix(stats):
    nd = np.asarray(stats["ND"], float)
    nad = np.asarray(stats["NAD"], float)
    rd = np.asarray(stats["NroleD"], float)
    ra = np.asarray(stats["NroleAD"], float)
    rc = np.asarray(stats["role_code"])
    identified = rc < 4
    valid = nd > 0
    rv = identified & (rd > 0)
    frequency = np.divide(nad, nd, out=np.zeros_like(nd), where=valid)
    mass = np.divide(stats["mass_sum"], nd, out=np.zeros_like(nd), where=valid)
    rfrequency = np.divide(ra, rd, out=np.zeros_like(rd), where=rv)
    rmass = np.divide(stats["role_mass_sum"], rd, out=np.zeros_like(rd), where=rv)
    if (
        np.any(nad > nd)
        or np.any(ra > rd)
        or np.any(mass < -1e-09)
        or np.any(mass > frequency + 1e-09)
        or np.any(rmass < -1e-09)
        or np.any(rmass > rfrequency + 1e-09)
    ):
        raise ValueError("Source-equal mass/count denominator or scope mismatch")
    X = np.column_stack(
        (
            np.log1p(nd),
            np.log1p(nad),
            frequency,
            mass,
            ~valid,
            valid & (nad == 0),
            np.log1p(rd),
            np.log1p(ra),
            rfrequency,
            rmass,
            identified & (rd == 0),
            rv & (ra == 0),
            rc == 4,
            rc == 5,
            identified,
            ~rv,
        )
    ).astype(np.float32)
    if X.shape != (len(nd), 16) or not np.isfinite(X).all():
        raise ValueError("Finite16column schema mismatch")
    return X
