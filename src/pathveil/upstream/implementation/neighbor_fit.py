"""Unified255 neighbor fit computations."""
import lightgbm as lgb
import numpy as np

from ..resources import positive_int


SEED = 20261021


def add_relative(X, queries, schema):
    keys = [
        "route_edge_forward",
        "route_edge_reverse",
        "route_edge_evidence_sources",
        "route_reverse_evidence_sources",
        "ext_link_number_paths_log1p",
        "ext_candidate_cone_asns_log1p",
        "route_prefix_forward",
        "route_prefix_reverse",
    ]
    extra = np.zeros((len(X), len(keys)), dtype=np.float32)
    off = 0
    for q in queries:
        n = q["size"]
        for j, key in enumerate(keys):
            v = X[off : off + n, schema.index(key)]
            (values, inv, counts) = np.unique(v, return_inverse=True, return_counts=True)
            lower = np.cumsum(counts) - counts
            extra[off : off + n, j] = lower[inv] / max(1, n - 1)
        off += n
    return (np.concatenate((X, extra), axis=1), schema + ["relative_" + k for k in keys])


def source_weights(queries):
    counts = {}
    for q in queries:
        if not q["root"]:
            counts[q["source"]] = counts.get(q["source"], 0) + 1
    return np.concatenate(
        [
            np.full(
                q["size"],
                (1.0 if counts.get(q["source"], 0) == 0 else 0.5)
                if q["root"]
                else 0.5 / counts[q["source"]],
                dtype=np.float32,
            )
            for q in queries
        ]
    )


def fit_one(spec, X, labels, queries, keep_sources, schema, threads=4):
    threads = positive_int(threads, "cpu_threads")
    (indices, groups, off) = ([], [], 0)
    for q in queries:
        size = q["size"]
        keep = q["source"] in keep_sources and (not spec["root_only"] or q["root"])
        if (
            keep
            and size
            and (spec["kind"] != "rank" or len(np.unique(labels[off : off + size])) > 1)
        ):
            indices.extend(range(off, off + size))
            groups.append(size)
        off += size
    idx = np.asarray(indices, dtype=np.int64)
    w = source_weights(queries)[idx]
    w /= np.mean(w)
    params = dict(
        n_estimators=240,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.04,
        min_child_samples=30,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=threads,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    if spec["kind"] == "rank":
        model = lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=spec["gains"],
            lambdarank_truncation_level=20,
            **params,
        )
        model.fit(X[idx], labels[idx], group=groups, sample_weight=w, feature_name=schema)
    elif spec["kind"] == "present":
        model = lgb.LGBMClassifier(objective="binary", **params)
        model.fit(X[idx], labels[idx], sample_weight=w, feature_name=schema)
    else:
        model = lgb.LGBMRegressor(objective="regression_l1", **params)
        model.fit(X[idx], labels[idx], sample_weight=w, feature_name=schema)
    return model


def predict(model, X, kind, threads=4):
    threads = positive_int(threads, "cpu_threads")
    return (
        model.predict_proba(X, num_threads=threads)[:, 1]
        if kind == "present"
        else model.predict(X, num_threads=threads)
    )
