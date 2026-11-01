"""Unified255 distribution computations."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import numpy as np

from ..resources import positive_int


LENGTHS = tuple(map(str, range(3, 21))) + ("21+",)


def softmax_queries(logits, offsets, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = np.asarray(logits, dtype=float)
    if offsets[0] != 0 or offsets[-1] != len(logits):
        raise ValueError("Context offsets do not span prediction rows")
    result = np.empty_like(logits)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        if stop <= start:
            continue
        values = logits[start:stop] / temperature
        values = np.exp(values - values.max())
        result[start:stop] = values / values.sum()
    return result


class GroupedSoftmaxObjective:
    """Proper cross entropy with fractional within-context branch labels.

    Diagonal Hessian approximation is exposed for tree boosting. The numeric
    gradient equals the exact grouped objective gradient, tested independently
    by finite differences. A zero-reference context must be excluded by caller.
    """

    def __init__(self, offsets, context_weights):
        self.offsets = np.asarray(offsets, dtype=int)
        self.weights = np.asarray(context_weights, dtype=float)
        if len(self.weights) + 1 != len(self.offsets) or np.any(self.weights <= 0):
            raise ValueError("Invalid context weights")
        self.row_weights = np.repeat(self.weights, np.diff(self.offsets))

    def __call__(self, truth, logits):
        truth = np.asarray(truth, dtype=float)
        probabilities = softmax_queries(logits, self.offsets)
        gradient = (probabilities - truth) * self.row_weights
        hessian = np.maximum(probabilities * (1.0 - probabilities), 1e-08) * self.row_weights
        return (gradient, hessian)

    def loss(self, truth, logits):
        p = softmax_queries(logits, self.offsets)
        return float(-(np.asarray(truth) * np.log(np.maximum(p, 1e-300)) * self.row_weights).sum())


def aligned_targets(queries, source_labels):
    """Build row-aligned train labels without using known formal/val identities.

    Missing candidate-pool mass is retained in metadata. Cross entropy is
    conditional on the fixed candidate pool; it cannot prove that pool covers
    all actual neighbors. Labels entirely outside that pool give no fit target.
    """
    branch = []
    length = []
    sources = []
    records = []
    for q in queries:
        source = str(q["source"])
        if source not in source_labels:
            raise ValueError("Every fit query must belong to an allowed TRAIN source")
        if q["root"]:
            context = source_labels[source]["root"]
        else:
            context = source_labels[source]["conditional"].get(str(q["center"]), {})
        marginal = context.get("branch_mass", {})
        joint = context.get("branch_length_mass", {})
        values = np.array([marginal.get(str(n), 0.0) for n in q["candidates"]], dtype=float)
        mass = float(values.sum())
        start = len(branch)
        branch.extend(values / mass if mass else values)
        for neighbor, probability in zip(q["candidates"], values):
            raw = np.array([joint.get(str(neighbor), {}).get(k, 0.0) for k in LENGTHS], dtype=float)
            length.append(raw / probability if probability > 0 else np.zeros(len(LENGTHS)))
            sources.append(int(source))
        records.append(
            dict(
                source=int(source),
                center=int(q["center"]),
                root=bool(q["root"]),
                start=start,
                stop=len(branch),
                observed_candidate_mass=mass,
                outside_candidate_mass=max(0.0, 1.0 - mass) if marginal else None,
                length_reference_destinations=context.get("destinations", 0),
                reference_unknown=not bool(marginal),
                unknown=not bool(mass),
            )
        )
    return (np.asarray(branch), np.asarray(length), np.asarray(sources), records)


def select_training_contexts(records, allowed_sources):
    allowed_sources = set(map(int, allowed_sources))
    selected = [
        q
        for q in records
        if q["source"] in allowed_sources and q["stop"] > q["start"] and (not q["unknown"])
    ]
    nonroot = Counter((q["source"] for q in selected if not q["root"]))
    rows = []
    offsets = [0]
    weights = []
    for q in selected:
        rows.extend(range(q["start"], q["stop"]))
        offsets.append(len(rows))
        weights.append(
            (0.5 if nonroot[q["source"]] else 1.0) if q["root"] else 0.5 / nonroot[q["source"]]
        )
    weights = np.asarray(weights, dtype=float)
    if not len(weights):
        raise ValueError("No informative training contexts")
    weights /= weights.mean()
    return (np.asarray(rows, dtype=int), np.asarray(offsets), weights, selected)


def context_features(matrix, queries):
    """Evidence-only feature summaries for predicting unavailable mass."""
    features = []
    offset = 0
    for q in queries:
        rows = matrix[offset : offset + len(q["candidates"])]
        offset += len(rows)
        if len(rows):
            features.append(
                np.r_[rows.mean(axis=0), rows.max(axis=0), np.log1p(len(rows)), float(q["root"])]
            )
        else:
            features.append(np.r_[np.zeros(matrix.shape[1] * 2), 0.0, float(q["root"])])
    if offset != len(matrix):
        raise ValueError("Context feature query alignment changed")
    return np.asarray(features, dtype=np.float32)


@dataclass
class DistributionHead:
    seed: int = 20261023
    threads: int = 4
    branch_trees: int = 160
    length_trees: int = 96
    temperature: float = 1.0

    def fit(self, matrix, queries, source_labels, allowed_sources):
        from lightgbm import LGBMRegressor
        from sklearn.ensemble import ExtraTreesRegressor

        self.threads = positive_int(self.threads, "cpu_threads")
        (y, length_y, sources, records) = aligned_targets(queries, source_labels)
        if matrix.shape[0] != len(y):
            raise ValueError("Feature/label row alignment changed")
        (rows, offsets, context_weights, selected) = select_training_contexts(
            records, allowed_sources
        )
        objective = GroupedSoftmaxObjective(offsets, context_weights)
        self.branch_model = LGBMRegressor(
            objective=objective,
            n_estimators=self.branch_trees,
            num_leaves=15,
            max_depth=5,
            min_child_samples=60,
            reg_lambda=1.0,
            learning_rate=0.04,
            n_jobs=self.threads,
            random_state=self.seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        self.branch_model.fit(matrix[rows], y[rows])
        (positive_rows, length_weights) = ([], [])
        for q, weight in zip(selected, context_weights):
            indices = np.arange(q["start"], q["stop"])
            indices = indices[length_y[indices].sum(axis=1) > 0]
            if len(indices):
                positive_rows.extend(indices)
                length_weights.extend([weight / len(indices)] * len(indices))
        positive_rows = np.asarray(positive_rows, dtype=int)
        if not len(positive_rows):
            raise ValueError("No observed TRAIN branches with length labels")
        length_weights = np.asarray(length_weights)
        length_weights /= length_weights.mean()
        self.length_model = ExtraTreesRegressor(
            n_estimators=self.length_trees,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.5,
            n_jobs=self.threads,
            random_state=self.seed,
        )
        self.length_model.fit(
            matrix[positive_rows], length_y[positive_rows], sample_weight=length_weights
        )
        context_x = context_features(matrix, queries)
        allowed = set(map(int, allowed_sources))
        coverage_rows = [
            i
            for (i, q) in enumerate(records)
            if q["source"] in allowed and (not q["reference_unknown"])
        ]
        coverage_y = np.asarray(
            [np.clip(records[i]["observed_candidate_mass"], 0.0, 1.0) for i in coverage_rows]
        )
        coverage_sizes = Counter((records[i]["source"] for i in coverage_rows))
        coverage_weights = np.asarray(
            [1.0 / coverage_sizes[records[i]["source"]] for i in coverage_rows]
        )
        coverage_weights /= coverage_weights.mean()
        self.coverage_model = ExtraTreesRegressor(
            n_estimators=self.length_trees,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.5,
            n_jobs=self.threads,
            random_state=self.seed,
        )
        self.coverage_model.fit(
            context_x[coverage_rows], coverage_y, sample_weight=coverage_weights
        )
        self.fit_info = dict(
            fit_sources=sorted(set(map(int, allowed_sources))),
            branch_rows=len(rows),
            length_positive_rows=len(positive_rows),
            contexts=len(selected),
            length_bins=list(LENGTHS),
            coverage_contexts=len(coverage_rows),
            no_source_asn_identity_feature_added=True,
            zero_observation_is_physical_absence=False,
            label_unit="Source-equal grouped branch CE; conditional length histograms",
            branch_reference_population="3+ AS observations conditional on fixed candidate pool",
        )
        return self

    def predict_arrays(self, matrix, queries):
        offsets = np.r_[0, np.cumsum([len(q["candidates"]) for q in queries])]
        if offsets[-1] != matrix.shape[0]:
            raise ValueError("Prediction query/matrix alignment changed")
        logits = self.branch_model.predict(matrix, raw_score=True, num_threads=self.threads)
        mass = softmax_queries(logits, offsets, self.temperature)
        conditional_length = np.maximum(0.0, self.length_model.predict(matrix))
        totals = conditional_length.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Length prediction returned an unknown allzero histogram")
        conditional_length /= totals[:, None]
        return (mass, conditional_length)

    def predict_candidate_coverage(self, matrix, queries):
        return np.clip(self.coverage_model.predict(context_features(matrix, queries)), 0.0, 1.0)

    def predict(self, matrix, queries):
        (mass, conditional_length) = self.predict_arrays(matrix, queries)
        coverage = self.predict_candidate_coverage(matrix, queries)
        offsets = np.r_[0, np.cumsum([len(q["candidates"]) for q in queries])]
        results = []
        for q, start, stop, candidate_mass in zip(queries, offsets[:-1], offsets[1:], coverage):
            joint = mass[start:stop, None] * conditional_length[start:stop]
            results.append(
                dict(
                    source=int(q["source"]),
                    center=int(q["center"]),
                    root=bool(q["root"]),
                    branch_probability={
                        str(n): float(v * candidate_mass)
                        for (n, v) in zip(q["candidates"], mass[start:stop])
                    },
                    candidate_conditional_probability={
                        str(n): float(v) for (n, v) in zip(q["candidates"], mass[start:stop])
                    },
                    predicted_candidate_observation_mass=float(candidate_mass),
                    predicted_outside_candidate_observation_mass=float(1.0 - candidate_mass),
                    length_probability=dict(zip(LENGTHS, map(float, joint.sum(axis=0)))),
                    joint_probability={
                        str(n): dict(zip(LENGTHS, map(float, row * candidate_mass)))
                        for (n, row) in zip(q["candidates"], joint)
                    },
                    reference_labels_read=False,
                    condition="Normalized over ALL frozen candidates; selected Top10 mass must be reported separately",
                )
            )
        return results
