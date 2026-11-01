"""Unified255 prior interface computations."""
from __future__ import annotations
import numpy as np
from .distribution import LENGTHS


def as_prior(query, conditional_branch, length_given_branch, candidate_mass, selected=None):
    """Restriction keeps all excluded/unknown mass explicitly in metadata."""
    candidates = list(map(str, query["candidates"]))
    conditional_branch = np.asarray(conditional_branch, dtype=float)
    length_given_branch = np.asarray(length_given_branch, dtype=float)
    if conditional_branch.shape != (len(candidates),) or length_given_branch.shape != (
        len(candidates),
        len(LENGTHS),
    ):
        raise ValueError("Prior candidate/length row alignment differs")
    if not np.isclose(conditional_branch.sum(), 1.0) or not np.allclose(
        length_given_branch.sum(1), 1.0
    ):
        raise ValueError("Conditional predictions are not probability distributions")
    if not 0 <= candidate_mass <= 1:
        raise ValueError("Predicted candidate observation mass is invalid")
    selected_names = set(candidates) if selected is None else set(map(str, selected))
    indices = [i for (i, n) in enumerate(candidates) if n in selected_names]
    branch = {candidates[i]: float(candidate_mass * conditional_branch[i]) for i in indices}
    joint = {
        candidates[i]: dict(
            zip(
                LENGTHS, map(float, candidate_mass * conditional_branch[i] * length_given_branch[i])
            )
        )
        for i in indices
    }
    selected_mass = sum(branch.values())
    histogram = sum(
        (conditional_branch[i] * length_given_branch[i] for i in indices), np.zeros(len(LENGTHS))
    )
    histogram = histogram / histogram.sum() if histogram.sum() else histogram
    return dict(
        source=int(query["source"]),
        center=int(query["center"]),
        root=bool(query["root"]),
        branch_probability=branch,
        joint_probability=joint,
        length_probability=dict(zip(LENGTHS, map(float, histogram))),
        predicted_all_candidate_observation_mass=float(candidate_mass),
        predicted_selected_observation_mass=float(selected_mass),
        predicted_unselected_candidate_observation_mass=float(candidate_mass - selected_mass),
        predicted_outside_candidate_observation_mass=float(1.0 - candidate_mass),
        selected_not_in_candidate_pool=sorted(selected_names - set(candidates)),
        prior_condition="Selector may condition allocation on selected branches; discarded/missing mass remains explicit here",
        labels_read_at_prediction=False,
    )
