"""Dependency injection for the self-contained unified255 computations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from .implementation import (
    candidates,
    construction_budget,
    destination_features,
    distribution,
    donor_stats,
    evidence_index,
    final_features,
    folds,
    frontier,
    frontier_adapter,
    frontier_policy,
    joint_labels,
    joint_selector,
    neighbor_features,
    neighbor_fit,
    neighbor_struct_fit,
    next_features,
    path_features,
    path_labels,
    policy_features,
    prefix_features,
    prior_interface,
    prior_spec,
    public,
    reference_neighbors,
    retrieval,
    role_labels,
    root,
)


class MethodLibrary:
    """Canonical computational API; no source snapshots or experiment folders."""

    def __init__(self, project_root=None, **_):
        self.project = Path(project_root or Path(__file__).resolve().parents[3]).resolve()

    def core(self):
        return SimpleNamespace(
            public=public,
            base=neighbor_features,
            legacy=SimpleNamespace(source_folds=folds.source_folds),
            retrieval=retrieval,
            root=root,
            experiment=neighbor_fit,
            structural_experiment=neighbor_struct_fit,
            frontier=frontier,
            candidates=candidates,
            frontier_policy=frontier_policy,
            adapter=frontier_adapter,
            budget=construction_budget,
            pathbase=path_features,
            final=final_features,
            destination=destination_features,
            selection=joint_selector,
            specs=prior_spec,
            training=path_labels,
            index=evidence_index,
            top=reference_neighbors,
            rpsl=policy_features,
            distribution=distribution,
            distribution_head=distribution,
            joint_labels=joint_labels,
            role_labels=role_labels,
            prior_interface=prior_interface,
            donor_stats=donor_stats,
        )

    def receipt(self):
        files = sorted(Path(__file__).parent.rglob("*.py"))
        return dict(
            method_id="unified255_v1",
            self_contained=True,
            computational_sources_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
            historical_launchers_executed=False,
        )
