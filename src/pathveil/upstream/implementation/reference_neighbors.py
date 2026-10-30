"""Unified255 reference neighbors computations."""
from __future__ import annotations


K = 10
TOL = 1e-12


def top_target(weights):
    ordered = sorted(
        ((int(n), float(w)) for (n, w) in weights.items() if w > 0),
        key=lambda item: (-round(item[1], 12), item[0]),
    )
    top = ordered[:K]
    cutoff = top[-1][1] if top else None
    high = {n for (n, w) in ordered if cutoff is not None and w > cutoff + TOL}
    boundary = {n for (n, w) in ordered if cutoff is not None and abs(w - cutoff) <= TOL}
    return dict(
        strict=[n for (n, _) in top],
        size=len(top),
        high=high,
        boundary=boundary,
        positive=high | boundary,
        cutoff=cutoff,
        boundary_tie=len(ordered) > K and abs(ordered[K - 1][1] - ordered[K][1]) <= TOL,
    )
