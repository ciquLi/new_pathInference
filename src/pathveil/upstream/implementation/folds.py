"""Unified255 folds computations."""
from __future__ import annotations
import hashlib


def source_folds(sources):
    """Five balanced folds from source identities alone, without reading labels."""
    salt = "20261021:neighbor_oof:"
    ordered = sorted(
        map(int, sources), key=lambda source: hashlib.sha256(f"{salt}{source}".encode()).hexdigest()
    )
    return {source: index % 5 for (index, source) in enumerate(ordered)}
