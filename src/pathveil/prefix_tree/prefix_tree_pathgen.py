"""Portable numerical API extracted from the verified Tree B implementation."""
from __future__ import annotations


import argparse


from collections import Counter, defaultdict, deque


import csv


import hashlib


import json


import math


import os


from pathlib import Path


import shutil


import sys


from pathveil.utils import read_json, sha256, utc_now, write_json


def path_set_signature(paths):
    return hashlib.sha256(
        ("\n".join(sorted("->".join(map(str, p)) for p in paths)) + "\n").encode()
    ).hexdigest()


def valid_path(path, source=None, destination=None):
    path = tuple(map(str, path))
    return (
        2 <= len(path) <= 20
        and len(path) == len(set(path))
        and all(n.isascii() and n.isdecimal() and 0 < int(n) < 2**32 for n in path)
        and (source is None or path[0] == str(source))
        and (destination is None or path[-1] == str(destination))
    )


def balanced_candidates(paths, k, prefix_weights=None):
    """Round-robin first hops, then their actual three-AS prefixes.

    k is the existing per-query GMPI slot count. It never truncates the source
    augmentation library. Stable deduplication precedes slot allocation.
    """
    if k < 1:
        raise ValueError("The frozen GMPI query slot count must be positive")
    grouped, seen = defaultdict(lambda: defaultdict(deque)), set()
    for path in paths:
        path = tuple(map(str, path))
        if not valid_path(path) or path in seen:
            continue
        seen.add(path)
        grouped[path[1]][path[:3]].append(path)
    default_weight = 0.0 if prefix_weights else 1.0
    prefix_weights = prefix_weights or {}
    root_queues = []
    for root in sorted(grouped, key=int):
        order = sorted(
            grouped[root],
            key=lambda prefix: (
                -prefix_weights.get(prefix, prefix_weights.get(prefix[:2], default_weight)),
                prefix,
            ),
        )
        branches = deque(grouped[root][prefix] for prefix in order)
        root_queues.append(branches)
    roots = deque(root_queues)
    output = []
    while roots and len(output) < k:
        branches = roots.popleft()
        queue = branches.popleft()
        output.append(list(queue.popleft()))
        if queue:
            branches.append(queue)
        if branches:
            roots.append(branches)
    return output
