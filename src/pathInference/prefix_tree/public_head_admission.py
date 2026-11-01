"""Portable numerical API extracted from the verified Tree B implementation."""
from __future__ import annotations


import argparse


from collections import Counter, defaultdict


import csv


import fcntl


import hashlib


import json


import os


from pathlib import Path


import shutil


import sys


import time


from types import SimpleNamespace


from . import public_prefix_tree


from .public_prefix_tree import commercial_phases


HIGH = "fully_public_observed_commercial_compatible"


UNKNOWN = "public_unknown_or_role_conflict"


INCOMPATIBLE = "public_commercial_abstraction_incompatible"


def classify_prefix(prefix, evidence):
    if len(evidence) != len(prefix) - 1:
        raise ValueError("Public evidence edge count differs")
    for (left, right), edge in zip(zip(prefix, prefix[1:]), evidence):
        if str(edge["left"]) != left or str(edge["right"]) != right:
            raise ValueError("Public edge evidence endpoints differ")
    roles = [edge["public_role"] for edge in evidence]
    if any(role not in {"c2p", "p2c", "p2p", "s2s", "unknown", "conflict"} for role in roles):
        raise ValueError("Unknown public role code")
    compatibility = commercial_phases(roles)
    fully_observed = all(edge["public_edge_observed"] is True for edge in evidence)
    if fully_observed and compatibility["status"] == "compatible":
        tier = HIGH
    elif compatibility["status"] == "incompatible":
        tier = INCOMPATIBLE
    else:
        tier = UNKNOWN
    return dict(
        prefix=list(prefix),
        tier=tier,
        fully_public_observed=fully_observed,
        commercial_compatibility=compatibility,
        public_roles=roles,
        public_role_conflict=any(role == "conflict" for role in roles),
        unknown_is_not_absence=True,
        commercial_incompatible_is_not_route_nonexistence=True,
    )
