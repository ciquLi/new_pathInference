"""Explicit date bindings and checked acquisition for pathInference data."""

from .registry import load_registry, project_root
from .preflight import check_date

__all__ = ["load_registry", "project_root", "check_date"]
