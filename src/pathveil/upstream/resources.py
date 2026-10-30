"""Explicit CPU resources; numerical model and feature parameters stay separate."""
from collections.abc import Mapping

DEFAULT_RUNTIME_RESOURCES = {
    "cpu_threads": 4,
    "case_predict_threads": 2,
    "case_workers": 2,
}


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def runtime_resources(config=None):
    config = config or {}
    values = config.get("runtime_resources", {})
    if not isinstance(values, Mapping):
        raise ValueError("runtime_resources must be a mapping")
    unexpected = set(values) - set(DEFAULT_RUNTIME_RESOURCES)
    if unexpected:
        raise ValueError("Unknown runtime resources: " + ", ".join(sorted(unexpected)))
    return {
        name: positive_int(values.get(name, default), name)
        for name, default in DEFAULT_RUNTIME_RESOURCES.items()
    }
