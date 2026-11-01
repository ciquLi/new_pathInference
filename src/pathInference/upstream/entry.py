"""Public upstream API used by the one-date project runner."""
from __future__ import annotations

from .pipeline import run_upstream as _run


def run_upstream(
    data_manifest,
    run_dir,
    method_config=None,
    project_root=None,
    migration_replay=False,
    sources=None,
    split_manifest=None,
):
    """Run fresh date-local stages; cached March replay is an explicit choice."""
    return _run(
        data_manifest,
        run_dir,
        method_config,
        project_root=project_root,
        sources=sources,
        fresh=not migration_replay,
        split_manifest=split_manifest,
    )
