"""Explicit read mappings for immutable inputs on filesystems without links.

Mappings apply only to reads. Generated files always use their requested local
path, so an input mapping can never redirect a write into the original corpus.
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path

_manifest_path = None
_manifest = None


def read_path(path):
    global _manifest_path, _manifest
    if not isinstance(path, (str, bytes, os.PathLike)):
        return path
    configured = os.environ.get("PATHVEIL_DATA_MANIFEST")
    if not configured:
        return path
    if configured != _manifest_path:
        with builtins.open(configured) as stream:
            _manifest = json.load(stream)
        _manifest_path = configured
    from pathveil.data.prepare import resolve_read_path

    return resolve_read_path(_manifest, os.fsdecode(path))


def open_data(file, mode="r", *args, **kwargs):
    if not any(character in mode for character in "wax+"):
        file = read_path(file)
    return builtins.open(file, mode, *args, **kwargs)


def read_exists(path):
    return os.path.exists(read_path(path))


class PandasReadProxy:
    """Map a module's CSV reads, without monkey-patching pandas globally."""

    def __init__(self, pandas):
        self._pandas = pandas

    def read_csv(self, path, *args, **kwargs):
        return self._pandas.read_csv(read_path(path), *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._pandas, name)
