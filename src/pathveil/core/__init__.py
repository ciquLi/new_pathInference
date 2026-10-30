"""Preserved numerical implementation and its compatibility imports.

Legacy pickled GMPI checkpoints name top-level modules. The private import
path preserves these names without changing model architecture or tensors.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def activate(data_root=None, asrank_links=None, data_manifest=None):
    if data_manifest is not None:
        import json

        path = Path(data_manifest).resolve()
        manifest = json.loads(path.read_text())
        data_root = manifest["source_data_root"]
        asrank_links = manifest["public_files"]["asrank_links"]
        os.environ["PATHVEIL_DATA_MANIFEST"] = str(path)
        os.environ["PATHVEIL_GRAPH_DIR"] = manifest["graph_dir"]
    if data_root is not None:
        root = str(Path(data_root).resolve())
        previous = os.environ.get("ASINFER_FILES_ROOT")
        if "environment" in sys.modules and previous and previous != root:
            raise RuntimeError("Use a fresh subprocess when changing routing snapshots")
        os.environ["ASINFER_FILES_ROOT"] = root
    if asrank_links is not None:
        os.environ["PATHVEIL_ASRANK_LINKS"] = str(Path(asrank_links).resolve())
    os.environ["PATHVEIL_STRICT_DATA"] = "1"
    here = str(Path(__file__).resolve().parent)
    if here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)
    importlib.invalidate_caches()
    return Path(here)
