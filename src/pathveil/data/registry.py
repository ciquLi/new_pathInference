"""Date registries bind routing and auxiliary snapshots independently.

Only paths explicitly listed in a date configuration can be selected. There
is deliberately no nearest-date lookup in this module.
"""

import datetime as dt
import json
import os
from pathlib import Path

from ..utils import project_root as find_project_root


def project_root():
    return find_project_root()


def normalize_date(value):
    text = str(value).replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError("Date must use YYYYMMDD or YYYY-MM-DD")
    dt.datetime.strptime(text, "%Y%m%d")
    return text


def load_registry(date, project_root=None, files_root=None):
    """Load a checked registry and expand its explicit path templates."""
    date = normalize_date(date)
    root = Path(project_root).resolve() if project_root else globals()["project_root"]()
    path = root / "configs" / "dates" / (date + ".json")
    if not path.is_file():
        raise ValueError("No registered experiment date: " + date)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("date") != date:
        raise ValueError("Invalid date registry: " + str(path))
    normalize_date(config["routing_date"])
    for value in config["auxiliary_dates"].values():
        normalize_date(value)
    configured_root = files_root or os.environ.get("ASINFER_FILES_ROOT") or config["files_root"]
    variables = {"project_root": str(root), "files_root": str(Path(configured_root).resolve())}
    for group in ["raw_inputs", "derived_inputs"]:
        for asset in config.get(group, []):
            if asset.get("snapshot_date"):
                normalize_date(asset["snapshot_date"])
            asset["candidates"] = [
                str(Path(item.format(**variables))) for item in asset["candidates"]
            ]
    config["registry_path"] = str(path)
    config["project_root"] = str(root)
    config["files_root"] = variables["files_root"]
    if config.get("routing_files_root"):
        config["routing_files_root"] = str(Path(config["routing_files_root"].format(**variables)).resolve())
    return config


def select_asset(asset):
    """Return the first nonempty explicit candidate; never search by date."""
    directory = asset.get("type") == "directory"
    for name in asset["candidates"]:
        path = Path(name)
        if directory:
            if path.is_dir() and any(path.iterdir()):
                return path
        elif path.is_file() and path.stat().st_size > 0:
            return path
    return None
