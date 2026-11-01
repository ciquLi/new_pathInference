"""Small shared helpers; no numerical or machine-learning dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("PATHVEIL_PROJECT_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_date(value: str) -> str:
    if len(value) != 8 or not value.isascii() or not value.isdecimal():
        raise ValueError("Date must use YYYYMMDD, for example 20260501")
    datetime.strptime(value, "%Y%m%d")
    return value


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def object_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def file_record(path: Path | str) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_record(record: dict) -> Path:
    path = Path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError("Frozen input/output changed: " + str(path))
    return path
