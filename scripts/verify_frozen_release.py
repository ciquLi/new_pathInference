#!/usr/bin/env python3
"""Read-only verification of frozen code and optional local baseline assets.

Never starts experiments, repairs metadata, or rewrites historical identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT_FILES = ("AGENTS.md", ".gitignore", "pyproject.toml", "requirements.lock.txt")
SOURCE_DIRS = ("src", "scripts", "configs", "tests")


def source_files(root):
    """Include added files as well as tracked files in the release code scope."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "--", *SOURCE_DIRS, *ROOT_FILES],
        cwd=root, check=True, stdout=subprocess.PIPE,
    )
    return set(result.stdout.decode().rstrip("\0").split("\0")) - {""}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_hashes(root, expected):
    errors = []
    for name, value in expected.items():
        path = root / name
        if not path.is_file():
            errors.append("missing: " + name)
        elif digest(path) != value:
            errors.append("changed: " + name)
    return errors


def verify(root, manifest, assets=False):
    expected = manifest["source_files_sha256"]
    errors = check_hashes(root, expected)
    for name in sorted(source_files(root) - set(expected)):
        errors.append("unfrozen source: " + name)
    count = len(expected)
    if assets:
        errors.extend(check_hashes(root, manifest["baseline_assets_sha256"]))
        count += len(manifest["baseline_assets_sha256"])
    return dict(
        status="FAIL" if errors else "PASS",
        release=manifest["release_tag"],
        scope="code_and_baseline_assets" if assets else "code_only",
        checked_files=count,
        errors=errors,
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "docs/releases/pats-v1.0-21x5.json",
    )
    parser.add_argument(
        "--assets", action="store_true",
        help="Also hash the registered local inputs, models, candidates and predictions.",
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        result = verify(root, manifest, assets=args.assets)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        result = dict(status="FAIL", errors=[str(error)])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
