"""Inspect real inputs; completion-marker files never establish readiness."""

import hashlib
import json
from pathlib import Path

from .registry import load_registry, normalize_date, select_asset


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_jsonl(path, date, kind, expected_count=None):
    """Validate every JSONL row's date and unique ASN/link identity."""
    seen = set()
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            row = json.loads(line)
            if normalize_date(row.get("date", "")) != normalize_date(date):
                raise ValueError("ASRank date mismatch at row %s" % count)
            key = (
                str(row["asn"])
                if kind == "asns"
                else (str(row["asn0"]["asn"]), str(row["asn1"]["asn"]))
            )
            if key in seen:
                raise ValueError("Duplicate ASRank identity at row %s" % count)
            seen.add(key)
    if not count or (expected_count is not None and count != expected_count):
        raise ValueError("ASRank row count mismatch: %s != %s" % (count, expected_count))
    return {
        "rows": count,
        "sha256": sha256_file(path),
        "all_dates_verified": True,
        "unique_rows": len(seen),
    }


def inspect_asset(asset, deep=False):
    path = select_asset(asset)
    required = asset.get("required", True)
    purpose = asset.get("purpose", "experiment_input")
    result = {
        "id": asset["id"],
        "snapshot_date": asset.get("snapshot_date"),
        "candidates": asset["candidates"],
        "required": required,
        "optional": not required,
        "purpose": purpose,
        "ready": path is not None,
    }
    if path is None:
        result["status"] = "missing_or_empty" if required else "optional_missing_or_empty"
        return result
    result.update(path=str(path), status="present" if required else "optional_present")
    if purpose == "archive_only":
        # RIBs are retained to reproduce the historical extraction, but the
        # current workflow consumes the extracted paths and graph instead.
        # Do not enumerate, parse or hash potentially huge archived RIBs.
        result["validation"] = "archive_presence_only"
        result["used_by_experiment"] = False
        return result
    if path.is_dir():
        files = [item for item in path.iterdir() if item.is_file()]
        result.update(
            file_count=len(files),
            bytes=sum(item.stat().st_size for item in files),
            empty_files=[item.name for item in files if item.stat().st_size == 0],
        )
        result["ready"] = bool(files) and any(item.stat().st_size > 0 for item in files)
    else:
        result["bytes"] = path.stat().st_size
        try:
            if asset.get("download"):
                if deep:
                    spec = asset["download"]
                    result.update(
                        inspect_jsonl(path, spec["date"], spec["kind"], spec.get("expected_count"))
                    )
                else:
                    with path.open(encoding="utf-8") as handle:
                        row = json.loads(next(handle))
                    if normalize_date(row.get("date", "")) != asset["snapshot_date"]:
                        raise ValueError("First ASRank row does not match the registered date")
                    result["validation"] = "first_row_date_only"
            elif asset["id"] == "peeringdb" and deep:
                data = json.loads(path.read_text())
                if not data.get("net", {}).get("data"):
                    raise ValueError("PeeringDB net collection is empty")
                result["network_count"] = len(data["net"]["data"])
            if deep and "sha256" not in result:
                result["sha256"] = sha256_file(path)
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            result.update(ready=False, status="invalid", error=str(exc))
    return result


def classify_empty_csv(path, reports_dir=None):
    """Empty CSVs require a parser report, even for an IPv6 collector."""
    path = Path(path)
    candidates = [
        path.with_suffix(path.suffix + ".report.json"),
        path.with_suffix(path.suffix + ".parse.json"),
        path.with_suffix(".parse.json"),
    ]
    if reports_dir:
        candidates.insert(0, Path(reports_dir) / (path.name + ".json"))
    for report in candidates:
        if not report.is_file():
            continue
        data = json.loads(report.read_text())
        source = Path(data["source"]) if data.get("source") else None
        bound = (
            source is not None
            and source.is_file()
            and source.stat().st_size == data.get("source_size")
            and source.stat().st_mtime_ns == data.get("source_mtime_ns")
        )
        # Require a nonempty completed parse bound to the current RIB. A
        # collector's name or an unrelated stale report is not sufficient.
        if (
            bound
            and data.get("status") in {"complete", "success", "done", "no_usable_paths"}
            and data.get("records", 0) > 0
            and data.get("paths") == 0
            and not data.get("parse_errors", 0)
        ):
            return {
                "path": str(path),
                "status": "valid_empty_after_filtering",
                "report": str(report),
            }
        return {
            "path": str(path),
            "status": "empty_parse_report_requires_review",
            "report": str(report),
        }
    return {
        "path": str(path),
        "status": "empty_unverified",
        "possible_ipv6_only": "route-views6" in path.name,
    }


def check_date(date, project_root=None, files_root=None, deep=False):
    config = load_registry(date, project_root=project_root, files_root=files_root)
    raw = [inspect_asset(asset, deep=deep) for asset in config["raw_inputs"]]
    derived = [inspect_asset(asset, deep=False) for asset in config["derived_inputs"]]
    missing = [asset["id"] for asset in raw if asset["required"] and not asset["ready"]]
    optional_unavailable = [
        asset["id"] for asset in raw if asset["optional"] and not asset["ready"]
    ]
    downloads = [
        asset["download"]
        for asset, result in zip(config["raw_inputs"], raw)
        if asset.get("required", True) and asset.get("download") and not result["ready"]
    ]
    converted = Path(config["files_root"]) / "data/aspath/bgp" / config["routing_date"]
    empties = [
        classify_empty_csv(path) for path in converted.glob("*/*.csv") if path.stat().st_size == 0
    ]
    for parser_report in converted.glob("*/*.csv.parse.json"):
        csv_path = Path(str(parser_report)[: -len(".parse.json")])
        if not csv_path.exists():
            classification = classify_empty_csv(csv_path)
            if classification["status"] == "valid_empty_after_filtering":
                empties.append(classification)
    test_dir = (
        Path(config["files_root"])
        / "data/aspath/path_by_vantage"
        / config["routing_date"]
        / "path_without_prefix/test"
    )
    aggregates = {
        name: (test_dir / name).is_file() and (test_dir / name).stat().st_size > 0
        for name in [
            "merged_paths.txt",
            "expand_paths.txt",
            "pathWithFreq.json",
            "pathWithKey.pkl",
        ]
    }
    warnings = []
    if optional_unavailable:
        warnings.append(
            "Optional archive inputs are unavailable; extracted paths and graph must still pass "
            "run-local preparation: " + ", ".join(optional_unavailable)
        )
    if empties:
        warnings.append(
            "Converted empty CSVs are classified separately; they do not establish raw RIB corruption."
        )
    if config["mode"] == "bgp" and not all(aggregates.values()):
        warnings.append("Test aggregates need isolated rebuilding; source ASN CSVs remain usable.")
    if config["date"] in {"20260501", "20260601"}:
        warnings.append(
            "Existing AS_Feature and AS embedding lack same-date ASRank provenance and must be rebuilt."
        )
    source_ids = sorted(path.stem for path in test_dir.glob("*.csv"))
    if source_ids and source_ids != sorted(config["test_sources"]):
        warnings.append(
            "Existing test cohort differs from registry; source-level isolation must be checked."
        )
    return {
        "date": config["date"],
        "mode": config["mode"],
        "routing_date": config["routing_date"],
        "auxiliary_dates": config["auxiliary_dates"],
        "registry_path": config["registry_path"],
        "ready": not missing,
        "raw_ready": not missing,
        "derived_ready": False,
        "derived_available": all(item["ready"] for item in derived),
        "derived_readiness_note": "Run-local preparation verifies provenance; legacy done markers are ignored.",
        "raw_inputs": raw,
        "derived_inputs": derived,
        "required_downloads": downloads,
        "optional_inputs_unavailable": optional_unavailable,
        "errors": ["Missing or invalid required input: " + name for name in missing],
        "warnings": warnings,
        "empty_converted_csvs": empties,
        "test_aggregates": aggregates,
        "test_sources": config["test_sources"],
        "existing_test_sources": source_ids,
        "common9": config["common9"],
    }
