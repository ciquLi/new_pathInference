"""Resumable, exact-date ASRank downloads with complete pagination checks."""

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .preflight import check_date, inspect_jsonl, sha256_file
from .registry import load_registry, normalize_date, project_root

ENDPOINT = "https://api.asrank.caida.org/v2/graphql"
FIELDS = {
    "asns": """date asn asnName rank organization { orgId orgName }
      cliqueMember seen longitude latitude cone { numberAsns numberPrefixes numberAddresses }
      country { iso name } asnDegree { provider peer customer total transit sibling }
      announcing { numberPrefixes numberAddresses }""",
    "asnLinks": "date relationship numberPaths asn0 { asn } asn1 { asn }",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def query(kind, date, offset, page_size):
    if kind not in FIELDS:
        raise ValueError("ASRank kind must be asns or asnLinks")
    text = normalize_date(date)
    api_date = "%s-%s-%s" % (text[:4], text[4:6], text[6:])
    return (
        '{%s(first:%d,offset:%d,dateStart:"%s",dateEnd:"%s")'
        "{totalCount pageInfo{first offset hasNextPage status} edges{node{%s}}}}"
    ) % (kind, page_size, offset, api_date, api_date, FIELDS[kind])


def request_page(kind, date, offset, page_size, endpoint=ENDPOINT, timeout=120, retries=5):
    request = Request(
        endpoint,
        data=json.dumps({"query": query(kind, date, offset, page_size)}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "PathVeil-reproducible-data/1"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if payload.get("errors"):
                raise ValueError("ASRank GraphQL returned errors: " + str(payload["errors"]))
            connection = payload["data"][kind]
            if not isinstance(connection.get("totalCount"), int):
                raise ValueError("ASRank response has no integer totalCount")
            return connection
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(
                    "ASRank request failed at offset %s: %s" % (offset, exc)
                ) from exc
            time.sleep(min(2**attempt, 30))


def _validate_page(connection, kind, date, offset, page_size, total):
    if connection.get("totalCount") != total:
        raise ValueError("ASRank totalCount changed during pagination")
    rows = [edge["node"] for edge in connection.get("edges", [])]
    expected = min(page_size, total - offset)
    if len(rows) != expected:
        raise ValueError(
            "%s offset %s: expected %s rows, got %s" % (kind, offset, expected, len(rows))
        )
    for row in rows:
        if normalize_date(row.get("date", "")) != normalize_date(date):
            raise ValueError("ASRank page contains a different date")
    page_info = connection.get("pageInfo", {})
    if "hasNextPage" in page_info and bool(page_info["hasNextPage"]) != (
        offset + len(rows) < total
    ):
        raise ValueError("ASRank pageInfo hasNextPage contradicts the row count")
    return rows


def download_asrank(
    date,
    kind,
    output_root=None,
    expected_count=None,
    page_size=5000,
    workers=3,
    endpoint=ENDPOINT,
    progress=None,
    fetcher=None,
):
    """Publish JSONL only after every row, page, date, identity and hash passes.

    Checked page caches survive a failed invocation. Existing nonempty final
    files are validated in full and reused, never accepted on filename alone.
    """
    date = normalize_date(date)
    if kind not in FIELDS or page_size <= 0 or workers < 1:
        raise ValueError("Invalid ASRank kind, page size or worker count")
    root = Path(output_root) if output_root else project_root() / "data/asrank"
    root.mkdir(parents=True, exist_ok=True)
    filename = "asns%s.jsonl" % date if kind == "asns" else "%slinks.jsonl" % date
    target = root / filename
    manifest_path = target.with_suffix(".manifest.json")
    if target.is_file() and target.stat().st_size:
        stats = inspect_jsonl(target, date, kind, expected_count)
        manifest = {
            "status": "complete",
            "date": date,
            "kind": kind,
            "path": str(target.resolve()),
            "source_url": endpoint,
            "bytes": target.stat().st_size,
            **stats,
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("sha256") != stats["sha256"]:
                raise ValueError("Existing download differs from its manifest: " + str(target))
            return existing
        manifest["origin"] = "preexisting_file_fully_validated"
        atomic_json(manifest_path, manifest)
        return manifest
    fetch = fetcher or (
        lambda kind, date, offset, size: request_page(kind, date, offset, size, endpoint=endpoint)
    )
    first = fetch(kind, date, 0, page_size)
    total = first.get("totalCount")
    if not isinstance(total, int) or total <= 0:
        raise ValueError("ASRank returned an empty or invalid totalCount")
    if expected_count is not None and total != expected_count:
        raise ValueError("Registered ASRank count differs: %s != %s" % (total, expected_count))
    cache = root / ".pages" / date / kind / str(page_size)
    cache.mkdir(parents=True, exist_ok=True)
    offsets = list(range(0, total, page_size))

    def page(offset):
        path = cache / ("%09d.json" % offset)
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("date") != date or data.get("kind") != kind or data.get("offset") != offset:
                raise ValueError("Cached ASRank page binding mismatch: " + str(path))
            connection = data["connection"]
        else:
            connection = first if offset == 0 else fetch(kind, date, offset, page_size)
        rows = _validate_page(connection, kind, date, offset, page_size, total)
        content_hash = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if path.exists() and data.get("rows_sha256") != content_hash:
            raise ValueError("Cached ASRank page hash mismatch: " + str(path))
        atomic_json(
            path,
            {
                "date": date,
                "kind": kind,
                "offset": offset,
                "rows_sha256": content_hash,
                "connection": connection,
            },
        )
        if progress:
            progress("%s %s offset %d: %d/%d" % (date, kind, offset, offset + len(rows), total))
        return {
            "offset": offset,
            "count": len(rows),
            "rows_sha256": content_hash,
            "path": str(path.resolve()),
        }

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(
        manifest_path,
        {
            "status": "running",
            "date": date,
            "kind": kind,
            "expected_rows": total,
            "started_utc": started,
            "source_url": endpoint,
        },
    )
    temporary = target.with_suffix(".jsonl.part")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
            pages = list(pool.map(page, offsets))
        with temporary.open("w", encoding="utf-8") as handle:
            for offset in offsets:
                connection = json.loads((cache / ("%09d.json" % offset)).read_text())["connection"]
                for row in _validate_page(connection, kind, date, offset, page_size, total):
                    handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        stats = inspect_jsonl(temporary, date, kind, total)
        os.replace(temporary, target)
        manifest = {
            "status": "complete",
            "date": date,
            "kind": kind,
            "date_start": date,
            "date_end": date,
            "source_url": endpoint,
            "path": str(target.resolve()),
            "bytes": target.stat().st_size,
            "expected_rows": total,
            "page_size": page_size,
            "pages": pages,
            "started_utc": started,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **stats,
        }
        atomic_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        atomic_json(
            manifest_path,
            {
                "status": "failed",
                "date": date,
                "kind": kind,
                "source_url": endpoint,
                "expected_rows": total,
                "error": str(exc),
                "started_utc": started,
            },
        )
        raise


def download_required(date, project_root=None, files_root=None, **kwargs):
    config = load_registry(date, project_root=project_root, files_root=files_root)
    report = check_date(date, project_root=project_root, files_root=files_root)
    output_root = kwargs.pop("output_root", Path(config["project_root"]) / "data/asrank")
    return [
        download_asrank(
            spec["date"],
            spec["kind"],
            output_root=output_root,
            expected_count=spec.get("expected_count"),
            **kwargs,
        )
        for spec in report["required_downloads"]
    ]
