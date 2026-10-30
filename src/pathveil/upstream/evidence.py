"""Explicit dated public files and original-origin-masked routing evidence."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
from types import FunctionType

from .contracts import read, require, sha, validate_date, verify, write


def value(mapping):
    return read(mapping) if isinstance(mapping, (str, Path)) else mapping


def public_hashes(data_manifest):
    data = value(data_manifest)
    return {
        str(Path(data["public_files"].get(name, name)).resolve()): expected
        for name, expected in data.get("public_input_sha256", {}).items()
    }


def make_public(data_manifest, core):
    """Reuse the public parsers with an explicit file selector, never fallback."""
    data = value(data_manifest)
    date = validate_date(data["date"])
    files = {k: Path(v).resolve() for k, v in data["public_files"].items()}
    require(
        set(files) >= {"asrel", "asrank_nodes", "asrank_links", "peeringdb"},
        "Dated public inputs are incomplete",
    )
    for name, path in files.items():
        require(path.is_file() and path.stat().st_size > 0, "Missing dated public file: " + name)
    verify(public_hashes(data))
    for name, day in data.get("auxiliary_dates", {}).items():
        require(validate_date(day) <= date, "Future public snapshot: " + name)

    def explicit(_directories, pattern, _date):
        names = {
            "*.as-rel.txt": "asrel",
            "*links.jsonl": "asrank_links",
            "asns*.jsonl": "asrank_nodes",
            "peeringdb_2_dump_*.json": "peeringdb",
        }
        require(pattern in names, "Unexpected public input selector: " + pattern)
        return files[names[pattern]]

    parent = core.public.PublicKnowledge

    def loader(name):
        original = getattr(parent, name)
        # A private globals dictionary keeps the unchanged parser independent
        # of process-global monkey patches and concurrent dated runs.
        return FunctionType(
            original.__code__,
            dict(original.__globals__, _past_file=explicit),
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )

    dated = type(
        "ExplicitPublicKnowledge",
        (parent,),
        {name: loader(name) for name in ("_load_asrel", "_load_asrank", "_load_peeringdb")},
    )
    result = dated(Path("."), Path("."), date)
    require(
        result.skipped_future_rows == {"asrank_links": 0, "asrank_nodes": 0},
        "ASRank file contains future rows",
    )
    return result


def split_roles(split_manifest):
    manifest = value(split_manifest)
    split = manifest.get("split", manifest)
    train = set(map(int, split["train"]))
    validation = set(map(int, split["validation"]))
    formal = set(map(int, split["formal_test"]))
    require(train and not train & (validation | formal), "Training source roles overlap")
    require(not validation & formal, "Validation and formal source roles overlap")
    excluded = {
        int(s) for role, sources in split.items() if isinstance(sources, list) for s in sources
    }
    excluded.update(map(int, manifest.get("excluded_task_origins", [])))
    require(
        train <= excluded and validation <= excluded and formal <= excluded,
        "Every task origin must be excluded before evidence expansion",
    )
    return manifest, split, train, excluded


def source_fold_map(split_manifest, training_sources, fallback=None):
    """Validate explicit source folds, preserving identities across dated subsets."""
    manifest = value(split_manifest)
    train = set(map(int, training_sources))
    registered = manifest.get("source_to_fold")
    if registered is None:
        require(fallback is not None, "The source split has no registered fold mapping")
        registered = fallback(sorted(train))
    try:
        folds = {str(int(source)): int(fold) for source, fold in registered.items()}
    except (AttributeError, TypeError, ValueError):
        require(False, "Invalid source-disjoint fold mapping")
    require(
        len(folds) == len(registered) and set(folds) == set(map(str, train)),
        "Source fold keys must match all and only the registered training sources",
    )
    require(
        all(
            not isinstance(fold, bool) and str(fold) in {"0", "1", "2", "3", "4"}
            for fold in registered.values()
        )
        and set(folds.values()) == set(range(5)),
        "Exactly five nonempty source-disjoint training folds are required",
    )
    return folds


def select_dated_sources(data_manifest, split_manifest, requested, allow_date_template=False):
    """Freeze the observed subset of registered training origins for this date.

    Availability comes only from actual first ASNs in the canonical original
    stream. It cannot be inferred from expanded per-AS CSVs or held-out labels.
    Missing origins remain masked from evidence; no reserve identities are fit.
    """
    data, original = value(data_manifest), value(split_manifest)
    date = validate_date(data["date"])
    registered_date = validate_date(original.get("date", date))
    require(
        registered_date == date or allow_date_template,
        "Explicit source split date differs from the requested date",
    )
    requested = list(map(int, requested))
    require(requested and len(requested) == len(set(requested)), "Invalid formal source list")
    canonical = Path(data["merged_train"]).resolve()
    require(
        canonical.is_file() and canonical.stat().st_size > 0,
        "Canonical original merged stream is missing or empty",
    )
    before = canonical.stat()
    digest, origins, malformed = hashlib.sha256(), Counter(), 0
    with canonical.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            try:
                first, remainder = raw.split(b"->", 1)
                source = int(first)
                if not 0 < source < 2**32 or not remainder.strip():
                    raise ValueError("Invalid canonical origin")
            except ValueError:
                malformed += 1
                continue
            origins[source] += 1
    after = canonical.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "Canonical original stream changed while selecting source roles",
    )
    canonical_sha = digest.hexdigest()
    previous = original.get("source_availability")
    if previous is not None:
        require(
            registered_date == date
            and previous.get("canonical_sha256") == canonical_sha
            and previous.get("canonical_path") == str(canonical),
            "Dated source split belongs to a different canonical original stream",
        )
        requested_train = set(map(int, previous["requested_training_sources"]))
    else:
        requested_train = set(map(int, original["split"]["train"]))
    require(
        not requested_train & set(requested),
        "Requested source was a fitted training identity",
    )
    available = requested_train & set(origins)
    unavailable = requested_train - set(origins)
    split = {role: list(sources) for role, sources in original["split"].items()}
    split["train"] = sorted(available)
    split["formal_test"] = requested
    excluded = set(map(int, original.get("excluded_task_origins", [])))
    excluded.update(int(source) for sources in original["split"].values() for source in sources)
    excluded.update(requested)
    selected = dict(original, date=date, split=split, excluded_task_origins=sorted(excluded))
    assigned = set()
    for role, sources in split.items():
        members = set(map(int, sources))
        require(
            len(members) == len(sources) and not assigned & members,
            "Source identities overlap registered task roles: " + role,
        )
        assigned.update(members)
    registered = original.get("source_to_fold")
    if registered is not None:
        expected = available if previous is not None else requested_train
        source_fold_map(original, expected)
        selected["source_to_fold"] = {
            str(source): int(registered.get(str(source), registered.get(source)))
            for source in sorted(available)
        }
        source_fold_map(selected, available)
    require(available, "No registered training origins have same-date canonical observations")
    split_roles(selected)
    audit = dict(
        policy="registered_training_origins_intersect_same_date_canonical_no_replacement_v1",
        template_date=previous["template_date"] if previous is not None else registered_date,
        canonical_path=str(canonical),
        canonical_sha256=canonical_sha,
        requested_training_sources=sorted(requested_train),
        available_training_sources=sorted(available),
        unavailable_training_sources=sorted(unavailable),
        requested_training_source_count=len(requested_train),
        available_training_source_count=len(available),
        actual_original_source_count=len(origins),
        malformed_origin_rows=malformed,
        training_source_path_counts={str(s): origins[s] for s in sorted(available)},
        fold_policy="preserve_registered_source_to_fold_subset"
        if registered is not None
        else "original_deterministic_source_folds",
        held_out_labels_read=False,
        replacement_sources=[],
    )
    if previous is not None:
        require(previous == audit, "Dated source availability receipt changed")
        require(
            set(map(int, original["split"]["train"])) == available,
            "Dated training source membership differs from its availability receipt",
        )
    selected["source_availability"] = audit
    return selected


def prepare_evidence(data_manifest, split_manifest, work_dir, core):
    """Build original-source-isolated evidence and training-only references.

    Only the actual first ASN in the canonical merged stream assigns origin.
    Per-target CSV filenames never establish the original observation source.
    """
    data = value(data_manifest)
    manifest, split, train, excluded = split_roles(split_manifest)
    date = validate_date(data["date"])
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    canonical = Path(data["merged_train"]).resolve()
    require(canonical.is_file(), "Canonical original merged stream is missing")
    marker = work / "EVIDENCE_COMPLETED.json"
    binding = {
        "date": date,
        "canonical_sha256": sha(canonical),
        "split": split,
        "excluded_task_origins": sorted(excluded),
        "source_to_fold": manifest.get("source_to_fold"),
    }
    if manifest.get("source_availability"):
        require(
            manifest["source_availability"]["canonical_sha256"] == binding["canonical_sha256"],
            "Canonical original stream changed after dated source selection",
        )
    if marker.exists():
        cached = read(marker)
        require(
            cached["binding"] == binding,
            "Evidence cache belongs to different inputs or source roles",
        )
        verify(cached["files_sha256"])
        return cached

    labels_dir = work / "training_origins"
    labels_dir.mkdir(exist_ok=True)
    handles = {}
    origins = Counter()
    malformed = 0
    try:
        with canonical.open("rb") as stream:
            for raw in stream:
                try:
                    source = int(raw.split(b"->", 1)[0])
                except ValueError:
                    malformed += 1
                    continue
                origins[source] += 1
                if source in train:
                    if source not in handles:
                        handles[source] = (labels_dir / f"{source}.csv").open("wb")
                    handles[source].write(raw)
    finally:
        for handle in handles.values():
            handle.close()
    require(
        train <= set(handles),
        "Some registered training origins have no same-date canonical observations: "
        + ",".join(map(str, sorted(train - set(handles)))),
    )
    residual = set(origins) - excluded
    require(residual, "No residual original observations remain after task masking")
    isolated = {
        "tasks": [{"source": s} for s in sorted(excluded)],
        "evidence": [{"source": s} for s in sorted(residual)],
        "canonical_evidence_input": {"path": str(canonical)},
    }
    core.base.build_evidence(isolated, work)
    with canonical.open("rb") as stream:
        index, index_audit = core.index.build_index(stream, excluded, residual)
    audit = read(work / "evidence_audit.json")
    for key in (
        "canonical_sha256",
        "total_lines",
        "total_paths",
        "source_path_counts",
        "withheld_lines",
        "withheld_source_sha256",
        "excluded",
    ):
        require(
            audit[key] == json.loads(json.dumps(index_audit[key])),
            "Independent evidence/index scans disagree: " + key,
        )
    require(
        not any(audit["withheld_start_counts"].values()), "Task source entered residual evidence"
    )
    index_path = work / "safe_index.pkl"
    with index_path.open("wb") as stream:
        pickle.dump(index, stream, protocol=5)
    write(work / "safe_index_audit.json", index_audit)
    refs, joint, training_files = {}, {}, {}
    for source in sorted(train):
        path = labels_dir / f"{source}.csv"
        ref = core.base.read_reference(path, source)
        require(
            ref["counts"].get("paths", 0) > 0,
            "Training source has no valid same-date routes: " + str(source),
        )
        refs[str(source)] = ref
        truth, info = core.joint_labels.read_training_reference(path, source, ref["sha256"])
        joint[str(source)] = dict(
            source=source, provenance=info, **core.joint_labels.source_labels(truth)
        )
        require(
            ref["sha256"] == audit["withheld_source_sha256"].get(str(source)),
            "Extracted training bytes differ from canonical source mask",
        )
        training_files[str(path)] = ref["sha256"]
    refs_path, joint_path = work / "training_references.json", work / "training_joint_labels.json"
    write(refs_path, refs)
    write(joint_path, joint)
    files = [
        work / n
        for n in [
            "evidence.pkl",
            "evidence_audit.json",
            "safe_index.pkl",
            "safe_index_audit.json",
            "training_references.json",
            "training_joint_labels.json",
        ]
    ]
    result = {
        "status": "PASS",
        "binding": binding,
        "date": date,
        "evidence_pickle": str(work / "evidence.pkl"),
        "safe_index_pickle": str(index_path),
        "training_references": str(refs_path),
        "training_joint_labels": str(joint_path),
        "files_sha256": {str(p): sha(p) for p in files},
        "training_origin_files_sha256": training_files,
        "actual_original_source_counts": dict(origins),
        "malformed_origin_rows": malformed,
        "formal_labels_read": False,
        "validation_labels_read": False,
        "source_identity": "actual first ASN of canonical original observation, never filename",
        "no_reverse_or_synthetic_segments": True,
    }
    write(marker, result)
    return result


def unknown_policy():
    return {
        "valid": False,
        "neighbors": {},
        "rows": 0,
        "unresolved_terms": 0,
        "snapshot_from": None,
        "status": "unknown_missing_dated_rpsl",
    }


def load_policies(data_manifest, sources):
    data = value(data_manifest)
    date = validate_date(data["date"])
    policies = {str(s): unknown_policy() for s in sources}
    path = data.get("parsed_policy_file")
    if not path:
        return policies, {"status": "unknown_missing_dated_rpsl", "date": date}
    raw = read(path)
    declared_day = raw.get("snapshot_date") if isinstance(raw, dict) else None
    records = raw.get("policies", raw)
    for source in policies:
        record = records.get(source)
        if not isinstance(record, dict):
            continue
        stamp = record.get("snapshot_from") or declared_day
        if not stamp:
            continue
        day = str(stamp)[:10].replace("-", "")
        require(validate_date(day) <= date, "Future RPSL snapshot for source " + source)
        policies[source] = record
    return policies, {
        "status": "dated_partial_or_complete",
        "date": date,
        "path": str(Path(path).resolve()),
        "sha256": sha(path),
        "valid_sources": sum(bool(p["valid"]) for p in policies.values()),
    }


def policy_features(policy, neighbor, date, core):
    result = core.rpsl.rpsl_features(policy, neighbor)
    stamp = policy.get("snapshot_from")
    snapshot = (
        datetime.fromisoformat(stamp.replace("Z", "+00:00")).replace(tzinfo=None) if stamp else None
    )
    age = (
        ((datetime.strptime(date, "%Y%m%d") - snapshot).total_seconds() / 86400)
        if snapshot
        else 0.0
    )
    require(age >= 0, "Future RPSL feature timestamp")
    result["rpsl_source_snapshot_age_days_log1p"] = math.log1p(age)
    return result
