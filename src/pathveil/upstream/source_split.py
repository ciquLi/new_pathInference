"""Freeze date-local source roles from all sufficiently observed original paths.

Eligibility uses unique valid full paths in the canonical original stream,
never per-AS expanded CSVs. This is a new protocol, independent of the legacy
160-source template and its balanced folds. No reference labels are used to
rank sources, choose a role, or decide a model policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from .contracts import read, require, sha, validate_date, write
from .implementation.neighbor_features import parse_path


PROTOCOL = "all_canonical_unique_valid_paths_gt_threshold_v1"
FOLD_POLICY = "fixed_ASN_SHA256_modulo_5_v1"
PATH_RULE = dict(
    parser="pathveil.upstream.implementation.neighbor_features.parse_path",
    minimum_AS_count=2,
    maximum_AS_count=None,
    ASN_range="0 < ASN < 2**32",
    repeated_ASN="reject_entire_path",
    identity="ordered_full_integer_ASN_tuple",
    normalization="integer conversion only; no reversal, expansion or segment extraction",
    source="first ASN of original canonical AS_PATH, not prefix-origin final ASN",
)


def _load(value):
    return read(value) if isinstance(value, (str, Path)) else value


def _ids(values, name):
    result = list(map(int, values))
    require(
        bool(result)
        and len(result) == len(set(result))
        and all(0 < source < 2**32 for source in result),
        "Invalid or duplicated " + name,
    )
    return sorted(result)


class _InterleavedSource(Exception):
    pass


def inventory_sources(canonical, progress=None):
    """Use a grouped fast path, falling back to exact on-disk deduplication."""
    try:
        return _inventory_grouped(canonical, progress)
    except _InterleavedSource:
        if progress:
            progress(dict(stage="restart_exact_SQLite_dedup", reason="interleaved_sources"))
        return _inventory_sqlite(canonical, progress)


def _inventory_grouped(canonical, progress=None):
    """Scan once with bounded per-source deduplication and raw-byte receipts.

    Every valid-source row contributes to its raw SHA, including rows rejected
    by the existing reference parser. A noncontiguous source triggers a full
    exact SQLite rescan instead of becoming an eligibility restriction.
    """
    canonical = Path(canonical).resolve()
    require(
        canonical.is_file() and canonical.stat().st_size > 0,
        "Canonical original merged stream is missing or empty",
    )
    before = canonical.stat()
    digest, sources = hashlib.sha256(), {}
    totals = Counter()
    closed, seen = set(), set()
    current, raw_digest, counts = None, None, Counter()

    def flush():
        if current is None:
            return
        reference = hashlib.sha256()
        for text in sorted("->".join(map(str, nodes)).encode("ascii") for nodes in seen):
            reference.update(text + b"\n")
        row = dict(
            source=current,
            raw_rows=counts["raw_rows"],
            valid_rows=counts["valid_rows"],
            invalid_rows=counts["invalid_rows"],
            unique_valid_paths=len(seen),
            duplicate_valid_rows=counts["valid_rows"] - len(seen),
            raw_sha256=raw_digest.hexdigest(),
            reference_unique_sha256=reference.hexdigest(),
        )
        sources[str(current)] = row
        totals["unique_valid_paths"] += len(seen)
        totals["duplicate_valid_rows"] += row["duplicate_valid_rows"]
        closed.add(current)

    with canonical.open("rb") as stream:
        for raw in stream:
            totals["raw_rows"] += 1
            digest.update(raw)
            try:
                source = int(raw.split(b"->", 1)[0])
                if not 0 < source < 2**32:
                    raise ValueError("invalid ASN")
            except ValueError:
                totals["malformed_source_rows"] += 1
                continue
            if source != current:
                flush()
                if source in closed:
                    raise _InterleavedSource()
                current, raw_digest, counts, seen = source, hashlib.sha256(), Counter(), set()
            counts["raw_rows"] += 1
            totals["valid_source_rows"] += 1
            raw_digest.update(raw)
            try:
                nodes = parse_path(raw.decode("utf-8"))
            except UnicodeError:
                nodes = None
            if nodes is None:
                counts["invalid_rows"] += 1
                totals["invalid_path_rows"] += 1
            else:
                require(nodes[0] == source, "Parsed canonical source changed")
                counts["valid_rows"] += 1
                totals["valid_rows"] += 1
                seen.add(nodes)
            if progress and totals["raw_rows"] % 1000000 == 0:
                progress(
                    dict(
                        raw_rows=totals["raw_rows"],
                        completed_sources=len(sources),
                        current_source=source,
                        current_source_unique_paths=len(seen),
                    )
                )
    flush()
    return _inventory_receipt(canonical, before, digest.hexdigest(), totals, sources, True)


def _inventory_receipt(canonical, before, digest, totals, sources, grouped):
    after = canonical.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "Canonical stream changed during inventory",
    )
    for name in (
        "raw_rows",
        "valid_source_rows",
        "malformed_source_rows",
        "valid_rows",
        "invalid_path_rows",
        "unique_valid_paths",
        "duplicate_valid_rows",
    ):
        totals.setdefault(name, 0)
    return dict(
        schema_version=1,
        canonical_path=str(canonical),
        canonical_sha256=digest,
        canonical_bytes=after.st_size,
        canonical_mtime_ns=after.st_mtime_ns,
        source_count=len(sources),
        source_count_with_valid_paths=sum(
            row["unique_valid_paths"] > 0 for row in sources.values()
        ),
        totals=dict(totals),
        path_rule=PATH_RULE,
        unique_reference_encoding="byte lexical sort; canonical decimal ASNs joined by ->; LF",
        canonical_deduplication_assumed=False,
        canonical_is_unique_for_valid_paths=totals["duplicate_valid_rows"] == 0,
        source_grouped=grouped,
        dedup_backend="per_source_integer_tuple_set"
        if grouped
        else "SQLite_exact_unique_path_BLOB",
        scan_passes=1 if grouped else 2,
        expanded_CSVs_read=False,
        sources=sources,
    )


def _inventory_sqlite(canonical, progress=None):
    """Bounded memory exact fallback; the temporary database never edits input."""
    canonical = Path(canonical).resolve()
    before = canonical.stat()
    digest, totals, counts, raw_digests = hashlib.sha256(), Counter(), {}, {}
    with tempfile.TemporaryDirectory(prefix="pathveil_source_inventory_") as directory:
        connection = sqlite3.connect(str(Path(directory) / "unique_paths.sqlite"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute(
                "CREATE TABLE paths(source INTEGER, path BLOB, PRIMARY KEY(source,path)) WITHOUT ROWID"
            )
            batch = []
            with canonical.open("rb") as stream:
                for raw in stream:
                    digest.update(raw)
                    totals["raw_rows"] += 1
                    try:
                        source = int(raw.split(b"->", 1)[0])
                        if not 0 < source < 2**32:
                            raise ValueError("invalid ASN")
                    except ValueError:
                        totals["malformed_source_rows"] += 1
                        continue
                    if source not in counts:
                        counts[source], raw_digests[source] = Counter(), hashlib.sha256()
                    counts[source]["raw_rows"] += 1
                    totals["valid_source_rows"] += 1
                    raw_digests[source].update(raw)
                    try:
                        nodes = parse_path(raw.decode("utf-8"))
                    except UnicodeError:
                        nodes = None
                    if nodes is None:
                        counts[source]["invalid_rows"] += 1
                        totals["invalid_path_rows"] += 1
                    else:
                        require(nodes[0] == source, "Parsed canonical source changed")
                        counts[source]["valid_rows"] += 1
                        totals["valid_rows"] += 1
                        batch.append((source, "->".join(map(str, nodes)).encode("ascii")))
                        if len(batch) >= 10000:
                            connection.executemany("INSERT OR IGNORE INTO paths VALUES(?,?)", batch)
                            batch.clear()
                    if progress and totals["raw_rows"] % 1000000 == 0:
                        progress(
                            dict(
                                raw_rows=totals["raw_rows"],
                                seen_sources=len(counts),
                                backend="SQLite",
                            )
                        )
            if batch:
                connection.executemany("INSERT OR IGNORE INTO paths VALUES(?,?)", batch)
            connection.commit()
            unique_counts, references = Counter(), {}
            for source, text in connection.execute(
                "SELECT source,path FROM paths ORDER BY source,path"
            ):
                unique_counts[source] += 1
                if source not in references:
                    references[source] = hashlib.sha256()
                references[source].update(text + b"\n")
            sources = {}
            for source, row in counts.items():
                unique = unique_counts[source]
                duplicates = row["valid_rows"] - unique
                sources[str(source)] = dict(
                    source=source,
                    raw_rows=row["raw_rows"],
                    valid_rows=row["valid_rows"],
                    invalid_rows=row["invalid_rows"],
                    unique_valid_paths=unique,
                    duplicate_valid_rows=duplicates,
                    raw_sha256=raw_digests[source].hexdigest(),
                    reference_unique_sha256=references.get(source, hashlib.sha256()).hexdigest(),
                )
                totals["unique_valid_paths"] += unique
                totals["duplicate_valid_rows"] += duplicates
            return _inventory_receipt(canonical, before, digest.hexdigest(), totals, sources, False)
        finally:
            connection.close()


def split_sources(inventory, date, formal_sources, min_paths=10000, seed=20261021):
    """Use every eligible source, assigning roles using source identity only."""
    date = validate_date(str(date))
    require(
        isinstance(min_paths, int) and not isinstance(min_paths, bool) and min_paths >= 0,
        "Invalid exclusive path threshold",
    )
    require(isinstance(seed, int) and not isinstance(seed, bool), "Seed must be a fixed integer")
    formal = _ids(formal_sources, "formal sources")
    rows = inventory["sources"]
    qualified = {
        int(source) for source, row in rows.items() if row["unique_valid_paths"] > min_paths
    }
    eligible = qualified - set(formal)
    salt = f"{seed}:source:"
    ordered = sorted(
        eligible, key=lambda source: hashlib.sha256(f"{salt}{source}".encode()).hexdigest()
    )
    require(len(ordered) >= 8, "Too few eligible sources for train/validation/test plus five folds")
    ntrain, nvalidation = len(ordered) * 2 // 3, len(ordered) // 6
    split = dict(
        train=sorted(ordered[:ntrain]),
        validation=sorted(ordered[ntrain : ntrain + nvalidation]),
        test=sorted(ordered[ntrain + nvalidation :]),
        formal_test=formal,
    )
    fold_salt = f"{seed}:neighbor_oof_stable_source_v1:"
    folds = {
        str(source): int(hashlib.sha256(f"{fold_salt}{source}".encode()).hexdigest(), 16) % 5
        for source in split["train"]
    }
    require(set(folds.values()) == set(range(5)), "Training sources do not populate all five folds")
    excluded = eligible | set(formal)
    background = {int(s): row for s, row in rows.items() if int(s) not in excluded}
    background_totals = {
        name: sum(row[name] for row in background.values())
        for name in (
            "raw_rows",
            "valid_rows",
            "unique_valid_paths",
            "duplicate_valid_rows",
            "invalid_rows",
        )
    }
    background_totals.update(
        source_count=len(background),
        source_count_with_valid_paths=sum(
            row["unique_valid_paths"] > 0 for row in background.values()
        ),
    )
    require(
        background_totals["unique_valid_paths"] > 0,
        "No independent original background paths remain after masking all task sources",
    )
    return dict(
        schema_version=1,
        protocol=PROTOCOL,
        date=date,
        seed=seed,
        split=split,
        excluded_task_origins=sorted(excluded),
        source_to_fold=folds,
        eligibility=dict(
            path_count_field="unique_valid_paths",
            operator=">",
            threshold=min_paths,
            qualified_before_formal_exclusion=sorted(qualified),
            formal_sources_removed=sorted(qualified & set(formal)),
            eligible_sources=sorted(eligible),
            eligible_count=len(eligible),
            all_eligible_sources_assigned=True,
            legacy_160_template_used=False,
            expanded_CSVs_read=False,
        ),
        role_assignment=dict(
            policy="stable_source_SHA256_order_all_eligible_2over3_1over6_remainder_v1",
            salt=salt,
            ordered_eligible_sources=ordered,
            training_count=ntrain,
            validation_count=nvalidation,
            test_count=len(ordered) - ntrain - nvalidation,
            formal_reference_files_read=False,
            candidate_coverage_or_model_scores_read=False,
            content_use="Original path validity and unique count determine eligibility only; roles use ASN hashes only",
        ),
        fold_assignment=dict(
            policy=FOLD_POLICY,
            salt=fold_salt,
            hash_interpretation="SHA256 digest as big-endian unsigned integer modulo 5",
            stable_across_date_source_subsets=True,
            legacy_balanced_folds_reused=False,
        ),
        canonical_evidence_input=dict(
            path=inventory["canonical_path"],
            sha256=inventory["canonical_sha256"],
            bytes=inventory["canonical_bytes"],
        ),
        source_reference_hashes={
            source: dict(
                raw_sha256=row["raw_sha256"], reference_unique_sha256=row["reference_unique_sha256"]
            )
            for source, row in rows.items()
            if int(source) in excluded
        },
        remaining_background=background_totals,
        background_sufficiency="Counts only; nonempty background is required, adequacy for retrieval/training is not established",
        path_rule=PATH_RULE,
    )


def prepare_source_split(
    data_manifest, output_dir, min_paths=10000, seed=20261021, formal_sources=None, progress=None
):
    """Write an immutable inventory/split and return the registered split path."""
    data = _load(data_manifest)
    date = validate_date(str(data["date"]))
    canonical, output = Path(data["merged_train"]).resolve(), Path(output_dir).resolve()
    require(canonical.is_file(), "Canonical original merged stream is missing")
    require(
        canonical != output and canonical not in output.parents,
        "Output must not be inside an input file",
    )
    configuration = dict(
        protocol=PROTOCOL,
        date=date,
        canonical_path=str(canonical),
        formal_sources=_ids(
            formal_sources if formal_sources is not None else data["sources"], "formal sources"
        ),
        min_paths_exclusive=min_paths,
        seed=seed,
        code_sha256={
            str(path): sha(path)
            for path in (
                Path(__file__),
                Path(__file__).parent / "implementation/neighbor_features.py",
            )
        },
        path_rule=PATH_RULE,
    )
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "SOURCE_SPLIT_COMPLETED.json"
    if marker.exists():
        previous = read(marker)
        require(
            previous["configuration"] == configuration,
            "Completed source protocol configuration changed",
        )
        require(
            sha(canonical) == previous["canonical_sha256"],
            "Canonical stream changed after split freeze",
        )
        for path, expected in previous["outputs_sha256"].items():
            require(
                Path(path).is_file() and sha(path) == expected,
                "Frozen source split output changed: " + path,
            )
        return output / "split.json"
    require(
        not any((output / name).exists() for name in ("inventory.json", "split.json")),
        "Incomplete source protocol outputs exist; select a new isolated output directory",
    )
    inventory = inventory_sources(canonical, progress)
    inventory["date"] = date
    split = split_sources(inventory, date, configuration["formal_sources"], min_paths, seed)
    write(output / "inventory.json", inventory)
    split["inventory"] = dict(
        path=str(output / "inventory.json"), sha256=sha(output / "inventory.json")
    )
    write(output / "split.json", split)
    write(
        marker,
        dict(
            status="PASS",
            configuration=configuration,
            canonical_sha256=inventory["canonical_sha256"],
            outputs_sha256={
                str(output / name): sha(output / name) for name in ("inventory.json", "split.json")
            },
            role_counts={name: len(sources) for name, sources in split["split"].items()},
            remaining_background=split["remaining_background"],
            no_model_training=True,
            original_corpus_modified=False,
        ),
    )
    return output / "split.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-paths", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    split_path = prepare_source_split(
        args.data_manifest,
        args.out_dir,
        args.min_paths,
        args.seed,
        progress=lambda value: print(
            json.dumps({"stage": "source_inventory", **value}), flush=True
        ),
    )
    result = read(split_path)
    print(
        json.dumps(
            dict(
                status="PASS",
                split=str(split_path),
                role_counts={k: len(v) for k, v in result["split"].items()},
                remaining_background=result["remaining_background"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
