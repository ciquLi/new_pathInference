"""Build a private data root while retaining immutable source evidence.

Legacy data are read only. Run-local feature files are ordinary copies, never
symlinks that a training/preprocessing function could write through.
"""

import csv
import json
import os
from pathlib import Path
import pickle
import shutil

from .downloader import atomic_json
from .preflight import check_date, sha256_file
from .registry import load_registry, normalize_date, select_asset

DEFAULT_FEATURE = [100000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
AGGREGATES = ("merged_paths.txt", "expand_paths.txt", "pathWithFreq.json", "pathWithKey.pkl")


def read_double_json(path):
    value = json.loads(Path(path).read_text())
    return json.loads(value) if isinstance(value, str) else value


def write_double_json(path, value):
    path = Path(path)
    if path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(json.dumps(value, separators=(",", ":"))) + "\n")


def rebuild_test_aggregates(test_dir, source_csv_dir=None):
    """Reproduce legacy merge/expand/index semantics in an isolated directory.

    The original expansion counts a path once per >100000-item flush block,
    not once per raw occurrence. Retaining that rule preserves experiment
    semantics while sorting emitted sets removes incidental hash-order noise.
    """
    test_dir = Path(test_dir)
    for name in AGGREGATES:
        path = test_dir / name
        if path.is_symlink():
            path.unlink()
    merged = test_dir / "merged_paths.txt"
    with merged.open("w") as output:
        for path in Path(source_csv_dir or test_dir).glob("*.csv"):
            with path.open(newline="") as handle:
                for row in csv.reader(handle):
                    if row:
                        output.write(row[0] + "\n")
    frequencies = {}
    block = []

    def flush():
        for value in set(block):
            frequencies[value] = frequencies.get(value, 0) + 1
        block.clear()

    with merged.open() as handle:
        for line in handle:
            path = line.strip()
            nodes = path.split("->")
            if len(nodes) < 3:
                block.append(path)
            else:
                for start in range(len(nodes) - 1):
                    for end in range(start + 2, len(nodes) + 1):
                        block.append("->".join(nodes[start:end]))
            if len(block) > 100000:
                flush()
    flush()
    index = {}
    with (test_dir / "expand_paths.txt").open("w") as output:
        for path in sorted(frequencies):
            output.write(path + "\n")
            nodes = path.split("->")
            index.setdefault(nodes[0] + "-" + nodes[-1], []).append(path)
    write_double_json(test_dir / "pathWithFreq.json", frequencies)
    with (test_dir / "pathWithKey.pkl").open("wb") as handle:
        pickle.dump(index, handle)
    return {
        "method": "legacy_chunk_dedup_expansion",
        "unique_expanded_paths": len(frequencies),
        "distinct_source_destination_pairs": len(index),
        "files": {name: sha256_file(test_dir / name) for name in AGGREGATES},
    }


def rebuild_as_features(asrank_path, source_features, edge_frequency, output_dir, snapshot_date):
    """Use the graph's original first-encounter order for categorical codes."""
    nodes = {}
    with Path(asrank_path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if normalize_date(row.get("date", "")) != normalize_date(snapshot_date):
                raise ValueError("ASRank node date does not match registered auxiliary date")
            nodes[str(row["asn"])] = row
    old_features = read_double_json(source_features)
    countries, organizations, features = {}, {}, {}
    missing = 0
    for asn in old_features:
        try:
            row = nodes[str(asn)]
            degree = row["asnDegree"]
            country = row["country"]["iso"]
            countries.setdefault(country, len(countries) + 1)
            organization = row["organization"]["orgId"]
            organizations.setdefault(organization, len(organizations) + 1)
            features[asn] = [
                row["rank"],
                degree["total"],
                degree["transit"],
                degree["provider"],
                degree["peer"],
                degree["customer"],
                countries[country],
                organizations[organization],
                round(row["longitude"], 2),
                round(row["latitude"], 2),
            ]
        except (KeyError, TypeError):
            features[asn] = DEFAULT_FEATURE[:]
            missing += 1
    output = Path(output_dir)
    write_double_json(output / "AS_Feature.json", features)
    edges = read_double_json(edge_frequency)
    edge_features = {}
    for edge, frequency in edges.items():
        left, right = edge.split("-")
        edge_features[edge] = (
            [frequency] + features.get(left, DEFAULT_FEATURE) + features.get(right, DEFAULT_FEATURE)
        )
    write_double_json(output / "edge_feature.json", edge_features)
    return {
        "nodes": len(features),
        "default_feature_nodes": missing,
        "snapshot_date": snapshot_date,
        "categorical_encoding": "original_graph_node_encounter_order",
    }


def rebuild_as_embedding(embedding_path, features_path, output_path):
    """Keep BGP node2vec vectors and concatenate normalized same-date features."""
    import numpy as np

    features = read_double_json(features_path)
    embeddings = {}
    with Path(embedding_path).open() as handle:
        header = next(handle).split()
        count, dim = map(int, header)
        for line in handle:
            tokens = line.split()
            if len(tokens) != dim + 1:
                raise ValueError("Malformed node2vec embedding row")
            key = tokens[0]
            if key in embeddings:
                raise ValueError("Duplicate node2vec node")
            vector = np.asarray(tokens[1:], dtype=np.float32)
            raw = np.asarray(features.get(key, DEFAULT_FEATURE))
            norm = np.linalg.norm(raw)
            if not np.isfinite(norm) or norm == 0 or not np.all(np.isfinite(vector)):
                raise ValueError("Invalid feature/vector norm for " + key)
            embeddings[key] = np.concatenate((raw / norm, vector))
    if len(embeddings) != count:
        raise ValueError("Node2vec header count does not match rows")
    for key in features.keys() - embeddings.keys():
        raw = np.asarray(features[key])
        norm = np.linalg.norm(raw)
        if not np.isfinite(norm) or not norm:
            raise ValueError("Invalid feature norm for " + key)
        embeddings[key] = np.concatenate((raw / norm, np.zeros(dim)))
    path = Path(output_path)
    if path.is_symlink():
        path.unlink()
    with path.open("wb") as handle:
        pickle.dump(embeddings, handle)
    return {
        "nodes": len(embeddings),
        "node2vec_dimension": dim,
        "feature_dimension": 10,
        "sha256": sha256_file(path),
    }


def resolve_read_path(manifest, path):
    """Resolve a legacy read through explicit mappings; never use for writes."""
    value = str(Path(path).absolute())
    candidate = Path(value)
    exact = manifest.get("read_path_map", {})
    if value in exact:
        return Path(exact[value])
    if candidate.is_file():
        return candidate
    prefixes = sorted(manifest.get("path_prefix_map", {}), key=len, reverse=True)
    for prefix in prefixes:
        if value == prefix or value.startswith(prefix + os.sep):
            return Path(manifest["path_prefix_map"][prefix]) / os.path.relpath(value, prefix)
    return candidate


def _copy_small(source, target, budget):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise ValueError("Required legacy input missing: " + str(source))
    size = source.stat().st_size
    if size > budget[0]:
        raise ValueError("Run-local copy budget exceeded; use explicit read path mapping")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    budget[0] -= size


def prepare_date(
    date,
    run_dir,
    project_root=None,
    files_root=None,
    deep=False,
    copy_budget_bytes=600 * 1024 * 1024,
):
    """Prepare local writable features and explicit external read mappings.

    This works on exFAT: no symbolic/hard links and no copies of the large
    training corpora. Consumers must use ``resolve_read_path`` for legacy
    input paths, or read ``train_dir``/``test_dir``/``public_files`` directly.
    """
    config = load_registry(date, project_root=project_root, files_root=files_root)
    report = check_date(date, project_root=project_root, files_root=files_root, deep=deep)
    if not report["raw_ready"]:
        raise ValueError("Required exact-date inputs are missing: " + "; ".join(report["errors"]))
    run = Path(run_dir).resolve()
    root = run / "data/files"
    source_root = Path(config["files_root"]).resolve()
    if root == source_root or source_root in root.parents:
        raise ValueError("Derived output must not be written inside the external data root")
    source = source_root / "data/aspath/path_by_vantage" / config["routing_date"]
    source_paths = source / "path_without_prefix"
    corpus_inputs = [source_paths / "train/merged_paths.txt"]
    if config["mode"] == "bgp":
        corpus_inputs.extend(
            source_paths / "test" / (str(asn) + ".csv") for asn in config["test_sources"]
        )
    for path in corpus_inputs:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("Required processed routing input missing or empty: " + str(path))
    corpus_hashes = {str(path): sha256_file(path) for path in corpus_inputs}
    manifest_path = run / "data/data_manifest.json"
    selected = {asset["id"]: select_asset(asset) for asset in config["raw_inputs"]}
    public = {
        "asrank_nodes": selected["asrank_asns"],
        "asrank_links": selected["asrank_asnLinks"],
        "asrel": selected["relationships"],
        "peeringdb": selected["peeringdb"],
    }
    input_hashes = {key: sha256_file(path) for key, path in public.items()}
    registry_hash = sha256_file(config["registry_path"])
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("date") != config["date"]
            or existing.get("public_input_sha256") != input_hashes
        ):
            raise ValueError("Prepared input dates or hashes changed; use a fresh run directory")
        if existing.get("registry_sha256") != registry_hash:
            raise ValueError("Date registry changed; use a fresh run directory")
        if existing.get("corpus_input_sha256") != corpus_hashes:
            raise ValueError("Processed corpus inputs changed; use a fresh run directory")
        holdout = existing.get("source_holdout")
        if holdout and sha256_file(holdout["path"]) != holdout["sha256"]:
            raise ValueError("Dated BGP source holdout manifest changed; use a fresh run directory")
        for item in existing.get("graph_inputs", []):
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError("Legacy graph/node2vec input changed; use a fresh run directory")
        for name, digest in existing.get("generated_sha256", {}).items():
            if not Path(name).is_file() or sha256_file(name) != digest:
                raise ValueError("Prepared derived output changed: " + name)
        return existing
    date = config["routing_date"]
    if not source.is_dir():
        raise ValueError("No legacy date data to prepare: " + str(source))
    target = root / "data/aspath/path_by_vantage" / date
    graph, source_graph = target / "RL_Graph", source / "RL_Graph"
    graph.mkdir(parents=True, exist_ok=True)
    expected = sorted(config["test_sources"])
    observed = sorted(path.stem for path in (source_paths / "test").glob("*.csv"))
    source_holdout = None
    if config["mode"] == "traceroute":
        holdout_path = source_paths / "source_holdout.json"
        if not holdout_path.is_file():
            raise ValueError(
                "Traceroute preparation requires the dated BGP source holdout manifest"
            )
        holdout = json.loads(holdout_path.read_text())
        if (
            not set(expected) <= set(map(str, holdout.get("requested_sources", [])))
            or holdout.get("training_source_paths") != 0
        ):
            raise ValueError(
                "Traceroute source holdout is incomplete or retained original training paths"
            )
        for source_asn in expected:
            original = source_paths / "train" / (source_asn + ".csv")
            if original.is_file():
                with original.open() as handle:
                    if any(line.strip() for line in handle):
                        raise ValueError(
                            "Traceroute source remains in its original training CSV: " + source_asn
                        )
        source_holdout = dict(
            path=str(holdout_path),
            sha256=sha256_file(holdout_path),
            requested_sources=expected,
            original_training_source_paths=0,
            reference_kind="independent_traceroute_measurements",
        )
    elif observed != expected:
        raise ValueError(
            "Legacy test sources do not match frozen registry: %s != %s" % (observed, expected)
        )
    # Filename overlap can result from forward derivation from other original
    # BGP sources. It is recorded, not retrospectively relabelled as leakage.
    named_training_files = [
        {
            "filename_asn": asn,
            "path": str(source_paths / "train" / (asn + ".csv")),
            "bytes": (source_paths / "train" / (asn + ".csv")).stat().st_size,
        }
        for asn in expected
        if (source_paths / "train" / (asn + ".csv")).exists()
        and (source_paths / "train" / (asn + ".csv")).stat().st_size > 0
    ]
    read_map, prefixes = {}, {
        str(target / "path_without_prefix"): str(source_paths),
        str(graph): str(source_graph),
    }
    for key, name in [
        ("asrank_nodes", "asns" + config["auxiliary_dates"]["asrank"] + ".jsonl"),
        ("asrank_links", config["auxiliary_dates"]["asrank"] + "links.jsonl"),
    ]:
        read_map[str(root / "data/ASRank" / name)] = str(public[key].resolve())
    read_map[str(root / "data/aspath" / (date + ".as-rel.txt"))] = str(public["asrel"].resolve())
    # Env keeps the external root for large BGP corpora. Bind its routing-date
    # relationship alias too, especially for trace 1015 -> auxiliary 1001.
    read_map[str(source_root / "data/aspath" / (date + ".as-rel.txt"))] = str(
        public["asrel"].resolve()
    )
    read_map[str(target / "AS_frequency.json")] = str(source / "AS_frequency.json")
    budget = [int(copy_budget_bytes)]
    rebuild_features = date in {"20260501", "20260601"}
    mutable_features = {
        "AS_Feature.json",
        "edge_feature.json",
        "ASEmbedding_collectedPath_256.txt",
    }
    # Small foundational inputs also work for legacy constructors that open
    # these files before the runtime installs the complete resolver mapping.
    for name in ["edge_frequency.json"]:
        _copy_small(source_graph / name, graph / name, budget)
    _copy_small(source / "AS_frequency.json", target / "AS_frequency.json", budget)
    if not rebuild_features:
        for name in mutable_features:
            _copy_small(source_graph / name, graph / name, budget)
    graph_inputs = [
        {"path": str(source_graph / name), "sha256": sha256_file(source_graph / name)}
        for name in ["edge_frequency.json", date + "_246_embeddings_collectedPath.emb"]
        if (source_graph / name).is_file()
    ]
    repairs = {}
    if config["mode"] == "bgp" and not all(
        (source_paths / "test" / name).is_file() and (source_paths / "test" / name).stat().st_size
        for name in AGGREGATES
    ):
        local_test = target / "path_without_prefix/test"
        local_test.mkdir(parents=True, exist_ok=True)
        # Rebuilding only needs a stream of the source CSVs. No corpus copy
        # and no link is required; the helper accepts an explicit source dir.
        repairs["test_aggregates"] = rebuild_test_aggregates(
            local_test, source_csv_dir=source_paths / "test"
        )
        for name in AGGREGATES:
            read_map[str(source_paths / "test" / name)] = str(local_test / name)
    if rebuild_features:
        repairs["as_features"] = rebuild_as_features(
            public["asrank_nodes"],
            source_graph / "AS_Feature.json",
            graph / "edge_frequency.json",
            graph,
            config["auxiliary_dates"]["asrank"],
        )
        repairs["as_embedding"] = rebuild_as_embedding(
            source_graph / (date + "_246_embeddings_collectedPath.emb"),
            graph / "AS_Feature.json",
            graph / "ASEmbedding_collectedPath_256.txt",
        )
    generated = [graph / name for name in mutable_features]
    generated += [target / "AS_frequency.json", graph / "edge_frequency.json"]
    generated += [
        target / "path_without_prefix/test" / name
        for name in AGGREGATES
        if (target / "path_without_prefix/test" / name).is_file()
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "date": date,
        "mode": config["mode"],
        "routing_date": date,
        "auxiliary_dates": config["auxiliary_dates"],
        "data_root": str(root),
        "source_data_root": str(source_root),
        "graph_dir": str(graph),
        "merged_train": str(source_paths / "train/merged_paths.txt"),
        "train_dir": str(source_paths / "train"),
        "test_dir": str(source_paths / "test"),
        "read_path_map": read_map,
        "path_prefix_map": prefixes,
        "public_files": {key: str(path.resolve()) for key, path in public.items()},
        "public_input_sha256": input_hashes,
        "corpus_input_sha256": corpus_hashes,
        "sources": config["test_sources"],
        "common9": config["common9"],
        "legacy_source_root": str(source_root),
        "source_mutation": False,
        "registry_path": config["registry_path"],
        "registry_sha256": registry_hash,
        "graph_inputs": graph_inputs,
        "node2vec_policy": "reuse_legacy_BGP_graph_and_walk_vectors; same-date ASRank does not train node2vec",
        "legacy_feature_reuse": not rebuild_features,
        "source_holdout": source_holdout,
        "nonempty_test_named_training_files": named_training_files,
        "training_source_boundary": "Filename/start-AS overlap does not establish original BGP source identity; source derivation must be verified separately.",
        "legacy_provenance_boundary": "Original legacy training provenance is unavailable; reuse is explicit and hashed, not retrospectively certified.",
        "copy_budget_bytes": copy_budget_bytes,
        "copied_bytes": copy_budget_bytes - budget[0],
        "repairs": repairs,
        "generated_sha256": {str(path): sha256_file(path) for path in generated if path.is_file()},
    }
    atomic_json(manifest_path, manifest)
    return manifest
