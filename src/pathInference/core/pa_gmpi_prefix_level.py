"""Prefix-level adapter for the PA-GMPI downstream path inference stage.

The personalized generator still produces AS-level candidate paths once per
source/destination-AS pair.  This module binds each pair to a destination
prefix and builds prefix-specific observed positive paths (OPPs) from global
training routes.  Target-VAS routes are used only to define test queries and
ground truth; they are excluded from the OPP index.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FILES_ROOT = "/run/media/lxl/data/asinfer/files"


def _load_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as fp:
        value = json.load(fp)
    if isinstance(value, str):
        value = json.loads(value)
    return value


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(value, fp, separators=(",", ":"), sort_keys=True)
    os.replace(tmp_path, path)


def _date_dir(files_root, date):
    return os.path.join(files_root, "data", "aspath", "path_by_vantage", str(date))


def _artifact_dir_from_base(base_dir):
    return os.path.join(base_dir, "prefixes_pa_gmpi")


def artifact_dir_for_date(files_root, date):
    return _artifact_dir_from_base(os.path.join(_date_dir(files_root, date), "RL_Graph"))


def _read_path_lines(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            value = line.strip().split(",")[0].strip()
            if value and "->" in value:
                rows.append(value)
    return rows


def _select_prefix(prefixes):
    values = sorted({str(prefix) for prefix in prefixes if str(prefix)})
    return values[0] if values else ""


def _build_test_queries(date_dir, vas, desired_dests):
    split_path = os.path.join(date_dir, "path_without_prefix", "test", "%s.csv" % vas)
    raw_path = os.path.join(date_dir, "path", "%s.csv" % vas)
    test_sequences = set(_read_path_lines(split_path))
    desired_dests = set(str(dest) for dest in desired_dests)
    prefixes_by_dest = defaultdict(set)
    truth_by_dest_prefix = defaultdict(set)

    if test_sequences and os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                value = line.strip().split(",")[0].strip()
                if "->" not in value:
                    continue
                parts = value.split("->")
                if len(parts) < 3:
                    continue
                prefix = parts[-1]
                as_nodes = parts[:-1]
                sequence = "->".join(as_nodes)
                dest = as_nodes[-1]
                if sequence not in test_sequences or dest not in desired_dests:
                    continue
                prefixes_by_dest[dest].add(prefix)
                truth_by_dest_prefix[(dest, prefix)].add(sequence)

    query_prefixes = {}
    truth_paths = {}
    for dest in sorted(desired_dests):
        prefix = _select_prefix(prefixes_by_dest.get(dest, ()))
        if not prefix:
            continue
        query_prefixes[dest] = prefix
        paths = sorted(truth_by_dest_prefix.get((dest, prefix), ()))
        if paths:
            truth_paths["%s-%s" % (vas, dest)] = [path.split("->") for path in paths]
    return query_prefixes, truth_paths


def _build_train_queries(date_dir, base_dir, vas, as_prefix):
    augmented_path = os.path.join(base_dir, "support_data", "train", "%s_augmented_paths.csv" % vas)
    query_prefixes = {}
    first_hops = set()
    internal_counts = Counter()
    for sequence in _read_path_lines(augmented_path):
        nodes = sequence.split("->")
        if len(nodes) < 2 or nodes[0] != str(vas):
            continue
        first_hops.add(nodes[1])
        internal_counts.update(nodes[1:-1])
        dest = nodes[-1]
        prefix = _select_prefix(as_prefix.get(dest, ()))
        if prefix:
            query_prefixes[dest] = prefix
    # PA-GMPI uses both direct neighbors and high-frequency convergent
    # vertices.  For a zero-path VAS, a sparse first hop may be invisible in
    # other collectors while its recursive anchor is well observed.
    convergent_vertices = {asn for asn, _count in internal_counts.most_common(10)}
    return query_prefixes, first_hops | convergent_vertices


def _load_train_sequence_set(date_dir, source_vas):
    path = os.path.join(date_dir, "path_without_prefix", "train", "%s.csv" % source_vas)
    return set(_read_path_lines(path))


def _neighbor_pattern(neighbors):
    escaped = "|".join(re.escape(asn) for asn in sorted(neighbors))
    return r"(^|->)(%s)->" % escaped


def _iter_neighbor_routes(raw_dir, neighbors):
    if not neighbors:
        return
    if shutil.which("rg"):
        process = subprocess.Popen(
            [
                "rg",
                "-H",
                "-N",
                "--no-messages",
                "-g",
                "*.csv",
                "-e",
                _neighbor_pattern(neighbors),
                raw_dir,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr = ""
        try:
            for output in process.stdout:
                filename, separator, line = output.partition(":")
                if separator:
                    yield filename, line.strip()
        finally:
            process.stdout.close()
            process.wait()
            if process.stderr:
                stderr = process.stderr.read()
                process.stderr.close()
        if process.returncode not in (0, 1):
            raise RuntimeError("rg prefix scan failed: %s" % stderr.strip())
        return

    for raw_path in sorted(Path(raw_dir).glob("*.csv")):
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                value = line.strip()
                nodes = value.split("->")
                if any(node in neighbors for node in nodes[:-1]):
                    yield str(raw_path), value


def build_prefix_opp_index(
    date_dir, selected_prefixes, source_neighbors, excluded_vases, max_paths_per_key=200
):
    raw_dir = os.path.join(date_dir, "path")
    selected_prefixes = set(selected_prefixes)
    source_neighbors = set(source_neighbors)
    excluded_vases = set(str(vas) for vas in excluded_vases)
    index = defaultdict(list)
    seen = defaultdict(set)
    current_source = None
    train_sequences = set()

    for filename, value in _iter_neighbor_routes(raw_dir, source_neighbors):
        source_vas = os.path.basename(filename).split(".")[0]
        if source_vas in excluded_vases:
            continue
        if source_vas != current_source:
            current_source = source_vas
            train_sequences = _load_train_sequence_set(date_dir, source_vas)
        if not train_sequences or "->" not in value:
            continue
        parts = value.split("->")
        if len(parts) < 3:
            continue
        prefix = parts[-1]
        if prefix not in selected_prefixes:
            continue
        as_nodes = parts[:-1]
        sequence = "->".join(as_nodes)
        if sequence not in train_sequences:
            continue
        for idx, asn in enumerate(as_nodes[:-1]):
            if asn not in source_neighbors:
                continue
            key = "%s-%s" % (asn, prefix)
            suffix = "->".join(as_nodes[idx:])
            if suffix in seen[key] or len(index[key]) >= max_paths_per_key:
                continue
            seen[key].add(suffix)
            index[key].append(suffix)
    return dict(index)


def prepare_prefix_artifacts(
    date,
    target_vases,
    files_root=DEFAULT_FILES_ROOT,
    max_paths_per_key=200,
    rebuild=False,
    build_opps=True,
):
    date = str(date)
    target_vases = sorted({str(vas) for vas in target_vases if str(vas)})
    date_dir = _date_dir(files_root, date)
    base_dir = os.path.join(date_dir, "RL_Graph")
    artifact_dir = _artifact_dir_from_base(base_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    as_prefix = _load_json(os.path.join(date_dir, "as_prefix.json"))
    test_dests = _load_json(os.path.join(base_dir, "vantage_dst_test.json"))
    selected_prefixes = set()
    source_neighbors = set()

    for vas in target_vases:
        train_queries, first_hops = _build_train_queries(date_dir, base_dir, vas, as_prefix)
        test_queries, truth_paths = _build_test_queries(date_dir, vas, test_dests.get(vas, ()))
        _write_json(os.path.join(artifact_dir, "train_%s_dest_prefix.json" % vas), train_queries)
        _write_json(os.path.join(artifact_dir, "test_%s_dest_prefix.json" % vas), test_queries)
        _write_json(os.path.join(artifact_dir, "test_%s_truth_paths.json" % vas), truth_paths)
        _write_json(os.path.join(artifact_dir, "source_nodes_%s.json" % vas), sorted(first_hops))
        selected_prefixes.update(train_queries.values())
        selected_prefixes.update(test_queries.values())
        source_neighbors.update(first_hops)

    signature_data = json.dumps(
        {
            "date": date,
            "target_vases": target_vases,
            "selected_prefixes": sorted(selected_prefixes),
            "source_neighbors": sorted(source_neighbors),
            "max_paths_per_key": int(max_paths_per_key),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    signature = hashlib.sha256(signature_data).hexdigest()
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    old_metadata = _load_json(metadata_path)
    opp_path = os.path.join(artifact_dir, "train_prefix_opp.json")

    if build_opps and (
        rebuild or old_metadata.get("signature") != signature or not os.path.exists(opp_path)
    ):
        prefix_opp = build_prefix_opp_index(
            date_dir,
            selected_prefixes,
            source_neighbors,
            target_vases,
            max_paths_per_key=max_paths_per_key,
        )
        _write_json(opp_path, prefix_opp)

    metadata = {
        "date": date,
        "target_vases": target_vases,
        "selected_prefix_count": len(selected_prefixes),
        "source_node_count": len(source_neighbors),
        "signature": signature,
        "prefix_opp_built": bool(build_opps),
        "max_paths_per_key": int(max_paths_per_key),
    }
    _write_json(metadata_path, metadata)
    return metadata


def _runtime_artifact_dir(env):
    return _artifact_dir_from_base(env.baseDir.rstrip("/"))


def get_query_prefix(env, vas, dest, data_type):
    cache = getattr(env, "_prefix_query_cache", None)
    if cache is None:
        cache = {}
        env._prefix_query_cache = cache
    key = (str(data_type), str(vas))
    if key not in cache:
        path = os.path.join(_runtime_artifact_dir(env), "%s_%s_dest_prefix.json" % key)
        cache[key] = _load_json(path)
    return str(cache[key].get(str(dest), ""))


def get_test_truth_paths(env, vas):
    cache = getattr(env, "_prefix_truth_cache", None)
    if cache is None:
        cache = {}
        env._prefix_truth_cache = cache
    vas = str(vas)
    if vas not in cache:
        path = os.path.join(_runtime_artifact_dir(env), "test_%s_truth_paths.json" % vas)
        cache[vas] = _load_json(path)
    return cache[vas]


def get_prefix_source_nodes(env, vas):
    cache = getattr(env, "_prefix_source_node_cache", None)
    if cache is None:
        cache = {}
        env._prefix_source_node_cache = cache
    vas = str(vas)
    if vas not in cache:
        path = os.path.join(_runtime_artifact_dir(env), "source_nodes_%s.json" % vas)
        cache[vas] = [str(asn) for asn in _load_json(path, default=[])]
    return cache[vas]


def get_prefix_observed_paths(env, source_nodes, prefix, limit):
    cache = getattr(env, "_prefix_opp_cache", None)
    if cache is None:
        path = os.path.join(_runtime_artifact_dir(env), "train_prefix_opp.json")
        cache = _load_json(path)
        env._prefix_opp_cache = cache
    paths = []
    for source in source_nodes:
        for path in cache.get("%s-%s" % (source, prefix), ()):
            nodes = path.split("->")
            if len(nodes) >= 2:
                paths.append(nodes)
                if len(paths) >= limit:
                    return paths
    return paths


def _parse_target_vases(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--target-vases", default="")
    parser.add_argument("--meta-test-as-num", type=int, default=10)
    parser.add_argument("--files-root", default=DEFAULT_FILES_ROOT)
    parser.add_argument("--max-paths-per-key", type=int, default=200)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-prefix-opp", action="store_true")
    args = parser.parse_args()

    target_vases = _parse_target_vases(args.target_vases)
    if not target_vases:
        import commonFunction as common_fun

        target_vases = common_fun.select_meta_vases(args.date, args.meta_test_as_num)
    metadata = prepare_prefix_artifacts(
        args.date,
        target_vases,
        files_root=args.files_root,
        max_paths_per_key=args.max_paths_per_key,
        rebuild=args.rebuild,
        build_opps=not args.skip_prefix_opp,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
