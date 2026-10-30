import commonFunction as commonFun
import networkx as nx
import networkit as nk
import os
import csv
import shelve
import ujson
import torch
from multiprocessing import Pool
from functools import partial
import pandas as pd
from collections import Counter
import math

current_path = os.getcwd()  # 返回当前文件所在的目录
parent_path = os.path.dirname(current_path)  # 获得current_path所在的目录,即父级目录
files_path = os.path.join(
    os.environ.get("ASINFER_FILES_ROOT", "/run/media/lxl/data/asinfer/files"), ""
)


def _support_data_dir(env):
    return getattr(env, "support_data_dir", env.baseDir + "support_data/")


def _local_knowledge_path_file(env, vas):
    """Return the source-specific paths used by personalized generation."""
    augmented = os.path.join(_support_data_dir(env), "train", "%s_augmented_paths.csv" % vas)
    if read_exists(augmented):
        return augmented
    return os.path.join(
        files_path,
        "data",
        "aspath",
        "path_by_vantage",
        env.parent,
        "path_without_prefix",
        "train",
        "%s.csv" % vas,
    )


def _external_observation_path_file(env, vas):
    return os.path.join(_support_data_dir(env), "train", "%s_external_observed_paths.csv" % vas)


def _external_observation_is_sufficient(env, vas):
    """Use raw external paths only for the VAS class approved upstream."""
    cache = getattr(env, "_external_observation_sufficiency_cache", None)
    if cache is None:
        cache = {}
        run_dir = os.path.dirname(_support_data_dir(env).rstrip("/"))
        profile_paths = (
            os.path.join(run_dir, "neighbor_selection", "vas_observation_profiles.csv"),
            os.path.join(run_dir, "vas_observation_profiles.csv"),
        )
        for profile_path in profile_paths:
            if not read_exists(profile_path):
                continue
            with open(profile_path, newline="", encoding="utf-8") as fp:
                for row in csv.DictReader(fp):
                    cache[str(row.get("vas", ""))] = str(
                        row.get("external_observation_sufficient", "")
                    ).strip().lower() in {"1", "true", "yes"}
            break
        env._external_observation_sufficiency_cache = cache
    return bool(cache.get(str(vas), False))


def _read_source_paths(path_file, vas):
    paths = []
    if not read_exists(path_file):
        return paths
    with open(path_file, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            path = line.strip().split(",")[0].strip()
            if not path or "->" not in path:
                continue
            nodes = [node.strip() for node in path.split("->") if node.strip()]
            if len(nodes) >= 2 and nodes[0] == vas:
                paths.append(nodes)
    return paths


def load_local_knowledge_paths(env, vas):
    """Merge path_augmentation and expanded source paths without double-counting overlap."""
    vas = str(vas)
    cache = getattr(env, "_pg_local_path_cache", None)
    if cache is None:
        cache = {}
        env._pg_local_path_cache = cache
    knowledge_mode = getattr(env, "pg_local_knowledge_mode", "augmentation_only")
    cache_key = (vas, knowledge_mode)
    if cache_key in cache:
        return cache[cache_key]

    local_file = _local_knowledge_path_file(env, vas)
    path_augmentation_paths = _read_source_paths(local_file, vas)
    external_paths = []
    all_external_paths = []
    external_file = _external_observation_path_file(env, vas)
    if knowledge_mode == "augmentation_plus_external_observation":
        all_external_paths = _read_source_paths(external_file, vas)
        if _external_observation_is_sufficient(env, vas):
            external_paths = all_external_paths
    elif knowledge_mode != "augmentation_only" and hasattr(env, "iter_expand_paths"):
        for path in env.iter_expand_paths("train", source_filter={vas}):
            nodes = [str(node).strip() for node in path if str(node).strip()]
            if len(nodes) >= 2 and nodes[0] == vas:
                external_paths.append(nodes)

    # A repeated path within one source carries frequency. If the same path is
    # present in both sources, the larger count avoids treating file overlap as
    # independent observations.
    path_augmentation_counts = Counter(tuple(path) for path in path_augmentation_paths)
    external_counts = Counter(tuple(path) for path in external_paths)
    merged_counts = path_augmentation_counts | external_counts
    paths = [list(path) for path, count in merged_counts.items() for _ in range(count)]
    source_profile_cache = getattr(env, "_pg_local_source_profile_cache", None)
    if source_profile_cache is None:
        source_profile_cache = {}
        env._pg_local_source_profile_cache = source_profile_cache
    source_profile_cache[cache_key] = {
        "path_augmentation_paths": path_augmentation_paths,
        "path_augmentation_path_count": len(path_augmentation_paths),
        "expand_path_count": len(external_paths),
        "external_observation_path_file": external_file,
        "external_observation_path_count": len(all_external_paths or external_paths),
        "external_observation_enabled": bool(external_paths),
        "external_observation_used_path_count": len(external_paths),
        "overlap_path_count": sum(
            min(path_augmentation_counts[path], external_counts[path])
            for path in path_augmentation_counts.keys() & external_counts.keys()
        ),
    }
    cache[cache_key] = paths
    return paths


def get_local_path_profile(env, vas):
    """Summarize the source paths that define PG's local knowledge."""
    merged_paths = load_local_knowledge_paths(env, vas)
    knowledge_mode = getattr(env, "pg_local_knowledge_mode", "augmentation_only")
    cache_key = (str(vas), knowledge_mode)
    source_profile = getattr(env, "_pg_local_source_profile_cache", {}).get(cache_key, {})
    path_augmentation_paths = source_profile.get("path_augmentation_paths", [])
    if knowledge_mode == "augmentation_only":
        paths = path_augmentation_paths
        first_hop_paths = path_augmentation_paths
        dgi_paths = path_augmentation_paths
    elif knowledge_mode in {"merged_structure", "augmentation_plus_external_observation"}:
        first_hop_paths = path_augmentation_paths
        admitted_first_hops = {path[1] for path in path_augmentation_paths if len(path) >= 2}
        paths = [path for path in merged_paths if len(path) >= 2 and path[1] in admitted_first_hops]
        dgi_paths = (
            paths if knowledge_mode == "augmentation_plus_external_observation" else merged_paths
        )
    else:
        paths = merged_paths
        first_hop_paths = merged_paths
        dgi_paths = merged_paths

    first_hop_counts = Counter(path[1] for path in first_hop_paths if len(path) >= 2)
    avg_length = sum(len(path) for path in dgi_paths) / len(dgi_paths) if dgi_paths else 0.0
    return {
        "knowledge_mode": knowledge_mode,
        "path_file": _local_knowledge_path_file(env, str(vas)),
        "paths": paths,
        "path_count": len(paths),
        "dgi_path_count": len(dgi_paths),
        "path_augmentation_path_count": source_profile.get("path_augmentation_path_count", 0),
        "expand_path_count": source_profile.get("expand_path_count", 0),
        "external_observation_path_file": source_profile.get("external_observation_path_file", ""),
        "external_observation_path_count": source_profile.get("external_observation_path_count", 0),
        "overlap_path_count": source_profile.get("overlap_path_count", 0),
        "merged_path_count": len(merged_paths),
        "avg_length": avg_length,
        "first_hop_counts": first_hop_counts,
        "first_hops": [
            asn
            for asn, _count in sorted(
                first_hop_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }


def _sort_neighbors_by_local_frequency(neighbors, first_hop_counts):
    weighted = [(int(first_hop_counts.get(neighbor, 0)), neighbor) for neighbor in neighbors]
    total = sum(weight for weight, _neighbor in weighted)
    if total <= 0:
        if not weighted:
            return [], {}
        ratio = 1.0 / len(weighted)
        ranked = [[ratio, neighbor] for _weight, neighbor in sorted(weighted)]
        return ranked, {neighbor: ratio for _weight, neighbor in weighted}
    ranked = [[weight / total, neighbor] for weight, neighbor in weighted]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked, {neighbor: weight / total for weight, neighbor in weighted}


def networkit_sp(env, srcASN, dests):
    if srcASN not in env.nk_asn2id:
        return {}
    srcid = env.nk_asn2id[srcASN]
    Dijkstra = nk.distance.Dijkstra(env.G_nk, source=srcid)
    Dijkstra.run()
    npath = {}
    for dest in dests:
        if dest not in env.nk_asn2id:
            continue
        destid = env.nk_asn2id[dest]
        # NetworKit returns [] for both an unreachable node and the source.
        if destid == srcid:
            npath[dest] = [srcASN]
            continue
        nk_paths = Dijkstra.getPath(destid)
        if nk_paths:
            npath[dest] = [env.nk_id2asn[asn] for asn in nk_paths]
    return npath


def get_highFrequencyVertex_and_segment(env, file):
    top_n = env.top_n2  # number of high frequent vertex
    # read the file
    df = pd.read_csv(file, header=None)
    all_asn = []
    for row in df.values.tolist():
        path = row[0]
        pathList = path.split("->")
        for asn in pathList:
            all_asn.append(asn)

    asn_count = Counter(all_asn)
    sorted_asn_count = sorted(asn_count.items(), key=lambda x: x[1], reverse=True)
    top_asn = sorted_asn_count[1 : top_n + 1]  # remove the first one, which is the VAS
    highFreqVertex = [x[0] for x in top_asn]

    segments = []
    for row in df.values.tolist():
        path = row[0]
        pathList = path.split("->")
        for idx, asn in enumerate(pathList):
            if asn in highFreqVertex:
                segment = pathList[: idx + 1]
                segments.append("->".join(segment))

    segments = list(set(segments))
    segments_dict = {}
    for segment in segments:
        seg_list = segment.split("->")
        ap = seg_list[0] + "-" + seg_list[-1]
        try:
            tmp = segments_dict[ap]
            tmp.append(seg_list)
            segments_dict[ap] = tmp
        except:
            segments_dict[ap] = [seg_list]
    return highFreqVertex, segments_dict


# do not contain neighbors of vas
def get_highFrequencyVertex_and_segment2(env, file, high_frequency_neighbors):
    top_n = env.top_n2  # number of high frequent vertex
    # read the file
    df = pd.read_csv(file, header=None)
    all_asn = []
    for row in df.values.tolist():
        path = row[0]
        pathList = path.split("->")
        for asn in pathList:
            all_asn.append(asn)

    asn_count = Counter(all_asn)
    sorted_asn_count = sorted(asn_count.items(), key=lambda x: x[1], reverse=True)
    highFreqVertex = []
    vas = sorted_asn_count[0][0]
    for item in sorted_asn_count[1:]:  # remove the first one, which is the VAS
        asn = item[0]
        if asn not in high_frequency_neighbors:
            highFreqVertex.append(asn)
        if len(highFreqVertex) >= top_n:
            break
    # top_asn = sorted_asn_count[1:top_n+1]    # remove the first one, which is the VAS
    # highFreqVertex = [x[0] for x in top_asn]

    segments = []
    for row in df.values.tolist():
        path = row[0]
        pathList = path.split("->")
        for idx, asn in enumerate(pathList):
            if asn in highFreqVertex:
                segment = pathList[: idx + 1]
                segments.append("->".join(segment))

    segments = list(set(segments))
    segments_dict = {}
    for segment in segments:
        seg_list = segment.split("->")
        ap = seg_list[0] + "-" + seg_list[-1]
        try:
            tmp = segments_dict[ap]
            tmp.append(seg_list)
            segments_dict[ap] = tmp
        except:
            segments_dict[ap] = [seg_list]
    return highFreqVertex, segments_dict


def get_highFrequencyVertex_and_segment2_from_paths(env, paths, high_frequency_neighbors):
    """Identify PG convergent vertices directly from merged local paths."""
    all_asn = [asn for path in paths for asn in path]
    if not all_asn:
        return [], {}
    asn_count = Counter(all_asn)
    vas = paths[0][0]
    highFreqVertex = []
    for asn, _count in sorted(asn_count.items(), key=lambda item: (-item[1], item[0])):
        if asn == vas or asn in high_frequency_neighbors:
            continue
        highFreqVertex.append(asn)
        if len(highFreqVertex) >= env.top_n2:
            break

    segments_dict = {}
    seen_segments = set()
    high_frequency_set = set(highFreqVertex)
    for path in paths:
        for idx, asn in enumerate(path):
            if asn not in high_frequency_set:
                continue
            segment = tuple(path[: idx + 1])
            if segment in seen_segments:
                continue
            seen_segments.add(segment)
            key = segment[0] + "-" + segment[-1]
            segments_dict.setdefault(key, []).append(list(segment))
    return highFreqVertex, segments_dict


def generate_kpath(env, k, vas, dests):
    kpaths = generate_kpath_nosave(env, k, vas, dests)
    kpath_file = (
        env.curkpathDir
        + "/%s_%s_kpaths_k=%d_k2=%d_top_n=%d_top_n2=%d_neighborNum=%d_neighborNum2=%d.json"
        % (
            vas,
            env.dataType,
            k,
            env.k2,
            env.top_n,
            env.top_n2,
            env.neighbor_num_1,
            env.neighbor_num_2,
        )
    )
    ujson.dump(kpaths, open(kpath_file, "w"))
    return kpaths


def generate_kpath_nosave(env, k, vas, dests):
    added_external_edges = env.augment_graph_for_external_nodes(dests)
    if added_external_edges:
        print("external topology edges added", added_external_edges)
    profile = get_local_path_profile(env, vas)
    avgLen = profile["avg_length"]
    env.asn_collected_path_avglen[vas] = avgLen
    configure_pg_strategy(env, avgLen)

    if avgLen > env.length_border:
        kpaths = highFreqBasedPathGen_long(env, k, vas, dests)
    else:
        kpaths = highFreqBasedPathGen_short(env, k, vas, dests)
    return kpaths


def configure_pg_strategy(env, avg_length):
    """Apply the ToN PG parameters using the configured DGI threshold."""
    env.top_n = 1
    env.neighbor_num_2 = 100
    if avg_length > env.length_border:
        env.top_n2 = 10
        env.k2 = 5
    else:
        env.top_n2 = 3
        env.k2 = 2


def _valid_query_path(path, source, dest):
    """A candidate must describe this query and contain no repeated AS."""
    return bool(path) and path[0] == source and path[-1] == dest and len(path) == len(set(path))


def _merge_candidate_results(results, vas, dests, k=0):
    """Preserve candidate order while counting each valid path only once."""
    merged = {}
    for dest in dests:
        key = vas + "-" + dest
        seen = set()
        paths = []
        for result in results:
            for path in result.get(key, []):
                candidate = tuple(path)
                if candidate in seen or not _valid_query_path(path, vas, dest):
                    continue
                seen.add(candidate)
                paths.append(list(path))
                if k > 0 and len(paths) >= k:
                    break
            if k > 0 and len(paths) >= k:
                break
        if paths:
            merged[key] = paths
    return merged


def compute_left_k(results, vas, dests, k):
    curkpaths = _merge_candidate_results(results, vas, dests, k)
    if k == 0:
        # Zero means unlimited, rather than a destination already completed.
        return {dest: 0 for dest in dests}, not bool(dests)
    left_k_dict = {}
    for dest in dests:
        remaining = k - len(curkpaths.get(vas + "-" + dest, []))
        left_k_dict[dest] = remaining if remaining > 0 else -1
    return left_k_dict, all(value < 0 for value in left_k_dict.values())


def get_k_dict(dests, k):
    k_dict = {}
    for dest in dests:
        k_dict[dest] = k
    return k_dict


# 问题： 从high-frequency convergent vertex 往 high-frequency neighbors 跳的问题
def highFreqBasedPathGen_long(env, k, vas, dests):
    profile = get_local_path_profile(env, vas)
    vas_local_neighbors = profile["first_hops"]
    # The path_augmentation stage already audited first-hop admission. PG may reorder
    # these branches and discover deeper convergent vertices, but it must not
    # silently reintroduce unrelated topology neighbors.
    vas_global_neighbors = vas_local_neighbors.copy()
    top_n = env.top_n

    # 2. top_n global neighbors
    (
        edge_ratio_sorted_global_neighbors,
        global_neighbor_ratio_dict,
    ) = _sort_neighbors_by_local_frequency(vas_global_neighbors, profile["first_hop_counts"])
    highFreq_global_neighbors = [x[1] for x in edge_ratio_sorted_global_neighbors[:top_n]]

    # 3. high-frequency local neighbors
    # 3.1 top_n local neighbors
    (
        edge_ratio_sorted_local_neighbors,
        local_neighbor_ratio_dict,
    ) = _sort_neighbors_by_local_frequency(vas_local_neighbors, profile["first_hop_counts"])
    highFreq_local_neighbors = [x[1] for x in edge_ratio_sorted_local_neighbors[:top_n]]
    # 3.2 high-frequency local neighbors from high-frequency vertex
    # for asn in highFreq_ConvergentVertex.copy():
    #     if env.G.has_edge(vas,asn):
    #         highFreq_local_neighbors.append(asn)
    #         highFreq_ConvergentVertex.remove(asn)

    # 4. common local neighbors
    common_local_neighbors = []
    for asn in vas_local_neighbors:
        if asn not in highFreq_local_neighbors and asn not in highFreq_global_neighbors:
            common_local_neighbors.append(asn)

    # 4. other global neighbors except the above three kinds of neighbors
    other_global_neighbors = []

    high_frequency_neighbors = list(
        dict.fromkeys(highFreq_global_neighbors + highFreq_local_neighbors)
    )
    if len(high_frequency_neighbors) > 1:
        print("high_frequency_neighbors long", high_frequency_neighbors)

    # 1. high-frequency convergent vertex from collected paths
    try:
        highFreq_ConvergentVertex, segments_dict = get_highFrequencyVertex_and_segment2_from_paths(
            env, profile["paths"], high_frequency_neighbors
        )
    except:
        highFreq_ConvergentVertex = []
        segments_dict = {}

    results = []
    print("vas", vas)
    # high-frequency neighbors
    print("high-frequency neighbors")

    for neighbor in high_frequency_neighbors:
        ratio = global_neighbor_ratio_dict[neighbor]
        # 往上取整
        curk = math.ceil(ratio * k)
        cur_left_k_dict = get_k_dict(dests, curk)
        # used_k += curk
        root_path = [vas, neighbor]
        sp_dict = {}
        neighbors = list(env.G.neighbors(neighbor))
        # remove intersected asns of neighbors and vas_global_neighbors from neighbors, which may cause return back to vas_global_neighbors
        # neighbors = list(set(neighbors) - set(vas_global_neighbors))
        neighbor_num = env.neighbor_num_2
        sorted_nodes_by_rank = sort_neighbors_by_rank(neighbors, env)  # be careful
        res = get_all_k_paths_sort_by_rank(
            neighbor_num, root_path, dests, env, sorted_nodes_by_rank, sp_dict, cur_left_k_dict
        )
        results.append(res)
        # used_neighbors.append(neighbor)

    # high Frequency Convergent Vertex, top_n2, k2 should set a small value
    print("high Frequency Convergent Vertex")
    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    if not isFinished:
        k2 = env.k2  # k value allocated to each convergent vertex
        neighbor_num = env.neighbor_num_2
        cache_sp_dict = {}
        for asn in highFreq_ConvergentVertex:
            key = vas + "-" + asn
            root_paths = segments_dict[key]
            curk = k2
            # print('curk',curk)
            sp_dict = {}
            neighbors = list(env.G.neighbors(asn))
            # neighbors = list(set(neighbors) - set(vas_global_neighbors))
            sorted_neighbors = sort_neighbors_by_rank(neighbors, env)[:neighbor_num]
            if len(sorted_neighbors) < len(dests):
                for rank, neighbor in sorted_neighbors:
                    try:
                        sp = cache_sp_dict[neighbor]
                        sp_dict[neighbor] = sp
                    except:
                        # sp = nx.shortest_path(env.G,neighbor)
                        sp = networkit_sp(env, neighbor, dests)
                        sp_dict[neighbor] = sp
            else:
                for dest in dests:
                    try:
                        sp = cache_sp_dict[dest]
                        sp_dict[dest] = sp
                    except:
                        # sp = nx.shortest_path(env.G,dest)
                        sp = networkit_sp(env, dest, [item[1] for item in sorted_neighbors])
                        sp_dict[dest] = sp
            cache_sp_dict = {}
            # allocate k value to each root_path
            root_paths.sort(key=lambda x: len(x), reverse=True)  # longer path first
            if len(root_paths) > curk:
                root_paths = root_paths[:curk]
                curks = [1] * len(root_paths)
            else:
                curks = [int(curk / len(root_paths))] * len(root_paths)
                for i in range(curk % len(root_paths)):
                    curks[i] += 1
            for idx, root_path in enumerate(root_paths):
                print("root_path", root_path)
                cur_k_dict = get_k_dict(dests, curks[idx])
                res = get_all_k_paths_sort_by_rank(
                    neighbor_num, root_path, dests, env, sorted_neighbors, sp_dict, cur_k_dict
                )
                # used_k += curk
                if len(res) > 0:
                    results.append(res)
                else:
                    print("no path")
            cache_sp_dict = sp_dict.copy()

    # print('all left neighbors')
    # all_left_neighbors = list(set(other_global_neighbors + common_local_neighbors))
    # left_k_dict, isFinished = compute_left_k(results,vas,dests,k)
    # if not isFinished:
    #     sp_dict = {}
    #     root_path = [vas]
    #     neighbor_num = env.neighbor_num_1
    #     sorted_nodes_by_rank = sort_neighbors_by_rank(all_left_neighbors,env)
    #     res = get_all_k_paths_sort_by_rank(neighbor_num,root_path,dests,env,sorted_nodes_by_rank,sp_dict,left_k_dict)
    #     if len(res) > 0:
    #         results.append(res)
    #     else:
    #         print('no path')

    # common local neighbors
    print("common local neighbors")
    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    if not isFinished:
        sp_dict = {}
        root_path = [vas]
        neighbor_num = env.neighbor_num_1
        sorted_nodes_by_rank = sort_neighbors_by_rank(common_local_neighbors, env)
        res = get_all_k_paths_sort_by_rank(
            neighbor_num, root_path, dests, env, sorted_nodes_by_rank, cache_sp_dict, left_k_dict
        )
        if len(res) > 0:
            results.append(res)
        else:
            print("no path")

    print("other global neighbors")
    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    if not isFinished:
        sp_dict = {}
        root_path = [vas]
        neighbor_num = env.neighbor_num_1
        sorted_nodes_by_rank = sort_neighbors_by_rank(other_global_neighbors, env)
        res = get_all_k_paths_sort_by_rank(
            neighbor_num, root_path, dests, env, sorted_nodes_by_rank, cache_sp_dict, left_k_dict
        )
        if len(res) > 0:
            results.append(res)
        else:
            print("no path")

    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    return _merge_candidate_results(results, vas, dests, k)


def highFreqBasedPathGen_short(env, k, vas, dests):
    profile = get_local_path_profile(env, vas)
    vas_local_neighbors = profile["first_hops"]
    vas_global_neighbors = vas_local_neighbors.copy()
    top_n = env.top_n

    # 2. top_n global neighbors
    (
        edge_ratio_sorted_global_neighbors,
        global_neighbor_ratio_dict,
    ) = _sort_neighbors_by_local_frequency(vas_global_neighbors, profile["first_hop_counts"])
    highFreq_global_neighbors = [x[1] for x in edge_ratio_sorted_global_neighbors[:top_n]]

    # 3. high-frequency local neighbors
    # 3.1 top_n local neighbors
    (
        edge_ratio_sorted_local_neighbors,
        local_neighbor_ratio_dict,
    ) = _sort_neighbors_by_local_frequency(vas_local_neighbors, profile["first_hop_counts"])
    # highFreq_local_neighbors = [x[1] for x in edge_ratio_sorted_local_neighbors[:top_n]]
    highFreq_local_neighbors = []
    # 3.2 high-frequency local neighbors from high-frequency vertex
    # for asn in highFreq_ConvergentVertex.copy():
    #     if env.G.has_edge(vas,asn):
    #         highFreq_local_neighbors.append(asn)
    #         highFreq_ConvergentVertex.remove(asn)

    # 4. common local neighbors
    common_local_neighbors = []
    for asn in vas_local_neighbors:
        if asn not in highFreq_local_neighbors and asn not in highFreq_global_neighbors:
            common_local_neighbors.append(asn)

    # 4. other global neighbors except the above three kinds of neighbors
    other_global_neighbors = []

    high_frequency_neighbors = list(
        dict.fromkeys(highFreq_global_neighbors + highFreq_local_neighbors)
    )
    if len(high_frequency_neighbors) > 1:
        print("high_frequency_neighbors short", high_frequency_neighbors)

    # 1. high-frequency convergent vertex from collected paths
    try:
        highFreq_ConvergentVertex, segments_dict = get_highFrequencyVertex_and_segment2_from_paths(
            env, profile["paths"], high_frequency_neighbors
        )
    except:
        highFreq_ConvergentVertex = []
        segments_dict = {}

    results = []
    print("vas", vas)

    # used_k = 0
    # high-frequency neighbors
    print("high-frequency neighbors")

    for neighbor in high_frequency_neighbors:
        ratio = global_neighbor_ratio_dict[neighbor]
        # 往上取整
        curk = math.ceil(ratio * k)
        cur_left_k_dict = get_k_dict(dests, curk)
        # used_k += curk
        root_path = [vas, neighbor]
        sp_dict = {}
        neighbors = list(env.G.neighbors(neighbor))
        # remove intersected asns of neighbors and vas_global_neighbors from neighbors, which may cause return back to vas_global_neighbors
        # neighbors = list(set(neighbors) - set(vas_global_neighbors))
        neighbor_num = env.neighbor_num_1
        sorted_nodes_by_rank = sort_neighbors_by_rank(neighbors, env)  # be careful
        res = get_all_k_paths_sort_by_rank(
            neighbor_num, root_path, dests, env, sorted_nodes_by_rank, sp_dict, cur_left_k_dict
        )
        results.append(res)

    # common local neighbors
    print("common local neighbors")
    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    if not isFinished:
        sp_dict = {}
        root_path = [vas]
        neighbor_num = env.neighbor_num_1
        sorted_nodes_by_rank = sort_neighbors_by_rank(common_local_neighbors, env)
        res = get_all_k_paths_sort_by_rank(
            neighbor_num, root_path, dests, env, sorted_nodes_by_rank, sp_dict, left_k_dict
        )
        if len(res) > 0:
            results.append(res)
        else:
            print("no path")

    print("other global neighbors")
    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    if not isFinished:
        sp_dict = {}
        root_path = [vas]
        neighbor_num = env.neighbor_num_1
        sorted_nodes_by_rank = sort_neighbors_by_rank(other_global_neighbors, env)
        res = get_all_k_paths_sort_by_rank(
            neighbor_num, root_path, dests, env, sorted_nodes_by_rank, {}, left_k_dict
        )
        if len(res) > 0:
            results.append(res)
        else:
            print("no path")

    # high Frequency Convergent Vertex, top_n2, k2 should set a small value
    print("high Frequency Convergent Vertex")
    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    if not isFinished:
        k2 = env.k2  # k value allocated to each convergent vertex
        neighbor_num = env.neighbor_num_2
        cache_sp_dict = {}
        for asn in highFreq_ConvergentVertex:
            key = vas + "-" + asn
            root_paths = segments_dict[key]
            curk = k2
            # print('curk',curk)
            sp_dict = {}
            neighbors = list(env.G.neighbors(asn))
            # neighbors = list(set(neighbors) - set(vas_global_neighbors))
            sorted_neighbors = sort_neighbors_by_rank(neighbors, env)[:neighbor_num]
            if len(sorted_neighbors) < len(dests):
                for rank, neighbor in sorted_neighbors:
                    try:
                        sp = cache_sp_dict[neighbor]
                        sp_dict[neighbor] = sp
                    except:
                        # sp = nx.shortest_path(env.G,neighbor)
                        sp = networkit_sp(env, neighbor, dests)
                        sp_dict[neighbor] = sp
            else:
                for dest in dests:
                    try:
                        sp = cache_sp_dict[dest]
                        sp_dict[dest] = sp
                    except:
                        # sp = nx.shortest_path(env.G,dest)
                        sp = networkit_sp(env, dest, [item[1] for item in sorted_neighbors])
                        sp_dict[dest] = sp
            cache_sp_dict = {}
            # allocate k value to each root_path
            root_paths.sort(key=lambda x: len(x), reverse=True)  # longer path first
            if len(root_paths) > curk:
                root_paths = root_paths[:curk]
                curks = [1] * len(root_paths)
            else:
                curks = [int(curk / len(root_paths))] * len(root_paths)
                for i in range(curk % len(root_paths)):
                    curks[i] += 1
            for idx, root_path in enumerate(root_paths):
                print("root_path", root_path)
                cur_k_dict = get_k_dict(dests, curks[idx])
                res = get_all_k_paths_sort_by_rank(
                    neighbor_num, root_path, dests, env, sorted_neighbors, sp_dict, cur_k_dict
                )
                # used_k += curk
                if len(res) > 0:
                    results.append(res)
                else:
                    print("no path")
            cache_sp_dict = sp_dict.copy()
    cache_sp_dict = {}
    left_k_dict, isFinished = compute_left_k(results, vas, dests, k)
    print(
        "vas",
        vas,
        "unfinished_dests",
        sum(1 for remaining in left_k_dict.values() if remaining > 0),
    )
    return _merge_candidate_results(results, vas, dests, k)


def sort_neighbors_by_ratio(G, ASN, neighbors):
    weight_node = []
    total_weight = 0
    ratio_node_dict = {}
    for neighbor in neighbors:
        weight = G[ASN][neighbor]["weight"]
        total_weight += weight
        weight_node.append([weight, neighbor])
    ratio_node = []
    for weight, node in weight_node:
        ratio = weight / total_weight
        ratio_node.append([ratio, node])
        ratio_node_dict[node] = ratio
    ratio_node.sort(reverse=True)
    # sorted_node = [node for _, node in ratio_node]
    return ratio_node, ratio_node_dict


def sort_neighbors_by_rank(neighbors, env):
    neighbor_sortby_rank = []
    for neighbor in neighbors:
        try:
            rank = env.node_features[neighbor][0]
        except:
            rank = 100000
        neighbor_sortby_rank.append([rank, neighbor])
    # 按rank由小到大排序
    neighbor_sortby_rank.sort(reverse=False)
    return neighbor_sortby_rank


def get_path_forward(root_path, path, rel_dict):
    if not root_path or not path:
        return []
    newpath = list(root_path) + list(path)
    if len(newpath) != len(set(newpath)):
        return []
    # Keep the existing tail policy; this change only checks path integrity.
    return newpath if commonFun.valley_free(path, rel_dict) else []


def get_path_backward(root_path, path, rel_dict):
    # Cached shortest paths can be reused by several roots. Never reverse one
    # in place, or the next query observes a path in the opposite direction.
    return get_path_forward(root_path, list(reversed(path)), rel_dict)


# def get_kpath_for_node_pair_forward(G,root_path,rel_dict,rootAS,dests,rank_and_neighbor):
#     neighbor = rank_and_neighbor[1]
#     result = {}
#     try:
#         # npath = nx.single_source_shortest_path(G,neighbor)
#         npath = nx.shortest_path(G,neighbor)
#     except:
#         return result
#     for dest in dests:
#         key = rootAS + '-' + dest
#         if G.has_edge(rootAS,dest):
#             # kpaths[key] = [[rootAS,dest]]
#             continue
#         else:
#             try:
#                 path = npath[dest]
#             except:
#                 continue
#             ret = get_path_forward(root_path,path,rel_dict)
#             if len(ret) > 0:
#                 result[key] = [rank_and_neighbor[0],ret]
#     return result


def get_kpath_for_node_pair_backward(
    G, k, NeighborNum, root_path, rel_dict, rootAS, neighbor_sortby_rank, dest
):
    if G.has_edge(rootAS, dest):
        return {}
    try:
        npath = nx.shortest_path(G, dest)
    except (nx.NetworkXException, KeyError):
        return {}
    paths, seen = [], set()
    for item in neighbor_sortby_rank:
        path = npath.get(item[1])
        if not path:
            continue
        candidate = get_path_backward(root_path, path, rel_dict)
        candidate_key = tuple(candidate)
        if not _valid_query_path(candidate, rootAS, dest) or candidate_key in seen:
            continue
        seen.add(candidate_key)
        paths.append(candidate)
        if k > 0 and len(paths) >= k:
            break
    return {rootAS + "-" + dest: paths}


# 获得一个vantage as 的所有dest的 k条路径,k=0为不限制条数，neighborNum=0，使用所有邻居
# sortet_neighbors: neighbors of vas = root_path[-1]
def _cached_shortest_paths(env, sp_dict, source, dests):
    """Extend partial caches when another root asks for new destinations."""
    paths = sp_dict.setdefault(source, {})
    missing = [dest for dest in dests if dest not in paths]
    if missing:
        found = networkit_sp(env, source, missing)
        for dest in missing:
            # Retain negative cache entries so repeated roots do not rerun
            # Dijkstra for the same disconnected destination.
            paths[dest] = found.get(dest, [])
    return paths


def get_all_k_paths_sort_by_rank(
    NeighborNum, root_path, all_dests, env, sorted_neighbors, sp_dict, left_k_dict
):
    if not root_path:
        return {}
    G, rel_dict = env.G, env.rel_dict
    rootAS = root_path[0]
    if NeighborNum > 0:
        sorted_neighbors = sorted_neighbors[:NeighborNum]
    dests = list(all_dests)
    kpaths, seen = {}, {}

    def append_candidate(dest, path):
        if not _valid_query_path(path, rootAS, dest):
            return
        key = rootAS + "-" + dest
        candidate = tuple(path)
        seen_for_pair = seen.setdefault(key, set())
        if candidate not in seen_for_pair:
            seen_for_pair.add(candidate)
            kpaths.setdefault(key, []).append(path)

    def complete(dest):
        limit = left_k_dict[dest]
        return (
            limit < 0
            or G.has_edge(rootAS, dest)
            or (limit > 0 and len(kpaths.get(rootAS + "-" + dest, [])) >= limit)
        )

    if len(sorted_neighbors) < len(dests):
        for _rank, neighbor in sorted_neighbors:
            unfinished = [dest for dest in dests if not complete(dest)]
            if not unfinished:
                break
            npath = _cached_shortest_paths(env, sp_dict, neighbor, unfinished)
            for dest in unfinished:
                tail = npath.get(dest)
                if not tail:
                    continue
                candidate = get_path_forward(root_path, tail, rel_dict)
                append_candidate(dest, candidate)
    else:
        neighbors = [item[1] for item in sorted_neighbors]
        for dest in dests:
            if complete(dest):
                continue
            npath = _cached_shortest_paths(env, sp_dict, dest, neighbors)
            for neighbor in neighbors:
                tail = npath.get(neighbor)
                if not tail:
                    # An unreachable neighbor must not hide later reachable ones.
                    continue
                candidate = get_path_backward(root_path, tail, rel_dict)
                append_candidate(dest, candidate)
                if complete(dest):
                    break
    return kpaths


# Input paths are resolved only for reads; numerical operations are unchanged.
from pathveil.core.io import open_data as open, read_exists, PandasReadProxy

pd = PandasReadProxy(pd)
