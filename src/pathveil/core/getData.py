import commonFunction as commonFun
import os
import csv
from multiprocessing import Pool
import ujson
import pandas as pd
import ast
import copy
import time
import pa_gmpi_prefix_level as prefix_level

pathNum = 100000
_PATHGEN_WORKER_ENV = None


def _support_data_dir(env):
    return getattr(env, "support_data_dir", env.baseDir + "support_data/")


def _external_observed_path_file(env, vas):
    return os.path.join(_support_data_dir(env), "train", "%s_external_observed_paths.csv" % vas)


def _neighbor_competition_path_file(env, vas):
    return os.path.join(_support_data_dir(env), "train", "%s_neighbor_competition_paths.json" % vas)


def _valid_model_path(env, path, expected_pair=None):
    """Return a fresh, encodable path, or None for an unusable cache entry.

    Never replace an unknown ASN with padding: that changes both the path and
    its endpoints.  Validate before applying candidate/history count limits so
    corrupt early cache entries cannot displace valid later ones.
    """
    if isinstance(path, str):
        path = path.split("->")
    if not isinstance(path, (list, tuple)):
        return None
    nodes = [str(node).strip() for node in path]
    if not 2 <= len(nodes) <= getattr(env, "maxLen", 20):
        return None
    if any((not node or not node.isdigit() or int(node) <= 0 for node in nodes)):
        return None
    if len(set(nodes)) != len(nodes):
        return None
    if expected_pair is not None and (nodes[0], nodes[-1]) != tuple(
        (str(node) for node in expected_pair)
    ):
        return None
    stoi = getattr(env, "stoi", None)
    if stoi is not None and any((node not in stoi or stoi[node] == 0 for node in nodes)):
        return None
    return nodes


def _filter_model_paths(env, paths, expected_pair=None, limit=None):
    """Stable deduplication and validation without modifying the source cache."""
    (kept, seen) = ([], set())
    if limit is not None and limit <= 0:
        return kept
    for path in paths:
        nodes = _valid_model_path(env, path, expected_pair)
        if nodes is None or tuple(nodes) in seen:
            continue
        kept.append(nodes)
        seen.add(tuple(nodes))
        if limit is not None and len(kept) >= limit:
            break
    return kept


def _external_observation_is_sufficient(env, vas):
    cache = getattr(env, "_external_observation_sufficiency_cache", None)
    if cache is None:
        cache = {}
        run_dir = os.path.dirname(_support_data_dir(env).rstrip("/"))
        for profile_path in (
            os.path.join(run_dir, "neighbor_selection", "vas_observation_profiles.csv"),
            os.path.join(run_dir, "vas_observation_profiles.csv"),
        ):
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


def load_train_extra_truth_paths(env, vas):
    """Return train-extra truth paths from pure augmentation plus observations.

    The two sources remain separate on disk.  This union is only the
    in-memory truth set used to emit the unchanged GlobalPath train-extra
    schema.  Paths are deduplicated per VAS--destination so repeated raw
    observations cannot overweight an otherwise identical label.
    """
    vas = str(vas)
    mode = getattr(env, "train_extra_data_mode", "augmentation_only")
    cache = getattr(env, "_train_extra_truth_path_cache", None)
    if cache is None:
        cache = {}
        env._train_extra_truth_path_cache = cache
    cache_key = (vas, mode)
    if cache_key in cache:
        return cache[cache_key]
    if mode == "neighbor_competition":
        merged_file = _neighbor_competition_path_file(env, vas)
        merged = commonFun.load_json(merged_file) if read_exists(merged_file) else {}
        profiles = getattr(env, "train_extra_truth_path_profiles", None)
        if profiles is None:
            profiles = {}
            env.train_extra_truth_path_profiles = profiles
        profiles[vas] = {
            "vas": vas,
            "train_extra_data_mode": mode,
            "pure_augmented_paths": 0,
            "pure_augmented_destinations": 0,
            "pure_augmented_paths_removed_for_external_destination": 0,
            "pure_augmented_paths_retained": 0,
            "external_observed_paths": 0,
            "external_observed_destinations": 0,
            "external_added_unique_paths": 0,
            "merged_truth_paths": sum((len(paths) for paths in merged.values())),
            "merged_destinations": len(merged),
        }
        cache[cache_key] = merged
        return merged
    train_dir = os.path.join(_support_data_dir(env), "train")
    pure_file = os.path.join(train_dir, "%s_augmented_paths.json" % vas)
    pure_data = commonFun.load_json(pure_file) if read_exists(pure_file) else {}
    pure_by_pair = {}
    pure_seen = {}
    pure_path_count = 0
    for key, paths in pure_data.items():
        for path in paths:
            nodes = [str(node) for node in path if str(node)]
            if len(nodes) < 2 or nodes[0] != vas:
                continue
            pair = "%s-%s" % (vas, nodes[-1])
            signature = tuple(nodes)
            pure_by_pair.setdefault(pair, [])
            pure_seen.setdefault(pair, set())
            if signature not in pure_seen[pair]:
                pure_by_pair[pair].append(nodes)
                pure_seen[pair].add(signature)
                pure_path_count += 1
    external_path_count = 0
    external_by_pair = {}
    external_seen = {}
    external_destinations = set()
    if mode == "augmentation_plus_external_observation":
        external_file = _external_observed_path_file(env, vas)
        if read_exists(external_file):
            with open(external_file, encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    nodes = [
                        node.strip()
                        for node in line.strip().split(",", 1)[0].split("->")
                        if node.strip()
                    ]
                    if len(nodes) < 2 or nodes[0] != vas:
                        continue
                    external_path_count += 1
                    pair = "%s-%s" % (vas, nodes[-1])
                    external_destinations.add(nodes[-1])
                    signature = tuple(nodes)
                    external_by_pair.setdefault(pair, [])
                    external_seen.setdefault(pair, set())
                    if signature not in external_seen[pair]:
                        external_by_pair[pair].append(nodes)
                        external_seen[pair].add(signature)
    merged = {}
    seen = {}
    pure_removed_path_count = 0

    def add_paths(pair, paths):
        merged.setdefault(pair, [])
        seen.setdefault(pair, set())
        added = 0
        for nodes in paths:
            signature = tuple(nodes)
            if signature not in seen[pair]:
                merged[pair].append(nodes)
                seen[pair].add(signature)
                added += 1
        return added

    for pair, paths in pure_by_pair.items():
        add_paths(pair, paths)
    external_added_path_count = 0
    for pair, paths in external_by_pair.items():
        external_added_path_count += add_paths(pair, paths)
    profiles = getattr(env, "train_extra_truth_path_profiles", None)
    if profiles is None:
        profiles = {}
        env.train_extra_truth_path_profiles = profiles
    profiles[vas] = {
        "vas": vas,
        "train_extra_data_mode": mode,
        "pure_augmented_paths": pure_path_count,
        "pure_augmented_destinations": len(pure_data),
        "pure_augmented_paths_removed_for_external_destination": pure_removed_path_count,
        "pure_augmented_paths_retained": pure_path_count - pure_removed_path_count,
        "external_observed_paths": external_path_count,
        "external_observed_destinations": len(external_destinations),
        "external_added_unique_paths": external_added_path_count,
        "merged_truth_paths": sum((len(paths) for paths in merged.values())),
        "merged_destinations": len(merged),
    }
    cache[cache_key] = merged
    return merged


def write_train_extra_truth_path_profiles(env):
    profiles = list((getattr(env, "train_extra_truth_path_profiles", {}) or {}).values())
    if not profiles:
        return ""
    output = os.path.join(_support_data_dir(env), "pathgen_train_extra_input_profiles.csv")
    with open(output, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(sorted(profiles, key=lambda row: row["vas"]))
    return output


def _init_pathgen_worker(env):
    """Install the fork-inherited read-only environment once per worker."""
    global _PATHGEN_WORKER_ENV
    _PATHGEN_WORKER_ENV = env


def _run_pathgen_task(task):
    (vas, dests) = task
    if _PATHGEN_WORKER_ENV is None:
        raise RuntimeError("path-generation worker environment is not initialized")
    return pathGeneration_PerformanceAnalysis([vas, dests, _PATHGEN_WORKER_ENV])


def multi_process(tasks, env, num_thread):
    p = Pool(num_thread, initializer=_init_pathgen_worker, initargs=(env,))
    result = p.map(_run_pathgen_task, tasks)
    p.close()
    p.join()
    return result


def multi_main_sparse_intensive(env, vas_dst={}):
    analyze_data_dir = env.analyze_data_dir + "%s/" % env.dataType
    if not read_exists(analyze_data_dir):
        os.mkdir(analyze_data_dir)
    analyze_data_dir = analyze_data_dir + "%s/" % env.dataAttr
    if not read_exists(analyze_data_dir):
        os.mkdir(analyze_data_dir)
    generation_data_type = getattr(env, "generation_data_type", env.dataType)
    kpathDir = env.curDate_baseDir + "%s_kpath-k=%d-border=%d-%s" % (
        env.pathGenAlgorithm,
        env.k,
        env.border,
        generation_data_type,
    )
    refresh_kpath_cache = bool(getattr(env, "refresh_kpath_cache", False)) or not bool(
        getattr(env, "isGetData", False)
    )
    if not read_exists(kpathDir):
        os.mkdir(kpathDir)
    kpathDir = kpathDir + "/%s" % env.dataAttr
    if not read_exists(kpathDir):
        os.mkdir(kpathDir)
    elif refresh_kpath_cache:
        commonFun.del_file(kpathDir)
    env.refresh_kpath_cache = False
    env.curkpathDir = kpathDir
    paras = []
    for vas in vas_dst.keys():
        dests = list(dict.fromkeys(vas_dst[vas]))
        max_dests = int(getattr(env, "max_dests", 0) or 0)
        if max_dests > 0:
            dests = dests[:max_dests]
        paras.append((vas, dests))
    if env.num_thread == 1:
        for para in paras:
            pathGeneration_PerformanceAnalysis([para[0], para[1], env])
    else:
        multi_process(paras, env, env.num_thread)


def expand_path(aslist):
    l = len(aslist)
    expand_paths = []
    if l < 2:
        return expand_paths
    for i in range(1, l):
        path = aslist[: i + 1]
        expand_paths.append("->".join(path))
    return expand_paths


def get_edge_from_path(path_list):
    edges = []
    for i in range(len(path_list) - 1):
        edges.append((path_list[i], path_list[i + 1]))
    return edges


def get_vas_paths(env, vases, vas):
    paths = []
    edges = []
    path_without_dir = (
        env.root_dir + "data/aspath/path_by_vantage/" + env.parent + "/path_without_prefix/"
    )
    for vas in vases:
        file = path_without_dir + vas + ".csv"
        df = pd.read_csv(file)
        for row in df.values.tolist():
            path = row[0].split("->")
            paths.append(path)
            edges.extend(get_edge_from_path(path))
    augmented_data_file = _support_data_dir(env) + "train/%s_augmented_paths.csv" % vas
    df = pd.read_csv(augmented_data_file)
    for row in df.values.tolist():
        path = row[0].split("->")
        paths.append(path)
        edges.extend(get_edge_from_path(path))
    edges_freq = {}
    for edge in edges:
        if edge in edges_freq:
            edges_freq[edge] += 1
        else:
            edges_freq[edge] = 1
    expanded_paths = []
    for path in paths:
        expanded_paths.extend(expand_path(path))
    expanded_paths = list(set(expanded_paths))
    expanded_paths = [path.split("->") for path in expanded_paths]
    expanded_paths_dict = {}
    for path in expanded_paths:
        src = path[0]
        dst = path[-1]
        ap = "%s-%s" % (src, dst)
        if ap in expanded_paths_dict:
            expanded_paths_dict[ap].append("->".join(path))
        else:
            expanded_paths_dict[ap] = ["->".join(path)]
    return (expanded_paths, expanded_paths_dict, edges_freq)


" \npath generation and performance analysis\nisGetData = True: generated and observed data both are represented by index and saved as pt file\n"


def pathGeneration_PerformanceAnalysis(paras):
    vas = paras[0]
    dests = list(paras[1])
    print("numDests:%d" % len(dests))
    env = paras[2]
    generation_data_type = getattr(env, "generation_data_type", env.dataType)
    is_train_extra = generation_data_type == "train_extra_data"
    if is_train_extra:
        if env.dataType != "train":
            raise ValueError('train-extra labels require dataType="train"')
        exist_path = load_train_extra_truth_paths(env, vas)
    elif getattr(env, "prefix_level", False) and env.dataType == "test":
        exist_path = prefix_level.get_test_truth_paths(env, vas)
    else:
        exist_path = env.get_exist_path_by_ASN(vas, env.dataType)
    copied_dests = dests.copy()
    for dest in copied_dests:
        key_pair = vas + "-" + dest
        if key_pair not in exist_path:
            dests.remove(dest)
    analyze_data_dir = env.analyze_data_dir + "%s/" % env.dataType
    analyze_data_dir = analyze_data_dir + "%s/" % env.dataAttr
    if not read_exists(analyze_data_dir):
        os.mkdir(analyze_data_dir)
    if is_train_extra:
        generated_data_path = analyze_data_dir + "%s_train_extra_data_generated_data.csv" % vas
    else:
        generated_data_path = analyze_data_dir + "%s_generated_data.csv" % vas
    if read_exists(generated_data_path) and env.isGetData:
        os.remove(generated_data_path)
    time1 = time.time()
    all_similariy = []
    two_hops_similarity = []
    available_APs = 0
    haspath = 0
    if env.pathGenAlgorithm in ("HFVG", "PG"):
        import HighFrequencyVertexBasedPathGen as topk

        try:
            env.length_border = env.better_borderLen_dict[vas]
            print("vas:%s,length_border:%d" % (vas, env.length_border))
        except:
            pass
        avglen = topk.get_local_path_profile(env, vas)["avg_length"]
        env.asn_collected_path_avglen[vas] = avglen
        topk.configure_pg_strategy(env, avglen)
        k = env.k
        top_n = env.top_n
        neighborNum = env.neighbor_num_1
        kpath_file = (
            env.curkpathDir
            + "/%s_%s_kpaths_k=%d_k2=%d_top_n=%d_top_n2=%d_neighborNum=%d_neighborNum2=%d.json"
            % (
                vas,
                env.dataType,
                k,
                env.k2,
                top_n,
                env.top_n2,
                env.neighbor_num_1,
                env.neighbor_num_2,
            )
        )
        if not read_exists(kpath_file):
            kpath = topk.generate_kpath(env, k, vas, dests)
        else:
            with open(kpath_file) as cache_file:
                kpath = ujson.load(cache_file)
    elif env.pathGenAlgorithm == "WG":
        raise ValueError("The formal method supports PG only")
    exact_same_index = []
    num = 0
    cnt = 0
    data = []
    neighborPathNum = 0
    total_length = 0
    result_dict = {}
    for dest in dests:
        index_in_kpaths = 0
        similarity = 0
        tmp = []
        hisPathLabel = []
        hisAPLabel = []
        kpathLable = []
        key_pair = vas + "-" + dest
        epaths = exist_path[key_pair]
        epath_sets = [set(ep) for ep in epaths]
        total_length += len(epaths[0])
        if env.G.has_edge(vas, dest):
            inferPath = [vas, dest]
            infer_set = set(inferPath)
            for ep_set in epath_sets:
                union = infer_set | ep_set
                s = len(infer_set & ep_set) / len(union) if union else 0
                if s > similarity:
                    similarity = s
            exact_same_index.append(index_in_kpaths)
            available_APs += 1
            neighborPathNum += 1
            two_hops_similarity.append(similarity)
        else:
            kpaths = _filter_model_paths(env, kpath.get(key_pair, []), (vas, dest), limit=k)
            if not kpaths:
                all_similariy.append(0)
                try:
                    if env.G.has_path(vas, dest):
                        haspath += 1
                except:
                    pass
                continue
            available_APs += 1
            for path in kpaths:
                isBreak = False
                curpath_similarity = 0.0
                path_set = set(path)
                for ep_set in epath_sets:
                    union = path_set | ep_set
                    s = len(path_set & ep_set) / len(union) if union else 0
                    if s > similarity:
                        similarity = s
                    if s > curpath_similarity:
                        curpath_similarity = s
                    if s >= 1:
                        if not isBreak:
                            exact_same_index.append(index_in_kpaths)
                        isBreak = True
                        break
                tmp.append(round(curpath_similarity, 3))
                if env.isGetData:
                    labeled_path = path_represent_by_idx(env, path)
                    kpathLable.append("->".join([str(p) for p in labeled_path]))
                index_in_kpaths += 1
        all_similariy.append(similarity)
        epath_len = len(epaths[0])
        try:
            tmp1 = result_dict[epath_len]
            tmp1["num"] += 1
            tmp1["similarity"] += similarity
            result_dict[epath_len] = tmp1
        except:
            num = 1
            tmp1 = {"num": num, "similarity": similarity}
            result_dict[epath_len] = tmp1
        if len(tmp) > 0 and env.isGetData:
            cnt += 1
            prefix = ""
            if getattr(env, "prefix_level", False):
                prefix = prefix_level.get_query_prefix(env, vas, dest, env.dataType)
            target = [vas, dest, prefix] if prefix else [vas, dest]
            (hisPaths, hisPairs) = getObservedPaths(env, target)
            for hpath in hisPaths:
                hisPathLabel.append("->".join([str(p) for p in path_represent_by_idx(env, hpath)]))
            for hpair in hisPairs:
                hisAPLabel.append("->".join([str(p) for p in AP_represent_by_idx(env, hpair)]))
            sim = [str(s) for s in tmp]
            data.append(
                [
                    "|".join(kpathLable),
                    "|".join(hisPathLabel),
                    "|".join(hisAPLabel),
                    "|".join(sim),
                    "->".join([vas, dest]),
                ]
            )
            if cnt % pathNum == 0 and cnt > 0:
                df = pd.DataFrame(data)
                df.to_csv(generated_data_path, header=False, index=False, mode="a")
                data = []
        num += 1
    if len(data) > 0 and env.isGetData:
        df = pd.DataFrame(data)
        df.to_csv(generated_data_path, header=False, index=False, mode="a")
        data = []
    time2 = time.time()
    same = 0
    record = 0
    for s in all_similariy:
        if s >= 1.0:
            same += 1
        record += s
    two_hops_same = 0
    two_hops_record = 0
    for s in two_hops_similarity:
        if s >= 1.0:
            two_hops_same += 1
        two_hops_record += s
    drop_duplicate = list(set(exact_same_index))
    drop_duplicate.sort()
    count_dict = {}
    for i in drop_duplicate:
        num = exact_same_index.count(i)
        count_dict[str(i)] = num
    if env.isGetData:
        total_dests = len(dests)
        coverage = available_APs / total_dests if total_dests else 0.0
        exact_ratio = same / total_dests if total_dests else 0.0
        avg_length = total_length / total_dests if total_dests else 0.0
        sorted_result_by_length = {}
        for length in sorted(result_dict):
            item = dict(result_dict[length])
            item["accuracy"] = round(item["similarity"] / item["num"], 4)
            item["similarity"] = round(item["similarity"], 4)
            sorted_result_by_length[length] = item
        res_dict = {
            "num_dests": total_dests,
            "total_available_APs": available_APs,
            "haspath": haspath,
            "coverage": coverage,
            "all_similarity": record / total_dests if total_dests else 0.0,
            "num_vantages": 1,
            "time_cost": time2 - time1,
            "isGetData": int(env.isGetData),
            "total_exact_same": same,
            "exact_ratio": exact_ratio,
            "two_hops": neighborPathNum,
            "two_hops_same": two_hops_same,
            "two_hops_similarity": two_hops_record,
            "avg_length": avg_length,
            "sorted_total_result_by_length": str(sorted_result_by_length),
        }
    else:
        dataLable = env.dataLabel + "-vas=%s" % vas
        result_file = env.raw_result_dir + "/%s_%d_%d_%d_%d.txt" % (
            dataLable,
            k,
            top_n,
            neighborNum,
            env.node2vecOutputDim,
        )
        if read_exists(result_file):
            os.remove(result_file)
        f = open(result_file, "a")
        line = (
            "vantage:%s" % vas
            + "\t"
            + "k-value:%d" % k
            + "\t"
            + "threshold_trigered:%d" % 0
            + "\t"
            + "num_dests:%d" % len(dests)
            + "\t"
            + "available_APs:%d" % available_APs
            + "\t"
            + "haspath:%d" % haspath
            + "\t"
            + "neighborPathNum:%d" % neighborPathNum
            + "\t"
            + "exact_same:%d" % same
            + "\t"
            + "totalSimilarity:%.4f" % record
            + "\t"
            + "jaccardSimilarity:%.4f" % (record / len(dests))
            + "\t"
            + "two_hops_same:%d" % two_hops_same
            + "\t"
            + "two_hops_similarity:%.4f" % two_hops_record
            + "\t"
            + "mean_length:%.2f" % (total_length / len(dests))
            + "\t"
            + "total_length:%d" % total_length
        )
        print(line)
        f.write(line + "\n")
        count_dict_str = "%s" % str(count_dict)
        f.write(count_dict_str + "\n")
        result_dict_str = "%s" % str(result_dict)
        f.write("result_by_length--" + result_dict_str + "\n")
        f.write("time_cost:%.4f\tisGetData:%d" % (time2 - time1, env.isGetData) + "\n")
        f.close()
        (low, resultTxt, res_dict) = analyze_raw_resultTxt(result_file)
        resultTxtFile = env.result_dir + "/%s-result.txt" % env.pathGenAlgorithm
        f = open(resultTxtFile, "a")
        f.write("%s\n" % dataLable)
        f.write(
            "k:%d,k2:%d,top_n:%d,top_n2:%d,neighborNum:%d,neighborNumCovergentVertex:%d\n"
            % (env.k, env.k2, top_n, env.top_n2, neighborNum, env.neighbor_num_2)
        )
        for txt in resultTxt:
            f.write(txt + "\n")
        f.close()
    save_dict = {"vantage": vas}
    save_dict.update(res_dict)
    save_dict["k"] = k
    save_dict["k2"] = env.k2
    save_dict["top_n"] = top_n
    save_dict["top_n2"] = env.top_n2
    save_dict["neighborNum"] = neighborNum
    save_dict["neighborNum2"] = env.neighbor_num_2
    result_csv = env.result_dir + "/%s-%s-result.csv" % (env.pathGenAlgorithm, env.dataType)
    df_result = pd.DataFrame.from_dict(save_dict, orient="index").T
    df_result.to_csv(result_csv, header=False, index=False, mode="a")


def path_represent_by_idx(env, aslist):
    nodes = [str(asn) for asn in aslist]
    unknown = [
        asn
        for asn in nodes
        if not asn.isdigit() or int(asn) <= 0 or asn not in env.stoi or (env.stoi[asn] == 0)
    ]
    if unknown:
        raise ValueError("ASNs cannot be encoded as real path nodes: %s" % ", ".join(unknown))
    return [env.stoi[asn] for asn in nodes]


def AP_represent_by_idx(env, AP):
    if len(AP) != 2:
        raise ValueError("AS pair must have exactly two endpoints")
    return path_represent_by_idx(env, AP)


def getObservedPaths(env, targetASPair):
    src = str(targetASPair[0])
    dst = str(targetASPair[1])
    prefix = targetASPair[2] if len(targetASPair) > 2 else ""
    try:
        srcNeighbors = list(env.G.neighbors(src))
    except Exception:
        srcNeighbors = []
    try:
        dstNeighbors = list(env.G.neighbors(dst))
    except Exception:
        dstNeighbors = []
    hisPaths = []
    hisPairs = []
    seen = set()
    limit = env.observed_path_num
    if limit <= 0:
        return (hisPaths, hisPairs)

    def add_history(path, expected_pair=None):
        nodes = _valid_model_path(env, path, expected_pair)
        if nodes is not None and tuple(nodes) not in seen:
            hisPaths.append(nodes)
            hisPairs.append([nodes[0], nodes[-1]])
            seen.add(tuple(nodes))
        return len(hisPaths) >= limit

    if prefix and getattr(env, "prefix_level", False):
        prefix_sources = list(
            dict.fromkeys(prefix_level.get_prefix_source_nodes(env, src) + srcNeighbors)
        )
        prefix_paths = prefix_level.get_prefix_observed_paths(
            env, prefix_sources, prefix, float("inf")
        )
        for path in prefix_paths:
            if add_history(path):
                return (hisPaths, hisPairs)

    def related_pairs():
        for node in dstNeighbors:
            yield (src, str(node))
        for node in srcNeighbors:
            yield (str(node), dst)
        for snode in srcNeighbors:
            for dnode in dstNeighbors:
                yield (str(snode), str(dnode))

    for pair in related_pairs():
        key = "%s-%s" % pair
        for path in env.existPath.get(key, ()):
            if add_history(path, pair):
                return (hisPaths, hisPairs)
    return (hisPaths, hisPairs)


def analyze_raw_resultTxt(file):
    fp = open(file, "r")
    lines = fp.readlines()
    total_dests = 0
    total_available_APs = 0
    total_haspath = 0
    total_exact_same = 0
    all_similarity = 0
    num_vantages = 0
    neighborPathNum = 0
    two_hops_same = 0
    two_hops_similarity = 0.0
    total_length = 0
    time_cost = 0
    isGetData = 0
    vases = []
    n = 0.8
    lowSimilarity = []
    total_result_by_length = {}
    for line in lines:
        if line.find("vantage") >= 0:
            num_vantages += 1
            keys = line.split("\t")
            line_dict = {}
            for k in keys:
                a = k.split(":")
                key = a[0]
                item = a[1]
                line_dict[key] = item
            jaccardSimilarity = float(line_dict["jaccardSimilarity"])
            if jaccardSimilarity < n:
                lowSimilarity.append(
                    [line_dict["vantage"], int(line_dict["num_dests"]), jaccardSimilarity]
                )
            total_dests += int(line_dict["num_dests"])
            total_available_APs += int(line_dict["available_APs"])
            total_exact_same += int(line_dict["exact_same"])
            all_similarity += float(line_dict["totalSimilarity"])
            vases.append(line_dict["vantage"])
            neighborPathNum += int(line_dict["neighborPathNum"])
            total_haspath += int(line_dict["haspath"])
            two_hops_same += int(line_dict["two_hops_same"])
            two_hops_similarity += float(line_dict["two_hops_similarity"])
            total_length += int(line_dict["total_length"])
        elif line.find("result_by_length") >= 0:
            result_str = line.split("--")[1]
            result_str = result_str[:-1]
            result_dict = ast.literal_eval(result_str)
            for key in result_dict.keys():
                item = result_dict[key]
                try:
                    t_item = total_result_by_length[key]
                    t_item["num"] += item["num"]
                    t_item["similarity"] += item["similarity"]
                    total_result_by_length[key] = t_item
                except:
                    total_result_by_length[key] = item
        elif line.find("time_cost") >= 0:
            line = line[:-1]
            rstr = line.split("\t")
            time_cost = float(rstr[0].split(":")[1])
            isGetData = int(rstr[1].split(":")[1])
    coverage = total_available_APs / total_dests
    accuracy = all_similarity / total_dests
    outText1 = "num_dests:%d, total_available_APs:%d, haspath:%d, coverage:%.4f" % (
        total_dests,
        total_available_APs,
        total_haspath,
        coverage,
    )
    outText2 = "all_similarity:%.4f" % accuracy
    outText3 = "num_vantages:%d, time_cost:%.4f, isGeteData:%d" % (
        num_vantages,
        time_cost,
        isGetData,
    )
    outText4 = "total_exact_same:%d, exact_ratio:%.4f" % (
        total_exact_same,
        total_exact_same / total_dests,
    )
    outText5 = "two_hops:%d, two_hops_same:%d, two_hops_similarity:%.4f" % (
        neighborPathNum,
        two_hops_same,
        two_hops_similarity,
    )
    outText6 = "avg_length:%.2f" % (total_length / total_dests)
    keys = list(total_result_by_length.keys())
    keys.sort()
    sorted_total_result_by_length = {}
    for key in keys:
        item = total_result_by_length[key]
        num = item["num"]
        sim = item["similarity"]
        avg_accuracy = round(sim / num, 4)
        item["accuracy"] = avg_accuracy
        item["similarity"] = round(sim, 4)
        sorted_total_result_by_length[key] = item
    outText7 = "%s" % str(sorted_total_result_by_length)
    ret_dict = {}
    ret_dict["num_dests"] = total_dests
    ret_dict["total_available_APs"] = total_available_APs
    ret_dict["haspath"] = total_haspath
    ret_dict["coverage"] = coverage
    ret_dict["all_similarity"] = accuracy
    ret_dict["num_vantages"] = num_vantages
    ret_dict["time_cost"] = time_cost
    ret_dict["isGetData"] = isGetData
    ret_dict["total_exact_same"] = total_exact_same
    ret_dict["exact_ratio"] = total_exact_same / total_dests
    ret_dict["two_hops"] = neighborPathNum
    ret_dict["two_hops_same"] = two_hops_same
    ret_dict["two_hops_similarity"] = two_hops_similarity
    ret_dict["avg_length"] = total_length / total_dests
    ret_dict["sorted_total_result_by_length"] = outText7
    return (
        lowSimilarity,
        [outText1, outText2, outText3, outText4, outText5, outText6, outText7],
        ret_dict,
    )


def get_intensive_vas_by_percentile(validate_intensive_vas_dests, percentiles):
    numdests_vas = []
    for vas in validate_intensive_vas_dests.keys():
        dests = validate_intensive_vas_dests[vas]
        numdests_vas.append([len(dests), vas])
    numdests_vas.sort()
    vases = []
    for p in percentiles:
        idx = int(p * (len(numdests_vas) - 1))
        vases.append(numdests_vas[idx][1])
    return vases


def get_meta_vas_dests(vases, measured_vantage_dst, unmeasure_vantage_dst):
    train_vas_dests = {}
    test_vas_dests = {}
    for vas in vases:
        try:
            dests = measured_vantage_dst[vas]
            train_vas_dests[vas] = dests
        except:
            pass
        try:
            dests = unmeasure_vantage_dst[vas]
            test_vas_dests[vas] = dests
        except:
            pass
    return (train_vas_dests, test_vas_dests)


def get_extra_training_data(env, vases):
    vas_dests = {}
    for vas in vases:
        extra_data = load_train_extra_truth_paths(env, vas)
        vas_dests[vas] = list(dict.fromkeys((key.split("-", 1)[1] for key in extra_data.keys())))
    return vas_dests


def _configure_workers(env):
    env.existPath = env.getExistPath("train")
    env.num_thread = 1
    num_thread_override = getattr(env, "num_thread_override", 0)
    if num_thread_override and num_thread_override > 0:
        env.num_thread = num_thread_override


def _ensure_result_csv(env, headers):
    result_csv = env.result_dir + "/%s-%s-result.csv" % (env.pathGenAlgorithm, env.dataType)
    if not read_exists(result_csv):
        df = pd.DataFrame([headers])
        df.to_csv(result_csv, index=False, mode="a", header=False)


def main(env, vases=[]):
    measured_vantage_dst = commonFun.get_measured_vantage_dst(env.baseDir)
    unmeasure_vas_dests = commonFun.get_unmeasured_vantage_dst(env.baseDir, "test")
    if not getattr(env, "result_dir_override", "") and (not read_exists("result")):
        os.mkdir("result")
    env.result_dir = getattr(env, "result_dir_override", "") or "result/" + env.curDate
    if not read_exists(env.result_dir):
        os.mkdir(env.result_dir)
    print("begin main")
    env.raw_result_dir = env.result_dir + "/" + env.pathGenAlgorithm
    if not read_exists(env.raw_result_dir):
        os.mkdir(env.raw_result_dir)
    env.analyze_data_dir = (
        getattr(env, "analyze_data_dir_override", "")
        or env.curDate_baseDir + "model_input_data_%s/" % env.pathGenAlgorithm
    )
    if not read_exists(env.analyze_data_dir):
        os.makedirs(env.analyze_data_dir)
    headers = [
        "vantage",
        "num_dests",
        "total_available_APs",
        "haspath",
        "coverage",
        "all_similarity",
        "num_vantages",
        "time_cost",
        "isGetData",
        "total_exact_same",
        "exact_ratio",
        "two_hops",
        "two_hops_same",
        "two_hops_similarity",
        "avg_length",
        "sorted_total_result_by_length",
    ]
    paras = ["k", "k2", "top_n", "top_n2", "neighborNum", "neighborNum2"]
    headers.extend(paras)
    print("meta vases:%d" % len(vases))
    print(vases)
    (train_vas_dest, test_vas_dest) = get_meta_vas_dests(
        vases, measured_vantage_dst, unmeasure_vas_dests
    )
    _configure_workers(env)
    env.generation_data_type = "standard"
    env.dataAttr = "meta-%s" % env.as_type
    for data_type, vas_dests in (("train", train_vas_dest), ("test", test_vas_dest)):
        if data_type == "test" and getattr(env, "reuse_test_data", False):
            print("reuse existing test generated data")
            continue
        env.dataType = data_type
        env.dataLabel = "%s_%s_%s" % (env.pathGenAlgorithm, env.dataType, env.dataAttr)
        _ensure_result_csv(env, headers)
        multi_main_sparse_intensive(env, vas_dests)
    env.existPath = {}


def main_extra_training(env, vases=None):
    vases = vases or []
    env.result_dir = (
        getattr(env, "meta_result_dir_override", "") or "result/" + env.curDate + "/meta"
    )
    os.makedirs(env.result_dir, exist_ok=True)
    env.raw_result_dir = env.result_dir + "/" + env.pathGenAlgorithm
    os.makedirs(env.raw_result_dir, exist_ok=True)
    env.analyze_data_dir = (
        getattr(env, "analyze_data_dir_override", "")
        or env.curDate_baseDir + "model_input_data_%s/" % env.pathGenAlgorithm
    )
    os.makedirs(env.analyze_data_dir, exist_ok=True)
    headers = [
        "vantage",
        "num_dests",
        "total_available_APs",
        "haspath",
        "coverage",
        "all_similarity",
        "num_vantages",
        "time_cost",
        "isGetData",
        "total_exact_same",
        "exact_ratio",
        "two_hops",
        "two_hops_same",
        "two_hops_similarity",
        "avg_length",
        "sorted_total_result_by_length",
        "k",
        "k2",
        "top_n",
        "top_n2",
        "neighborNum",
        "neighborNum2",
    ]
    _configure_workers(env)
    env.dataType = "train"
    env.generation_data_type = "train_extra_data"
    env.dataAttr = "meta-%s" % env.as_type
    env.dataLabel = "%s_%s_%s" % (env.pathGenAlgorithm, env.dataType, env.dataAttr)
    _ensure_result_csv(env, headers)
    extra_training_data = get_extra_training_data(env, vases)
    print("train-extra truth-input profiles:", getattr(env, "train_extra_truth_path_profiles", {}))
    print("train-extra truth-input profile:", write_train_extra_truth_path_profiles(env))
    multi_main_sparse_intensive(env, extra_training_data)
    env.existPath = {}


from pathveil.core.io import open_data as open, read_exists, PandasReadProxy

pd = PandasReadProxy(pd)
