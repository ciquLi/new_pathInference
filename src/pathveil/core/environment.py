import commonFunction as commonFunc
import os
import ujson
import networkx as nx
import pandas as pd
import pickle
import numpy as np
import scipy.sparse as sp
import networkit as nk
import re
import shutil
import subprocess
import csv
import json
from pathlib import Path


current_path = os.getcwd()  # 返回当前文件所在的目录
parent_path = os.path.dirname(current_path)  # 获得current_path所在的目录,即父级目录

# parent_path = '/pdlfs/home/yqhe/lxl'
# parent_path = '/datasets/aote'
files_path = os.path.join(
    os.environ.get("ASINFER_FILES_ROOT", "/run/media/lxl/data/asinfer/files"), ""
)


class Env:
    def __init__(self, parent, curDate):
        self.parent = parent
        self.root_dir = files_path
        self.baseDir = os.path.join(
            os.environ.get(
                "PATHVEIL_GRAPH_DIR",
                files_path + "data/aspath/path_by_vantage/" + parent + "/RL_Graph/",
            ),
            "",
        )
        self.curDate_baseDir = (
            self.baseDir
            if curDate == parent
            else files_path + "data/aspath/path_by_vantage/" + curDate + "/RL_Graph/"
        )
        self.train_vas_divergence_file = self.baseDir + "train_vas_firsthop_divergence.csv"
        self.support_data_dir = self.baseDir + "support_data/"
        self.result_file = ""
        self.raw_result_dir = ""
        self.curkpathDir = ""
        self.analyze_data_dir = ""
        # self.meta_split = 0.01
        self.cur_vas = ""
        # hyperparameters
        self.curDate = curDate
        self.node2vecOutputDim = 246
        self.asrankFeatureDim = 10
        self.hiddedNum = 512
        self.nodeEmbeddingGrad = True
        self.train_loader_params = {
            "batch_size": 16,
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
            "pin_memory": True,
        }
        self.test_loader_params = {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
            "pin_memory": True,
        }
        self.training_print_num = 100
        self.lr = 0.0001
        self.local_lr = 0.0001
        self.siamese_epoch = 50
        self.siamese_same_path_prob = 0.8  # the probability of selecting the same path
        self.beta = 0.3
        self.num_local_updates = 3
        self.num_inferred_paths = 1

        self.sampled_support_num = 150
        self.train_similar_support_num = 1  # number of support ap for each query ap
        self.test_support_num = 3
        self.meta_split_num = 500
        self.as_type = "intensive"
        self.isGetData = False
        self.prefix_level = False
        self.max_dests = 0
        self.G = self.build_graph()  # 根据自己生成的图和边的频率来构建图
        self.G_nk, self.nk_id2asn, self.nk_asn2id = self.build_networkit_graph()
        self.itos, self.stoi = self.generate_asn_idx(self.G)
        # self.vas_cneighbors = self.get_vas_cneighbors()
        self.node_features = (
            self.get_node_features()
        )  # AS的特征，一共10个特征返回为一个list:[rank, total_degree, transit_degree, provider, peer, customer,isoCode,orgCode,lon,lat]
        self.rel_dict = self.relationship_dict()
        # self.better_borderLen_dict = self.get_better_borderLen_dict()
        self.asn_collected_path_avglen = {}
        self.existPath = {}
        self.border = 7000
        self.length_border = 4
        self.pg_local_knowledge_mode = "augmentation_only"
        self.graph_selected_neighbors_csv = ""
        self.graph_recursive_anchors_csv = ""
        self._external_topology_checked = set()
        self.pathGenAlgorithm = ""
        self.walksType = "collectedPath"  # or '_randomWalk'
        self.dataType = ""  # train or test
        self.dataLabel = ""
        self.dataAttr = ""
        self.maxLen = 20

        self.observed_path_num = 200  # number of observed paths
        self.k = 200
        self.k2 = 3  # k of the convergent vertex
        self.neighbor_num_1 = 0  # neighbor num of the vas
        self.neighbor_num_2 = 100  # neighbor num of the convergent vertex
        self.top_n = 1  # top_n most frequent neighbors of the vas
        self.top_n2 = 10  # number of high frequent vertex
        self.num_thread = 6

    def get_train_vas_divergence(self):
        """Load the date-level train first-hop divergence profiles, if built."""
        if not read_exists(self.train_vas_divergence_file):
            return {}
        with open(self.train_vas_divergence_file, newline="", encoding="utf-8") as fp:
            return {str(row["vas"]): row for row in csv.DictReader(fp)}

    # 通过aspath数据构建的图

    def build_graph(self):
        edge_path = self.baseDir + "edge_frequency.json"
        fp = open(edge_path)
        str_js = ujson.load(fp)
        edge_dict = ujson.loads(str_js)
        fp.close()

        G = nx.Graph()
        # DiGraph = nx.DiGraph()
        edges = []
        for key in edge_dict.keys():
            edge = key.split("-")
            u = edge[0]
            v = edge[1]
            w = edge_dict[key]
            edges.append([u, v, w])
            # G.add_edge(u, v, weight=w, feature=self.edge_features[key])
            # 权重为u-v + v-u
            try:
                weight = G[u][v]["weight"]
                w += weight
            except:
                pass
            G.add_edge(u, v, weight=w)
        degrees = list(G.degree())
        deg = 0
        for degree in degrees:
            d = degree[1]
            deg += d
        avgDegree = deg / len(degrees)
        print(
            "baseDate:%s,curDate:%s,numNodes:%d,numEdges:%d,avgDegree%.2f"
            % (self.parent, self.curDate, len(list(G.nodes)), len(list(G.edges)), avgDegree)
        )
        # df = pd.DataFrame(edges)
        # df.to_csv('%s-ASGraph.csv'%(self.parent),index=False,header=None)
        return G

    def build_networkit_graph(self):
        # 创建一个图
        graph = self.G
        G_nk = nk.Graph()
        nk_id2asn = {}
        nk_asn2id = {}
        # 添加节点
        for node in graph.nodes():
            id = G_nk.addNode()
            nk_id2asn[id] = node
            nk_asn2id[node] = id
        # 添加边
        for edge in graph.edges():
            src = nk_asn2id[edge[0]]
            dst = nk_asn2id[edge[1]]
            G_nk.addEdge(src, dst)
        return G_nk, nk_id2asn, nk_asn2id

    def rebuild_networkit_graph(self):
        self.G_nk, self.nk_id2asn, self.nk_asn2id = self.build_networkit_graph()

    def _add_graph_edge(self, left, right, weight=1.0):
        left, right = str(left), str(right)
        if not left or not right or left == right or self.G.has_edge(left, right):
            return False
        self.G.add_edge(left, right, weight=max(float(weight or 1.0), 1.0))
        return True

    def augment_graph_from_pipeline_priors(self):
        """Add first-hop and recursive-anchor edges accepted upstream."""
        added = 0
        selected_file = getattr(self, "graph_selected_neighbors_csv", "")
        if selected_file and read_exists(selected_file):
            with open(selected_file, newline="", encoding="utf-8") as fp:
                for row in csv.DictReader(fp):
                    if str(row.get("selected", "1")) not in {"1", "1.0", "True", "true"}:
                        continue
                    added += self._add_graph_edge(
                        row.get("center_as"),
                        row.get("neighbor_as"),
                        row.get("asrank_link_number_paths_vu", 1.0),
                    )

        anchor_file = getattr(self, "graph_recursive_anchors_csv", "")
        if anchor_file and read_exists(anchor_file):
            with open(anchor_file, newline="", encoding="utf-8") as fp:
                for row in csv.DictReader(fp):
                    added += self._add_graph_edge(
                        row.get("first_hop"),
                        row.get("second_hop"),
                        row.get("asrank_number_paths", 1.0),
                    )
        if added:
            self.rebuild_networkit_graph()
        return added

    def _nearest_asrank_link_file(self):
        explicit = os.environ.get("PATHVEIL_ASRANK_LINKS")
        if not explicit:
            raise ValueError("Date-bound workflow requires PATHVEIL_ASRANK_LINKS")
        path = Path(explicit)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError("Configured ASRank links missing: " + explicit)
        return path

    def augment_graph_for_external_nodes(self, nodes):
        """Attach graph-external nodes using ASRank and as-rel agreement."""
        targets = {
            str(node)
            for node in nodes
            if str(node) not in self.G and str(node) not in self._external_topology_checked
        }
        if not targets:
            return 0
        self._external_topology_checked.update(targets)
        link_file = self._nearest_asrank_link_file()
        if link_file is None:
            return 0

        added = 0
        with link_file.open(encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                try:
                    row = json.loads(line)
                    left = str(row.get("asn0", {}).get("asn", "") or "")
                    right = str(row.get("asn1", {}).get("asn", "") or "")
                except Exception:
                    continue
                if left not in targets and right not in targets:
                    continue
                if left not in self.G and right not in self.G:
                    continue
                if commonFunc.get_as_relationship(self.rel_dict, left, right) == 2:
                    continue
                added += self._add_graph_edge(left, right, row.get("numberPaths", 1.0))
        if added:
            self.rebuild_networkit_graph()
        return added

    def generate_asn_idx(self, G):
        stoi = {}
        itos = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"]
        AS_freq_path = (
            files_path + "data/aspath/path_by_vantage/" + self.parent + "/AS_frequency.json"
        )
        fp = open(AS_freq_path)
        str_js = ujson.load(fp)
        AS_freq = ujson.loads(str_js)
        fp.close()
        nodes = list(AS_freq.keys())
        # nodes = list(G.nodes())
        nodes.sort()
        itos.extend(nodes)
        for idx, asn in enumerate(itos):
            stoi[asn] = idx
        return itos, stoi

    def empty_vector(self):
        asnEmbedding = np.zeros(self.node2vecOutputDim)
        asnFeatures = np.array([100000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        asnFeatures = asnFeatures / np.linalg.norm(asnFeatures)  # 归一化
        asnEmbedding = np.concatenate((asnFeatures, asnEmbedding), axis=0)
        return asnEmbedding

    def get_node_vectors(self):
        asn_embedding = self.load_embed()
        # asn_embedding = self.tmp_load_snn_embed()
        emptyEmbedding = self.empty_vector()
        nodeVectors = []
        for node in self.itos:
            try:
                nodeVectors.append(asn_embedding[node])
            except:
                nodeVectors.append(emptyEmbedding)
        return nodeVectors

    # def get_gmpi_node_vectors(self):
    #     asn_embedding = self.load_embed()
    #     emptyEmbedding = self.empty_vector()
    #     nodeVectors = []
    #     for node in self.itos:
    #         if node in asn_embedding:
    #             nodeVectors.append(asn_embedding[node])
    #         else:
    #             nodeVectors.append(emptyEmbedding)
    #     return nodeVectors

    def load_embed(self):
        # FILES
        filePath = self.baseDir + "ASEmbedding%s_%d.txt" % (
            self.walksType,
            self.node2vecOutputDim + self.asrankFeatureDim,
        )
        f = open(filePath, "rb")
        asnEmbedding = pickle.load(f)
        f.close()
        return asnEmbedding

    def tmp_load_snn_embed(self):
        asEmbedding_path = self.baseDir + "SNN-ASEmbedding-metatRatio=0.01.json"
        fp = open(asEmbedding_path, "r")
        str_js = ujson.load(fp)
        asEmbedding = ujson.loads(str_js)
        fp.close()
        return asEmbedding

    def get_node_features(self):
        asnFeature_path = self.baseDir + "AS_Feature.json"
        fp = open(asnFeature_path)
        str_js = ujson.load(fp)
        node_dict = ujson.loads(str_js)
        fp.close()
        return node_dict

    def getExistPath(self, dataType):
        pathFile = (
            files_path
            + "data/aspath/path_by_vantage/"
            + self.parent
            + "/path_without_prefix/%s/pathWithKey.pkl" % (dataType)
        )
        f = open(pathFile, "rb")
        existPath = pickle.load(f)
        f.close()
        return existPath

    def _truth_path_file(self, VASN, dataType):
        """Resolve the requested observation split without an unsplit fallback.

        An explicit empty split selects the unsplit observations for callers
        that intentionally need them. Train/test callers must use the actual
        split, including when its source-AS CSV is absent.
        """
        if dataType not in ("", "train", "test"):
            raise ValueError("unsupported observation split: %r" % dataType)
        parts = [
            files_path,
            "data",
            "aspath",
            "path_by_vantage",
            self.curDate,
            "path_without_prefix",
        ]
        if dataType:
            parts.append(dataType)
        parts.append(str(VASN) + ".csv")
        return os.path.join(*parts)

    # 用于计算生成路径的准确率；train/test 必须使用各自独立的观测。
    def get_exist_path_by_ASN(self, VASN, dataType):
        df = pd.read_csv(self._truth_path_file(VASN, dataType), names=["aspath"])
        src_dst_path = {}
        for row in df["aspath"].values:
            aspath = row
            as_list = aspath.split("->")
            # is valley_free
            # if not commonFunc.valley_free(as_list,self.rel_dict):
            #     continue
            dst = as_list[-1]
            key = VASN + "-" + dst
            if key in src_dst_path:
                src_dst_path[key].append(as_list)
            else:
                src_dst_path[key] = [as_list]
            # try:
            #     tmp = src_dst_path[key]
            #     tmp.append(as_list)
            #     src_dst_path[key] = tmp
            # except:
            #     src_dst_path[key] = [as_list]
        return src_dst_path

    def get_exist_path_by_ASN_valley_free(self, VASN, dataType):
        df = pd.read_csv(self._truth_path_file(VASN, dataType), names=["aspath"])
        src_dst_path = {}
        for row in df["aspath"].values:
            aspath = row
            as_list = aspath.split("->")
            # whether valley_free
            if not commonFunc.valley_free(as_list, self.rel_dict):
                continue
            dst = as_list[-1]
            key = VASN + "-" + dst
            if key in src_dst_path:
                src_dst_path[key].append(as_list)
            else:
                src_dst_path[key] = [as_list]
            # try:
            #     tmp = src_dst_path[key]
            #     tmp.append(as_list)
            #     src_dst_path[key] = tmp
            # except:
            #     src_dst_path[key] = [as_list]
        return src_dst_path

    def relationship_dict(self):
        as_relationship_file = files_path + "data/aspath/%s.as-rel.txt" % (self.parent)
        df = pd.read_csv(as_relationship_file, names=["rel"])
        rel = df["rel"].tolist()
        d = {}
        for r in rel:
            if r.startswith("#"):
                continue
            as1 = r.split("|")[0]
            as2 = r.split("|")[1]
            relationship = int(r.split("|")[2])
            try:
                tmp = d[as1]
                tmp[as2] = relationship
                d[as1] = tmp
            except:
                d1 = {}
                d1[as2] = relationship
                d[as1] = d1
        return d

    def getMaxLen(self, existPath):
        maxLen = 0
        for key in existPath.keys():
            paths = existPath[key]
            for path in paths:
                path = path.split("->")
                l = len(path)
                if l > maxLen:
                    maxLen = l
        print("maxLen:%d" % (maxLen))
        self.maxLen = maxLen
        return maxLen

    def get_vas_pathAvgLen(self):
        train_avglen_file = self.baseDir + "avglen.csv"
        if read_exists(train_avglen_file):
            train_avglen = pd.read_csv(train_avglen_file, header=None)
            asn_avglen = train_avglen.values.tolist()
            asn_avglen = [[str(int(row[0])), row[1], row[2]] for row in asn_avglen]
        else:
            filedir = (
                files_path
                + "data/aspath/path_by_vantage/"
                + self.parent
                + "/path_without_prefix/%s/" % ("train")
            )
            files = os.listdir(filedir)
            asn_avglen = []
            for file in files:
                if file.find("expand") >= 0 or file.find("merge") >= 0:
                    continue
                asn = file.split(".")[0]
                file = filedir + file
                try:
                    df = pd.read_csv(file, header=None)
                except:
                    continue
                # if len(df) < self.border:
                #     continue
                total_len = 0
                for row in df.values.tolist():
                    path = row[0]
                    total_len += len(path.split("->"))
                avg_len = total_len / len(df)
                asn_avglen.append([asn, avg_len, len(df)])
                df = pd.DataFrame(asn_avglen)
                df.to_csv(train_avglen_file, header=None, index=False)
        return asn_avglen

    def get_neighbors_from_collected_path(self, file):
        df = pd.read_csv(file, header=None)
        neighbors = []
        for row in df.values.tolist():
            path = row[0]
            pathList = path.split("->")
            neighbors.append(pathList[1])
        neighbors = list(set(neighbors))
        return neighbors

    def get_vas_cneighbors(self):
        vas_cneighbors_path = self.baseDir + "vas_cneighbors.json"
        if read_exists(vas_cneighbors_path):
            fp = open(vas_cneighbors_path)
            str_js = ujson.load(fp)
            vas_cneighbors = ujson.loads(str_js)
            fp.close()
        else:
            vantage_dst_path = self.baseDir + "vantage_dst.json"
            fp = open(vantage_dst_path)
            str_js = ujson.load(fp)
            vantage_dst = ujson.loads(str_js)
            fp.close()
            vas_cneighbors = {}
            for vas in vantage_dst.keys():
                collected_path_file = (
                    files_path
                    + "data/aspath/path_by_vantage/%s/path_without_prefix/%s/%s.csv"
                    % (self.parent, "train", vas)
                )
                cneighbors = self.get_neighbors_from_collected_path(collected_path_file)
                cneighbors = list(set(cneighbors))
                vas_cneighbors[vas] = cneighbors
            # save vas_cneighbors to json file
            fp = open(vas_cneighbors_path, "w")
            str_js = ujson.dumps(vas_cneighbors)
            ujson.dump(str_js, fp)
            fp.close()
        return vas_cneighbors

    def iter_expand_paths(self, dataType, source_filter=None):
        expand_path_file = (
            files_path
            + "data/aspath/path_by_vantage/%s/path_without_prefix/%s/expand_paths.txt"
            % (self.parent, dataType)
        )
        source_filter = set(source_filter) if source_filter else None
        split_dir = files_path + "data/aspath/path_by_vantage/%s/path_without_prefix/%s" % (
            self.parent,
            dataType,
        )
        if source_filter and os.path.isdir(split_dir):
            missing_sources = set()
            for src in sorted(source_filter):
                csv_file = os.path.join(split_dir, "%s.csv" % src)
                if not read_exists(csv_file):
                    missing_sources.add(src)
                    continue
                with open(csv_file, "r", encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        path = line.strip()
                        if not path:
                            continue
                        pathList = path.split("->")
                        if pathList[0] not in source_filter:
                            continue
                        yield pathList
            if not missing_sources:
                return
            source_filter = missing_sources
        if not read_exists(expand_path_file):
            if source_filter:
                candidate_files = [
                    os.path.join(split_dir, "%s.csv" % src) for src in sorted(source_filter)
                ]
            else:
                candidate_files = (
                    [
                        os.path.join(split_dir, name)
                        for name in sorted(os.listdir(split_dir))
                        if name.endswith(".csv")
                    ]
                    if os.path.isdir(split_dir)
                    else []
                )
            for csv_file in candidate_files:
                if not read_exists(csv_file):
                    continue
                with open(csv_file, "r", encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        path = line.strip()
                        if not path:
                            continue
                        pathList = path.split("->")
                        if source_filter is not None and pathList[0] not in source_filter:
                            continue
                        yield pathList
            return
        if source_filter and shutil.which("rg"):
            escaped_sources = [re.escape(src) for src in sorted(source_filter)]
            pattern = r"^(%s)->" % "|".join(escaped_sources)
            process = subprocess.Popen(
                ["rg", "--no-messages", "-N", "-e", pattern, expand_path_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for line in process.stdout:
                    path = line.strip()
                    if not path:
                        continue
                    yield path.split("->")
            finally:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
                process.wait()
            return

        with open(expand_path_file, "r", encoding="utf-8") as fp:
            for line in fp:
                path = line.strip()
                if not path:
                    continue
                pathList = path.split("->")
                if source_filter is not None and pathList[0] not in source_filter:
                    continue
                yield pathList

    def get_expand_paths(self, dataType, source_filter=None):
        print("get expand paths...")
        expand_paths = []
        for pathList in self.iter_expand_paths(dataType, source_filter=source_filter):
            expand_paths.append(pathList)
        return expand_paths

    def get_path_freq(self, dataType):
        cache_file = self.baseDir + "pathWithFreq_%s.pkl" % dataType
        if read_exists(cache_file):
            with open(cache_file, "rb") as fp:
                return pickle.load(fp)

        path_freq_file = (
            files_path
            + "data/aspath/path_by_vantage/%s/path_without_prefix/%s//pathWithFreq.json"
            % (self.parent, dataType)
        )
        fp = open(path_freq_file)
        str_js = ujson.load(fp)
        path_freq = ujson.loads(str_js)
        fp.close()
        with open(cache_file, "wb") as fp:
            pickle.dump(path_freq, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return path_freq

    def get_better_borderLen_dict(self):
        file = self.baseDir + "better_borderLen_dict.json"
        if read_exists(file):
            with open(file, "r") as f:
                str_js = ujson.load(f)
                better_borderLen_dict = ujson.loads(str_js)
        else:
            better_borderLen_dict = {}
        return better_borderLen_dict

    def convert_expaned_paths_to_idx(self):
        expanded_paths = self.get_expand_paths("train")
        output_file = self.baseDir + "expand_paths_idx.csv"
        for epath in expanded_paths:
            idx_path = []
            for asn in epath:
                idx = self.stoi[asn]
                idx_path.append(idx)
            with open(output_file, "a") as f:
                f.write(str(idx_path) + "\n")


# Input paths are resolved only for reads; numerical operations are unchanged.
from pathveil.core.io import open_data as open, read_exists, PandasReadProxy

pd = PandasReadProxy(pd)
