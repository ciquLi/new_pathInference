from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import math
import csv
from copy import deepcopy
import itertools
import commonFunction as commonFun
import pandas as pd


# read the expaned paths from the file
class siameseDataset(Dataset):
    def __init__(self, env, file):
        self.env = env
        self.file = file
        self.maxLen = env.maxLen
        self.paths = self.read_file()
        self.len = len(self.paths)
        self.vas_paths = self.path_by_vas()
        self.prob_same = env.siamese_same_path_prob
        # self.prob = env.siamese_random_prob

    # def read_expaned_paths(self):
    #     expanded_paths = self.env.get_expand_paths('train')
    #     print('expanded paths loaded successfully')
    #     return expanded_paths

    def read_file(self):
        print("reading file %s" % (self.file))
        df = pd.read_csv(self.file, header=None)
        paths = []
        for row in df.values.tolist():
            path = row[0]
            pathList = path.split("->")
            paths.append(pathList)
        print("file read successfully")
        return paths

    def path_by_vas(self):
        vas_paths = {}
        for path in self.paths:
            vas = path[0]
            if vas not in vas_paths:
                vas_paths[vas] = []
            vas_paths[vas].append(path)
        return vas_paths

    def __len__(self):
        """
        返回长度
        """
        return len(self.paths)

    # def random_replace_asn(self,path):
    #     path_idx = [self.env.stoi['<pad>']]*self.maxLen
    #     random_path_idx = path_idx.copy()
    #     if np.random.rand() <= self.prob_same:
    #         for i,asn in enumerate(path):
    #             if asn in self.env.stoi:
    #                 path_idx[i] = self.env.stoi[asn]
    #                 if np.random.rand() < self.prob:
    #                     random_path_idx[i] = self.env.stoi[asn]
    #                 else:
    #                     random_path_idx[i] = np.random.randint(0,len(self.env.itos))
    #     aspair_idx = [path_idx[0],path_idx[len(path)-1]]
    #     last_second_hop = path_idx[len(path)-2]
    #     random_aspair_idx = [random_path_idx[0],random_path_idx[len(path)-1]]
    #     random_last_second_hop = random_path_idx[len(path)-2]
    #     path_true_len = len(path)
    #     simalarity = commonFun.jaccard_index(path_idx[:path_true_len],random_path_idx[:path_true_len])
    #     return path_idx, aspair_idx, last_second_hop, random_path_idx, random_aspair_idx, random_last_second_hop, simalarity

    def random_select_path(self, path):
        path_idx = [self.env.stoi["<pad>"]] * self.maxLen
        for i, asn in enumerate(path):
            if asn in self.env.stoi:
                path_idx[i] = self.env.stoi[asn]
        # random select one path from self.paths
        random_path = self.paths[np.random.randint(0, len(self.paths))]
        random_path_idx = [self.env.stoi["<pad>"]] * self.maxLen
        for i, asn in enumerate(random_path):
            if asn in self.env.stoi:
                random_path_idx[i] = self.env.stoi[asn]
        aspair_idx = [path_idx[0], path_idx[len(path) - 1]]
        last_second_hop = path_idx[len(path) - 2]
        random_aspair_idx = [random_path_idx[0], random_path_idx[len(random_path) - 1]]
        random_last_second_hop = random_path_idx[len(random_path) - 2]
        similarity = commonFun.jaccard_index(path, random_path)
        return (
            path_idx,
            aspair_idx,
            last_second_hop,
            random_path_idx,
            random_aspair_idx,
            random_last_second_hop,
            similarity,
        )

    def random_select_path_by_vas(self, path):
        vas = path[0]
        path_idx = [self.env.stoi["<pad>"]] * self.maxLen
        for i, asn in enumerate(path):
            if asn in self.env.stoi:
                path_idx[i] = self.env.stoi[asn]
        if np.random.rand() <= self.prob_same:
            # random select one path from self.vas_paths
            random_path = self.vas_paths[vas][np.random.randint(0, len(self.vas_paths[vas]))]
            random_path_idx = [self.env.stoi["<pad>"]] * self.maxLen
            for i, asn in enumerate(random_path):
                if asn in self.env.stoi:
                    random_path_idx[i] = self.env.stoi[asn]
        else:
            random_path = path.copy()
            random_path_idx = path_idx.copy()
        aspair_idx = [path_idx[0], path_idx[len(path) - 1]]
        last_second_hop = path_idx[len(path) - 2]
        random_aspair_idx = [random_path_idx[0], random_path_idx[len(random_path) - 1]]
        random_last_second_hop = random_path_idx[len(random_path) - 2]
        similarity = commonFun.jaccard_index(path, random_path)
        return (
            path_idx,
            aspair_idx,
            last_second_hop,
            random_path_idx,
            random_aspair_idx,
            random_last_second_hop,
            similarity,
        )

    def __getitem__(self, idx):
        """
        根据idx获取一个文件的数据并处理
        """
        # if self.step == 'pretrain':

        # elif self.step == 'finetune':
        #     path_idx, aspair_idx, last_second_hop_idx, random_path_idx, random_aspair_idx, random_last_second_hop_idx, similarity = self.random_select_path(self.paths[idx])

        (
            path_idx,
            aspair_idx,
            last_second_hop_idx,
            random_path_idx,
            random_aspair_idx,
            random_last_second_hop_idx,
            similarity,
        ) = self.random_select_path_by_vas(self.paths[idx])
        # convert to tensor
        path_idx = torch.tensor(path_idx)
        ap_idx = torch.tensor(aspair_idx)
        last_sec_hop_idx = torch.tensor([last_second_hop_idx])
        random_epath_idx = torch.tensor(random_path_idx)
        random_ap_idx = torch.tensor(random_aspair_idx)
        random_last_sec_hop_idx = torch.tensor([random_last_second_hop_idx])
        similarity = torch.tensor([similarity])
        # add a dimension at axis 0
        epath_idx = path_idx.unsqueeze(0)
        ap_idx = ap_idx.unsqueeze(0)
        last_sec_hop_idx = last_sec_hop_idx.unsqueeze(0)
        random_epath_idx = random_epath_idx.unsqueeze(0)
        random_ap_idx = random_ap_idx.unsqueeze(0)
        random_last_sec_hop_idx = random_last_sec_hop_idx.unsqueeze(0)
        similarity = similarity.unsqueeze(0)
        return [
            epath_idx,
            ap_idx,
            last_sec_hop_idx,
            random_epath_idx,
            random_ap_idx,
            random_last_sec_hop_idx,
            similarity,
        ]


class metaIterableDatasetShuffle(IterableDataset):
    def __init__(
        self, filename, env, maxLen, k, buffersize, seed, support_data_dict, shuffle=False
    ):
        self.env = env
        self.maxLen = maxLen
        self.k = k
        # Store the filename in object's memory
        self.filename = filename
        self.buffer_size = buffersize
        self.generator = np.random.default_rng(seed=seed)
        self.shuffle = shuffle
        self.support_data_dict = support_data_dict

    def __len__(self):
        length = 0
        with open(self.filename, "r") as fin:
            for _ in enumerate(fin):
                length += 1
        return length

    def pad_empty_path(self, maxLen):
        vecLen = self.env.node2vecOutputDim
        # if len(aslist) < maxLen:
        #     padASN = ['0']*(maxLen-len(aslist))
        #     aslist = aslist + padASN
        result = []
        for i in range(maxLen):
            asnEmbedding = np.zeros(vecLen)
            asnFeatures = np.array([100000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            asnFeatures = asnFeatures / np.linalg.norm(asnFeatures)  # 归一化
            asnEmbedding = np.concatenate((asnFeatures, asnEmbedding), axis=0)
            result.append(asnEmbedding)
        result = np.array([result])
        return result

    def pad_paths_to_maxLength(self, paths, maxLen):
        if len(paths) > 0:
            for path in paths:
                if len(path) < maxLen:
                    path.extend([self.env.stoi["<pad>"]] * (maxLen - len(path)))
        else:
            paths = [[self.env.stoi["<pad>"]] * maxLen]

        try:
            paths = np.array(paths)
        except Exception as e:
            print(e)
            for path in paths:
                if len(path) > maxLen:
                    print(path)

        return paths

    def process_string_paths(self, string):
        strlist = string.split("|")
        result = []
        last_second_hops = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            last_second_hops.append(pathList[-2])
            result.append(pathList)
        return result, last_second_hops

    def process_string_aspairs(self, string):
        strlist = string.split("|")
        result = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            result.append(pathList)
        return result

    def convertToTensor(self, item):
        k = self.k
        hisMax = k
        maxLen = self.maxLen
        try:
            kpaths, hisPaths, hisPathsAP, similarity, aspair_str = item
        except:
            print(item)
            kpaths, hisPaths, hisPathsAP, similarity, aspair_str = "", "", "", "0|0", "0->0"

        if kpaths == "":
            kpaths, k_last_second_hops = [], []
        else:
            kpaths, k_last_second_hops = self.process_string_paths(kpaths)
        generatedPathsNum = len(kpaths)
        # if isinstance(hisPaths, float):
        if hisPaths == "":
            hisPaths, his_last_second_hops = [], []
        else:
            hisPaths, his_last_second_hops = self.process_string_paths(hisPaths)
        hisPathsNum = len(hisPaths)
        # if isinstance(hisPathsAP, float):
        if hisPathsAP == "":
            hisPathsAP = []
        else:
            hisPathsAP = self.process_string_aspairs(hisPathsAP)

        similarity = [float(i) for i in similarity.split("|")]
        aspair_str = aspair_str.split("->")

        # Obtaing AS pair represented by AS index
        try:
            src_idx = self.env.stoi[aspair_str[0]]
        except:
            src_idx = self.env.stoi["<pad>"]
        try:
            dst_idx = self.env.stoi[aspair_str[1]]
        except:
            dst_idx = self.env.stoi["<pad>"]
        aspair_label = [src_idx, dst_idx]

        # The recieced k paths in kpaths(hisPasths) are of different length, padding them to the same length.
        kpaths = self.pad_paths_to_maxLength(kpaths, maxLen)
        hisPaths = self.pad_paths_to_maxLength(hisPaths, maxLen)
        # Convert to 2 dim
        similarity = np.array(similarity).reshape(-1, 1)
        # Covert to numpy array
        if len(hisPathsAP) > 0:
            hisPathsAP = np.array(hisPathsAP)
        else:
            hisPathsAP = np.array([[self.env.stoi["<pad>"], self.env.stoi["<pad>"]]])

        kpathsAP = np.tile(np.array(aspair_label).reshape(1, -1), (kpaths.shape[0], 1))
        hislengthAP = np.tile(np.array(aspair_label).reshape(1, -1), (hisPaths.shape[0], 1))
        if (k - kpaths.shape[0]) > 0:
            # 放在kpaths填充以前
            kpathsAP = np.concatenate(
                (kpathsAP, np.zeros((k - kpaths.shape[0], 2), dtype=int)), axis=0
            )
            similarity = np.concatenate(
                (similarity, np.zeros((k - kpaths.shape[0], 1), dtype=int)), axis=0
            )
            kpaths = np.concatenate(
                (kpaths, np.zeros((k - kpaths.shape[0], maxLen), dtype=int)), axis=0
            )  # 补齐成k*maxLen
        kpaths = torch.from_numpy(kpaths).int()
        kpathsAP = torch.from_numpy(kpathsAP).int()
        similarity = torch.from_numpy(similarity).float()
        if (hisMax - hisPaths.shape[0]) > 0:
            hisPathsAP = np.concatenate(
                (hisPathsAP, np.zeros((hisMax - hisPaths.shape[0], 2), dtype=int)), axis=0
            )
            hislengthAP = np.concatenate(
                (hislengthAP, np.zeros((hisMax - hisPaths.shape[0], 2), dtype=int)), axis=0
            )
            hisPaths = np.concatenate(
                (hisPaths, np.zeros((hisMax - hisPaths.shape[0], maxLen), dtype=int)), axis=0
            )

        hisPaths = torch.from_numpy(hisPaths).int()
        hisPathsAP = torch.from_numpy(hisPathsAP).int()

        k_last_second_hops.extend([0] * (k - len(k_last_second_hops)))
        k_last_second_hops = np.array(k_last_second_hops).reshape(-1, 1)
        k_last_second_hops = torch.from_numpy(k_last_second_hops).int()

        his_last_second_hops.extend([0] * (hisMax - len(his_last_second_hops)))
        his_last_second_hops = np.array(his_last_second_hops).reshape(-1, 1)
        his_last_second_hops = torch.from_numpy(his_last_second_hops).int()

        ap = aspair_str[0] + "-" + aspair_str[1]
        return [
            kpaths[: self.k],
            k_last_second_hops[: self.k],
            kpathsAP[: self.k],
            hislengthAP[:hisMax],
            hisPaths[:hisMax],
            his_last_second_hops[:hisMax],
            hisPathsAP[:hisMax],
            similarity[: self.k],
            ap,
            generatedPathsNum,
            hisPathsNum,
        ]

    def line_mapper(self, line):
        item = line.strip().split(",")
        query_data = self.convertToTensor(item)
        ap = query_data[-3]
        ap = ap.replace("-", "->")
        if ap in self.support_data_dict:
            support_data_list = self.support_data_dict[ap]
            support_data = self.convertToTensor(support_data_list[0])
            support_data_label = True
        else:
            support_data = ["", "", "", "0|0", "0->0"]
            support_data = self.convertToTensor(support_data)
            support_data_label = False
        return support_data, query_data, support_data_label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
        else:
            worker_id = 0
            worker_total_num = 1

        # Create an iterator
        file_itr = open(self.filename)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        # Add multiworker functionality
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

        if self.shuffle:
            return self._shuffle(mapped_itr)
        else:
            return mapped_itr

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000):
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def _shuffle(self, ex_iterable):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = []
        for x in ex_iterable:
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        rng.shuffle(mem_buffer)
        yield from mem_buffer


class metaMultiSupportIterableDatasetShuffle(IterableDataset):
    def __init__(
        self,
        filename,
        env,
        maxLen,
        k,
        buffersize,
        seed,
        support_data_dict,
        support_num,
        shuffle=False,
    ):
        self.env = env
        self.maxLen = maxLen
        self.k = k
        # Store the filename in object's memory
        self.filename = filename
        self.buffer_size = buffersize
        self.generator = np.random.default_rng(seed=seed)
        self.shuffle = shuffle
        self.support_data_dict = support_data_dict
        self.support_num = support_num

    def __len__(self):
        length = 0
        with open(self.filename, "r") as fin:
            for _ in enumerate(fin):
                length += 1
        return length

    def pad_empty_path(self, maxLen):
        vecLen = self.env.node2vecOutputDim
        # if len(aslist) < maxLen:
        #     padASN = ['0']*(maxLen-len(aslist))
        #     aslist = aslist + padASN
        result = []
        for i in range(maxLen):
            asnEmbedding = np.zeros(vecLen)
            asnFeatures = np.array([100000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            asnFeatures = asnFeatures / np.linalg.norm(asnFeatures)  # 归一化
            asnEmbedding = np.concatenate((asnFeatures, asnEmbedding), axis=0)
            result.append(asnEmbedding)
        result = np.array([result])
        return result

    def pad_paths_to_maxLength(self, paths, maxLen):
        if len(paths) > 0:
            for path in paths:
                if len(path) < maxLen:
                    path.extend([self.env.stoi["<pad>"]] * (maxLen - len(path)))
        else:
            paths = [[self.env.stoi["<pad>"]] * maxLen]

        try:
            paths = np.array(paths)
        except Exception as e:
            print(e)
            for path in paths:
                if len(path) > maxLen:
                    print(path)

        return paths

    def process_string_paths(self, string):
        strlist = string.split("|")
        result = []
        last_second_hops = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            last_second_hops.append(pathList[-2])
            result.append(pathList)
        return result, last_second_hops

    def process_string_aspairs(self, string):
        strlist = string.split("|")
        result = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            result.append(pathList)
        return result

    def convertToTensor(self, item):
        k = self.k
        hisMax = k
        maxLen = self.maxLen
        try:
            kpaths, hisPaths, hisPathsAP, similarity, aspair_str = item
        except:
            print(item)
            kpaths, hisPaths, hisPathsAP, similarity, aspair_str = "", "", "", "0|0", "0->0"

        if kpaths == "":
            kpaths, k_last_second_hops = [], []
        else:
            kpaths, k_last_second_hops = self.process_string_paths(kpaths)
        generatedPathsNum = len(kpaths)
        # if isinstance(hisPaths, float):
        if hisPaths == "":
            hisPaths, his_last_second_hops = [], []
        else:
            hisPaths, his_last_second_hops = self.process_string_paths(hisPaths)
        hisPathsNum = len(hisPaths)
        # if isinstance(hisPathsAP, float):
        if hisPathsAP == "":
            hisPathsAP = []
        else:
            hisPathsAP = self.process_string_aspairs(hisPathsAP)

        similarity = [float(i) for i in similarity.split("|")]
        aspair_str = aspair_str.split("->")

        # Obtaing AS pair represented by AS index
        try:
            src_idx = self.env.stoi[aspair_str[0]]
        except:
            src_idx = self.env.stoi["<pad>"]
        try:
            dst_idx = self.env.stoi[aspair_str[1]]
        except:
            dst_idx = self.env.stoi["<pad>"]
        aspair_label = [src_idx, dst_idx]

        # The recieced k paths in kpaths(hisPasths) are of different length, padding them to the same length.
        kpaths = self.pad_paths_to_maxLength(kpaths, maxLen)
        hisPaths = self.pad_paths_to_maxLength(hisPaths, maxLen)
        # Convert to 2 dim
        similarity = np.array(similarity).reshape(-1, 1)
        # Covert to numpy array
        if len(hisPathsAP) > 0:
            hisPathsAP = np.array(hisPathsAP)
        else:
            hisPathsAP = np.array([[self.env.stoi["<pad>"], self.env.stoi["<pad>"]]])

        kpathsAP = np.tile(np.array(aspair_label).reshape(1, -1), (kpaths.shape[0], 1))
        hislengthAP = np.tile(np.array(aspair_label).reshape(1, -1), (hisPaths.shape[0], 1))
        if (k - kpaths.shape[0]) > 0:
            # 放在kpaths填充以前
            kpathsAP = np.concatenate(
                (kpathsAP, np.zeros((k - kpaths.shape[0], 2), dtype=int)), axis=0
            )
            similarity = np.concatenate(
                (similarity, np.zeros((k - kpaths.shape[0], 1), dtype=int)), axis=0
            )
            kpaths = np.concatenate(
                (kpaths, np.zeros((k - kpaths.shape[0], maxLen), dtype=int)), axis=0
            )  # 补齐成k*maxLen
        kpaths = torch.from_numpy(kpaths).int()
        kpathsAP = torch.from_numpy(kpathsAP).int()
        similarity = torch.from_numpy(similarity).float()
        if (hisMax - hisPaths.shape[0]) > 0:
            hisPathsAP = np.concatenate(
                (hisPathsAP, np.zeros((hisMax - hisPaths.shape[0], 2), dtype=int)), axis=0
            )
            hislengthAP = np.concatenate(
                (hislengthAP, np.zeros((hisMax - hisPaths.shape[0], 2), dtype=int)), axis=0
            )
            hisPaths = np.concatenate(
                (hisPaths, np.zeros((hisMax - hisPaths.shape[0], maxLen), dtype=int)), axis=0
            )

        hisPaths = torch.from_numpy(hisPaths).int()
        hisPathsAP = torch.from_numpy(hisPathsAP).int()

        k_last_second_hops.extend([0] * (k - len(k_last_second_hops)))
        k_last_second_hops = np.array(k_last_second_hops).reshape(-1, 1)
        k_last_second_hops = torch.from_numpy(k_last_second_hops).int()

        his_last_second_hops.extend([0] * (hisMax - len(his_last_second_hops)))
        his_last_second_hops = np.array(his_last_second_hops).reshape(-1, 1)
        his_last_second_hops = torch.from_numpy(his_last_second_hops).int()

        ap = aspair_str[0] + "-" + aspair_str[1]
        return [
            kpaths[: self.k],
            k_last_second_hops[: self.k],
            kpathsAP[: self.k],
            hislengthAP[:hisMax],
            hisPaths[:hisMax],
            his_last_second_hops[:hisMax],
            hisPathsAP[:hisMax],
            similarity[: self.k],
            ap,
            generatedPathsNum,
            hisPathsNum,
        ]

    def line_mapper(self, line):
        item = line.strip().split(",")
        query_data = self.convertToTensor(item)
        ap = query_data[-3]
        ap = ap.replace("-", "->")
        support_data = []
        support_data_label = []
        if ap in self.support_data_dict:
            support_data_list = self.support_data_dict[ap]
            for item in support_data_list:
                support_data.append(self.convertToTensor(item))
                support_data_label.append(True)

        if len(support_data) < self.support_num:
            for i in range(3 - len(support_data)):
                tmp = ["", "", "", "0|0", "0->0"]
                support_data.append(self.convertToTensor(tmp))
                support_data_label.append(False)
        return support_data, query_data, support_data_label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
        else:
            worker_id = 0
            worker_total_num = 1

        # Create an iterator
        file_itr = open(self.filename)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        # Add multiworker functionality
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

        if self.shuffle:
            return self._shuffle(mapped_itr)
        else:
            return mapped_itr

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000):
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def _shuffle(self, ex_iterable):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = []
        for x in ex_iterable:
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        rng.shuffle(mem_buffer)
        yield from mem_buffer


def _convert_model_input(dataset, item):
    """Validate one generated five-column CSV record and retain its 11-field contract.

    Paths/pairs in the first three columns use vocabulary indices; the final
    query column uses actual ASNs. Invalid query candidates are removed together
    with their labels, and identical candidates retain their first occurrence.
    """
    filename = getattr(dataset, "filename", getattr(dataset, "file_path", "<memory>"))
    query = item[4] if isinstance(item, (list, tuple)) and len(item) > 4 else "<unknown>"

    def fail(message):
        raise ValueError("%s: query %r: %s" % (filename, query, message))

    if not isinstance(item, (list, tuple)) or len(item) != 5:
        fail("expected exactly 5 CSV fields")
    if not all(isinstance(value, str) for value in item):
        fail("all CSV fields must be strings")
    if dataset.k < 1 or dataset.maxLen < 2:
        fail("k must be positive and maxLen must be at least 2")
    k = dataset.k
    max_length = dataset.maxLen
    pad = dataset.env.stoi["<pad>"]

    # Cache once per reader: special tokens, including padding, cannot be hops.
    if not hasattr(dataset, "_valid_path_indices"):
        dataset._valid_path_indices = {
            index
            for asn, index in dataset.env.stoi.items()
            if str(asn).isascii()
            and str(asn).isdecimal()
            and 0 < int(asn) <= 4294967295
            and index != pad
        }
    valid_indices = dataset._valid_path_indices
    actual_pair = item[4].split("->")
    if len(actual_pair) != 2 or any(
        not value.isascii() or not value.isdecimal() or not 0 < int(value) <= 4294967295
        for value in actual_pair
    ):
        fail("query must contain exactly two positive numeric ASNs")
    try:
        query_pair = [dataset.env.stoi[value] for value in actual_pair]
    except KeyError as exc:
        fail("query ASN %s is absent from the vocabulary" % exc.args[0])
    if any(value not in valid_indices for value in query_pair):
        fail("query maps to a special or unknown vocabulary index")

    def parse_sequences(value, field, is_pair=False):
        if value == "":
            return []
        result = []
        for position, encoded in enumerate(value.split("|")):
            tokens = encoded.split("->")
            if any(not token.isascii() or not token.isdecimal() for token in tokens):
                fail("%s[%d] contains a nonnumeric vocabulary index" % (field, position))
            sequence = [int(token) for token in tokens]
            if is_pair and len(sequence) != 2:
                fail("%s[%d] must contain exactly two indices" % (field, position))
            if not is_pair and not 2 <= len(sequence) <= max_length:
                fail(
                    "%s[%d] has length %d; expected 2..%d (paths are never truncated)"
                    % (field, position, len(sequence), max_length)
                )
            if any(index not in valid_indices for index in sequence):
                fail(
                    "%s[%d] contains padding, a special token, or an unknown vocabulary index"
                    % (field, position)
                )
            result.append(sequence)
        return result

    candidates = parse_sequences(item[0], "candidates")
    histories = parse_sequences(item[1], "history")
    history_pairs = parse_sequences(item[2], "history_pairs", is_pair=True)
    try:
        labels = [float(value) for value in item[3].split("|")] if item[3] else []
    except ValueError:
        fail("candidate labels must be numeric")
    if len(candidates) != len(labels):
        fail("candidate/label count mismatch (%d versus %d)" % (len(candidates), len(labels)))
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in labels):
        fail("candidate labels must be finite Jaccard values in [0, 1]")
    if len(histories) != len(history_pairs):
        fail(
            "history/history-pair count mismatch (%d versus %d)"
            % (len(histories), len(history_pairs))
        )
    for position, (path, pair) in enumerate(zip(histories, history_pairs)):
        if [path[0], path[-1]] != pair:
            fail("history[%d] endpoints disagree with its own history pair" % position)

    unique_candidates = []
    unique_labels = []
    seen = {}
    for path, label in zip(candidates, labels):
        if [path[0], path[-1]] != query_pair:
            continue
        key = tuple(path)
        if key in seen:
            if seen[key] != label:
                fail("duplicate candidate has conflicting labels")
            continue
        seen[key] = label
        unique_candidates.append(path)
        unique_labels.append(label)
    if not unique_candidates:
        fail("no usable candidates remain for the query")

    # Counts refer to valid observations before slot capping or padding.
    generated_count = len(unique_candidates)
    history_count = len(histories)
    candidate_tensor = np.full((k, max_length), pad, dtype=np.int32)
    candidate_hops = np.full((k, 1), pad, dtype=np.int32)
    candidate_pairs = np.full((k, 2), pad, dtype=np.int32)
    history_query_pairs = np.full((k, 2), pad, dtype=np.int32)
    history_tensor = np.full((k, max_length), pad, dtype=np.int32)
    history_hops = np.full((k, 1), pad, dtype=np.int32)
    history_pair_tensor = np.full((k, 2), pad, dtype=np.int32)
    similarities = np.zeros((k, 1), dtype=np.float32)
    for position, (path, label) in enumerate(zip(unique_candidates[:k], unique_labels[:k])):
        candidate_tensor[position, : len(path)] = path
        candidate_hops[position, 0] = path[-2]
        candidate_pairs[position] = query_pair
        similarities[position, 0] = label
    for position, (path, pair) in enumerate(zip(histories[:k], history_pairs[:k])):
        history_tensor[position, : len(path)] = path
        history_hops[position, 0] = path[-2]
        history_pair_tensor[position] = pair
        history_query_pairs[position] = query_pair
    ap = "-".join(actual_pair)
    tensors = [
        candidate_tensor,
        candidate_hops,
        candidate_pairs,
        history_query_pairs,
        history_tensor,
        history_hops,
        history_pair_tensor,
        similarities,
    ]
    return [torch.from_numpy(value) for value in tensors] + [ap, generated_count, history_count]


import sys


class IterableDatasetShuffle(IterableDataset):
    def __init__(self, filename, env, maxLen, k, buffersize, seed, shuffle=False):
        self.env = env
        self.maxLen = maxLen
        self.k = k
        # Store the filename in object's memory
        self.filename = filename
        self.buffer_size = buffersize
        self.generator = np.random.default_rng(seed=seed)
        self.shuffle = shuffle

    def __len__(self):
        length = 0
        with open(self.filename, "r", newline="") as fin:
            length = sum(1 for _ in csv.reader(fin, strict=True))
        return length

    def pad_empty_path(self, maxLen):
        vecLen = self.env.node2vecOutputDim
        # if len(aslist) < maxLen:
        #     padASN = ['0']*(maxLen-len(aslist))
        #     aslist = aslist + padASN
        result = []
        for i in range(maxLen):
            asnEmbedding = np.zeros(vecLen)
            asnFeatures = np.array([100000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            asnFeatures = asnFeatures / np.linalg.norm(asnFeatures)  # 归一化
            asnEmbedding = np.concatenate((asnFeatures, asnEmbedding), axis=0)
            result.append(asnEmbedding)
        result = np.array([result])
        return result

    def pad_paths_to_maxLength(self, paths, maxLen):
        if len(paths) > 0:
            for path in paths:
                if len(path) < maxLen:
                    path.extend([self.env.stoi["<pad>"]] * (maxLen - len(path)))
        else:
            paths = [[self.env.stoi["<pad>"]] * maxLen]

        try:
            paths = np.array(paths)
        except Exception as e:
            print(e)
            for path in paths:
                if len(path) > maxLen:
                    print(path)

        return paths

    def process_string_paths(self, string):
        strlist = string.split("|")
        result = []
        last_second_hops = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            result.append(pathList)
            last_second_hops.append(pathList[-2])

        return result, last_second_hops

    def process_string_aspairs(self, string):
        strlist = string.split("|")
        result = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            result.append(pathList)
        return result

    def convertToTensor(self, item):
        return _convert_model_input(self, item)

    def line_mapper(self, line):
        return self.convertToTensor(next(csv.reader([line], strict=True)))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        worker_count = worker_info.num_workers if worker_info else 1

        def records():
            with open(self.filename, "r", newline="") as stream:
                for row_number, fields in enumerate(csv.reader(stream, strict=True)):
                    if row_number % worker_count != worker_id:
                        continue
                    try:
                        yield self.convertToTensor(fields)
                    except ValueError as exc:
                        raise ValueError(
                            "%s: record %d: %s" % (self.filename, row_number + 1, exc)
                        ) from exc

        iterator = records()
        return self._shuffle(iterator) if self.shuffle else iterator

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000):
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def _shuffle(self, ex_iterable):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = []
        for x in ex_iterable:
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        rng.shuffle(mem_buffer)
        yield from mem_buffer


class SummaryDataset(IterableDataset):
    def __init__(self, file_path, env, maxLen, k):
        super().__init__()
        self.env = env
        self.maxLen = maxLen
        self.k = k
        self.file_path = file_path
        self.info = self._get_file_info(file_path)
        self.start = self.info["start"]
        self.end = self.info["end"]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker
            iter_start = self.start
            iter_end = self.end
        else:  # multiple workers
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        sample_iterator = self._sample_generator(iter_start, iter_end)
        return sample_iterator

    def __len__(self):
        return self.end - self.start

    def _get_file_info(self, file_path):
        info = {
            "start": 0,
            "end": 0,
            "kpaths_colum": 0,
            "obsPaths_colum": 1,
            "obsAPs_colum": 2,
            "similarity_colum": 3,
            "aspair_colum": 4,
        }
        with open(file_path, "r", newline="") as fin:
            info["end"] = sum(1 for _ in csv.reader(fin, strict=True))
        return info

    def pad_empty_path(self, maxLen):
        vecLen = self.env.node2vecOutputDim
        # if len(aslist) < maxLen:
        #     padASN = ['0']*(maxLen-len(aslist))
        #     aslist = aslist + padASN
        result = []
        for i in range(maxLen):
            asnEmbedding = np.zeros(vecLen)
            asnFeatures = np.array([100000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            asnFeatures = asnFeatures / np.linalg.norm(asnFeatures)  # 归一化
            asnEmbedding = np.concatenate((asnFeatures, asnEmbedding), axis=0)
            result.append(asnEmbedding)
        result = np.array([result])
        return result

    def pad_paths_to_maxLength(self, paths, maxLen):
        if len(paths) > 0:
            for path in paths:
                if len(path) < maxLen:
                    path.extend([self.env.stoi["<pad>"]] * (maxLen - len(path)))
        else:
            paths = [[self.env.stoi["<pad>"]] * maxLen]

        try:
            paths = np.array(paths)
        except Exception as e:
            print(e)
            for path in paths:
                if len(path) > maxLen:
                    print(path)

        return paths

    def process_string_paths(self, string):
        strlist = string.split("|")
        result = []
        last_second_hops = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            last_second_hops.append(pathList[-2])
            result.append(pathList)
        return result, last_second_hops

    def process_string_aspairs(self, string):
        strlist = string.split("|")
        result = []
        for pathstr in strlist:
            pathList = pathstr.split("->")
            pathList = [int(i) for i in pathList]
            result.append(pathList)
        return result

    def convertToTensor(self, item):
        return _convert_model_input(self, item)

    def _sample_generator(self, start, end):
        with open(self.file_path, "r", newline="") as fin:
            for row_number, fields in enumerate(csv.reader(fin, strict=True)):
                if row_number < start:
                    continue
                if row_number >= end:
                    return
                try:
                    yield self.convertToTensor(fields)
                except ValueError as exc:
                    raise ValueError(
                        "%s: record %d: %s" % (self.file_path, row_number + 1, exc)
                    ) from exc


from torch.utils.data import DataLoader

try:
    from prefetch_generator import BackgroundGenerator
except ImportError:

    class BackgroundGenerator:
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return iter(self.generator)

        def __next__(self):
            return next(self.generator)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
