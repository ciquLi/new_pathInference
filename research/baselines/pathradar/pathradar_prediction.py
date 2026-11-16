
from collections import defaultdict
import copy, os, math, pickle, random, argparse
import networkx as nx
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from data_store import DB_PATH, PathRadarStore

REPO_DIR = Path(os.environ.get('PATHRADAR_RUN_DIR', Path(__file__).resolve().parent.parent))
INTERMEDIATE_DIR = Path(os.environ.get('PATHRADAR_INTERMEDIATE_DIR', REPO_DIR / 'intermediate'))
TRAINED_DIR = REPO_DIR / 'trained'
TESTING_DIR = REPO_DIR / 'testing'

class Model_Sample(object):
    def __init__(self):
        self.prev_frequency, self.next_frequency = defaultdict(int), defaultdict(int)
        self.linkfreq = defaultdict(int)
        self.linkorder = defaultdict(list)

        with open(INTERMEDIATE_DIR / 'joinfreq_data.txt') as f:
            for line in f:
                [flag, key, num] = line.strip().split(' ')
                if flag == "prev":
                    self.prev_frequency[key] = int(num)
                else:
                    self.next_frequency[key] = int(num)
        
        with open(INTERMEDIATE_DIR / 'linkfreq_data.txt') as f:
            for line in f:
                [link, num] = line.strip().split(' ')
                self.linkfreq[tuple(link.split('|'))] = int(num)

        with open(INTERMEDIATE_DIR / 'linkorder_data.txt') as f:
            for line in f:
                [node, neighbor] = line.strip().split(' ')
                self.linkorder[node] = neighbor.split('|')

        self.graph = nx.Graph()
        self.relationship = dict()
        self.provider, self.peer, self.customer = defaultdict(set), defaultdict(set), defaultdict(set)
        with open(INTERMEDIATE_DIR / 'asrel_rib.txt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                [asn1, asn2, r] = line.strip().split('|')
                self.graph.add_edge(asn1, asn2)
                if r == '-1':
                    self.relationship[(asn1, asn2)] = 'p2c'
                    self.relationship[(asn2, asn1)] = 'c2p'
                    self.customer[asn1].add(asn2)
                    self.provider[asn2].add(asn1)
                else:
                    self.relationship[(asn1, asn2)] = 'p2p'
                    self.relationship[(asn2, asn1)] = 'p2p'
                    self.peer[asn1].add(asn2)
                    self.peer[asn2].add(asn1)
        
        self.core_cluster = set()
        with open(INTERMEDIATE_DIR / 'core_shell.txt') as f:
            for line in f:
                info = line.strip().split(' ')
                if int(info[1]) > 1000:
                    self.core_cluster.add(info[0])

    def get_sample(self, path):
        att = [len(path), self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)]
        att.extend([len(self.provider[path[-1]]), len(self.peer[path[-1]]), len(self.customer[path[-1]])])
        att.extend([len(self.provider[path[-2]]), len(self.peer[path[-2]]), len(self.customer[path[-2]])])

        if path[-1] in self.core_cluster:
            for i in range(len(path), 0, -1):
                if i == 1:
                    att.extend([1, 0])
                else:
                    temp = '|'.join(path[-i:])
                    if temp in self.next_frequency:
                        att.extend([i, self.next_frequency[temp]])
                        break 
            att.append(self.linkfreq[(path[-1], path[-2])])
        
        if path[-2] not in self.linkorder[path[-1]]:
            att.append(-1)
        else:
            att.append(self.linkorder[path[-1]].index(path[-2]))
        if path[-1] not in self.linkorder[path[-2]]:
            att.append(-1)
        else:
            att.append(self.linkorder[path[-2]].index(path[-1]))

        for i in range(len(path), 0, -1):
            if i == 1:
                att.extend([1, 0])
            else:
                temp = '|'.join(path[:i])
                if temp in self.prev_frequency:
                    att.extend([i, self.prev_frequency[temp]])
                    break
        
        rel = []
        for i in range(len(path) - 1):
            rel.append(self.relationship[(path[i], path[i + 1])])
        att.extend([rel.count('c2p'), rel.count('p2p'), rel.count('p2c')]) #3

        slen, slen1 = (self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)), (self.shortest[(path[0], path[-2])] if (path[0], path[-2]) in self.shortest else (len(path) - 1))
        att.extend([len(path) - slen, len(path) - 1 - slen1])
        
        return att

    def simulate(self, asn):
        self.AS_path = defaultdict(list)
        self.worst = defaultdict(list)
        self.path_flag = dict()
        self.AS_path[asn] = [asn]
        self.worst[asn] = [asn]
        self.path_flag[tuple([asn])] = 0
        self.stream(asn)

        self.shortest = dict()
        for key in self.AS_path:
            self.shortest[(asn, key)] = len(self.AS_path[key])
    
    def condition(self, current, neighbor):
        valleyfree = False
        if self.path_flag[current] == 0:
            valleyfree = True
        elif self.path_flag[current] != 0 and neighbor in self.customer[current[-1]]:
            valleyfree = True
        return valleyfree

    def smaller(self, link1, link2):
        (asn1, asn2) = link1
        (asn3, asn4) = link2
        if asn1 in self.customer[asn2] and (asn3 in self.peer[asn4] or asn3 in self.provider[asn4]):
            return True
        elif asn1 in self.peer[asn2] and asn3 in self.provider[asn4]:
            return True
        return False

    def stream(self, asn):
        active, candicate = [tuple([asn])], []
        relationship_list = [self.provider, self.peer, self.customer]
        
        while len(active) > 0:
            for i in range(3):
                relationship = relationship_list[i]
                for current in active:
                    for neighbor in relationship[current[-1]]:
                        if neighbor == asn:
                            continue
                        if self.condition(current, neighbor):
                            right_path = list(current) + [neighbor]
                            if len(self.AS_path[neighbor]) == 0 or len(right_path) < len(self.AS_path[neighbor]):
                                self.AS_path[neighbor] = copy.deepcopy(right_path)
                                self.path_flag[tuple(right_path)] = i
                                candicate.append(tuple(right_path))
                                if len(self.worst[neighbor]) == 0 or self.smaller((current[-1], neighbor), (self.worst[neighbor][-2], self.worst[neighbor][-1])):
                                    self.worst[neighbor] = copy.deepcopy(right_path)
                            elif self.smaller((current[-1], neighbor), (self.worst[neighbor][-2], self.worst[neighbor][-1])):
                                self.worst[neighbor] = copy.deepcopy(right_path)
                                self.path_flag[tuple(right_path)] = i
                                candicate.append(tuple(right_path))
            
            active = copy.deepcopy(candicate)
            active.sort(key = lambda x: int(x[-1]))
            candicate = []

class Routingtree(object):
    def __init__(self, rounds, threshold, modes, quick_mode, workers):
        self.ms = Model_Sample()
        self.rounds = rounds
        self.threshold = threshold
        self.modes = modes
        self.quick_mode = quick_mode
        self.workers = max(1, workers)
        self.store = PathRadarStore(DB_PATH) if DB_PATH.exists() else None

    def list_dataset(self, dataset_name):
        if self.store and dataset_name in {'rib', 'test'}:
            return self.store.list_asns(dataset_name)
        return os.listdir(INTERMEDIATE_DIR / dataset_name)

    def read_dataset(self, dataset_name, asn):
        if self.store and dataset_name in {'rib', 'test'}:
            return self.store.get_payload(dataset_name, asn) or ''
        path = INTERMEDIATE_DIR / dataset_name / asn
        if not path.exists():
            return ''
        return path.read_text(encoding='utf-8', errors='ignore')

    def simulate(self, asn):
        self.AS_path = defaultdict(list)
        self.AS_path[asn] = [tuple([asn])]
        self.spread_score = defaultdict(float)
        self.spread_score[tuple([asn])] = 1
        self.path_flag = dict()
        self.path_flag[tuple([asn])] = 0
        self.ms.simulate(asn)
        self.buffer = set([tuple([asn])])
        self.stream(asn)
        
    def condition(self, current, neighbor):
        if neighbor in current or len(current) >= 9:
            return False
        slen = self.ms.shortest[(current[0], neighbor)] if (current[0], neighbor) in self.ms.shortest else len(current) + 1
        if len(current) + 1 - slen > 2:
            return False

        valleyfree = False
        if self.path_flag[current] == 0:
            valleyfree = True
        elif self.path_flag[current] != 0 and neighbor in self.ms.customer[current[-1]]:
            valleyfree = True
        return valleyfree

    def get_model(self, paths, model_name):
        atts = list()
        for path in paths:
            atts.append(self.ms.get_sample(path))
        atts = np.array(atts)
        predict = self.model[model_name].predict_proba(atts)
        return (list([value[1] for value in predict]))

    def add_path(self, paths, candicate):
        for model_name in paths:
            result = self.get_model(paths[model_name], model_name)
            for i in range(len(result)):
                neighbor = paths[model_name][i][-1]
                self.spread_score[paths[model_name][i]] = result[i]

                if len(self.AS_path[neighbor]) == 0:
                    self.AS_path[neighbor].append(paths[model_name][i])
                    candicate.add(paths[model_name][i])
                else:
                    if self.spread_score[self.AS_path[neighbor][0]] <= self.threshold and self.spread_score[self.AS_path[neighbor][0]] < result[i]:
                        self.AS_path[neighbor][0] = paths[model_name][i]
                        candicate.add(paths[model_name][i])
                    elif self.spread_score[self.AS_path[neighbor][0]] > self.threshold and result[i] > self.threshold:
                        self.AS_path[neighbor].append(paths[model_name][i])
                        candicate.add(paths[model_name][i])
        return candicate

    def get_model_name(self, right_path):
        if right_path[-1] in self.ms.core_cluster:
            cate = 'core'
        else:
            cate = 'shell'
        slen = self.ms.shortest[(right_path[0], right_path[-1])] if (right_path[0], right_path[-1]) in self.ms.shortest else len(right_path)
        
        if slen <= 3:
            return cate + '1'
        elif slen <= 5:
            return cate + '2'
        else:
            return cate + '3'

    def stream(self, asn):
        active, candicate = [tuple([asn])], set()
        relationship_list = [self.ms.provider, self.ms.peer, self.ms.customer]
        
        for _ in range(9):
            paths = defaultdict(list)
            for i in range(3):
                relationship = relationship_list[i]
                for current in active:
                    for neighbor in relationship[current[-1]]:
                        if self.condition(current, neighbor):
                            right_path = tuple(list(current) + [neighbor])
                            self.path_flag[right_path] = i
                            paths[self.get_model_name(right_path)].append(right_path)
            candicate = self.add_path(paths, candicate)
            active = list(candicate)
            active.sort(key = lambda x: int(x[-1]))
            candicate = set()

    def evaluation(self, asn, accuracy):
        predict_right, predict_wrong = 0, 0
        recall_right, recall_wrong = 0, 0

        for key in self.valid_path_dict:
            for path in self.valid_path_dict[key]:
                if key in self.AS_path and path in self.AS_path[key]:
                    recall_right += 1
                else:
                    recall_wrong += 1

        for key in self.AS_path:
            if key in self.valid_path_dict:
                for path in self.AS_path[key]:
                    if path in self.valid_path_dict[key]:
                        predict_right += 1
                    else:
                        predict_wrong += 1
        print("AS " + asn + "\'s accuracy: " + str(predict_right / (predict_right + predict_wrong + 1e-5)) + ", completeness: " + str(recall_right / (recall_right + recall_wrong + 1e-5)))
        accuracy.put([predict_right, predict_wrong, recall_right, recall_wrong])

    def read_file(self, file, accuracy):
        self.model = dict()
        for cate in ['core', 'shell']:
            for num in ['1', '2', '3']:
                model_name = cate + num
                self.model[model_name] = pickle.load(open(TRAINED_DIR / 'models' / ('xgboost_' + str(self.rounds) + '_' + model_name + '.model'), 'rb'))
            
        self.valid_path_dict = dict()
        if file in self.valid_files:
            payload = self.read_dataset(self.valid_dataset, file)
            for line in payload.splitlines():
                paths = line.strip().split('&')[0].split(' ')
                for path in paths:
                    ases = tuple(path.split('|')[::-1])
                    if len(ases) < 2:
                        continue
                    if ases[-1] not in self.valid_path_dict:
                        self.valid_path_dict[ases[-1]] = [ases]
                    elif ases not in self.valid_path_dict[ases[-1]]:
                        self.valid_path_dict[ases[-1]].append(ases)

        self.simulate(file)
        self.evaluation(file, accuracy)

        if self.modes == 'train':
            result_dir = TESTING_DIR / ('result_round' + str(self.rounds))
            if not result_dir.exists():
                result_dir.mkdir(parents=True, exist_ok=True)
            output = set()
            for key in self.valid_path_dict:
                for path in self.AS_path[key]:
                    output.add('|'.join(path[::-1]) + ' ' + str(self.spread_score[path]))

            target = result_dir / (file + '.txt')
            temporary = result_dir / (file + '.txt.tmp')
            with open(temporary, 'w') as fout:
                fout.write('\n'.join(output))
            os.replace(temporary, target)

    def run(self):
        self.valid_dataset = 'rib' if self.modes == "train" else 'test'
        self.valid_files = self.list_dataset(self.valid_dataset)
        random.shuffle(self.valid_files)
        if self.quick_mode == 1:
            if self.rounds > 0 and self.modes == 'train':
                self.valid_files = os.listdir(TESTING_DIR / 'result_round0')
                self.valid_files = [self.valid_files[i][:-4] for i in range(len(self.valid_files))]
            else:
                self.valid_files = self.valid_files[:1000]

        if self.modes == 'train':
            result_dir = TESTING_DIR / ('result_round' + str(self.rounds))
            completed = {path.stem for path in result_dir.glob('*.txt')}
            self.valid_files = [file for file in self.valid_files if file not in completed]
            print(f'[resume] completed={len(completed)} remaining={len(self.valid_files)} workers={self.workers}', flush=True)

        processes, result, accuracy = list(), [0 for _ in range(4)], Queue()
        for file in self.valid_files:
            processes.append(Process(target=self.read_file, args=(file, accuracy)))
            if len(processes) == self.workers:
                [x.start() for x in processes]
                [x.join() for x in processes]
                processes = list()

                while accuracy.qsize() != 0:
                    info = accuracy.get()
                    for j in range(4):
                        result[j] += info[j]
                print('------------------------------------------------------------------------------------')
                print("overall accuracy: " + str(result[0] / (result[0] + result[1])) + ", completeness: " + str(result[2] / (result[2] + result[3])))
                
        [x.start() for x in processes]
        [x.join() for x in processes]

        while accuracy.qsize() != 0:
            info = accuracy.get()
            for j in range(4):
                result[j] += info[j]
        if result[0] + result[1] > 0 and result[2] + result[3] > 0:
            print('------------------------------------------------------------------------------------')
            print("overall accuracy: " + str(result[0] / (result[0] + result[1])) + ", completeness: " + str(result[2] / (result[2] + result[3])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predicting')
    parser.add_argument('-r', '--rounds', type=int, required=True)
    parser.add_argument('-t', '--threshold', type=float, required=True)
    parser.add_argument('-m', '--modes', required=True)
    parser.add_argument('-q', '--quick_mode', type=int, required=True)
    parser.add_argument('-w', '--workers', type=int, default=8)
    args = parser.parse_args()

    rt = Routingtree(args.rounds, args.threshold, args.modes, args.quick_mode, args.workers)
    rt.run()
