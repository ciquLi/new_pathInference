import os, math, pickle, argparse
import networkx as nx
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
from pathlib import Path
from data_store import DB_PATH, PathRadarStore

REPO_DIR = Path(os.environ.get('PATHRADAR_RUN_DIR', Path(__file__).resolve().parent.parent))
INTERMEDIATE_DIR = Path(os.environ.get('PATHRADAR_INTERMEDIATE_DIR', REPO_DIR / 'intermediate'))
TRAINED_DIR = REPO_DIR / 'trained'
TESTING_DIR = REPO_DIR / 'testing'

class Trainer(object):
    def __init__(self, rounds, quick_mode):
        self.graph = nx.Graph()
        self.relationship = dict()
        self.provider = defaultdict(set)
        self.peer = defaultdict(set)
        self.customer = defaultdict(set)
        self.rounds = rounds
        self.quick_mode = quick_mode
        self.store = PathRadarStore(DB_PATH) if DB_PATH.exists() else None
        self.read_topo()
        self.cal_features()

    def read_topo(self):
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

    def cal_features(self):
        self.linkfreq = defaultdict(int)
        self.linkorder = defaultdict(list)
        self.prev_frequency, self.next_frequency = defaultdict(int), defaultdict(int)
        self.shortest = dict()

        print('[train] loading shortest data', flush=True)
        if self.store:
            for asn, payload in self.store.iter_dataset('shortest'):
                self.shortest[(asn, asn)] = 1
                for line in payload.splitlines():
                    path = line.strip().split('|')[::-1]
                    if len(path) < 2:
                        continue
                    if (path[0], path[-1]) not in self.shortest or len(path) < self.shortest[(path[0], path[-1])]:
                        self.shortest[(path[0], path[-1])] = len(path)
        else:
            shortest_dir = INTERMEDIATE_DIR / 'shortest'
            for file in os.listdir(shortest_dir):
                with open(shortest_dir / file) as f:
                    asn = file[:-4]
                    self.shortest[(asn, asn)] = 1
                    for line in f:
                        path = line.strip().split('|')[::-1]
                        if (path[0], path[-1]) not in self.shortest or len(path) < self.shortest[(path[0], path[-1])]:
                            self.shortest[(path[0], path[-1])] = len(path)

        print(f'[train] shortest pairs={len(self.shortest)}', flush=True)
        print('[train] loading linkfreq/linkorder/joinfreq', flush=True)
        with open(INTERMEDIATE_DIR / 'linkfreq_train.txt') as f:
            for line in f:
                [link, num] = line.strip().split(' ')
                self.linkfreq[tuple(link.split('|'))] = int(num)
            
        with open(INTERMEDIATE_DIR / 'linkorder_train.txt') as f:
            for line in f:
                [node, neighbor] = line.strip().split(' ')
                self.linkorder[node] = neighbor.split('|')

        with open(INTERMEDIATE_DIR / 'joinfreq_train.txt') as f:
            for line in f:
                [flag, key, num] = line.strip().split(' ')
                if flag == "prev":
                    self.prev_frequency[key] = int(num)
                else:
                    self.next_frequency[key] = int(num)
        print(
            f"[train] loaded linkfreq={len(self.linkfreq)} linkorder={len(self.linkorder)} "
            f"prev={len(self.prev_frequency)} next={len(self.next_frequency)}",
            flush=True,
        )

    def get_paths(self, name):
        sample_paths = defaultdict(set)
        if name == 'train':
            positive_name = 'train'
        else:
            positive_name = 'validation'
        negative_name = ['shortest']
        if self.rounds != 0:
            for i in range(self.rounds):
                negative_name.append(TESTING_DIR / ('result_round' + str(i)))
        avoid_name = ['rib']
        sqlite_datasets = {'train', 'validation', 'test', 'rib', 'shortest', 'data', 'origintest'}

        def list_dataset(dataset_name):
            if self.store and dataset_name in sqlite_datasets:
                return self.store.list_asns(dataset_name)
            if dataset_name == 'validation':
                return os.listdir(INTERMEDIATE_DIR / 'validation')
            if dataset_name == 'test':
                return os.listdir(INTERMEDIATE_DIR / 'test')
            if dataset_name == 'rib':
                return os.listdir(INTERMEDIATE_DIR / 'rib')
            if dataset_name == 'shortest':
                return [name[:-4] for name in os.listdir(INTERMEDIATE_DIR / 'shortest') if name.endswith('.txt')]
            return []

        def read_dataset(dataset_name, asn):
            if self.store and dataset_name in sqlite_datasets:
                return self.store.get_payload(dataset_name, asn) or ''
            if dataset_name == 'shortest':
                path = INTERMEDIATE_DIR / 'shortest' / f'{asn}.txt'
            else:
                path = INTERMEDIATE_DIR / dataset_name / asn
            if not path.exists():
                return ''
            return path.read_text(encoding='utf-8', errors='ignore')

        def add_sample(path, cate):
            if len(path) < 2:
                return
            slen = self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)
            if slen <= 3:
                num = '1'
            elif slen <= 5:
                num = '2'
            else:
                num = '3'
            if path[-1] in self.core_cluster:
                sample_paths['core' + num].add((tuple(path), cate))
            if path[-1] not in self.graph.nodes() or self.graph.degree(path[-1]) < 300:
                sample_paths['shell' + num].add((tuple(path), cate))

        unfinished_name = set(list_dataset(negative_name[0]))
        for i in range(1, len(negative_name)):
            if isinstance(negative_name[i], Path):
                unfinished_name &= {name[:-4] for name in os.listdir(negative_name[i]) if name.endswith('.txt')}
            else:
                unfinished_name &= set(list_dataset(negative_name[i]))
        filenames = list_dataset(positive_name)
        
        cnt = 0
        for file in filenames:
            if file not in unfinished_name:
                continue
            cnt += 1
            if cnt > 1000 and self.quick_mode == 1:
                break
            if cnt % 100 == 0 or cnt == 1:
                print(
                    f"[train] get_paths name={name} processed_files={cnt} "
                    f"sample_groups={len(sample_paths)}",
                    flush=True,
                )
            filestore = set()
            sample_tuple = set()
            payload = read_dataset(positive_name, file)
            for line in payload.splitlines():
                paths = line.strip().split('&')[0].split(' ')
                for path in paths:
                    path = path.split('|')[::-1]
                    add_sample(path, 1)
                    filestore.add(tuple(path))
                    sample_tuple.add(path[-1])

            for name in avoid_name:
                existing = set(list_dataset(name))
                if file in existing:
                    payload = read_dataset(name, file)
                    for line in payload.splitlines():
                        paths = line.strip().split('&')[0].split(' ')
                        for path in paths:
                            path = path.split('|')[::-1]
                            filestore.add(tuple(path))

            for name in negative_name:
                if isinstance(name, Path):
                    neg_path = name / (file + '.txt')
                    if not neg_path.exists():
                        continue
                    payload = neg_path.read_text(encoding='utf-8', errors='ignore')
                else:
                    payload = read_dataset(name, file)
                for line in payload.splitlines():
                    info = line.strip().split(' ')
                    path = tuple(info[0].split('|')[::-1])
                    score = 0.5
                    if len(info) > 1:
                        score = float(info[1])
                    if path[-1] in sample_tuple and path not in filestore and score <= 0.8:
                        add_sample(path, 0)
        print(
            f"[train] get_paths name={name} finished files={cnt} "
            + ' '.join(f"{k}={len(v)}" for k, v in sorted(sample_paths.items())),
            flush=True,
        )
        return sample_paths

    def get_sample(self, path, cluster):
        att = [len(path), self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)]
        att.extend([len(self.provider[path[-1]]), len(self.peer[path[-1]]), len(self.customer[path[-1]])])
        att.extend([len(self.provider[path[-2]]), len(self.peer[path[-2]]), len(self.customer[path[-2]])])

        if 'core' in cluster:
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
            if (path[i], path[i + 1]) in self.relationship:
                rel.append(self.relationship[(path[i], path[i + 1])])
            else:
                rel.append('p2p')
        att.extend([rel.count('c2p'), rel.count('p2p'), rel.count('p2c')])

        slen, slen1 = (self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)), (self.shortest[(path[0], path[-2])] if (path[0], path[-2]) in self.shortest else (len(path) - 1))
        att.extend([len(path) - slen, len(path) - 1 - slen1])

        return att

    def get_data(self, key):
        train_atts, train_label = [], []
        for path_pair in self.train_paths[key]:
            (path, ground_truth) = path_pair
            train_label.append(ground_truth)
            train_atts.append(self.get_sample(path, key))
        
        test_atts, test_label = [], []
        for path_pair in self.test_paths[key]:
            (path, ground_truth) = path_pair
            test_label.append(ground_truth)
            test_atts.append(self.get_sample(path, key))
        return train_atts, train_label, test_atts, test_label

    def run(self):
        self.train_paths, self.test_paths = self.get_paths('train'), self.get_paths('test')
        print('data load finished...', flush=True)
        models_dir = TRAINED_DIR / 'models'
        if not models_dir.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
        for cate in ['shell', 'core']:
            for num in ['1', '3', '2']:
                key = cate + num
                train_atts, train_label, test_atts, test_label = self.get_data(key)
                train_atts, train_label = np.array(train_atts), np.array(train_label)
                test_atts, test_label = np.array(test_atts), np.array(test_label)
                print(
                    f"[train] bucket={key} train_pos={sum(train_label)} "
                    f"train_neg={len(train_label) - sum(train_label)} "
                    f"eval_pos={sum(test_label)} eval_neg={len(test_label) - sum(test_label)}",
                    flush=True,
                )
                if sum(train_label) == 0:
                    continue

                params = {
                    'eval_metric': ['aucpr'],
                    'scale_pos_weight': (len(train_label) - sum(train_label)) / sum(train_label),
                    'use_label_encoder': False,
                    'n_jobs': max(1, int(os.environ.get('PATHRADAR_WORKERS', '10'))),
                }#, 'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'
                
                eval_set=[(train_atts, train_label), (test_atts, test_label)]
                model = XGBClassifier(**params)
                print(f"[train] fitting bucket={key} samples={len(train_label)}", flush=True)
                model.fit(train_atts, train_label, eval_set=eval_set, verbose=0)

                print(model.feature_importances_, flush=True)
                pickle.dump(model, open(models_dir / ('xgboost_' + str(self.rounds) + '_' + str(key) + '.model'), 'wb'))
                print(f'[train] stored bucket={key}', flush=True)
                true_negative, false_positive = 0, 0
                predictions = list(model.predict(test_atts))
                print(f"[train] bucket={key} eval_size={len(test_label)} eval_neg={len(test_label) - sum(test_label)}", flush=True)
                for i in range(len(predictions)):
                    predictions[i] = round(predictions[i])
                    if test_label[i] == 1 and predictions[i] == 0:
                        true_negative += 1
                    if test_label[i] == 0 and predictions[i] == 1:
                        false_positive += 1
                print(key, accuracy_score(test_label, predictions), true_negative, false_positive, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('-r', '--rounds', type=int, required=True)
    parser.add_argument('-q', '--quick_modes', type=int, required=True)
    args = parser.parse_args()
    
    trainer = Trainer(args.rounds, args.quick_modes)
    trainer.run()
