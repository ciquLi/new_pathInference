"""Execution-only PathRadar acceleration; numerical routing methods stay upstream."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from types import MethodType

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import pathradar_prediction as original
from data_store import DB_PATH, PathRadarStore


class IndexedOrder(tuple):
    """Preserve list membership/first-index semantics for immutable link order."""
    def __new__(cls, values=()):
        obj = super().__new__(cls, values)
        obj.positions = {}
        for index, value in enumerate(obj):
            obj.positions.setdefault(value, index)
        return obj

    def __contains__(self, value):
        return value in self.positions

    def index(self, value, *bounds):
        if bounds:
            return super().index(value, *bounds)
        try:
            return self.positions[value]
        except KeyError:
            raise ValueError("value is not in link order") from None


def index_link_order(sample):
    sample.linkorder = defaultdict(IndexedOrder,
                                  ((key, IndexedOrder(values)) for key, values in sample.linkorder.items()))


def flat_path_stream(self, asn):
    """Original shortest-path traversal with shallow copies of immutable ASNs."""
    active, candicate = [tuple([asn])], []
    relationships = [self.provider, self.peer, self.customer]
    while len(active) > 0:
        for i in range(3):
            for current in active:
                for neighbor in relationships[i][current[-1]]:
                    if neighbor == asn:
                        continue
                    if self.condition(current, neighbor):
                        right_path = list(current) + [neighbor]
                        if len(self.AS_path[neighbor]) == 0 or len(right_path) < len(self.AS_path[neighbor]):
                            self.AS_path[neighbor] = right_path.copy()
                            self.path_flag[tuple(right_path)] = i
                            candicate.append(tuple(right_path))
                            if len(self.worst[neighbor]) == 0 or self.smaller((current[-1], neighbor), (self.worst[neighbor][-2], self.worst[neighbor][-1])):
                                self.worst[neighbor] = right_path.copy()
                        elif self.smaller((current[-1], neighbor), (self.worst[neighbor][-2], self.worst[neighbor][-1])):
                            self.worst[neighbor] = right_path.copy()
                            self.path_flag[tuple(right_path)] = i
                            candicate.append(tuple(right_path))
        active = candicate.copy()
        active.sort(key=lambda x: int(x[-1]))
        candicate = []


def accelerate_sample(sample):
    index_link_order(sample)
    sample.stream = MethodType(flat_path_stream, sample)


class Counts:
    def put(self, value):
        self.value = value


class PersistentTree(original.Routingtree):
    def load_models(self):
        self.model = {}
        for kind in ("core", "shell"):
            for number in ("1", "2", "3"):
                name = kind + number
                with (original.TRAINED_DIR / "models" / f"xgboost_{self.rounds}_{name}.model").open("rb") as stream:
                    self.model[name] = pickle.load(stream)

    def read_file(self, file, accuracy):
        # Same path parsing, numerical calls, scores and atomic publication as
        # Routingtree.read_file; only the model reload is moved to initialization.
        self.valid_path_dict = {}
        if file in self.valid_files:
            for line in self.read_dataset(self.valid_dataset, file).splitlines():
                for path in line.strip().split('&')[0].split(' '):
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
            directory = original.TESTING_DIR / ('result_round' + str(self.rounds))
            directory.mkdir(parents=True, exist_ok=True)
            output = set()
            for key in self.valid_path_dict:
                for path in self.AS_path[key]:
                    output.add('|'.join(path[::-1]) + ' ' + str(self.spread_score[path]))
            target = directory / (file + '.txt')
            temporary = directory / (file + '.txt.tmp')
            temporary.write_text('\n'.join(output))
            os.replace(temporary, target)


_TREE = None


def initialize_worker():
    # Each worker owns its SQLite connection; the large immutable feature maps
    # were loaded once in the parent and inherited using Linux copy-on-write.
    _TREE.store = PathRadarStore(DB_PATH) if DB_PATH.exists() else None
    _TREE.load_models()
    gc.enable()


def predict_destination(asn):
    started = time.monotonic()
    counts = Counts()
    _TREE.read_file(asn, counts)
    return dict(asn=asn, counts=counts.value, seconds=time.monotonic() - started)


def run(rounds, threshold, modes, workers):
    global _TREE
    started = time.monotonic()
    _TREE = PersistentTree(rounds, threshold, modes, 0, workers)
    accelerate_sample(_TREE.ms)
    _TREE.valid_dataset = 'rib' if modes == 'train' else 'test'
    all_files = _TREE.list_dataset(_TREE.valid_dataset)
    _TREE.valid_files = set(all_files)
    directory = original.TESTING_DIR / ('result_round' + str(rounds))
    completed = {path.stem for path in directory.glob('*.txt')} if modes == 'train' else set()
    pending = [asn for asn in all_files if asn not in completed]
    if _TREE.store:
        _TREE.store.close()
        _TREE.store = None
    print(json.dumps(dict(event='resume', completed=len(completed), remaining=len(pending),
                          workers=workers, initialized_seconds=time.monotonic()-started)), flush=True)
    if not pending:
        return
    gc.collect()
    gc.freeze()
    totals = [0] * 4
    done = 0
    try:
        with mp.get_context('fork').Pool(workers, initializer=initialize_worker, maxtasksperchild=256) as pool:
            for result in pool.imap_unordered(predict_destination, pending, chunksize=1):
                done += 1
                totals = [a+b for a, b in zip(totals, result['counts'])]
                print(json.dumps(dict(event='destination_complete', **result, completed=len(completed)+done,
                                      total=len(all_files), totals=totals)), flush=True)
    finally:
        gc.unfreeze()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--rounds', required=True, type=int)
    parser.add_argument('-t', '--threshold', required=True, type=float)
    parser.add_argument('-m', '--modes', required=True, choices=('train', 'test'))
    parser.add_argument('-q', '--quick_mode', type=int, default=0, choices=(0,))
    parser.add_argument('-w', '--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    run(args.rounds, args.threshold, args.modes, args.workers)


if __name__ == '__main__':
    main()
