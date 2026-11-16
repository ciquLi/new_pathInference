"""Load large frequency evidence once, directly into fork-shared native tables."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType

import pipeline_engine as previous
from build_loader import OUTPUT

record = json.loads((OUTPUT/'BUILD.json').read_text())
for path, expected in [(Path(__file__).with_name('loader_native.cpp'), record['source_sha256']),
                       (Path(record['binary']), record['binary_sha256'])]+[(Path(p), h) for p, h in record['dependencies_sha256'].items()]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Streaming loader build identity changed: '+str(path))
sys.path.insert(0, str(OUTPUT))
import _pathradar_loader as native

engine = previous.engine


class RoutingEvidenceSample(engine.original.Model_Sample):
    def __init__(self):
        self.prev_frequency, self.next_frequency = defaultdict(int), defaultdict(int)
        self.linkfreq = defaultdict(int)
        self.linkorder = defaultdict(list)
        directory = engine.original.INTERMEDIATE_DIR
        with (directory/'linkfreq_data.txt').open() as stream:
            for line in stream:
                link, count = line.strip().split(' ')
                self.linkfreq[tuple(link.split('|'))] = int(count)
        with (directory/'linkorder_data.txt').open() as stream:
            for line in stream:
                node, neighbors = line.strip().split(' ')
                self.linkorder[node] = neighbors.split('|')
        self.relationship = {}
        self.provider, self.peer, self.customer = (defaultdict(set) for _ in range(3))
        with (directory/'asrel_rib.txt').open() as stream:
            for line in stream:
                if line.startswith('#'):
                    continue
                a, b, relation = line.strip().split('|')
                if relation == '-1':
                    self.relationship[a, b] = 'p2c'
                    self.relationship[b, a] = 'c2p'
                    self.customer[a].add(b)
                    self.provider[b].add(a)
                else:
                    self.relationship[a, b] = self.relationship[b, a] = 'p2p'
                    self.peer[a].add(b)
                    self.peer[b].add(a)
        self.core_cluster = set()
        with (directory/'core_shell.txt').open() as stream:
            for line in stream:
                info = line.strip().split(' ')
                if int(info[1]) > 1000:
                    self.core_cluster.add(info[0])


def enable_sample(sample, release=True):
    sample._integer_graph = previous.previous.previous.native.pack(sample)
    sample._feature_evidence = native.pack_file(sample, str(engine.original.INTERMEDIATE_DIR/'joinfreq_data.txt'))
    sample._candidate_graph = native.pack_adjacency(sample._feature_evidence, sample)
    sample.__class__ = previous.PipelineSample
    sample.stream = MethodType(previous.previous.previous.shortest, sample)
    if release:
        previous.release_evidence(sample)


def configure():
    previous.native = native
    engine.original.Model_Sample = RoutingEvidenceSample
    engine.PersistentTree = previous.PipelineTree
    engine.accelerate_sample = enable_sample


def main():
    configure()
    engine.main()


if __name__ == '__main__':
    main()
