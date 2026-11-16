"""Ordered native candidate tables, immutable evidence sharing, unchanged XGBoost."""
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType

import features_engine as previous
from build_pipeline import OUTPUT

record = json.loads((OUTPUT/'BUILD.json').read_text())
for path, expected in [(Path(__file__).with_name('pipeline_native.cpp'), record['source_sha256']),
                       (Path(record['binary']), record['binary_sha256'])]+[(Path(p), h) for p, h in record['dependencies_sha256'].items()]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Pipeline build identity changed: '+str(path))
sys.path.insert(0, str(OUTPUT))
import _pathradar_pipeline as native

engine = previous.previous.bridge.engine
np = engine.original.np


class PipelineSample(previous.previous.IntegerSample):
    def simulate(self, asn):
        super().simulate(asn)
        self._feature_distances = native.distances(self._feature_evidence, self.shortest, asn)


def enable_sample(sample, release=False):
    sample._integer_graph = previous.previous.native.pack(sample)
    sample._feature_evidence = native.pack(sample)
    sample._candidate_graph = native.pack_adjacency(sample._feature_evidence, sample)
    sample.__class__ = PipelineSample
    sample.stream = MethodType(previous.previous.shortest, sample)
    if release:
        release_evidence(sample)


def release_evidence(sample):
    """Only native consumers remain; preserve graph for optional diagnostics."""
    for name in ('prev_frequency', 'next_frequency', 'linkfreq', 'linkorder', 'relationship', 'graph'):
        sample.__dict__.pop(name, None)
    gc.collect()
    # Return freed libc arenas where supported, before fork; no routing state changes.
    import ctypes
    libc = ctypes.CDLL(None)
    trim = getattr(libc, 'malloc_trim', None)
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)


class PipelineTree(previous.FeatureTree):
    def get_model(self, paths, model_name):
        buffer, rows, columns = native.features(self.ms._feature_evidence, self.ms._feature_distances, paths)
        atts = np.frombuffer(buffer, dtype=np.int64).reshape(rows, columns)
        predict = self.model[model_name].predict_proba(atts)
        return [value[1] for value in predict]

    def stream(self, asn):
        active = [(asn,)]
        for _ in range(9):
            paths = native.expand(self.ms._candidate_graph, self.ms._feature_distances, self, active)
            candidate = native.add_paths(self, paths, set())
            active = list(candidate)
            active.sort(key=lambda path: int(path[-1]))


def main():
    engine.PersistentTree = PipelineTree
    engine.accelerate_sample = lambda sample: enable_sample(sample, release=True)
    engine.main()


if __name__ == '__main__':
    main()
