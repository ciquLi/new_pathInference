"""Native feature evidence/matrices; original XGBoost models and calls unchanged."""
import hashlib
import json
from pathlib import Path
import sys

import shortest_engine as previous
from build_features import OUTPUT

record=json.loads((OUTPUT/'BUILD.json').read_text())
for path,key in [(Path(__file__).with_name('features_native.cpp'),'source_sha256'),(Path(record['binary']),'binary_sha256')]:
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record[key]:
        raise ValueError('Native feature build identity changed: '+str(path))
sys.path.insert(0,str(OUTPUT))
import _pathradar_features as native

np=previous.bridge.engine.original.np


def matrix(sample,paths):
    buffer,rows,columns=native.features(sample._feature_evidence,sample._feature_distances,paths)
    return np.frombuffer(buffer,dtype=np.int64).reshape(rows,columns)


class FeatureSample(previous.IntegerSample):
    def simulate(self,asn):
        super().simulate(asn)
        self._feature_distances=native.distances(self._feature_evidence,self.shortest,asn)


def enable_sample(sample):
    previous.enable_sample(sample)
    sample._feature_evidence=native.pack(sample)
    sample.__class__=FeatureSample


class FeatureTree(previous.IntegerTree):
    def get_model(self,paths,model_name):
        atts=matrix(self.ms,paths)
        predict=self.model[model_name].predict_proba(atts)
        return [value[1] for value in predict]


def main():
    previous.bridge.engine.PersistentTree=FeatureTree
    previous.bridge.engine.accelerate_sample=enable_sample
    previous.bridge.engine.main()


if __name__=='__main__':
    main()
