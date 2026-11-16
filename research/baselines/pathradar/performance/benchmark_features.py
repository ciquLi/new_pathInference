"""Verify every real feature value and full predictions, then time native matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import features_engine as fast
from benchmark_native import digest,state

np=fast.np


class MeasuredReference(fast.previous.IntegerTree):
    def get_model(self,paths,model_name):
        started=time.monotonic()
        atts=np.array(fast.previous.bridge.native.features(self.ms,paths))
        self.feature_seconds+=time.monotonic()-started
        started=time.monotonic()
        predict=self.model[model_name].predict_proba(atts)
        self.prediction_seconds+=time.monotonic()-started
        return [value[1] for value in predict]


class MeasuredNative(fast.FeatureTree):
    def get_model(self,paths,model_name):
        started=time.monotonic()
        atts=fast.matrix(self.ms,paths)
        self.feature_seconds+=time.monotonic()-started
        started=time.monotonic()
        predict=self.model[model_name].predict_proba(atts)
        self.prediction_seconds+=time.monotonic()-started
        return [value[1] for value in predict]


class CheckedNative(MeasuredNative):
    def get_model(self,paths,model_name):
        actual=fast.matrix(self.ms,paths)
        expected=np.array([self.ms.get_sample(path) for path in paths])
        if actual.dtype!=expected.dtype or not np.array_equal(actual,expected):
            raise ValueError('Native feature matrix differs from original Python')
        self.verified_features+=len(paths)
        self.verified_values+=actual.size
        predict=self.model[model_name].predict_proba(actual)
        return [value[1] for value in predict]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asns',nargs='+',default=['19529','328840','47608','141295'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    base=MeasuredReference(1,.5,'train',0,1)
    fast.previous.enable_sample(base.ms)
    base.valid_dataset='rib'
    base.valid_files=set(base.list_dataset('rib'))
    base.load_models()
    pack_started=time.monotonic()
    base.ms._feature_evidence=fast.native.pack(base.ms)
    pack_seconds=time.monotonic()-pack_started
    checked=CheckedNative.__new__(CheckedNative);checked.__dict__.update(base.__dict__)
    native=MeasuredNative.__new__(MeasuredNative);native.__dict__.update(base.__dict__)
    here=Path(__file__).resolve().parent
    report=dict(status='RUNNING',reference='verified live integer-graph predictor',
        initialization_seconds=time.monotonic()-started,pack_seconds=pack_seconds,comparisons=[],
        code_sha256={str(p):sha256(p) for p in (here/'features_native.cpp',here/'features_engine.py',
            here/'benchmark_features.py',here/'shortest_native.cpp',here/'shortest_engine.py',
            here/'native.cpp',here/'native_engine.py',here/'engine.py',here.parent/'pathradar_prediction.py')},
        build_sha256={str(p):sha256(p) for p in (fast.OUTPUT/'BUILD.json',fast.previous.OUTPUT/'BUILD.json',fast.previous.bridge.OUTPUT/'BUILD.json')},
        model_sha256={str(p):sha256(p) for p in (fast.previous.bridge.engine.original.TRAINED_DIR/'models').glob('xgboost_1_*.model')})
    print(json.dumps(dict(event='initialized',seconds=report['initialization_seconds'],pack_seconds=pack_seconds)),flush=True)
    for asn in args.asns:
        outcomes={}
        for name,tree in [('integer',base),('features_checked',checked),('features',native)]:
            tree.ms.__class__=fast.previous.IntegerSample if name=='integer' else fast.FeatureSample
            tree.feature_seconds=tree.prediction_seconds=tree.verified_features=tree.verified_values=0
            fast.previous.bridge.engine.original.TESTING_DIR=args.output/name
            sink=fast.previous.bridge.engine.Counts()
            started=time.monotonic();tree.read_file(asn,sink);elapsed=time.monotonic()-started
            output=fast.previous.bridge.engine.original.TESTING_DIR/'result_round1'/f'{asn}.txt'
            rows=output.read_text().splitlines()
            outcomes[name]=dict(seconds=elapsed,feature_seconds=tree.feature_seconds,
                prediction_seconds=tree.prediction_seconds,counts=sink.value,rows=len(rows),
                path_scores_sha256=digest(sorted(rows)),state=state(tree),
                verified_features=tree.verified_features,verified_values=tree.verified_values)
            print(json.dumps(dict(asn=asn,implementation=name,seconds=elapsed,
                feature_seconds=tree.feature_seconds,verified_features=tree.verified_features)),flush=True)
        expected=outcomes['integer']
        equal=all(outcomes[name][key]==expected[key] for name in ('features_checked','features')
                  for key in ('counts','rows','path_scores_sha256','state'))
        row=dict(asn=asn,equal=equal,speedup=expected['seconds']/outcomes['features']['seconds'],
                 feature_speedup=expected['feature_seconds']/outcomes['features']['feature_seconds'],**outcomes)
        report['comparisons'].append(row)
        (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(asn=asn,equal=equal,speedup=row['speedup'],feature_speedup=row['feature_speedup'])),flush=True)
        if not equal:
            raise ValueError('Native feature predictions or state differ; deployment prohibited')
    report['status']='PASS'
    report['total_speedup']=sum(r['integer']['seconds'] for r in report['comparisons'])/sum(r['features']['seconds'] for r in report['comparisons'])
    report['total_feature_speedup']=sum(r['integer']['feature_seconds'] for r in report['comparisons'])/sum(r['features']['feature_seconds'] for r in report['comparisons'])
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',speedup=report['total_speedup'],feature_speedup=report['total_feature_speedup'])),flush=True)


if __name__=='__main__':
    main()
