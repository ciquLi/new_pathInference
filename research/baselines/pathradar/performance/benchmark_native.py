"""Real-input equivalence and end-to-end timing for native PathRadar loops."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import native_engine as fast

engine=fast.engine
np=engine.original.np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CheckedTree(fast.NativeTree):
    def get_model(self,paths,model_name):
        actual=np.array(fast.native.features(self.ms,paths))
        expected=np.array([self.ms.get_sample(path) for path in paths])
        if not np.array_equal(actual,expected):
            raise ValueError('Native features differ from Python')
        self.verified_features+=len(paths)
        prediction=self.model[model_name].predict_proba(actual)
        return [value[1] for value in prediction]

    def stream(self,asn):
        active=[(asn,)]
        for _ in range(9):
            actual=fast.native.expand(self,active)
            expected=defaultdict(list)
            before=dict(self.path_flag)
            for relation,neighbors in enumerate((self.ms.provider,self.ms.peer,self.ms.customer)):
                for current in active:
                    for neighbor in neighbors[current[-1]]:
                        if self.condition(current,neighbor):
                            path=current+(neighbor,)
                            self.path_flag[path]=relation
                            expected[self.get_model_name(path)].append(path)
            if list(actual.items())!=list(expected.items()) or before!=self.path_flag:
                raise ValueError('Native expansion ordering or flags differ from Python')
            self.verified_layers+=1
            candidate=fast.native.add_paths(self,actual,set())
            active=list(candidate)
            active.sort(key=lambda path:int(path[-1]))


def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def state(tree):
    return dict(
        shortest=digest(list(tree.ms.shortest.items())),
        shortest_paths=digest(list(tree.ms.AS_path.items())),
        shortest_worst=digest(list(tree.ms.worst.items())),
        shortest_flags=digest(list(tree.ms.path_flag.items())),
        candidate_paths=digest(list(tree.AS_path.items())),
        candidate_scores=digest(list(tree.spread_score.items())),
        candidate_flags=digest(list(tree.path_flag.items())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--asns',nargs='+',default=['19529','328840','47608','141295'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    initialization=time.monotonic()
    base=engine.PersistentTree(args.round,.5,'train',0,1)
    engine.index_link_order(base.ms)
    base.valid_dataset='rib'
    base.valid_files=set(base.list_dataset('rib'))
    base.load_models()
    native=fast.NativeTree.__new__(fast.NativeTree);native.__dict__.update(base.__dict__)
    checked=CheckedTree.__new__(CheckedTree);checked.__dict__.update(base.__dict__)
    report=dict(status='RUNNING',rounds=args.round,threshold=.5,
                reference='persistent Python adapter, verified against original PathRadar',
                initialization_seconds=time.monotonic()-initialization,
                code_sha256={str(p):sha256(p) for p in [HERE/'native.cpp',HERE/'native_engine.py',
                            HERE/'engine.py',HERE/'benchmark_native.py',engine.SOURCE/'pathradar_prediction.py']},
                build_sha256=sha256(fast.OUTPUT/'BUILD.json'),
                model_sha256={str(p):sha256(p) for p in (engine.original.TRAINED_DIR/'models').glob(f'xgboost_{args.round}_*.model')},
                comparisons=[])
    print(json.dumps(dict(initialization_seconds=report['initialization_seconds'])),flush=True)
    for asn in args.asns:
        outcomes={}
        for name,tree in [('python',base),('native_checked',checked),('native',native)]:
            tree.ms.stream=MethodType(engine.flat_path_stream if name=='python' else fast.native_shortest,tree.ms)
            tree.verified_features=tree.verified_layers=0
            engine.original.TESTING_DIR=args.output/name
            sink=engine.Counts()
            started=time.monotonic();tree.read_file(asn,sink);elapsed=time.monotonic()-started
            output=engine.original.TESTING_DIR/f'result_round{args.round}'/f'{asn}.txt'
            rows=output.read_text().splitlines()
            outcomes[name]=dict(seconds=elapsed,counts=sink.value,rows=len(rows),
                                path_scores_sha256=digest(sorted(rows)),state=state(tree),
                                verified_features=tree.verified_features,verified_layers=tree.verified_layers)
            print(json.dumps(dict(asn=asn,implementation=name,seconds=elapsed,verified_features=tree.verified_features)),flush=True)
        expected=outcomes['python']
        equal=all(outcomes[name][key]==expected[key] for name in ('native_checked','native')
                  for key in ('counts','rows','path_scores_sha256','state'))
        result=dict(asn=asn,equal=equal,speedup=expected['seconds']/outcomes['native']['seconds'],**outcomes)
        report['comparisons'].append(result)
        (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(asn=asn,equal=equal,speedup=result['speedup'])),flush=True)
        if not equal:
            raise ValueError('Native routing state or predictions differ; deployment prohibited')
    report['status']='PASS'
    report['total_speedup']=sum(r['python']['seconds'] for r in report['comparisons'])/sum(r['native']['seconds'] for r in report['comparisons'])
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',total_speedup=report['total_speedup'])),flush=True)


if __name__=='__main__':
    main()
