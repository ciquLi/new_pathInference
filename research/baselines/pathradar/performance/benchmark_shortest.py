"""Compare the live CPython bridge with integer-graph shortest paths on real data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import shortest_engine as fast
from benchmark_native import digest,state


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shortest_state(sample):
    return {name:digest(list(getattr(sample,name).items()))
            for name in ('AS_path','worst','path_flag','shortest')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asns',nargs='+',default=['19529','328840','47608','141295'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    base=fast.bridge.NativeTree(1,.5,'train',0,1)
    fast.bridge.enable(base)
    base.valid_dataset='rib'
    base.valid_files=set(base.list_dataset('rib'))
    base.load_models()
    pack_started=time.monotonic()
    base.ms._integer_graph=fast.native.pack(base.ms)
    pack_seconds=time.monotonic()-pack_started
    here=Path(__file__).resolve().parent
    report=dict(status='RUNNING',reference='verified live native bridge',
        initialization_seconds=time.monotonic()-started,pack_seconds=pack_seconds,
        comparisons=[],code_sha256={str(p):sha256(p) for p in (
            here/'shortest_native.cpp',here/'shortest_engine.py',here/'benchmark_shortest.py',
            here/'native.cpp',here/'native_engine.py',here/'engine.py',here.parent/'pathradar_prediction.py')},
        build_sha256={str(p):sha256(p) for p in (fast.OUTPUT/'BUILD.json',fast.bridge.OUTPUT/'BUILD.json')},
        model_sha256={str(p):sha256(p) for p in (fast.bridge.engine.original.TRAINED_DIR/'models').glob('xgboost_1_*.model')})
    print(json.dumps(dict(event='initialized',seconds=report['initialization_seconds'],pack_seconds=pack_seconds)),flush=True)
    for asn in args.asns:
        outcomes={}
        for name,stream in [('bridge',fast.bridge.native_shortest),('integer',fast.shortest)]:
            base.ms.__class__=fast.bridge.engine.original.Model_Sample if name=='bridge' else fast.IntegerSample
            base.ms.stream=MethodType(stream,base.ms)
            started=time.monotonic();base.ms.simulate(asn);seconds=time.monotonic()-started
            shortest_result=dict(seconds=seconds,state=shortest_state(base.ms))
            fast.bridge.engine.original.TESTING_DIR=args.output/name
            sink=fast.bridge.engine.Counts()
            started=time.monotonic();base.read_file(asn,sink);elapsed=time.monotonic()-started
            output=fast.bridge.engine.original.TESTING_DIR/'result_round1'/f'{asn}.txt'
            outcomes[name]=dict(seconds=elapsed,counts=sink.value,state=state(base),
                                path_scores_sha256=digest(sorted(output.read_text().splitlines())),
                                shortest=shortest_result)
            print(json.dumps(dict(asn=asn,implementation=name,seconds=elapsed,shortest_seconds=seconds)),flush=True)
        reference,integer=outcomes['bridge'],outcomes['integer']
        equal=all(reference[key]==integer[key] for key in ('counts','state','path_scores_sha256'))
        equal=equal and reference['shortest']['state']==integer['shortest']['state']
        row=dict(asn=asn,equal=equal,speedup=reference['seconds']/integer['seconds'],
                 shortest_speedup=reference['shortest']['seconds']/integer['shortest']['seconds'],**outcomes)
        report['comparisons'].append(row)
        (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(asn=asn,equal=equal,speedup=row['speedup'],shortest_speedup=row['shortest_speedup'])),flush=True)
        if not equal:
            raise ValueError('Integer graph state or predictions differ; deployment prohibited')
    report['status']='PASS'
    report['total_speedup']=sum(r['bridge']['seconds'] for r in report['comparisons'])/sum(r['integer']['seconds'] for r in report['comparisons'])
    report['total_shortest_speedup']=sum(r['bridge']['shortest']['seconds'] for r in report['comparisons'])/sum(r['integer']['shortest']['seconds'] for r in report['comparisons'])
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',speedup=report['total_speedup'],shortest_speedup=report['total_shortest_speedup'])),flush=True)


if __name__=='__main__':
    main()
