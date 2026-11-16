"""Compare original and accelerated predictions using identical real inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine


def signature(lines):
    return hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--asns', nargs='+', default=['19529', '328840', '47608'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base = engine.original.Routingtree(args.round, .5, 'train', 0, 1)
    base.valid_dataset = 'rib'
    base.valid_files = set(base.list_dataset('rib'))
    fast = engine.PersistentTree.__new__(engine.PersistentTree)
    fast.__dict__.update(base.__dict__)
    original_order = base.ms.linkorder
    engine.index_link_order(base.ms)
    indexed_order = base.ms.linkorder
    get_sample = base.ms.get_sample
    current_digest = None
    def capture_features(path):
        features = get_sample(path)
        current_digest.update(json.dumps([path, features], separators=(',', ':')).encode())
        return features
    base.ms.get_sample = capture_features
    fast.load_models()
    report = dict(status='RUNNING', rounds=args.round, threshold=.5, initialization_seconds=time.monotonic()-started,
                  original_file=str(engine.SOURCE/'pathradar_prediction.py'),
                  original_sha256=hashlib.sha256((engine.SOURCE/'pathradar_prediction.py').read_bytes()).hexdigest(),
                  engine_sha256=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
                  model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (engine.original.TRAINED_DIR/'models').glob(f'xgboost_{args.round}_*.model')},
                  comparisons=[])
    print(json.dumps(dict(initialization_seconds=report['initialization_seconds'])), flush=True)
    for asn in args.asns:
        outcomes = {}
        for name, tree, order in [('original', base, original_order), ('accelerated', fast, indexed_order)]:
            tree.ms.linkorder = order
            tree.ms.stream = MethodType(engine.original.Model_Sample.stream if name=='original' else engine.flat_path_stream, tree.ms)
            engine.original.TESTING_DIR = args.output / name
            sink = engine.Counts()
            current_digest = hashlib.sha256()
            started = time.monotonic()
            tree.read_file(asn, sink)
            elapsed = time.monotonic()-started
            file = engine.original.TESTING_DIR / f'result_round{args.round}' / f'{asn}.txt'
            lines = file.read_text().splitlines()
            outcomes[name] = dict(seconds=elapsed, rows=len(lines), counts=sink.value,
                                  path_scores_sha256=signature(lines), features_sha256=current_digest.hexdigest(),
                                  shortest_sha256=signature(str(x) for x in tree.ms.shortest.items()))
        a,b = outcomes['original'], outcomes['accelerated']
        equal = all(a[k]==b[k] for k in ('rows','counts','path_scores_sha256','features_sha256','shortest_sha256'))
        result=dict(asn=asn,equal=equal,speedup=a['seconds']/b['seconds'],**outcomes)
        report['comparisons'].append(result)
        (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not equal:
            raise RuntimeError('Prediction equivalence failed; do not deploy')
    report['status']='PASS'
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
