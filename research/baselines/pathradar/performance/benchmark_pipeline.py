"""Check each routing layer and real evaluation outputs before pipeline deployment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import pipeline_engine as fast
import evaluate_pipeline as evaluation
from benchmark_native import digest, state


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CheckedTree(fast.PipelineTree):
    def stream(self, asn):
        active = [(asn,)]
        for _ in range(9):
            before = dict(self.path_flag)
            expected = fast.previous.previous.bridge.native.expand(self, active)
            flags = list(self.path_flag.items())
            self.path_flag = before
            actual = fast.native.expand(self.ms._candidate_graph, self.ms._feature_distances, self, active)
            if list(expected.items()) != list(actual.items()) or flags != list(self.path_flag.items()):
                raise ValueError('Native candidate ordering or flags differ')
            self.verified_layers += 1
            # Call the original prediction API once per branch, compare every score/state.
            saved_routes = copy.deepcopy(self.AS_path)
            saved_scores = dict(self.spread_score)
            expected_next = fast.previous.previous.bridge.native.add_paths(self, actual, set())
            routes, scores = list(self.AS_path.items()), list(self.spread_score.items())
            self.AS_path = saved_routes
            self.spread_score.clear()
            self.spread_score.update(saved_scores)
            next_paths = fast.native.add_paths(self, actual, set())
            if (list(expected_next) != list(next_paths) or routes != list(self.AS_path.items()) or
                    scores != list(self.spread_score.items())):
                raise ValueError('Native route update differs')
            active = list(next_paths)
            active.sort(key=lambda path: int(path[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asns', nargs='+', default=['19529', '328840', '47608', '141295'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base = fast.previous.FeatureTree(1, .5, 'train', 0, 1)
    fast.enable_sample(base.ms)
    base.valid_dataset = 'rib'
    base.valid_files = set(base.list_dataset('rib'))
    base.load_models()
    native = fast.PipelineTree.__new__(fast.PipelineTree)
    checked = CheckedTree.__new__(CheckedTree)
    native.__dict__.update(base.__dict__)
    checked.__dict__.update(base.__dict__)
    here = Path(__file__).resolve().parent
    files = [here/name for name in ('pipeline_native.cpp', 'pipeline_engine.py', 'build_pipeline.py',
             'benchmark_pipeline.py', 'evaluate_pipeline.py', 'features_native.cpp', 'features_engine.py',
             'shortest_native.cpp', 'shortest_engine.py', 'native.cpp', 'native_engine.py', 'engine.py')]
    files += [here.parent/'pathradar_prediction.py', here.parent/'evaluate_top1_pairs.py']
    report = dict(status='RUNNING', initialization_seconds=time.monotonic()-started,
                  code_sha256={str(p): sha256(p) for p in files},
                  build_sha256={str(p): sha256(p) for p in (fast.OUTPUT/'BUILD.json', fast.previous.OUTPUT/'BUILD.json',
                      fast.previous.previous.OUTPUT/'BUILD.json', fast.previous.previous.bridge.OUTPUT/'BUILD.json')},
                  model_sha256={str(p): sha256(p) for p in (fast.engine.original.TRAINED_DIR/'models').glob('xgboost_1_*.model')},
                  comparisons=[], evaluation_comparisons=[])
    print(json.dumps(dict(event='initialized', seconds=report['initialization_seconds'])), flush=True)
    for asn in args.asns:
        outcomes = {}
        for name, tree in [('features', base), ('pipeline_checked', checked), ('pipeline', native)]:
            fast.engine.original.TESTING_DIR = args.output/name
            tree.verified_layers = 0
            counts = fast.engine.Counts()
            started = time.monotonic()
            tree.read_file(asn, counts)
            seconds = time.monotonic()-started
            output = fast.engine.original.TESTING_DIR/'result_round1'/f'{asn}.txt'
            rows = output.read_text().splitlines()
            outcomes[name] = dict(seconds=seconds, counts=counts.value, rows=len(rows),
                                 path_scores_sha256=digest(sorted(rows)), state=state(tree),
                                 verified_layers=tree.verified_layers)
            print(json.dumps(dict(asn=asn, implementation=name, seconds=seconds)), flush=True)
        equal = all(outcomes[name][key] == outcomes['features'][key]
                    for name in ('pipeline_checked', 'pipeline')
                    for key in ('counts', 'rows', 'path_scores_sha256', 'state'))
        report['comparisons'].append(dict(asn=asn, equal=equal,
            speedup=outcomes['features']['seconds']/outcomes['pipeline']['seconds'], **outcomes))
        (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
        if not equal:
            raise ValueError('Pipeline changed output; deployment prohibited')
    # The exact upstream evaluator runs against both verified routing backends.
    run_dir = fast.engine.original.REPO_DIR
    date = run_dir.name
    config = json.loads((here.parents[3]/('configs/dates/'+date+'.json')).read_text())
    test = Path(config['files_root'])/'data/aspath/path_by_vantage'/date/'path_without_prefix/test'
    truth = evaluation.original.load_ground_truth(date, '', str(test))
    for src in sorted({pair[0] for pair in truth})[:3]:
        gt = {dst: [list(path) for path in sorted(paths)] for (source, dst), paths in truth.items() if source == src}
        payload = (1, .7, src, gt, 1, 100)
        rows = []
        for tree in (base, native):
            tree.threshold = .7
            with patch.object(evaluation.original, 'Routingtree', lambda *args: tree), patch.object(
                    evaluation.original, 'load_models', lambda rounds: tree.model):
                rows.append(evaluation.original.evaluate_source_task(payload))
        equal = rows[0] == rows[1]
        report['evaluation_comparisons'].append(dict(src=src, pairs=len(gt), equal=equal,
            result_sha256=digest(rows[0])))
        print(json.dumps(dict(event='evaluation_checked', src=src, pairs=len(gt), equal=equal)), flush=True)
        if not equal:
            raise ValueError('Evaluation changed')
    # Verify production evidence release on a complete real destination.
    fast.release_evidence(native.ms)
    native.threshold = .5
    fast.engine.original.TESTING_DIR = args.output/'released'
    counts = fast.engine.Counts()
    native.read_file(args.asns[0], counts)
    report['released_evidence_equal'] = state(native) == report['comparisons'][0]['pipeline']['state']
    if not report['released_evidence_equal']:
        raise ValueError('Releasing duplicate evidence changed routing')
    report['total_speedup'] = sum(r['features']['seconds'] for r in report['comparisons'])/sum(r['pipeline']['seconds'] for r in report['comparisons'])
    report['status'] = 'PASS'
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', speedup=report['total_speedup'])), flush=True)


if __name__ == '__main__':
    main()
