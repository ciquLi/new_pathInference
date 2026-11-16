"""Verify all frequency evidence, real outputs and evaluation after streaming load."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import loader_engine as fast
import evaluate_loader
from benchmark_native import digest, state


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_kib():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value, unit = line.split()
            values[key.rstrip(':')] = int(value)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asns', nargs='+', default=['19529', '328840', '47608'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference_start = time.monotonic()
    old = fast.previous.PipelineTree(1, .5, 'train', 0, 1)
    fast.previous.enable_sample(old.ms)
    reference_seconds = time.monotonic()-reference_start
    reference_memory = memory_kib()
    evidence = old.ms._feature_evidence
    fast.previous.release_evidence(old.ms)
    print(json.dumps(dict(event='reference_initialized', seconds=reference_seconds, memory_kib=reference_memory)), flush=True)
    started = time.monotonic()
    fast.configure()
    native = fast.previous.PipelineTree(1, .5, 'train', 0, 1)
    fast.enable_sample(native.ms, release=False)
    initialization_seconds = time.monotonic()-started
    evidence_equal = fast.native.evidence_equal(evidence, native.ms._feature_evidence)
    print(json.dumps(dict(event='stream_loaded', seconds=initialization_seconds, all_evidence_equal=evidence_equal)), flush=True)
    if not evidence_equal:
        raise ValueError('Streaming loader changed evidence')
    if old.store:
        old.store.close()
    del old, evidence
    gc.collect()
    fast.previous.release_evidence(native.ms)
    loader_memory = memory_kib()
    # Both backends read the same immutable evidence and relation sets in this process.
    reference = fast.previous.previous.FeatureTree.__new__(fast.previous.previous.FeatureTree)
    native.valid_dataset = 'rib'
    native.valid_files = set(native.list_dataset('rib'))
    native.load_models()
    reference.__dict__.update(native.__dict__)
    here = Path(__file__).resolve().parent
    files = [here/name for name in ('loader_native.cpp', 'loader_engine.py', 'build_loader.py', 'benchmark_loader.py',
        'evaluate_loader.py', 'pipeline_native.cpp', 'pipeline_engine.py', 'evaluate_pipeline.py',
        'features_native.cpp', 'features_engine.py', 'shortest_native.cpp', 'shortest_engine.py',
        'native.cpp', 'native_engine.py', 'engine.py')]
    files += [here.parent/'pathradar_prediction.py', here.parent/'evaluate_top1_pairs.py']
    proof = here.parents[3]/'runs/baselines/pathradar/performance/pipeline_equivalence_round1/REPORT.json'
    report = dict(status='RUNNING', all_evidence_equal=evidence_equal,
        reference_initialization_seconds=reference_seconds, initialization_seconds=initialization_seconds,
        reference_memory_kib=reference_memory, loader_memory_kib=loader_memory,
        pipeline_proof_sha256={str(proof): sha256(proof)},
        code_sha256={str(p): sha256(p) for p in files},
        build_sha256={str(fast.OUTPUT/'BUILD.json'): sha256(fast.OUTPUT/'BUILD.json')},
        model_sha256={str(p): sha256(p) for p in (fast.engine.original.TRAINED_DIR/'models').glob('xgboost_1_*.model')},
        comparisons=[], evaluation_comparisons=[])
    for asn in args.asns:
        outcomes = {}
        for name, tree in [('features', reference), ('loader', native)]:
            fast.engine.original.TESTING_DIR = args.output/name
            counts = fast.engine.Counts()
            started = time.monotonic()
            tree.read_file(asn, counts)
            seconds = time.monotonic()-started
            rows = (fast.engine.original.TESTING_DIR/'result_round1'/f'{asn}.txt').read_text().splitlines()
            outcomes[name] = dict(seconds=seconds, counts=counts.value, rows=len(rows),
                path_scores_sha256=digest(sorted(rows)), state=state(tree))
            print(json.dumps(dict(asn=asn, implementation=name, seconds=seconds)), flush=True)
        equal = all(outcomes['features'][key] == outcomes['loader'][key]
                    for key in ('counts', 'rows', 'path_scores_sha256', 'state'))
        report['comparisons'].append(dict(asn=asn, equal=equal, **outcomes))
        (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
        if not equal:
            raise ValueError('Streaming predictor changed full routing state')
    evaluator = evaluate_loader.evaluate_pipeline
    date = fast.engine.original.REPO_DIR.name
    config = json.loads((here.parents[3]/('configs/dates/'+date+'.json')).read_text())
    test = Path(config['files_root'])/'data/aspath/path_by_vantage'/date/'path_without_prefix/test'
    truth = evaluator.original.load_ground_truth(date, '', str(test))
    for src in sorted({pair[0] for pair in truth})[:3]:
        gt = {dst: [list(path) for path in sorted(paths)] for (source, dst), paths in truth.items() if source == src}
        outputs = []
        for tree in (reference, native):
            tree.threshold = .7
            with patch.object(evaluator, '_TREE', tree), patch.object(
                    evaluator.original, 'Routingtree', evaluator.tree_factory), patch.object(
                    evaluator.original, 'load_models', evaluator.cached_models):
                outputs.append(evaluator.original.evaluate_source_task((1, .7, src, gt, 1, 100)))
        equal = outputs[0] == outputs[1]
        report['evaluation_comparisons'].append(dict(src=src, pairs=len(gt), equal=equal,
            result_sha256=digest(outputs[0])))
        print(json.dumps(dict(event='evaluation_checked', src=src, pairs=len(gt), equal=equal)), flush=True)
        if not equal:
            raise ValueError('Streaming evaluation changed metrics or pair details')
    report['total_speedup'] = sum(r['features']['seconds'] for r in report['comparisons'])/sum(r['loader']['seconds'] for r in report['comparisons'])
    report['status'] = 'PASS'
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', speedup=report['total_speedup'], initialization_seconds=initialization_seconds,
        loader_memory_kib=loader_memory)), flush=True)


if __name__ == '__main__':
    main()
