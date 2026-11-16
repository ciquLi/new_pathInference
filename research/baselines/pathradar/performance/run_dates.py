"""Resume full baseline rounds using verified execution-only prediction changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PERFORMANCE = Path(__file__).resolve().parent
SOURCE = PERFORMANCE.parent
ROOT = SOURCE.parents[2]
sys.path.insert(0, str(SOURCE))
import run_baseline as baseline

DATES = ('20260201', '20260301', '20260401', '20260501', '20260601')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_equivalence():
    directory = ROOT / 'runs/baselines/pathradar/performance'
    files = [directory/'equivalence_round1_full/REPORT.json']
    reports = [json.loads(file.read_text()) for file in files]
    for report in reports:
        if report['status'] != 'PASS' or not all(row['equal'] for row in report['comparisons']):
            raise ValueError('Unverified prediction optimization')
        if report['engine_sha256'] != sha256(PERFORMANCE/'engine.py'):
            raise ValueError('Optimization changed after equivalence checks')
    if reports[0]['original_sha256'] != sha256(SOURCE/'pathradar_prediction.py'):
        raise ValueError('Original prediction code changed')
    return {str(path):sha256(path) for path in files}


def verify_native():
    """Require real-input equivalence for the exact native source and binary."""
    folder = ROOT/'runs/baselines/pathradar/performance'
    proof = folder/'native_equivalence_round1/REPORT.json'
    report = json.loads(proof.read_text())
    comparisons = report.get('comparisons', [])
    if report.get('status') != 'PASS' or len(comparisons) < 3:
        raise ValueError('Native implementation has not passed real-input checks')
    for row in comparisons:
        if not row['equal'] or not row['native_checked']['verified_features']:
            raise ValueError('Native implementation differs from Python')
    for name, expected in report['code_sha256'].items():
        if sha256(name) != expected:
            raise ValueError('Native verification code changed: '+name)
    for name, expected in report['model_sha256'].items():
        if sha256(name) != expected:
            raise ValueError('Native verification model changed: '+name)
    build = folder/'native_build/BUILD.json'
    record = json.loads(build.read_text())
    if (sha256(build) != report['build_sha256'] or
            sha256(record['binary']) != record['binary_sha256'] or
            sha256(PERFORMANCE/'native.cpp') != record['source_sha256']):
        raise ValueError('Native build changed after verification')
    return {str(proof):sha256(proof), str(build):sha256(build)}


def verify_shortest():
    folder = ROOT/'runs/baselines/pathradar/performance'
    proof = folder/'shortest_equivalence_round1/REPORT.json'
    report = json.loads(proof.read_text())
    rows = report.get('comparisons', [])
    if report.get('status') != 'PASS' or len(rows) < 3 or not all(r['equal'] for r in rows):
        raise ValueError('Integer-graph shortest paths have not passed real-input checks')
    for name, expected in report['code_sha256'].items():
        if sha256(name) != expected:
            raise ValueError('Shortest-path verification code changed: '+name)
    for name, expected in report['model_sha256'].items():
        if sha256(name) != expected:
            raise ValueError('Shortest-path verification model changed: '+name)
    for name, expected in report['build_sha256'].items():
        build = Path(name)
        if sha256(build) != expected:
            raise ValueError('Shortest-path verification build changed: '+name)
        record = json.loads(build.read_text())
        if sha256(record['binary']) != record['binary_sha256']:
            raise ValueError('Shortest-path binary changed: '+record['binary'])
    return {str(proof):sha256(proof), **report['build_sha256']}


def verify_features():
    folder = ROOT/'runs/baselines/pathradar/performance'
    proof = folder/'features_equivalence_round1/REPORT.json'
    report = json.loads(proof.read_text())
    rows = report.get('comparisons', [])
    if report.get('status') != 'PASS' or len(rows) < 3:
        raise ValueError('Native feature matrices have not passed real-input checks')
    for row in rows:
        if not row['equal'] or not row['features_checked']['verified_values']:
            raise ValueError('Native feature values or predictions differ from Python')
    for field in ('code_sha256', 'model_sha256'):
        for name, expected in report[field].items():
            if sha256(name) != expected:
                raise ValueError('Feature verification identity changed: '+name)
    for name, expected in report['build_sha256'].items():
        build = Path(name)
        if sha256(build) != expected:
            raise ValueError('Feature verification build changed: '+name)
        record = json.loads(build.read_text())
        if sha256(record['binary']) != record['binary_sha256']:
            raise ValueError('Feature binary changed: '+record['binary'])
    return {str(proof):sha256(proof), **report['build_sha256']}


def verify_pipeline():
    folder = ROOT/'runs/baselines/pathradar/performance'
    proof = folder/'pipeline_equivalence_round1/REPORT.json'
    report = json.loads(proof.read_text())
    rows = report.get('comparisons', [])
    evaluations = report.get('evaluation_comparisons', [])
    if (report.get('status') != 'PASS' or len(rows) < 3 or len(evaluations) < 3 or
            not report.get('released_evidence_equal')):
        raise ValueError('Candidate pipeline and evaluation have not passed real-input checks')
    if not all(row['equal'] and row['pipeline_checked']['verified_layers'] == 9 for row in rows):
        raise ValueError('Candidate order, state or probabilities differ')
    if not all(row['equal'] and row['pairs'] > 0 for row in evaluations):
        raise ValueError('Evaluation outputs differ')
    for field in ('code_sha256', 'model_sha256'):
        for name, expected in report[field].items():
            if sha256(name) != expected:
                raise ValueError('Pipeline verification identity changed: '+name)
    for name, expected in report['build_sha256'].items():
        build = Path(name)
        if sha256(build) != expected:
            raise ValueError('Pipeline build changed: '+name)
        record = json.loads(build.read_text())
        if sha256(record['binary']) != record['binary_sha256']:
            raise ValueError('Pipeline binary changed: '+record['binary'])
        for source, digest in record.get('dependencies_sha256', {}).items():
            if sha256(source) != digest:
                raise ValueError('Pipeline included source changed: '+source)
    return {str(proof):sha256(proof), **report['build_sha256']}


def verify_loader():
    folder = ROOT/'runs/baselines/pathradar/performance'
    proof = folder/'loader_equivalence_round1/REPORT.json'
    report = json.loads(proof.read_text())
    rows = report.get('comparisons', [])
    evaluations = report.get('evaluation_comparisons', [])
    if (report.get('status') != 'PASS' or not report.get('all_evidence_equal') or
            len(rows) < 3 or len(evaluations) < 3 or
            not all(row['equal'] for row in rows) or
            not all(row['equal'] and row['pairs'] > 0 for row in evaluations)):
        raise ValueError('Streaming loader has not passed complete evidence and output checks')
    for field in ('code_sha256', 'model_sha256', 'pipeline_proof_sha256'):
        for name, expected in report[field].items():
            if sha256(name) != expected:
                raise ValueError('Streaming loader verification identity changed: '+name)
    for name, expected in report['build_sha256'].items():
        build = Path(name)
        if sha256(build) != expected:
            raise ValueError('Streaming loader build changed: '+name)
        record = json.loads(build.read_text())
        if sha256(record['binary']) != record['binary_sha256']:
            raise ValueError('Streaming loader binary changed: '+record['binary'])
        for source, digest in record.get('dependencies_sha256', {}).items():
            if sha256(source) != digest:
                raise ValueError('Streaming loader included source changed: '+source)
    return {str(proof):sha256(proof), **report['build_sha256']}


def main():
    proof = verify_equivalence()
    use_native = os.environ.get('PATHRADAR_NATIVE', '0') == '1'
    use_integer = os.environ.get('PATHRADAR_SHORTEST', '0') == '1'
    use_features = os.environ.get('PATHRADAR_FEATURES', '0') == '1'
    use_pipeline = os.environ.get('PATHRADAR_PIPELINE', '0') == '1'
    use_loader = os.environ.get('PATHRADAR_LOADER', '1') == '1'
    use_pipeline = use_pipeline or use_loader
    use_features = use_features or use_pipeline
    use_integer = use_integer or use_features
    use_native = use_native or use_integer
    if use_native:
        proof.update(verify_native())
    if use_integer:
        proof.update(verify_shortest())
    if use_features:
        proof.update(verify_features())
    if use_pipeline:
        proof.update(verify_pipeline())
    if use_loader:
        proof.update(verify_loader())
    predictor = PERFORMANCE/('loader_engine.py' if use_loader else 'pipeline_engine.py' if use_pipeline else 'features_engine.py' if use_features else 'shortest_engine.py' if use_integer else 'native_engine.py' if use_native else 'engine.py')
    workers = int(os.environ.get('PATHRADAR_PREDICTION_WORKERS', '8'))
    if workers < 1:
        raise ValueError('workers must be positive')
    original_run_stage = baseline.run_stage
    def run_stage(run_dir, label, command, env):
        if label.startswith('predict_train_'):
            command = list(command)
            position = command.index(str(SOURCE/'pathradar_prediction.py'))
            command[position] = str(predictor)
            command[command.index('-w')+1] = str(workers)
        if label == 'evaluate' and use_pipeline:
            command = list(command)
            position = command.index(str(SOURCE/'evaluate_top1_pairs.py'))
            command[position] = str(PERFORMANCE/('evaluate_loader.py' if use_loader else 'evaluate_pipeline.py'))
            command[command.index('--workers')+1] = str(min(workers, 4))
        return original_run_stage(run_dir, label, command, env)
    baseline.run_stage = run_stage
    for date in DATES:
        directory = ROOT/'runs/baselines/pathradar'/date
        marker = directory/'COMPLETE.json'
        if marker.exists() and json.loads(marker.read_text()).get('round') == 2:
            print(date,'already complete',flush=True)
            continue
        directory.mkdir(parents=True,exist_ok=True)
        amendment = dict(
            scope='execution only: persistent dynamically scheduled workers with '+('verified streaming frequency loader, ordered candidate tables and shared evaluation' if use_loader else 'verified ordered candidate tables, released duplicate evidence and shared evaluation' if use_pipeline else 'verified native feature evidence and matrices, integer-graph shortest paths' if use_features else 'verified integer-graph shortest paths and native hot loops' if use_integer else 'verified native hot loops' if use_native else 'verified Python hot loops'),
            created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            original_identity_sha256=sha256(directory/'IDENTITY.json') if (directory/'IDENTITY.json').exists() else None,
            proofs_sha256=proof, code_sha256={str(p):sha256(p) for p in PERFORMANCE.glob('*.py')},
            workers=workers, original_candidates_and_models_preserved=True,
            rounds=[0,1,2], threshold=.5, final_evaluation_threshold=.7)
        target=directory/('EXECUTION_AMENDMENT_'+sha256(predictor)[:16]+'.json')
        if not target.exists():
            target.write_text(json.dumps(amendment,indent=2)+'\n')
        sys.argv=[str(SOURCE/'run_baseline.py'),date]
        print(date,'resuming with verified execution adapter',flush=True)
        baseline.main()
    subprocess.run([sys.executable,'-u',str(SOURCE.parent/'report.py')],check=True)


if __name__ == '__main__':
    main()
