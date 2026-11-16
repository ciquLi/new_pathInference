"""Re-evaluate a completed PathRadar date, preserving trained models and old scores."""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research/baselines/pathradar'
sys.path.insert(0, str(SOURCE))
import run_baseline as baseline
from performance.pair_details import persist, test_pairs


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def remove_temporary_evidence(control):
    """Delete only this tool's private directory; external evidence stays read-only."""
    directory = control / 'evidence'
    if directory.is_symlink():
        raise ValueError('Temporary evidence directory must not be a symlink')
    if directory.exists():
        shutil.rmtree(directory)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('date')
    parser.add_argument('--finalize-details', action='store_true',
                        help='Finalize an already evaluated, changed-score detail set separately from historical scores')
    args = parser.parse_args()
    os.sched_setaffinity(0, {4, 5})
    run = ROOT / 'runs/baselines/pathradar' / args.date
    control = run / 'control/pair_details_backfill'
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / '.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = run / 'testing/top1_eval_round2_full.json'
    old = json.loads(output.read_text())
    source = baseline.FILES / 'data/aspath/path_by_vantage' / args.date / 'path_without_prefix'
    identity = json.loads((run / 'IDENTITY.json').read_text())
    expected = test_pairs(source / 'test')
    completed = control / 'REEVALUATION_COMPLETE.json'
    if completed.exists():
        saved = json.loads(completed.read_text())
        for name, digest in saved['outputs_sha256'].items():
            if baseline.sha256(Path(name)) != digest:
                raise ValueError('Completed re-evaluation artifact changed: ' + name)
        print(json.dumps(saved['details']), flush=True)
        return
    if args.finalize_details:
        status = control / 'STATUS.json'
        record = json.loads(status.read_text())
        if record['phase'] != 'details_complete_historical_aggregate_differs':
            raise ValueError('No complete changed-score detail set is ready')
        if baseline.sha256(run / 'IDENTITY.json') != record['identity_sha256']:
            raise ValueError('Experiment identity changed')
        for name, digest in record['models_sha256'].items():
            if baseline.sha256(Path(name)) != digest:
                raise ValueError('Frozen model changed: ' + name)
        pending = Path(record['detail_output'])
        new = json.loads(pending.read_text())
        from performance.pair_details import validate
        validate(new, expected)
        final = output.with_name('top1_eval_round2_full_reevaluated.json')
        pending.replace(final)
        details = persist(final, new, expected)
        record.pop('error', None)
        record.update(phase='complete_reassessed', detail_output=str(final), details=details,
            historical_aggregate_preserved=True, historical_aggregate_sha256=baseline.sha256(output),
            score_difference_cause='Unresolved: original runtime ordering was not retained. '
                'Same dated inputs and round-2 model bytes verified. Use this evaluation for both curves and aggregate scores.',
            outputs_sha256={str(p): baseline.sha256(p) for p in
                (final, final.with_suffix('.pairs.csv'), final.with_suffix('.pairs.audit.json'))})
        for stale in (pending.with_suffix('.pairs.csv'), pending.with_suffix('.pairs.audit.json')):
            stale.unlink(missing_ok=True)
        record['temporary_evidence_removed'] = remove_temporary_evidence(control)
        record['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        completed.write_text(json.dumps(record, indent=2) + '\n')
        atomic_json(status, record)
        print(json.dumps(details), flush=True)
        return
    if old.get('include_pair_details'):
        print(json.dumps(persist(output, old, expected)), flush=True)
        return
    reuse_evidence = bool(identity.get('cached_intermediate'))
    evidence = (baseline.LEGACY / (args.date + '_devoid') / 'intermediate'
                if reuse_evidence else control / 'evidence')
    if not reuse_evidence:
        evidence.mkdir(exist_ok=True)
    status = control / 'STATUS.json'
    record = dict(date=args.date, phase='verify_inputs', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  models_retrained=False, shortest_paths_rebuilt=False)

    def publish(phase):
        record['phase'] = phase
        record['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_json(status, record)
        print(record['updated_utc'], phase, flush=True)

    try:
        publish('verify_inputs')
        asrel = baseline.FILES / 'data/aspath' / (args.date + '.as-rel.txt')
        if (baseline.csv_identity(source / 'train') != identity['train_csv']
                or baseline.csv_identity(source / 'test') != identity['test_csv']
                or baseline.sha256(asrel) != identity['asrel_sha256']):
            raise ValueError('Dated public inputs differ from the completed experiment')
        models = {str(p): baseline.sha256(p) for p in baseline.model_paths(run, 2)}
        record['models_sha256'] = models
        record['identity_sha256'] = baseline.sha256(run / 'IDENTITY.json')
        record['evidence_reused'] = reuse_evidence
        record['evidence_directory'] = str(evidence)
        builder = SOURCE / 'tools/build_from_split_csv.py'
        record['builder_sha256'] = baseline.sha256(builder)
        marker = control / 'EVIDENCE.json'
        names = ('asrel_rib.txt', 'core_shell.txt', 'joinfreq_data.txt', 'linkfreq_data.txt', 'linkorder_data.txt')
        if not marker.exists():
            if reuse_evidence:
                publish('verify_original_cached_evidence')
                prepared = json.loads((run / 'PREPARED.json').read_text())
                if baseline.sha256(evidence / 'pathradar_store.sqlite') != prepared['db_sha256']:
                    raise ValueError('Original cached database identity changed')
                record['cached_database_sha256'] = prepared['db_sha256']
            else:
                publish('rebuild_evaluation_evidence_only')
                spec = importlib.util.spec_from_file_location('original_evidence_builder', builder)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                paths = module.load_paths(source / 'train')
                module.build_auxiliary_files(paths, evidence, 'data')
                del paths
                shutil.copy2(asrel, evidence / 'asrel_rib.txt')
            hashes = {name: baseline.sha256(evidence / name) for name in names}
            marker.write_text(json.dumps(dict(inputs=identity, builder_sha256=record['builder_sha256'],
                evidence_reused=reuse_evidence, evidence_directory=str(evidence),
                files_sha256=hashes), indent=2) + '\n')
        saved = json.loads(marker.read_text())
        if saved['inputs'] != identity or saved['builder_sha256'] != record['builder_sha256']:
            raise ValueError('Rebuilt evidence identity changed')
        for name, digest in saved['files_sha256'].items():
            if baseline.sha256(evidence / name) != digest:
                raise ValueError('Rebuilt evidence hash mismatch: ' + name)
        from performance import run_dates
        for check in (run_dates.verify_equivalence, run_dates.verify_native, run_dates.verify_shortest,
                      run_dates.verify_features, run_dates.verify_pipeline, run_dates.verify_loader):
            check()
        pending = output.with_name('top1_eval_round2_full.details_pending.json')
        env = os.environ.copy()
        env.update(PATHRADAR_RUN_DIR=str(run), PATHRADAR_INTERMEDIATE_DIR=str(evidence),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        if reuse_evidence:
            env['PATHRADAR_READONLY_DB'] = '1'
        command = [sys.executable, '-B', '-u', str(SOURCE / 'performance/evaluate_loader.py'),
            '--rounds', str(old['rounds']), '--threshold', str(old['threshold']),
            '--date', args.date, '--test-dir', str(source / 'test'), '--sample-size', '0',
            '--workers', '2', '--include-pair-details', '1', '--pair-batch-size', '100',
            '--output', str(pending)]
        record['command'] = command
        publish('evaluate_with_frozen_round2_models')
        if not pending.exists():
            with (control / 'evaluate.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        new = json.loads(pending.read_text())
        if any(baseline.sha256(Path(p)) != digest for p, digest in models.items()):
            raise ValueError('Round 2 model bytes changed')
        publish('validate_query_details')
        from performance.pair_details import validate
        check = validate(new, expected)
        comparison = {key: dict(previous=old[key], reevaluated=new[key],
            equal=math.isclose(old[key], new[key], rel_tol=0, abs_tol=1e-12))
            for key in ('requested_pairs', 'coverage_count', 'exact_match_count', 'mean_jaccard_over_requested')}
        record['aggregate_comparison'] = comparison
        record['query_check'] = check
        backup = output.with_name('top1_eval_round2_aggregate_before_details.json')
        if not backup.exists(): shutil.copy2(output, backup)
        # Runtime ordering or rebuilt evidence may affect tied choices. Keep
        # historical aggregate scores separate if this evaluation differs.
        if not all(item['equal'] for item in comparison.values()):
            record['detail_output'] = str(pending)
            record['details'] = persist(pending, new, expected)
            publish('details_complete_historical_aggregate_differs')
            raise RuntimeError('Details complete but re-evaluated scores differ; old aggregate preserved. See STATUS.json')
        pending.replace(output)
        record['details'] = persist(output, new, expected)
        complete_file = run / 'COMPLETE.json'
        complete = json.loads(complete_file.read_text())
        complete['pair_details'] = record['details']
        complete_file.write_text(json.dumps(complete, indent=2) + '\n')
        record['temporary_evidence_removed'] = remove_temporary_evidence(control)
        publish('complete')
    except BaseException as exc:
        record['error'] = str(exc)
        publish(record['phase'])
        raise


if __name__ == '__main__':
    main()
