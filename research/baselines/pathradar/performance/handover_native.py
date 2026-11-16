"""Switch only the live Python predictor after native equivalence verification."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import run_dates
from handover import process


def live_children(parent):
    children = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                child = process(int(proc.name))
                if child['parent'] == parent and child['state'] != 'Z':
                    children.append(child)
            except (FileNotFoundError, ProcessLookupError):
                pass
    return children


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction_pid', type=int)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backend', choices=('native', 'integer', 'features', 'pipeline', 'loader'), default='native')
    args = parser.parse_args()
    proofs = run_dates.verify_equivalence()
    proofs.update(run_dates.verify_native())
    if args.backend in ('integer', 'features', 'pipeline', 'loader'):
        proofs.update(run_dates.verify_shortest())
    if args.backend in ('features', 'pipeline', 'loader'):
        proofs.update(run_dates.verify_features())
    if args.backend in ('pipeline', 'loader'):
        proofs.update(run_dates.verify_pipeline())
    if args.backend == 'loader':
        proofs.update(run_dates.verify_loader())
    folder = ROOT/'runs/baselines/pathradar/performance'
    proof_name = {'native':'native_equivalence_round1', 'integer':'shortest_equivalence_round1', 'features':'features_equivalence_round1', 'pipeline':'pipeline_equivalence_round1', 'loader':'loader_equivalence_round1'}[args.backend]
    proof = json.loads((folder/proof_name/'REPORT.json').read_text())
    if proof['total_speedup'] <= 1.1:
        raise ValueError('Measured native gain is too small to interrupt live work')
    prediction = process(args.prediction_pid)
    expected_predictor = HERE/{'native':'engine.py', 'integer':'native_engine.py', 'features':'shortest_engine.py', 'pipeline':'features_engine.py', 'loader':'features_engine.py'}[args.backend]
    if str(expected_predictor)+' -r ' not in prediction['command']:
        raise ValueError('Unexpected predictor; expected '+str(expected_predictor))
    supervisor = process(prediction['parent'])
    if str(HERE/'run_dates.py') not in supervisor['command']:
        raise ValueError('Unexpected prediction supervisor')
    raw_env = (Path('/proc')/str(args.prediction_pid)/'environ').read_bytes()
    env = dict(entry.decode().split('=', 1) for entry in raw_env.split(b'\0') if entry)
    run_dir = Path(env['PATHRADAR_RUN_DIR'])
    if run_dir.parent != ROOT/'runs/baselines/pathradar':
        raise ValueError('Unexpected run directory')
    # Check dated numerical/input identity before interrupting any process.
    subprocess.run([sys.executable, '-B', str(HERE.parent/'run_baseline.py'),
                    run_dir.name, '--prepare-only'], check=True)
    children = live_children(args.prediction_pid)
    plan = dict(previous_processes=[supervisor, prediction]+children,
                proofs_sha256=proofs, run_directory=str(run_dir),
                workers=int(os.environ.get('PATHRADAR_PREDICTION_WORKERS', '8')),
                completed_outputs_preserved=True,
                maximum_unpublished_destinations_recomputed=len(children),
                measured_speedup=proof['total_speedup'], backend=args.backend)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.apply:
        return
    stopped = []
    killed = False
    try:
        # Stop the parents before enumerating workers again: the pool can recycle
        # a child between the dry-run snapshot and the actual handover.
        for item in [supervisor, prediction]:
            current = process(item['pid'])
            if current['start_ticks'] != item['start_ticks']:
                raise ValueError('PID was reused before handover')
            os.kill(item['pid'], signal.SIGSTOP)
            stopped.append(item['pid'])
        children = live_children(args.prediction_pid)
        for item in children:
            current = process(item['pid'])
            if current['start_ticks'] != item['start_ticks']:
                raise ValueError('Worker PID was reused before handover')
            os.kill(item['pid'], signal.SIGSTOP)
            stopped.append(item['pid'])
        plan['previous_processes'] = [supervisor, prediction]+children
        plan['maximum_unpublished_destinations_recomputed'] = len(children)
        round_number = prediction['command'].split(' -r ', 1)[1].split()[0]
        plan['completed_predictions'] = sum(1 for _ in
            (run_dir/'testing'/('result_round'+round_number)).glob('*.txt'))
        plan['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (folder/(args.backend.upper()+'_HANDOVER.json')).write_text(json.dumps(plan, indent=2)+'\n')
        for item in children+[prediction, supervisor]:
            os.kill(item['pid'], signal.SIGKILL)
        killed = True
        launch_env = os.environ.copy()
        launch_env['PATHRADAR_NATIVE'] = '1'
        launch_env['PATHRADAR_SHORTEST'] = '1' if args.backend in ('integer', 'features', 'pipeline', 'loader') else '0'
        launch_env['PATHRADAR_FEATURES'] = '1' if args.backend in ('features', 'pipeline', 'loader') else '0'
        launch_env['PATHRADAR_PIPELINE'] = '1' if args.backend in ('pipeline', 'loader') else '0'
        launch_env['PATHRADAR_LOADER'] = '1' if args.backend == 'loader' else '0'
        launch_env['PATHRADAR_PREDICTION_WORKERS'] = str(plan['workers'])
        with (folder/'BATCH.log').open('ab') as log:
            worker = subprocess.Popen([sys.executable, '-B', '-u', str(HERE/'run_dates.py')],
                cwd=ROOT, env=launch_env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        (folder/'DISPATCHER.json').write_text(json.dumps(dict(pid=worker.pid,
            workers=plan['workers'], backend=args.backend,
            preserved_predictions=plan['completed_predictions']), indent=2)+'\n')
        print('Native dispatcher started:', worker.pid, flush=True)
    finally:
        if not killed:
            for pid in reversed(stopped):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    main()
