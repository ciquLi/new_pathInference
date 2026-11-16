"""Reload evaluation dispatch after the current shortest-path process finishes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from handover import process
import run_dates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    parent = process(int(sys.argv[1]))
    child = process(int(sys.argv[2]))
    if (str(HERE / 'run_dates.py') not in parent['command'] or child['parent'] != parent['pid']
            or 'pathradar_shortest_cpp' not in child['command']):
        raise ValueError('Expected a dispatcher waiting for an existing shortest-path task')
    # Check the frozen implementation and builds before touching the dispatcher.
    for check in (run_dates.verify_equivalence, run_dates.verify_native, run_dates.verify_shortest,
                  run_dates.verify_features, run_dates.verify_pipeline, run_dates.verify_loader):
        check()
    raw = (Path('/proc') / str(parent['pid']) / 'environ').read_bytes()
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    destination = ROOT / 'runs/baselines/pathradar/performance/EVALUATION_DETAILS_HANDOVER.json'
    record = dict(previous_dispatcher=parent, preserved_shortest_process=child,
                  include_pair_details=True, status='waiting_for_existing_shortest')
    stopped = False
    replaced = False
    try:
        if process(parent['pid'])['start_ticks'] != parent['start_ticks']:
            raise ValueError('Dispatcher PID changed')
        os.kill(parent['pid'], signal.SIGSTOP)
        stopped = True
        destination.write_text(json.dumps(record, indent=2) + '\n')
        print('Dispatcher paused; existing shortest-path process continues:', child['pid'], flush=True)
        while True:
            try:
                current = process(child['pid'])
                if current['start_ticks'] != child['start_ticks'] or current['state'] == 'Z':
                    break
            except (FileNotFoundError, ProcessLookupError):
                break
            time.sleep(15)
        # No running numerical child is interrupted. The new dispatcher verifies
        # existing caches and shortest --skip-existing before the following stage.
        if process(parent['pid'])['start_ticks'] != parent['start_ticks']:
            raise ValueError('Dispatcher PID changed during drain')
        os.kill(parent['pid'], signal.SIGKILL)
        replaced = True
        with (destination.parent / 'BATCH.log').open('ab') as log:
            worker = subprocess.Popen([sys.executable, '-B', '-u', str(HERE / 'run_dates.py')],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        record.update(status='dispatcher_reloaded', pid=worker.pid,
                      numerical_processes_interrupted=0)
        destination.write_text(json.dumps(record, indent=2) + '\n')
        print('Reloaded dispatcher:', worker.pid, flush=True)
    finally:
        if stopped and not replaced:
            os.kill(parent['pid'], signal.SIGCONT)


if __name__ == '__main__':
    main()
