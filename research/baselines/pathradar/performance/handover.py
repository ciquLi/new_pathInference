"""Drain the live prediction batch and resume verified caches with the adapter."""
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


def process(pid):
    root = Path('/proc')/str(pid)
    fields=(root/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, state=fields[0], parent=int(fields[1]), start_ticks=fields[19],
                command=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode())


def main():
    original_pid = int(sys.argv[1])
    # Verification finishes before the live job receives any signal.
    run_dates.verify_equivalence()
    prediction = process(original_pid)
    if 'pathradar_prediction.py -r 1 ' not in prediction['command']:
        raise ValueError('Unexpected prediction worker')
    parent=process(prediction['parent'])
    batch=process(parent['parent'])
    if 'run_baseline.py 20260201' not in parent['command'] or 'run_after_knownpath.py' not in batch['command']:
        raise ValueError('Unexpected baseline supervisors')
    records=[batch,parent,prediction]
    stopped=[]
    killed=False
    folder=ROOT/'runs/baselines/pathradar/performance'
    try:
        for item in records:
            if process(item['pid'])['start_ticks']!=item['start_ticks']:
                raise ValueError('PID identity changed')
            os.kill(item['pid'],signal.SIGSTOP)
            stopped.append(item['pid'])
        children=[]
        for proc in Path('/proc').iterdir():
            if proc.name.isdigit():
                try:
                    info=process(int(proc.name))
                    if info['parent']==original_pid and info['state']!='Z':
                        children.append(info)
                except (FileNotFoundError,ProcessLookupError):
                    pass
        print('Draining',len(children),'already running predictions; published files preserved',flush=True)
        deadline=time.monotonic()+600
        while children:
            pending=[]
            for child in children:
                try:
                    now=process(child['pid'])
                    if now['start_ticks']==child['start_ticks'] and now['state']!='Z':
                        pending.append(child)
                except (FileNotFoundError,ProcessLookupError):
                    pass
            children=pending
            if time.monotonic()>deadline:
                raise RuntimeError('Drain timeout: restore original processes; no forced redo')
            if children:
                time.sleep(1)
        complete=ROOT/'runs/baselines/pathradar/20260201/testing/result_round1'
        value=dict(previous_processes=records, completed_predictions=sum(1 for _ in complete.glob('*.txt')),
                   reason='verified execution acceleration; no model, data, threshold or round changes',
                   updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (folder/'HANDOVER.json').write_text(json.dumps(value,indent=2)+'\n')
        for item in records:
            os.kill(item['pid'],signal.SIGKILL)
        killed=True
        with (folder/'BATCH.log').open('ab') as log:
            worker=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'run_dates.py')],cwd=ROOT,
                                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        (folder/'DISPATCHER.json').write_text(json.dumps(dict(pid=worker.pid,workers=8,preserved_predictions=value['completed_predictions']),indent=2)+'\n')
        print('Adapter PID',worker.pid,'preserved predictions',value['completed_predictions'],flush=True)
    finally:
        if not killed:
            for pid in reversed(stopped):
                try:
                    os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    main()
