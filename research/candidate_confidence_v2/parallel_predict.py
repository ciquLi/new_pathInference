#!/usr/bin/env python3
"""Run one queued frozen prediction beside the live worker on the same two CPUs.

The original numerical worker continues. Its parent dispatcher is stopped only
until the ahead worker is terminal and its ordinary receipt has been verified.
No numerical source, model, candidate, training configuration or journal changes.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathInference.experiment.contracts import freeze, require, verify_files
from pathInference.utils import read_json, sha256, write_json, utc_now


def process(pid):
    directory = Path('/proc') / str(pid)
    if not directory.exists():
        return None
    # comm can contain spaces/parentheses, so parse after the final closing ')'.
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    arguments = (directory / 'cmdline').read_bytes().split(b'\0')
    return dict(pid=pid, state=fields[0], parent=int(fields[1]), start_ticks=int(fields[19]),
                arguments=[os.fsdecode(value) for value in arguments if value])


def validate_processes(parent, worker, runner, primary):
    require(parent is not None and worker is not None, 'Live dispatcher/worker required')
    require(parent['state'] not in ('T', 't', 'Z', 'X'), 'Dispatcher was already stopped or terminal')
    require(str(runner) in parent['arguments'] and '--stage' not in parent['arguments'],
            'PID is not the original dispatcher')
    require(worker['parent'] == parent['pid'] and worker['state'] not in ('Z', 'X')
            and str(runner) in worker['arguments'], 'Worker ownership differs')
    arguments = worker['arguments']
    require('--stage' in arguments and arguments[arguments.index('--stage') + 1] == 'predict'
            and '--source' in arguments and arguments[arguments.index('--source') + 1] == primary,
            'Unexpected active numerical stage')


def same_process(current, expected):
    return current is not None and current['pid'] == expected['pid'] and current['start_ticks'] == expected['start_ticks']


def stop_ahead_before_resume(child, parent, inspect=process, send=os.kill):
    """Never resume the serial queue while the ahead child can still write."""
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
    current = inspect(parent['pid'])
    if same_process(current, parent) and current['state'] in ('T', 't'):
        send(parent['pid'], signal.SIGCONT)


def load_queue():
    path = ROOT / 'research/candidate_confidence_v2/run.py'
    spec = importlib.util.spec_from_file_location('frozen_confidence_dispatch', path)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    return queue, path


def progress(queue, parent, primary, ahead, child, state, error=None):
    run, cfg = queue.experiment.RUN, queue.experiment.config()
    rows = []
    for source in cfg['sources']:
        status_file = run / ('as' + source) / 'STATUS.json'
        status = read_json(status_file) if status_file.exists() else {}
        rows.append(dict(status, source=source, done=queue.receipt_path(source, 'predict').is_file()))
    record = dict(updated_utc=utc_now(), stage=state, parent_dispatcher=parent,
                  primary_source=primary, ahead_source=ahead, ahead_pid=child.pid if child else None,
                  controller_pid=os.getpid(), cpu_affinity=[6, 7], sources=rows, error=error,
                  resource_dispatcher_sha256=sha256(Path(__file__)),
                  numerical_code_changed=False, candidates_regenerated=False)
    write_json(run / 'PARALLEL_PREDICTION.json', record)
    lines = ['# 20260201：冻结候选后的路径置信度评估', '', '更新时间：' + record['updated_utc'], '',
             '剩余两个预测共用 CPU 6、7；模型、候选、数值代码保持冻结。',
             '原预测继续运行；原调度器暂挂，待提前运行的源写好普通回执后恢复，避免重复任务。', '',
             '| AS | 预测完成 | 实际状态 | 已落盘查询 |', '|---|---|---|---:|']
    for row in rows:
        lines.append('| {} | {} | {} | {} |'.format(row['source'], '是' if row['done'] else '否',
                     row.get('stage', '等待'), row.get('committed_queries', '')))
    if error:
        lines += ['', '资源调度错误：' + error]
    lines += ['', '资源调度状态：' + state, '', '进程身份和完成回执见 PARALLEL_PREDICTION.json；最终评测仍由原调度器执行。']
    temporary = run / 'PROGRESS.md.parallel.tmp'
    temporary.write_text('\n'.join(lines) + '\n'); temporary.replace(run / 'PROGRESS.md')
    write_json(run / 'STATUS.json', dict(updated_utc=utc_now(), active=[ahead, 'predict'],
               parallel_active_sources=[primary, ahead], error=error, completed_sources=0,
               expected_sources=10, dispatcher_pid=parent['pid'], resource_dispatcher_pid=os.getpid()))


def dispatch(parent_pid, worker_pid, primary='199524', ahead='398465'):
    queue, runner = load_queue()
    run, cfg = queue.experiment.RUN, queue.experiment.config()
    os.sched_setaffinity(0, set(cfg['cpu_affinity']))
    lock = (run / '.parallel_prediction.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent, worker = process(parent_pid), process(worker_pid)
    # Existing launch used a relative runner path, while its worker uses absolute.
    if parent and 'research/candidate_confidence_v2/run.py' in parent['arguments']:
        parent['arguments'] = [str(runner) if a == 'research/candidate_confidence_v2/run.py' else a
                               for a in parent['arguments']]
    validate_processes(parent, worker, runner, primary)
    require(cfg['sources'][-2:] == [primary, ahead] and not queue.checked(primary, 'predict')
            and not queue.checked(ahead, 'predict'), 'Only the two pending final sources may run ahead')
    require(read_json(run / 'STATUS.json')['active'] == [primary, 'predict'], 'Producer is in another stage')
    require((run / 'ALL_MODELS_FROZEN.json').is_file() and not (run / 'ALL_PREDICTIONS_FROZEN.json').exists(),
            'Model/prediction barrier differs')
    for source in cfg['sources']:
        require(queue.checked(source, 'train'), 'Unverified frozen model: ' + source)
    for source in cfg['sources'][:-2]:
        require(queue.checked(source, 'predict'), 'Earlier prediction not verified: ' + source)
    child, suspended, error = None, False, None
    handlers = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(number, frame):
        raise InterruptedError('Resource dispatcher interrupted by signal ' + str(number))

    for number in handlers:
        signal.signal(number, interrupted)
    try:
        suspended = True
        os.kill(parent_pid, signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while True:
            current = process(parent_pid)
            require(same_process(current, parent), 'Dispatcher identity changed')
            if current['state'] in ('T', 't'):
                break
            require(time.monotonic() < deadline, 'Dispatcher did not suspend')
            time.sleep(.05)
        # Recheck after the gate closes; no serial launch may overlap this source.
        require(not queue.checked(primary, 'predict') and same_process(process(worker_pid), worker),
                'Primary stage advanced during resource handover')
        require(not queue.checked(ahead, 'predict'), 'Ahead prediction completed during preflight')
        command = [sys.executable, '-B', '-u', str(runner), '--stage', 'predict', '--source', ahead]
        started = time.monotonic()
        with (run / 'logs' / (ahead + '_predict.log')).open('a') as stream:
            child = subprocess.Popen(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
            while child.poll() is None:
                require(same_process(process(parent_pid), parent) and process(parent_pid)['state'] in ('T', 't'),
                        'Original queue was externally resumed')
                progress(queue, parent, primary, ahead, child, 'parallel_predicting')
                time.sleep(15)
        require(child.returncode == 0, 'Ahead predictor failed; exit=' + str(child.returncode))
        frozen = read_json(run / ('as' + ahead) / 'PREDICTIONS_FROZEN.json')
        model = read_json(run / ('as' + ahead) / 'MODEL_FROZEN.json')
        binding = read_json(run / 'SNAPSHOT.json')['sources'][ahead]
        require(frozen['model_sha256'] == model['model_sha256']
                and frozen['candidate_sha256'] == binding['candidate_sha256']
                and not frozen['final_labels_read'], 'Ahead prediction identity differs')
        outputs = {str(path): sha256(path) for path in queue.paths(ahead, 'predict')}
        freeze(queue.receipt_path(ahead, 'predict'), dict(source=ahead, stage='predict', status='PASS',
               configuration_sha256=sha256(queue.experiment.CONFIG), code_sha256=queue.code_identity(),
               outputs_sha256=outputs, wall_seconds=time.monotonic() - started, completed_utc=utc_now()))
        require(queue.checked(ahead, 'predict'), 'Ahead completion receipt did not verify')
        progress(queue, parent, primary, ahead, child, 'ahead_verified_resuming_original_queue')
    except BaseException as exc:
        error = str(exc)
        progress(queue, parent, primary, ahead, child, 'resource_dispatch_error', error)
        raise
    finally:
        if suspended:
            stop_ahead_before_resume(child, parent)
        for number, handler in handlers.items():
            signal.signal(number, handler)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatcher-pid', type=int, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    args = parser.parse_args()
    dispatch(args.dispatcher_pid, args.worker_pid)
