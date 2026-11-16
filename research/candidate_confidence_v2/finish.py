#!/usr/bin/env python3
"""Watch existing confidence jobs, then audit and report; never launch numerical work."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pathInference.utils import read_json, utc_now, write_json


def state(run, sources, variants):
    tasks = run / 'tasks'
    counts = {}
    for stage in ('train', 'predict', 'evaluate'):
        counts[stage] = sum((tasks / ('_'.join(x for x in (s, v, stage) if x) + '.json')).is_file()
                            for s in sources for v in variants)
    expected = len(sources) * len(variants)
    complete = (all(value == expected for value in counts.values())
                and all((run / file).is_file() for file in (
                    'ALL_MODELS_FROZEN.json', 'ALL_PREDICTIONS_FROZEN.json', 'SUMMARY.json', 'REPORT.md')))
    status = read_json(run / 'STATUS.json') if (run / 'STATUS.json').is_file() else {}
    return dict(counts=counts, expected=expected, ready_for_audit=complete,
                error=status.get('error'), producer_updated_utc=status.get('updated_utc'))


def main():
    os.sched_setaffinity(0, {6, 7})
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    destination = ROOT / 'runs/20260201/confidence_comparison'
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.finish.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = read_json(ROOT / 'research/candidate_confidence_v2/CONFIG.json')['sources']
    runs = {name: ROOT / 'runs/20260201' / name for name in ('gmpi_confidence_v1', 'candidate_confidence_v2')}
    commands = {
        'gmpi_confidence_v1': ['tools/validation/audit_gmpi_confidence.py', '--check-weights', '--check-candidates', '--rescore'],
        'candidate_confidence_v2': ['tools/validation/audit_candidate_confidence.py', '--validate-models', '--check-candidates', '--rescore'],
        'report': ['tools/validation/report_candidate_confidence.py']}
    audited, child, child_name, log = set(), None, None, None

    def publish(states, error=None, complete=False):
        record = dict(updated_utc=utc_now(), jobs=states, audited=sorted(audited),
                      active_check=child_name, complete=complete, error=error,
                      numerical_jobs_restarted=False, models_or_candidates_changed=False)
        write_json(destination / 'MONITOR.json', record)
        lines = ['# 20260201：候选置信度对照收尾进度', '', '更新时间：' + record['updated_utc'], '',
                 '| 实验 | 训练完成 | 预测完成 | 评测完成 | 独立审计 |', '|---|---:|---:|---:|---|']
        for name, item in states.items():
            count = item['counts']; total = item['expected']
            cells = [name, *[str(count[s]) + '/' + str(total) for s in ('train', 'predict', 'evaluate')],
                     '通过' if name in audited else '进行中' if name == child_name else '等待完整评测']
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += ['', '仅监测完成回执并运行独立审计和报告；不启动、重启或改动数值实验。',
                  '本表计数来自回执，不作为工作进程存活证明。']
        if error: lines += ['', '错误：' + error]
        if complete: lines += ['', '独立审计与统一汇总完成，见 REPORT.md、RESULTS.csv、SUMMARY.json。']
        temporary = destination / 'PROGRESS.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(destination / 'PROGRESS.md')

    try:
        while True:
            states = {name: state(run, sources, ['smooth_l1', 'query_cross_entropy']
                                 if name == 'gmpi_confidence_v1' else ['']) for name, run in runs.items()}
            if child is not None and child.poll() is not None:
                code = child.returncode
                log.close()
                if code:
                    raise RuntimeError(child_name + ' failed; see ' + child_name + '_completion.log')
                if child_name == 'report':
                    publish(states, complete=True)
                    return
                audit = read_json(runs[child_name] / 'COMPLETION_AUDIT.json')
                if audit['status'] != 'PASS' or not audit['final_labels_read']:
                    raise RuntimeError(child_name + ' has no complete independent rescore')
                audited.add(child_name)
                child, child_name = None, None
            for name, item in states.items():
                if item['error']:
                    raise RuntimeError(name + ': ' + str(item['error']))
            if child is None:
                child_name = next((name for name, item in states.items()
                                   if item['ready_for_audit'] and name not in audited), None)
                if child_name is None and len(audited) == len(runs):
                    child_name = 'report'
                if child_name:
                    log = (destination / (child_name + '_completion.log')).open('a')
                    child = subprocess.Popen([sys.executable, '-B', *commands[child_name]], cwd=ROOT,
                                             stdout=log, stderr=subprocess.STDOUT)
            publish(states)
            time.sleep(15)
    except BaseException as exc:
        publish(locals().get('states', {}), error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
