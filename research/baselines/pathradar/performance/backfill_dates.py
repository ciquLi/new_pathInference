"""Sequential low-resource queue for completed-date per-query evaluation backfills."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DATES = ('20260201', '20260401')
FOLDER = ROOT / 'runs/baselines/pathradar/detail_backfill'


def main():
    os.sched_setaffinity(0, {4, 5})
    FOLDER.mkdir(parents=True, exist_ok=True)
    lock = (FOLDER / '.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    states = {date: dict(stage='waiting') for date in DATES}

    def publish(error=None):
        record = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      dates=states, max_cpu=2, models_retrained=False, error=error)
        temporary = FOLDER / 'STATUS.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(FOLDER / 'STATUS.json')
        lines = ['# PathRadar 0201 / 0401 逐查询明细补齐', '', '更新时间：' + record['updated_utc'], '',
                 '顺序运行；最多 2 CPU；复用第 2 轮模型，不重训。', '',
                 '| 日期 | 阶段 | 实际工作 | 明细条数 | 测试查询数 |', '|---|---|---|---:|---:|']
        for date, state in states.items():
            status = ROOT / 'runs/baselines/pathradar' / date / 'control/pair_details_backfill/STATUS.json'
            detail = json.loads(status.read_text()).get('phase', '') if status.exists() else ''
            lines.append('| ' + ' | '.join(map(str, [date, state['stage'], detail,
                state.get('records', '—'), state.get('test_queries', '—')])) + ' |')
        if error: lines += ['', '错误：' + error]
        temporary = FOLDER / 'PROGRESS.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(FOLDER / 'PROGRESS.md')

    def execute(date, stage, command):
        states[date]['stage'] = stage
        log = FOLDER / (date + '_' + stage + '.log')
        with log.open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            states[date]['pid'] = child.pid
            while child.poll() is None:
                publish()
                time.sleep(15)
        states[date].pop('pid', None)
        return child.returncode

    publish()
    try:
        for date in DATES:
            command = [sys.executable, '-B', '-u', 'tools/validation/backfill_pathradar_details.py', date]
            code = execute(date, 'reevaluate', command)
            if code:
                status = ROOT / 'runs/baselines/pathradar' / date / 'control/pair_details_backfill/STATUS.json'
                record = json.loads(status.read_text()) if status.exists() else {}
                if record.get('phase') != 'details_complete_historical_aggregate_differs':
                    raise RuntimeError(date + ' backfill failed; see ' + str(FOLDER))
                # Keep changed-score results separately; never search seeds or retrain.
                if execute(date, 'finalize', command + ['--finalize-details']):
                    raise RuntimeError(date + ' detail finalization failed')
            if execute(date, 'audit', [sys.executable, '-B', '-u',
                    'tools/validation/audit_pathradar_pair_details.py', date]):
                raise RuntimeError(date + ' independent detail audit failed')
            audit = ROOT / 'runs/baselines/pathradar' / date / 'control/pair_details_backfill/INDEPENDENT_DETAILS_AUDIT.json'
            result = json.loads(audit.read_text())
            if result['status'] != 'PASS' or result['records'] != result['test_queries']:
                raise ValueError(date + ' detail count mismatch')
            states[date].update(stage='complete', records=result['records'], test_queries=result['test_queries'])
            publish()
    except BaseException as exc:
        publish(error=str(exc))
        raise


if __name__ == '__main__':
    main()
