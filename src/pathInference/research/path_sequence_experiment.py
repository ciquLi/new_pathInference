"""Architecture-only experiment using an isolated instance of the frozen engine.

Reuse the audited train/checkpoint/predict/journal APIs without editing their
source or rebinding the GMPI module used by another caller. Only model creation,
configuration and output paths are different in this private engine instance.
"""
from __future__ import annotations

import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import threading
import time

from pathInference.experiment.contracts import freeze, require, verify_files
from pathInference.utils import project_root, read_json, sha256, utc_now, write_json

ROOT = project_root()
CONFIG = ROOT / 'research/path_sequence_confidence_v1/CONFIG.json'
RUN = ROOT / 'runs/20260201/path_sequence_confidence_v1'


def code_identity():
    paths = [Path(__file__), ROOT / 'src/pathInference/research/path_sequence_confidence.py',
             ROOT / 'src/pathInference/research/gmpi_confidence.py',
             ROOT / 'src/pathInference/research/gmpi_confidence_loss.py',
             ROOT / 'research/path_sequence_confidence_v1/run.py']
    return {str(path): sha256(path) for path in paths}


def make_engine():
    # Import a separate module object: the completed control's globals stay intact.
    path = ROOT / 'src/pathInference/research/gmpi_confidence.py'
    spec = importlib.util.spec_from_file_location('pathInference.research._ordered_path_engine', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.CONFIG, engine.RUN = CONFIG, RUN

    def make_model(runtime, arguments):
        from pathInference.research.path_sequence_confidence import OrderedPathConfidence
        return OrderedPathConfidence(runtime['torch'].from_numpy(runtime['vectors']),
                                     **read_json(CONFIG)['architecture']).to(runtime['device'])

    engine.make_model = make_model
    return engine


def snapshot():
    cfg = read_json(CONFIG)
    control = ROOT / cfg['control_run']
    audit = read_json(control / 'COMPLETION_AUDIT.json')
    require(audit['status'] == 'PASS' and not audit['pending'] and audit['weights_checked'],
            'Frozen GMPI control has not passed independent audit')
    original = read_json(ROOT / cfg['source_snapshot'])
    require(cfg['sources'] == original['configuration']['sources']
            and cfg['seed'] == original['configuration']['seed']
            and cfg['date'] == original['configuration']['date']
            and cfg['label_temperature'] == original['configuration']['label_temperature'],
            'Architecture experiment changed data/seed/objective')
    require(not original['candidates_regenerated'] and not cfg['final_labels_used_for_tuning'],
            'Frozen experiment contract changed')
    files = dict(original['files_sha256'])
    files.update(code_identity())
    for path in (CONFIG, ROOT / cfg['source_snapshot'], control / 'COMPLETION_AUDIT.json'):
        files[str(path)] = sha256(path)
    controls = {}
    for source in cfg['sources']:
        directory = control / ('as' + source) / cfg['control_variant']
        paths = [directory / name for name in ('MODEL_FROZEN.json', 'PREDICTIONS_FROZEN.json',
                 'model.pt', 'ranked_predictions.csv', 'RESULTS.json')]
        controls[source] = {str(path): sha256(path) for path in paths}
        files.update(controls[source])
    verify_files(files)
    freeze(RUN / 'SNAPSHOT.json', dict(configuration=cfg, files_sha256=files,
        sources=original['sources'], data_manifest=original['data_manifest'],
        candidates_regenerated=False, test_labels_used_for_tuning=False,
        feature_schema='same GMPI 7 input tensors; ordered AS ids/endpoints/penultimate/history',
        control_assets_sha256=controls, historical_test_results_already_known=True))


def receipt(source, stage):
    return RUN / 'tasks' / ((source + '_ordered_path_' if source else '') + stage + '.json')


def checked(source, stage):
    path = receipt(source, stage)
    if not path.exists():
        return False
    record = read_json(path)
    require(record['code_sha256'] == code_identity() and record['config_sha256'] == sha256(CONFIG),
            'Architecture task code/config changed')
    verify_files(record['outputs_sha256'])
    return True


def report(error=None):
    cfg = read_json(CONFIG)
    active = read_json(RUN / 'ACTIVE.json') if (RUN / 'ACTIVE.json').exists() else {}
    counts = {stage: sum(receipt(s, stage).is_file() for s in cfg['sources'])
              for stage in ('train', 'predict', 'evaluate')}
    lines = ['# 20260201：路径顺序与历史交互网络', '', '更新时间：' + utc_now(), '',
             '唯一变化：评估网络结构；与已完成的 GMPI 查询内交叉熵组对照。',
             '候选、历史、初始 AS 嵌入、训练/验证划分、损失、训练参数均固定。',
             '单 GPU、2 CPU；全部模型冻结后预测，全部预测冻结后评测。', '',
             '| AS | 训练 | 预测 | 评测 | 当前状态 |', '|---|---|---|---|---|']
    for source in cfg['sources']:
        path = RUN / ('as' + source) / 'ordered_path/STATUS.json'
        status = read_json(path) if path.exists() else {}
        text = status.get('stage', '等待')
        if 'completed_epochs' in status:
            text += '；epoch ' + str(status['completed_epochs'])
        if 'committed_queries' in status:
            text += '；已落盘 ' + str(status['committed_queries']) + ' 查询'
        lines.append('| ' + ' | '.join([source, *['完成' if receipt(source, stage).exists() else '等待'
                                               for stage in counts], text]) + ' |')
    if error:
        lines += ['', '错误：' + error]
    lines += ['', '活动状态：' + json.dumps(active, ensure_ascii=False), '',
              '状态文件不单独证明进程存活；进程与控制台日志需一并核对。']
    temporary = RUN / 'PROGRESS.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(RUN / 'PROGRESS.md')
    write_json(RUN / 'STATUS.json', dict(updated_utc=utc_now(), counts=counts, active=active, error=error))


def finish():
    cfg = read_json(CONFIG)
    rows = []
    for source in cfg['sources']:
        new = read_json(RUN / ('as' + source) / 'ordered_path/RESULTS.json')
        base = read_json(ROOT / cfg['control_run'] / ('as' + source) / cfg['control_variant'] / 'RESULTS.json')
        require(new['candidate_sha256'] == base['candidate_sha256']
                and new['query_count'] == base['query_count']
                and new['fixed_identity'] == base['fixed_identity'], 'Comparison changed frozen queries')
        row = dict(new, control_jaccard=base['jaccard_accuracy'],
                   control_exact=base['ordered_exact_accuracy'],
                   jaccard_delta=new['jaccard_accuracy'] - base['jaccard_accuracy'],
                   exact_delta=new['ordered_exact_accuracy'] - base['ordered_exact_accuracy'],
                   oracle_gap=new['oracle_jaccard'] - new['jaccard_accuracy'])
        row['timing'] = {stage: read_json(RUN / ('as' + source) / 'ordered_path' / ('TIMING_' + stage + '.json'))
                         for stage in ('train', 'predict', 'evaluate')}
        rows.append(row)
    total = sum(row['query_count'] for row in rows)

    def aggregate(subset):
        count = sum(row['query_count'] for row in subset)
        if not count:
            return dict(queries=0)
        fields = ('jaccard_accuracy', 'ordered_exact_accuracy', 'control_jaccard', 'control_exact',
                  'jaccard_delta', 'exact_delta', 'oracle_jaccard', 'oracle_gap')
        return dict(queries=count, sources=len(subset),
                    **{field: math.fsum(row[field] * row['query_count'] for row in subset) / count
                       for field in fields})

    summary = dict(micro=aggregate(rows), macro={field: math.fsum(row[field] for row in rows) / len(rows)
                   for field in ('jaccard_accuracy', 'ordered_exact_accuracy', 'jaccard_delta', 'exact_delta')},
                   by_rho={name: aggregate([row for row in rows if (row['rho'] > 1) == high])
                           for name, high in [('high', True), ('low', False)]},
                   changed_factor=cfg['changed_factor'], queries=total,
                   historical_test_results_already_known=True, final_labels_used_for_tuning=False)
    write_json(RUN / 'RESULTS.json', rows)
    write_json(RUN / 'SUMMARY.json', summary)
    with (RUN / 'RESULTS.csv').open('w', newline='') as stream:
        fields = ['source', 'rho', 'query_count', 'control_jaccard', 'jaccard_accuracy', 'jaccard_delta',
                  'control_exact', 'ordered_exact_accuracy', 'exact_delta', 'oracle_jaccard', 'oracle_gap']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 20260201：评估网络结构对照', '',
             '| AS | GMPI CE | 顺序交互网络 | 差值 | Oracle 差距 | 训练秒 | 预测秒 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {} | {:.6f} | {:.6f} | {:+.6f} | {:.6f} | {:.1f} | {:.1f} |'.format(
            row['source'], row['control_jaccard'], row['jaccard_accuracy'], row['jaccard_delta'], row['oracle_gap'],
            row['timing']['train']['wall_seconds'], row['timing']['predict']['wall_seconds']))
    lines += ['', '全量加权：' + json.dumps(summary['micro'], ensure_ascii=False), '',
              '网络结构（含容量及归纳偏置）是一个整体实验因素；暂不分别归因于每个内部模块。',
              '分数用于候选排序，未证明是校准概率。旧测试结果已知，本轮属于探索性研究。',
              '独立审计结果见 COMPLETION_AUDIT.json；审计完成前不视为最终结论。']
    (RUN / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def dispatch():
    cfg = read_json(CONFIG)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'tasks').mkdir(exist_ok=True)
    lock = (RUN / '.dispatch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(cfg['cpu_affinity']))
    for key, folder in (('MPLCONFIGDIR', 'matplotlib'), ('CUDA_CACHE_PATH', 'cuda'), ('TORCH_HOME', 'torch')):
        path = RUN / 'runtime' / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    engine = make_engine()
    stop = threading.Event()

    def monitor():
        while not stop.wait(15):
            report()

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    error = None
    try:
        if not checked(None, 'snapshot'):
            snapshot()
            freeze(receipt(None, 'snapshot'), dict(code_sha256=code_identity(), config_sha256=sha256(CONFIG),
                   outputs_sha256={str(RUN / 'SNAPSHOT.json'): sha256(RUN / 'SNAPSHOT.json')}))
        verify_files(read_json(RUN / 'SNAPSHOT.json')['files_sha256'])
        for stage in ('train', 'predict', 'evaluate'):
            for source in cfg['sources']:
                if checked(source, stage):
                    continue
                engine.emit(source, 'ordered_path', 'starting_' + stage)
                start, cpu = time.monotonic(), time.process_time()
                if engine._RUNTIME is not None:
                    engine._RUNTIME['torch'].cuda.synchronize()
                getattr(engine, stage)(source, 'ordered_path')
                if engine._RUNTIME is not None:
                    engine._RUNTIME['torch'].cuda.synchronize()
                out = engine.directory(source, 'ordered_path')
                write_json(out / ('TIMING_' + stage + '.json'), dict(wall_seconds=time.monotonic() - start,
                           cpu_seconds=time.process_time() - cpu,
                           peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           measurement='stage clocks with CUDA synchronization; first training includes environment load'))
                names = {'train': ['MODEL_FROZEN.json', 'model.pt', 'TRAINING_HISTORY.json'],
                         'predict': ['PREDICTIONS_FROZEN.json', 'ranked_predictions.csv'],
                         'evaluate': ['RESULTS.json']}[stage] + ['TIMING_' + stage + '.json']
                freeze(receipt(source, stage), dict(code_sha256=code_identity(), config_sha256=sha256(CONFIG),
                       source=source, stage=stage, completed_utc=utc_now(),
                       outputs_sha256={str(out / name): sha256(out / name) for name in names}))
            if stage in ('train', 'predict'):
                name = 'ALL_MODELS_FROZEN.json' if stage == 'train' else 'ALL_PREDICTIONS_FROZEN.json'
                freeze(RUN / name, dict(snapshot_sha256=sha256(RUN / 'SNAPSHOT.json'),
                       receipts_sha256={str(receipt(source, stage)): sha256(receipt(source, stage))
                                        for source in cfg['sources']}))
        finish()
        # This standalone verifier is separate from the numerical implementation.
        import subprocess
        subprocess.run([__import__('sys').executable, '-B', str(ROOT / 'tools/validation/audit_path_sequence_confidence.py')],
                       cwd=str(ROOT), check=True)
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        stop.set(); watcher.join(); report(error); lock.close()
