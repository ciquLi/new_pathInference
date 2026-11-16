"""Architecture experiment reusing immutable GMPI training and prediction APIs."""
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
CONFIG = ROOT / 'research/edge_alignment_v1/CONFIG.json'
RUN = ROOT / 'runs/20260201/edge_alignment_v1'


def code_identity():
    paths = [ROOT / 'src/pathInference/research' / name for name in
             ('edge_alignment_confidence.py', 'edge_alignment_experiment.py',
              'gmpi_confidence.py', 'gmpi_confidence_loss.py')]
    paths += [ROOT / 'src/pathInference/core' / name for name in
              ('GMPI_pathLikelihoodEst.py', 'myDataset.py', 'PathInference_trainer.py', 'environment.py')]
    paths += [ROOT / 'research/edge_alignment_v1/run.py']
    return {str(path): sha256(path) for path in paths}


def make_engine():
    spec = importlib.util.spec_from_file_location('pathInference.research._edge_alignment_engine',
                                                 ROOT / 'src/pathInference/research/gmpi_confidence.py')
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    engine.CONFIG, engine.RUN = CONFIG, RUN
    engine.active_variant, engine.active_stage = None, None

    def make_model(runtime, arguments):
        from pathInference.research.edge_alignment_confidence import EdgeAlignmentConfidence
        cfg = read_json(CONFIG)
        require(engine.active_variant in cfg['variants'], 'Declare architecture before creating a model')
        model = EdgeAlignmentConfidence(runtime['torch'].from_numpy(runtime['vectors']),
            hidden_size=arguments['hidden_size'], use_base=engine.active_variant == 'gmpi_edge_residual',
            **cfg['architecture']).to(runtime['device'])
        if engine.active_stage == 'train':
            base = {name: value for name, value in model.state_dict().items() if not name.startswith('alignment.')}
            old = read_json(ROOT / cfg['control_run'] / ('as' + arguments['source_asn']) / 'INITIALIZATION.json')
            require(engine.tensor_hash(base) == old['state_dict_sha256'], 'Original GMPI cold initialization differs')
            parameters = {name: value for name, value in model.named_parameters()}
            freeze(engine.directory(arguments['source_asn'], engine.active_variant) / 'ARCHITECTURE.json',
                dict(use_base=model.use_base, input_semantics_version=model.input_semantics_version,
                     base_initial_sha256=engine.tensor_hash(base),
                     full_initial_sha256=engine.tensor_hash(model.state_dict()),
                     alignment_initial_sha256=engine.tensor_hash(model.alignment.state_dict()),
                     total_parameters=sum(p.numel() for p in parameters.values()),
                     trainable_parameters=sum(p.numel() for p in parameters.values() if p.requires_grad),
                     trainable_parameter_names=[name for name, p in parameters.items() if p.requires_grad],
                     dormant_modules=[] if model.use_base else ['pr', 'pm'],
                     final_correction_head_zero=bool(model.alignment.output[-1].weight.eq(0).all()
                                                     and model.alignment.output[-1].bias.eq(0).all()),
                     pretrained_checkpoint_loaded=False))
        return model

    engine.make_model = make_model
    return engine


def snapshot():
    cfg = read_json(CONFIG)
    original = read_json(ROOT / cfg['source_snapshot'])
    control = ROOT / cfg['control_run']
    resource_file = RUN / 'RESOURCE_CHECK.json'
    resource_check = read_json(resource_file)
    require(resource_check['config_sha256'] == sha256(CONFIG)
            and resource_check['model_code_sha256'] == sha256(ROOT / 'src/pathInference/research/edge_alignment_confidence.py')
            and resource_check['cpu_threads'] == cfg['cpu_threads']
            and [r['variant'] for r in resource_check['results']] == cfg['variants']
            and all(r['finite_forward_backward'] and r['initial_base_hash_matches_control']
                    and r['shape'] == dict(batch=64, candidates=200, histories=200, nodes=20)
                    for r in resource_check['results']), 'Current architecture must pass full CUDA resource check')
    audit = read_json(control / 'COMPLETION_AUDIT.json')
    require(audit['status'] == 'PASS' and not audit['pending'] and audit['weights_checked'],
            'GMPI CE control must pass independent audit')
    for field in ('sources', 'seed', 'date', 'label_temperature'):
        require(cfg[field] == original['configuration'][field], 'Changed control field: ' + field)
    require(not cfg['pretraining'] and not cfg['additional_evidence'] and not cfg['candidates_regenerated']
            and not cfg['final_labels_used_for_tuning'], 'Frozen architecture contract differs')
    files = dict(original['files_sha256']); files.update(code_identity())
    for path in (CONFIG, ROOT / cfg['source_snapshot'], control / 'COMPLETION_AUDIT.json', ROOT / cfg['raw_vector_cache'], resource_file):
        files[str(path)] = sha256(path)
    for source in cfg['sources']:
        directory = control / ('as' + source)
        for path in [directory / 'INITIALIZATION.json',
                     *[directory / cfg['control_variant'] / name for name in
                       ('MODEL_FROZEN.json', 'PREDICTIONS_FROZEN.json', 'model.pt', 'ranked_predictions.csv', 'RESULTS.json')]]:
            files[str(path)] = sha256(path)
    verify_files(files)
    freeze(RUN / 'SNAPSHOT.json', dict(configuration=cfg, sources=original['sources'], files_sha256=files,
        data_manifest=original['data_manifest'], candidates_regenerated=False, additional_evidence=False,
        test_labels_used_for_tuning=False, historical_test_results_already_known=True,
        feature_schema='same seven GMPI tensors; deterministic directed-fragment/suffix features are derived in the new network'))


def receipt(source, variant, stage):
    return RUN / 'tasks' / ('_'.join(p for p in (source, variant, stage) if p) + '.json')


def checked(source, variant, stage):
    path = receipt(source, variant, stage)
    if not path.exists():
        return False
    value = read_json(path)
    require(value['code_sha256'] == code_identity() and value['config_sha256'] == sha256(CONFIG),
            'Verified cache code/config changed')
    verify_files(value['outputs_sha256'])
    return True


def record_task(source, variant, stage, paths):
    freeze(receipt(source, variant, stage), dict(source=source, variant=variant, stage=stage,
        completed_utc=utc_now(), code_sha256=code_identity(), config_sha256=sha256(CONFIG),
        outputs_sha256={str(path): sha256(path) for path in paths}))


def progress(error=None):
    cfg = read_json(CONFIG)
    active = read_json(RUN / 'ACTIVE.json') if (RUN / 'ACTIVE.json').exists() else {}
    complete = (RUN / 'COMPLETION_AUDIT.json').exists() and read_json(RUN / 'COMPLETION_AUDIT.json')['status'] == 'PASS'
    counts = {stage: sum(receipt(s, v, stage).exists() for s in cfg['sources'] for v in cfg['variants'])
              for stage in ('train', 'predict', 'evaluate')}
    lines = ['# 20260201：有向片段逐段对齐网络', '', '更新时间：' + utc_now(), '',
             'PID：{}；1 GPU、2 CPU。'.format(os.getpid()),
             '两种结构：纯局部对齐评分、GMPI＋局部对齐修正；不是预训练对照。',
             '候选、公开证据、增强库、划分、AS 向量初值、损失和正式训练参数固定。', '',
             '| AS | 纯对齐：训练／预测／评测 | GMPI＋对齐：训练／预测／评测 |', '|---|---|---|']
    for source in cfg['sources']:
        cells = [source]
        for variant in cfg['variants']:
            cells.append('／'.join('完成' if receipt(source, variant, stage).exists() else '等待'
                                  for stage in ('train', 'predict', 'evaluate')))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '当前活动：' + json.dumps(active, ensure_ascii=False), '',
              '全部完成并独立审计通过。' if complete else '全部 20 模型冻结后预测；全部 20 份预测冻结后最终评测。',
              '每 15 秒刷新；实际工作量见 epoch 和已落盘查询数，心跳时间不代表工作量。', '',
              '结果：[REPORT.md](REPORT.md)；数值：[RESULTS.csv](RESULTS.csv)；审计：[COMPLETION_AUDIT.json](COMPLETION_AUDIT.json)。']
    if error:
        lines += ['', '错误：' + error]
    temporary = RUN / 'PROGRESS.md.tmp'; temporary.write_text('\n'.join(lines) + '\n'); temporary.replace(RUN / 'PROGRESS.md')
    write_json(RUN / 'STATUS.json', dict(updated_utc=utc_now(), pid=os.getpid(), counts=counts,
                                       active=active, error=error, complete=complete))


def finish():
    cfg = read_json(CONFIG)
    rows = []
    for source in cfg['sources']:
        base = read_json(ROOT / cfg['control_run'] / ('as' + source) / cfg['control_variant'] / 'RESULTS.json')
        for variant in cfg['variants']:
            out = RUN / ('as' + source) / variant
            value = read_json(out / 'RESULTS.json')
            require(value['candidate_sha256'] == base['candidate_sha256'] and value['query_count'] == base['query_count']
                    and value['fixed_identity'] == base['fixed_identity'], 'Compared different candidates or queries')
            rows.append(dict(value, control_jaccard=base['jaccard_accuracy'], control_exact=base['ordered_exact_accuracy'],
                jaccard_delta=value['jaccard_accuracy'] - base['jaccard_accuracy'],
                exact_delta=value['ordered_exact_accuracy'] - base['ordered_exact_accuracy'],
                oracle_gap=value['oracle_jaccard'] - value['jaccard_accuracy'],
                timing={stage: read_json(out / ('TIMING_' + stage + '.json')) for stage in ('train', 'predict', 'evaluate')}))

    def aggregate(values):
        count = sum(r['query_count'] for r in values)
        if not count:
            return dict(queries=0)
        return dict(queries=count, sources=len(values), **{key: math.fsum(r[key] * r['query_count'] for r in values) / count
            for key in ('jaccard_accuracy', 'ordered_exact_accuracy', 'control_jaccard', 'control_exact',
                        'jaccard_delta', 'exact_delta', 'oracle_jaccard', 'oracle_gap')})
    summary = dict(variants={}, final_labels_used_for_tuning=False, candidates_regenerated=False,
                   historical_test_results_already_known=True)
    for variant in cfg['variants']:
        subset = [r for r in rows if r['variant'] == variant]
        summary['variants'][variant] = dict(queries=sum(r['query_count'] for r in subset), micro=aggregate(subset),
            macro={key: math.fsum(r[key] for r in subset) / len(subset)
                   for key in ('jaccard_accuracy', 'ordered_exact_accuracy', 'jaccard_delta', 'exact_delta')},
            by_rho={label: aggregate([r for r in subset if (r['rho'] > 1) == high])
                    for label, high in [('high', True), ('low', False)]})
    write_json(RUN / 'RESULTS.json', rows); write_json(RUN / 'SUMMARY.json', summary)
    with (RUN / 'RESULTS.csv').open('w', newline='') as stream:
        fields = ['source', 'variant', 'rho', 'query_count', 'jaccard_accuracy', 'ordered_exact_accuracy',
                  'control_jaccard', 'control_exact', 'jaccard_delta', 'exact_delta', 'oracle_jaccard', 'oracle_gap']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    lines = ['# 20260201：逐段对齐评分网络结构对照', '',
             '| AS | GMPI CE | 纯逐段对齐 | GMPI＋逐段对齐 |', '|---|---:|---:|---:|']
    for source in cfg['sources']:
        a, b = [next(r for r in rows if r['source'] == source and r['variant'] == v) for v in cfg['variants']]
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} |'.format(source, a['control_jaccard'], a['jaccard_accuracy'], b['jaccard_accuracy']))
    lines += ['', '| 配置 | 查询数 | Jaccard | 完整匹配率 | Oracle 差距 |', '|---|---:|---:|---:|---:|']
    for variant in cfg['variants']:
        value = summary['variants'][variant]['micro']
        lines.append('| {} | {} | {:.6f} | {:.4%} | {:.6f} |'.format(variant, value['queries'],
            value['jaccard_accuracy'], value['ordered_exact_accuracy'], value['oracle_gap']))
    lines += ['', '| AS | 配置 | 训练秒 | 预测秒 | 评测秒 |', '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append('| {} | {} | {:.2f} | {:.2f} | {:.2f} |'.format(r['source'], r['variant'],
            *[r['timing'][stage]['wall_seconds'] for stage in ('train', 'predict', 'evaluate')]))
    lines += ['', '阶段计时含 I/O、初始化与检查点，不等于隔离条件下的纯 GPU 时间。第一源训练还包含环境加载。',
              '两组使用相同 AS 向量初值和相同局部网络初始权重；纯对齐组的 pr/pm 为未使用的冻结兼容参数，实际训练参数数见 ARCHITECTURE.json。',
              '该对照改变网络结构及其由相同输入派生的匹配特征，不单独归因于卷积、片段特征或残差中的某一项。',
              '固定候选、证据、增强库、划分、损失和正式训练超参数；没有新预训练，不按 test 逐源挑选配置。',
              '旧测试结果已知，本轮是探索性单种子对照。分数用于排序，不宣称经过真实性概率校准。',
              '独立审计结果见 COMPLETION_AUDIT.json，审计完成前不视为最终结论。']
    (RUN / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def dispatch():
    cfg = read_json(CONFIG)
    RUN.mkdir(parents=True, exist_ok=True); (RUN / 'tasks').mkdir(exist_ok=True)
    lock = (RUN / '.dispatch.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(cfg['cpu_affinity']))
    engine, stop = make_engine(), threading.Event()
    def watch():
        while not stop.wait(15):
            progress()
    watcher = threading.Thread(target=watch, daemon=True); watcher.start(); error = None
    progress()
    try:
        if not checked(None, None, 'snapshot'):
            engine.emit(None, None, 'verifying_frozen_inputs')
            snapshot(); record_task(None, None, 'snapshot', [RUN / 'SNAPSHOT.json'])
        verify_files(read_json(RUN / 'SNAPSHOT.json')['files_sha256'])
        for stage in ('train', 'predict', 'evaluate'):
            engine.active_stage = stage
            for source in cfg['sources']:
                for variant in cfg['variants']:
                    if checked(source, variant, stage):
                        continue
                    engine.active_variant = variant
                    engine.emit(source, variant, 'starting_' + stage)
                    if engine._RUNTIME is not None:
                        engine._RUNTIME['torch'].cuda.synchronize()
                    start, cpu = time.monotonic(), time.process_time()
                    getattr(engine, stage)(source, variant)
                    if engine._RUNTIME is not None:
                        engine._RUNTIME['torch'].cuda.synchronize()
                    out = engine.directory(source, variant)
                    timing = out / ('TIMING_' + stage + '.json')
                    write_json(timing, dict(wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        measurement='stage wall/CPU with CUDA synchronization; first train includes shared environment load'))
                    names = {'train': ['MODEL_FROZEN.json', 'model.pt', 'TRAINING_HISTORY.json', 'ARCHITECTURE.json'],
                             'predict': ['PREDICTIONS_FROZEN.json', 'ranked_predictions.csv'], 'evaluate': ['RESULTS.json']}[stage]
                    record_task(source, variant, stage, [out / name for name in names] + [timing])
            if stage in ('train', 'predict'):
                name = 'ALL_MODELS_FROZEN.json' if stage == 'train' else 'ALL_PREDICTIONS_FROZEN.json'
                freeze(RUN / name, dict(snapshot_sha256=sha256(RUN / 'SNAPSHOT.json'),
                    receipts_sha256={str(receipt(s, v, stage)): sha256(receipt(s, v, stage))
                                    for s in cfg['sources'] for v in cfg['variants']}))
        finish()
        engine.emit(None, None, 'independent_final_audit')
        import subprocess
        subprocess.run([__import__('sys').executable, '-B', str(ROOT / 'tools/validation/audit_edge_alignment.py')],
                       cwd=str(ROOT), check=True)
    except BaseException as exc:
        error = str(exc); raise
    finally:
        stop.set(); watcher.join(); progress(error); lock.close()
