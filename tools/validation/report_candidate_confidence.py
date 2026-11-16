#!/usr/bin/env python3
"""Combine independently verified candidate-confidence controls, without retuning."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pathInference.experiment.contracts import require, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json


def summarize(rows):
    require(rows and len({r['source'] for r in rows}) == len(rows), 'Duplicate or empty source scope')
    total = sum(r['queries'] for r in rows)
    require(total > 0, 'Empty evaluation')
    keys = ('jaccard', 'exact', 'oracle_jaccard', 'jaccard_oracle_gap', 'exact_available', 'exact_oracle_gap')
    result = dict(sources=len(rows), queries=total,
        micro={key: math.fsum(r[key] * r['queries'] for r in rows) / total for key in keys},
        macro={key: math.fsum(r[key] for r in rows) / len(rows) for key in keys})
    devices = {r['device'] for r in rows if r.get('device')}
    require(len(devices) <= 1, 'Method aggregate mixes hardware backends')
    if devices:
        result['device'] = next(iter(devices))
    for key in ('train_wall_seconds', 'train_cpu_seconds', 'predict_wall_seconds', 'predict_cpu_seconds'):
        known = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(measured_sources=len(known), total_sources=len(rows),
                           measured_sum=math.fsum(known) if known else None)
    return result


def measured(path, key):
    if not path.is_file():
        return None
    value = read_json(path)[key]
    require(math.isfinite(value) and value >= 0, 'Invalid recorded time')
    return value


def row(source, method, metric, oracle, timing_directory):
    result = dict(source=source, method=method, rho=metric['rho'], queries=metric['query_count'],
        jaccard=metric['jaccard_accuracy'], exact=metric['ordered_exact_accuracy'],
        oracle_jaccard=oracle['oracle_jaccard'], exact_available=oracle['candidate_exact_available'])
    result['jaccard_oracle_gap'] = result['oracle_jaccard'] - result['jaccard']
    result['exact_oracle_gap'] = result['exact_available'] - result['exact']
    require(result['jaccard_oracle_gap'] >= -1e-10 and result['exact_oracle_gap'] >= -1e-10,
            'Selection exceeds frozen candidate upper bound')
    for stage in ('train', 'predict'):
        for clock in ('wall', 'cpu'):
            result[stage + '_' + clock + '_seconds'] = measured(
                timing_directory / ('TIMING_' + stage + '.json'), clock + '_seconds')
    return result


def check_sequence_audit(audit, sources):
    records = audit['sources']
    require(audit['status'] == 'PASS' and audit['final_labels_read'] and audit['independent_rescore']
            and not audit['pending'] and len(records) == len(sources)
            and {r['source'] for r in records} == set(sources),
            'Ordered-network independent audit scope is incomplete')
    require(all(r['status'] == 'PASS' and r['independent_rescore']
                and r['candidate_membership']['status'] == 'PASS'
                and r['candidate_membership']['slot_capped_queries'] == 0 for r in records),
            'Ordered-network candidate pool/rescore differs')
    require(audit['queries'] == sum(r['candidate_membership']['queries'] for r in records),
            'Ordered-network audit denominator differs')


def build():
    root = ROOT / 'runs/20260201'
    ranking, gmpi, regression = [root / name for name in (
        'candidate_confidence_v2', 'gmpi_confidence_v1', 'candidate_scorer_v1')]
    configs = [read_json(ROOT / 'research' / name / 'CONFIG.json')
               for name in ('candidate_confidence_v2', 'gmpi_confidence_v1')]
    require(configs[0]['sources'] == configs[1]['sources'], 'Controlled source inventories differ')
    sources = configs[0]['sources']
    inputs = {}

    def bound(file):
        inputs[str(file)] = sha256(file)
        return read_json(file)

    audit_lgb = bound(ranking / 'COMPLETION_AUDIT.json')
    audit_gmpi = bound(gmpi / 'COMPLETION_AUDIT.json')
    audit_regression = bound(regression / 'COMPLETION_AUDIT.json')
    regression_records = {item['source']: item for item in audit_regression['completed_sources']}
    require(audit_regression['status'] == 'PASS' and not audit_regression['pending_sources']
            and len(audit_regression['completed_sources']) == len(sources)
            and set(regression_records) == set(sources)
            and all(item['status'] == 'PASS' and item['independent_rescore']
                    for item in regression_records.values()), 'Regression baseline audit is incomplete')
    runtime_gmpi = bound(gmpi / 'RUNTIME.json')
    require(runtime_gmpi['device_runtime']['effective'] == 'cuda', 'GMPI backend differs')
    require(audit_lgb['status'] == audit_gmpi['status'] == 'PASS'
            and audit_lgb['final_labels_read'] and audit_gmpi['final_labels_read'],
            'Both final independent rescores must pass')
    require(len(audit_lgb['completed_sources']) == 10 and len(audit_gmpi['final_results']) == 20
            and len(audit_lgb['candidate_membership']) == 10
            and audit_gmpi['weights_checked'] and len(audit_gmpi['candidate_membership']) == 10,
            'Independent verification scope is incomplete')
    require(all(r['slot_capped_queries'] == 0 for r in audit_gmpi['candidate_membership']),
            'GMPI sees a smaller candidate pool; compute its own oracle before combining results')
    sequence = root / 'path_sequence_confidence_v1'
    sequence_snapshot = None
    runtime_sequence = None
    if sequence.exists():
        sequence_audit = bound(sequence / 'COMPLETION_AUDIT.json')
        check_sequence_audit(sequence_audit, sources)
        sequence_cfg = bound(ROOT / 'research/path_sequence_confidence_v1/CONFIG.json')
        sequence_snapshot = bound(sequence / 'SNAPSHOT.json')
        runtime_sequence = bound(sequence / 'RUNTIME.json')
        require(runtime_sequence['device_runtime']['effective'] == 'cuda', 'Ordered-network backend differs')
        require(sequence_cfg['sources'] == sources
                and sequence_cfg['control_variant'] == 'query_cross_entropy'
                and sequence_cfg['label_temperature'] == configs[1]['label_temperature']
                and sequence_snapshot['sources'] == read_json(gmpi / 'SNAPSHOT.json')['sources'],
                'Ordered-network architecture control changed its inputs/objective')
        verify_files(sequence_snapshot['files_sha256'])
    rows = []
    for source in sources:
        original = bound(regression / ('as' + source) / 'RESULTS.json')
        require(sha256(regression / ('as' + source) / 'RESULTS.json')
                == regression_records[source]['result_sha256'], 'Audited regression result changed')
        ranking_metric = bound(ranking / ('as' + source) / 'RESULTS.json')
        snap_lgb = read_json(ranking / 'SNAPSHOT.json')['sources'][source]
        snap_gmpi = read_json(gmpi / 'SNAPSHOT.json')['sources'][source]
        require(snap_lgb['candidate_sha256'] == snap_gmpi['candidate_sha256'], 'Candidate bytes differ')
        metrics = [('lgb_regression', original, regression / ('as' + source)),
                   ('lgb_query_cross_entropy', ranking_metric, ranking / ('as' + source))]
        for variant in configs[1]['variants']:
            directory = gmpi / ('as' + source) / variant
            metrics.append(('gmpi_' + variant, bound(directory / 'RESULTS.json'), directory))
        if sequence_snapshot is not None:
            directory = sequence / ('as' + source) / 'ordered_path'
            require(sequence_snapshot['sources'][source]['candidate_sha256'] == snap_gmpi['candidate_sha256'],
                    'Ordered-network candidate bytes differ')
            for stage in ('train', 'predict', 'evaluate'):
                receipt = bound(sequence / 'tasks' / (source + '_ordered_path_' + stage + '.json'))
                verify_files(receipt['code_sha256']); verify_files(receipt['outputs_sha256'])
            metric = bound(directory / 'RESULTS.json')
            require(metric['candidate_sha256'] == snap_gmpi['candidate_sha256'],
                    'Ordered-network result binds other candidates')
            metrics.append(('ordered_path_network', metric, directory))
        for method, metric, directory in metrics:
            require(metric['query_count'] == original['query_count']
                    and metric['fixed_identity'] == original['fixed_identity'], 'Evaluation queries differ')
            item = row(source, method, metric, original, directory)
            if method.startswith('lgb_'):
                model = bound(directory / 'MODEL_FROZEN.json')
                model_file = Path(model['model'])
                require(sha256(model_file) == model['model_sha256']
                        and '[device_type: cpu]' in model_file.read_text(), 'LightGBM backend differs')
                inputs[str(model_file)] = model['model_sha256']
                item['device'] = 'CPU（' + str(model['parameters']['num_threads']) + ' 线程）'
            else:
                runtime = runtime_sequence if method == 'ordered_path_network' else runtime_gmpi
                item['device'] = runtime['gpu'] + ' / CUDA'
            item['candidate_sha256'] = snap_lgb['candidate_sha256']
            item['results_sha256'] = sha256(directory / 'RESULTS.json')
            for stage in ('train', 'predict'):
                file = directory / ('TIMING_' + stage + '.json')
                if file.is_file(): inputs[str(file)] = sha256(file)
            rows.append(item)
    methods = list(dict.fromkeys(r['method'] for r in rows))
    aggregates = {}
    for method in methods:
        selected = [r for r in rows if r['method'] == method]
        aggregates[method] = {'all': summarize(selected)}
        for label, high in (('rho_le_1', False), ('rho_gt_1', True)):
            subset = [r for r in selected if (r['rho'] > 1) == high]
            if subset: aggregates[method][label] = summarize(subset)
    comparisons = {}
    pairs = [('LightGBM', 'lgb_regression', 'lgb_query_cross_entropy', 'training objective only'),
             ('GMPI', 'gmpi_smooth_l1', 'gmpi_query_cross_entropy', 'training objective only')]
    if sequence_snapshot is not None:
        pairs.append(('Network', 'gmpi_query_cross_entropy', 'ordered_path_network',
                      'network architecture only; same GMPI inputs, query objective and training arguments'))
    for name, control, changed, factor in pairs:
        paired = []
        for source in sources:
            a = next(r for r in rows if r['source'] == source and r['method'] == control)
            b = next(r for r in rows if r['source'] == source and r['method'] == changed)
            paired.append(dict(source=source, rho=a['rho'], queries=a['queries'],
                jaccard_change=b['jaccard']-a['jaccard'], exact_change=b['exact']-a['exact']))
        comparisons[name] = dict(control=control, changed=changed, changed_factor=factor,
            sources=paired, jaccard_wins=sum(r['jaccard_change'] > 0 for r in paired),
            jaccard_losses=sum(r['jaccard_change'] < 0 for r in paired))
    return dict(date='20260201', created_utc=utc_now(), sources=sources, rows=rows, aggregates=aggregates,
        comparisons=comparisons, inputs_sha256=inputs,
        historical_test_results_already_known=True, final_labels_used_for_tuning=False,
        timing_notes='Sums of measured per-source stage wall/CPU clocks, not elapsed makespan for parallel work; '
                     'no invented estimates for unavailable historical timings. '
                     'GPU stage clocks include loading, validation and checkpoint I/O. RSS is cumulative process peak. '
                     'Legacy LightGBM training can include feature-matrix construction when uncached; '
                     'the objective control reuses audited matrices. Stage time differences are not isolated fit speedups.',
        hardware_notes='LightGBM uses CPU prediction/features; GMPI and the ordered network use the recorded CUDA GPU. '
                       'Cross-family stage times do not isolate hardware, feature extraction or architecture costs.',
        scope_notes='GMPI and LightGBM each have objective-only controls. The optional ordered network changes architecture '
                    'relative to GMPI cross entropy while retaining its inputs and objective. '
                    'Across GMPI and LightGBM, architecture/features also differ. '
                    'Same actual rebuilt candidate bytes are used here; historical deleted candidate sets are context only.')


def write_report(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    verify_files(result['inputs_sha256'])
    write_json(directory / 'SUMMARY.json', result)
    with (directory / 'RESULTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
        writer.writeheader(); writer.writerows(result['rows'])
    protocol = '全部十个 AS；复用同一冻结候选与增强库。GMPI、LightGBM 分别做训练目标对照。'
    if 'Network' in result['comparisons']:
        protocol += '顺序交互网络固定 GMPI 交叉熵目标及输入，只改变网络结构。'
    protocol += '跨模型家族同时存在结构、特征差异。'
    lines = ['# 20260201：冻结候选后的置信度评估对照', '', protocol, '',
        '| 方法 | 查询数 | Jaccard | 完整匹配率 | Oracle Jaccard | Jaccard 差距 |',
        '|---|---:|---:|---:|---:|---:|']
    for method, groups in result['aggregates'].items():
        summary = groups['all']; m = summary['micro']
        lines.append(f"| {method} | {summary['queries']} | {m['jaccard']:.6f} | {m['exact']:.4%} | "
                     f"{m['oracle_jaccard']:.6f} | {m['jaccard_oracle_gap']:.6f} |")
    lines += ['', '以下均为相同查询上的损失对照；差值为交叉熵组减原目标组。', '',
              '| AS | ρ | LGB Jaccard 差值 | LGB 完整匹配差值 | GMPI Jaccard 差值 | GMPI 完整匹配差值 |',
              '|---|---:|---:|---:|---:|---:|']
    for a, b in zip(result['comparisons']['LightGBM']['sources'], result['comparisons']['GMPI']['sources']):
        lines.append(f"| {a['source']} | {a['rho']:.4f} | {a['jaccard_change']:+.6f} | {a['exact_change']:+.4%} | "
                     f"{b['jaccard_change']:+.6f} | {b['exact_change']:+.4%} |")
    if 'Network' in result['comparisons']:
        network = result['comparisons']['Network']
        lines += ['', '网络结构对照：顺序交互网络减 GMPI 交叉熵组；不改变损失或输入。', '',
                  '| AS | ρ | Jaccard 差值 | 完整匹配差值 |', '|---|---:|---:|---:|']
        for item in network['sources']:
            lines.append(f"| {item['source']} | {item['rho']:.4f} | {item['jaccard_change']:+.6f} | "
                         f"{item['exact_change']:+.4%} |")
        lines += ['', f"网络对照中 Jaccard 提升 {network['jaccard_wins']} 源、下降 {network['jaccard_losses']} 源。"
                  '保留全部负结果，不按 test 为各源挑选最优模型。']
    lines += ['', '各源训练与预测阶段实测耗时之和（秒）；并行阶段之和不等于整体实验历时。'
              '未保留历史计时的项目记为缺失。LightGBM 使用 CPU，神经网络使用 GPU；'
              '跨家族耗时同时含硬件、特征处理和结构差异。'
              '旧 LightGBM 训练阶段在未缓存时包含特征矩阵构造，新排序训练复用已审计矩阵；'
              '不能把阶段耗时之差直接视为目标函数带来的纯训练加速。', '',
              '| 方法 | 硬件后端 | 训练墙钟合计 | 训练 CPU 合计 | 预测墙钟合计 | 预测 CPU 合计 |',
              '|---|---|---:|---:|---:|---:|']
    for method, groups in result['aggregates'].items():
        cells = []
        for key in ('train_wall_seconds', 'train_cpu_seconds', 'predict_wall_seconds', 'predict_cpu_seconds'):
            value = groups['all'][key]
            cells.append('缺失' if value['measured_sum'] is None else
                         f"{value['measured_sum']:.2f}（{value['measured_sources']}/10）")
        lines.append('| ' + ' | '.join([method, groups['all'].get('device', '见配置'), *cells]) + ' |')
    lines += ['', '逐 AS 原始数值、oracle 差距和实测计时见 RESULTS.csv。高/低 ρ 分组与宏平均见 SUMMARY.json。', '',
              '测试标签只用于全部预测冻结后的评测与独立复算。历史测试结果已知，本轮是探索性对照；未按测试结果调参。',
              '这里的分数用于选择路径，尚未证明是经过校准的真实概率。']
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    result = build()
    destination = ROOT / 'runs/20260201/confidence_comparison'
    write_report(result, destination)
    print(destination / 'REPORT.md')
