#!/usr/bin/env python3
"""Publish architecture comparisons and measured cost after the final audit.

This tool does not train, select checkpoints, generate candidates, or select a
different model per source. Numerical experiment identities remain untouched.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pathInference.experiment.contracts import require
from pathInference.utils import read_json, sha256, utc_now, write_json

MARKER = '<!-- edge-alignment-audited-conclusions -->'


def aggregate(rows):
    count = sum(r['query_count'] for r in rows)
    require(count > 0, 'Cannot aggregate an empty query group')
    return dict(queries=count, **{
        key: math.fsum(r[key] * r['query_count'] for r in rows) / count
        for key in ('jaccard_accuracy', 'ordered_exact_accuracy', 'oracle_jaccard')})


def check_scope(audit, cfg, rows):
    require(audit['status'] == 'PASS' and audit['independent_rescore']
            and audit['candidate_membership_checked'] and audit['aggregates_checked']
            and audit['csv_results_checked'] and audit['receipt_stage_order_checked']
            and audit['paired_initialization_and_effective_parameter_scope_checked']
            and audit['variants'] == cfg['variants']
            and [r['source'] for r in audit['sources']] == cfg['sources'],
            'A complete independent architecture audit is required')
    expected = {(source, variant) for source in cfg['sources'] for variant in cfg['variants']}
    require(len(rows) == len(expected)
            and {(r['source'], r['variant']) for r in rows} == expected,
            'Incomplete or duplicate source/architecture rows')
    for variant in cfg['variants']:
        require(sum(r['query_count'] for r in rows if r['variant'] == variant)
                == audit['queries_per_variant'], 'Audited query denominator differs')


def report():
    cfg = read_json(ROOT / 'research/edge_alignment_v1/CONFIG.json')
    run = ROOT / 'runs/20260201/edge_alignment_v1'
    audit = read_json(run / 'COMPLETION_AUDIT.json')
    rows = read_json(run / 'RESULTS.json')
    check_scope(audit, cfg, rows)
    controls = [read_json(ROOT / cfg['control_run'] / ('as' + s)
                         / cfg['control_variant'] / 'RESULTS.json') for s in cfg['sources']]
    for row in rows:
        base = next(r for r in controls if r['source'] == row['source'])
        require(row['candidate_sha256'] == base['candidate_sha256']
                and row['fixed_identity'] == base['fixed_identity']
                and row['query_count'] == base['query_count'], 'Comparison inputs differ')
    groups = {cfg['control_variant']: controls}
    groups.update({v: [r for r in rows if r['variant'] == v] for v in cfg['variants']})
    overall = {v: aggregate(values) for v, values in groups.items()}
    base = overall[cfg['control_variant']]
    comparisons = {}
    for variant in cfg['variants']:
        subset = groups[variant]
        deltas = [r['jaccard_accuracy'] - r['control_jaccard'] for r in subset]
        comparisons[variant] = dict(
            jaccard_delta=overall[variant]['jaccard_accuracy'] - base['jaccard_accuracy'],
            exact_percentage_point_delta=100 * (overall[variant]['ordered_exact_accuracy']
                                              - base['ordered_exact_accuracy']),
            source_wins=sum(d > 1e-12 for d in deltas),
            source_losses=sum(d < -1e-12 for d in deltas),
            source_ties=sum(abs(d) <= 1e-12 for d in deltas))
    costs = []
    for row in rows:
        directory = run / ('as' + row['source']) / row['variant']
        stages = {s: read_json(directory / ('TIMING_' + s + '.json'))
                  for s in ('train', 'predict', 'evaluate')}
        costs.append(dict(source=row['source'], variant=row['variant'], **{
            stage + '_' + key: value[key] for stage, value in stages.items()
            for key in ('wall_seconds', 'cpu_seconds')}))
    timing = {v: {stage + '_' + key: math.fsum(r[stage + '_' + key] for r in costs if r['variant'] == v)
                  for stage in ('train', 'predict', 'evaluate') for key in ('wall_seconds', 'cpu_seconds')}
              for v in cfg['variants']}
    write_json(run / 'TIMING_SUMMARY.json', dict(variants=timing,
        stage_wall_sum_seconds=math.fsum(sum(r[s + '_wall_seconds'] for s in ('train', 'predict', 'evaluate')) for r in costs),
        stage_cpu_sum_seconds=math.fsum(sum(r[s + '_cpu_seconds'] for s in ('train', 'predict', 'evaluate')) for r in costs),
        scope='measured stage sums including I/O, initialization and checkpoints; not isolated GPU time or elapsed makespan',
        hardware=read_json(run / 'RUNTIME.json')))
    with (run / 'TIMING.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(costs[0])); writer.writeheader(); writer.writerows(costs)
    conclusion = dict(updated_utc=utc_now(), overall=overall, comparisons=comparisons,
        final_labels_used_for_tuning=False, source_specific_test_selection=False,
        production_method_replaced=False, completion_audit_sha256=sha256(run / 'COMPLETION_AUDIT.json'),
        interpretation='Exploratory single-seed architecture comparison on an existing test set. No automatic production adoption.')
    write_json(run / 'CONCLUSION.json', conclusion)
    labels = {cfg['control_variant']: 'GMPI CE 参考', 'edge_only': '纯逐段对齐',
              'gmpi_edge_residual': 'GMPI＋逐段对齐'}
    lines = [MARKER, '', '独立审计通过后的统一结果（相同候选，按查询数加权）：', '',
             '| 配置 | 查询数 | Jaccard | 完整匹配率 | Oracle 差距 |', '|---|---:|---:|---:|---:|']
    for variant, value in overall.items():
        lines.append('| {} | {} | {:.6f} | {:.4%} | {:.6f} |'.format(labels[variant], value['queries'],
            value['jaccard_accuracy'], value['ordered_exact_accuracy'], value['oracle_jaccard'] - value['jaccard_accuracy']))
    lines += ['', '| 相对 GMPI CE | Jaccard 差值 | 完整匹配百分点差值 | 提升／下降／持平 AS 数 |',
              '|---|---:|---:|---:|']
    for variant, value in comparisons.items():
        lines.append('| {} | {:+.6f} | {:+.4f} | {}／{}／{} |'.format(labels[variant], value['jaccard_delta'],
            value['exact_percentage_point_delta'], value['source_wins'], value['source_losses'], value['source_ties']))
    lines += ['', '高低 rho 分组（各组按查询数加权）：', '',
              '| 配置 | rho 分组 | 查询数 | Jaccard | 完整匹配率 |', '|---|---|---:|---:|---:|']
    for variant, values in groups.items():
        for high, label in [(True, '>1'), (False, '≤1')]:
            value = aggregate([r for r in values if (r['rho'] > 1) == high])
            lines.append('| {} | {} | {} | {:.6f} | {:.4%} |'.format(labels[variant], label, value['queries'],
                value['jaccard_accuracy'], value['ordered_exact_accuracy']))
    lines += ['', '新结构实测阶段时间合计（秒）：', '',
              '| 配置 | 训练 wall／CPU | 预测 wall／CPU | 评测 wall／CPU |', '|---|---:|---:|---:|']
    for variant, value in timing.items():
        lines.append('| {} | {} |'.format(labels[variant], ' | '.join('{:.2f}／{:.2f}'.format(
            value[s + '_wall_seconds'], value[s + '_cpu_seconds']) for s in ('train', 'predict', 'evaluate'))))
    lines += ['', '计时包括 I/O、初始化与检查点，第一源训练含共享环境加载；阶段和不等于完整实验历时。',
              '逐 AS 的 wall／CPU 时间见 TIMING.csv。两组固定损失和训练参数，改变评分结构及同输入派生的匹配表示；'
              '不能分别归因于单个卷积、片段支持或残差模块。']
    for variant, value in comparisons.items():
        lines.append('{}相对 GMPI CE：总体 Jaccard {:+.6f}，完整匹配 {:+.4f} 个百分点，{} 个 AS 提升、{} 个下降。'.format(
            labels[variant], value['jaccard_delta'], value['exact_percentage_point_delta'], value['source_wins'], value['source_losses']))
    if all(v['jaccard_delta'] <= 0 for v in comparisons.values()):
        lines.append('本轮两种新结构均未提高总体 Jaccard，保留负结果，不替换现有方法。')
    else:
        lines.append('单种子结果只支持在本轮冻结条件下的观察，尚需独立日期或重复种子验证，生产方法未自动替换。')
    for variant, value in comparisons.items():
        if value['jaccard_delta'] <= 0 and value['exact_percentage_point_delta'] > 0:
            lines.append('{}的完整匹配率有所提高，平均 Jaccard 却未提高，属于指标间的取舍，不能称为全面优于对照。'.format(labels[variant]))
    lines += ['不按 test 逐源拼接最佳模型；最终排名分数尚未被证明是经过校准的真实路径概率。']
    path = run / 'REPORT.md'
    original = path.read_text().partition(MARKER)[0].rstrip()
    temporary = run / 'REPORT.derived.tmp.md'
    temporary.write_text(original + '\n\n' + '\n'.join(lines) + '\n'); temporary.replace(path)
    print('Published independently audited architecture comparisons and measured wall/CPU times', flush=True)


if __name__ == '__main__':
    report()
