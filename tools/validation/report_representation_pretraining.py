#!/usr/bin/env python3
"""Readable cost and controlled-comparison conclusions, after independent audit.

This tool never launches numerical work or selects models. Shared AS pretraining
is charged to each configuration for cost comparison, and counted only once in
the actual new-work total. Stage sums are not parallel experiment makespan.
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


def report():
    import numpy as np
    cfg = read_json(ROOT / 'research/representation_pretraining_v1/CONFIG.json')
    run = ROOT / 'runs/20260201/representation_pretraining_v1'
    audit = read_json(run / 'COMPLETION_AUDIT.json')
    require(audit['status'] == 'PASS' and audit['independently_rescored']
            and audit['aggregates_checked'] and audit['csv_results_checked']
            and [r['source'] for r in audit['sources']] == cfg['sources']
            and all(r['independently_rescored_variants'] == cfg['variants'] for r in audit['sources']),
            'Complete independent audit is required before publishing conclusions')
    rows = read_json(run / 'RESULTS.json')
    summary = read_json(run / 'SUMMARY.json')
    require(set(summary['variants']) == set(cfg['variants']), 'Summary configuration scope differs')
    corpus_rows = []
    for source in cfg['sources']:
        manifest = read_json(run / ('as' + source) / 'CORPUS.json')
        encoded = np.load(manifest['corpus'], allow_pickle=False)
        require(sha256(manifest['corpus']) == manifest['corpus_sha256'], 'Training corpus identity changed')
        vocabulary = read_json(ROOT / cfg['frozen_candidate_run'] / ('as' + source) / 'VOCABULARY.json')
        nodes = set(map(int, encoded[encoded != 0]))
        lengths = (encoded != 0).sum(1)
        edges = {(int(a), int(b)) for path in encoded for a, b in zip(path[:-1], path[1:]) if a and b}
        corpus_rows.append(dict(source=source, paths=len(encoded), distinct_asns=len(nodes),
            vocabulary_asns=len(vocabulary) - 5, vocabulary_fraction=len(nodes) / (len(vocabulary) - 5),
            directed_edges=len(edges), minimum_nodes=int(lengths.min()), median_nodes=float(np.median(lengths)),
            maximum_nodes=int(lengths.max()), corpus_sha256=manifest['corpus_sha256']))
    write_json(run / 'CORPUS_DIAGNOSTICS.json', dict(date=cfg['date'], sources=corpus_rows,
        scope='training corpus only; vocabulary coverage is not held-out route or candidate coverage', final_labels_read=False))
    metrics = {}
    for source in cfg['sources']:
        for name, file in [('cold', ROOT / cfg['cold_control'] / ('as' + source) / 'ordered_path/RESULTS.json'),
                           ('gmpi_ce', ROOT / cfg['gmpi_control'] / ('as' + source) / 'query_cross_entropy/RESULTS.json')]:
            metrics[source, name] = read_json(file)
        for variant in cfg['variants']:
            metrics[source, variant] = next(row for row in rows if row['source'] == source and row['variant'] == variant)
    costs, actual_new_wall, actual_new_cpu = [], 0., 0.
    for source in cfg['sources']:
        pre = {stage: read_json(run / ('as' + source) / ('pretrain_' + stage) / 'TIMING_pretrain.json')
               for stage in ('as_context', 'path_context')}
        actual_new_wall += sum(r['wall_seconds'] for r in pre.values())
        actual_new_cpu += sum(r['cpu_seconds'] for r in pre.values())
        for variant in cfg['variants']:
            stages = {stage: read_json(run / ('as' + source) / variant / ('TIMING_' + stage + '.json'))
                      for stage in ('train', 'predict', 'evaluate')}
            charge = ['as_context'] if variant == 'as_context' else ['as_context', 'path_context']
            row = dict(source=source, variant=variant,
                as_pretraining_wall_seconds=pre['as_context']['wall_seconds'],
                path_pretraining_wall_seconds=pre['path_context']['wall_seconds'] if variant == 'as_and_path' else 0.,
                pretraining_cpu_seconds=sum(pre[name]['cpu_seconds'] for name in charge),
                **{stage + '_' + clock: timing[clock] for stage, timing in stages.items()
                   for clock in ('wall_seconds', 'cpu_seconds')})
            row['configuration_wall_seconds'] = (row['as_pretraining_wall_seconds']
                + row['path_pretraining_wall_seconds'] + sum(r['wall_seconds'] for r in stages.values()))
            row['configuration_cpu_seconds'] = row['pretraining_cpu_seconds'] + sum(r['cpu_seconds'] for r in stages.values())
            costs.append(row)
            actual_new_wall += sum(r['wall_seconds'] for r in stages.values())
            actual_new_cpu += sum(r['cpu_seconds'] for r in stages.values())
    clock = dict(rows=costs, actual_new_stage_wall_sum_seconds=actual_new_wall,
        actual_new_stage_cpu_sum_seconds=actual_new_cpu,
        scope='measured stage sums including I/O and checkpoint work; not experiment elapsed makespan',
        shared_as_pretraining_counted_once_in_actual_total=True,
        shared_as_pretraining_charged_to_each_configuration=True,
        first_as_pretraining_includes_environment_load=True,
        hardware=read_json(run / 'RUNTIME.json'))
    write_json(run / 'TIMING_SUMMARY.json', clock)
    with (run / 'TIMING.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(costs[0])); writer.writeheader(); writer.writerows(costs)

    def aggregate(name, subset):
        values = [metrics[source, name] for source in subset]
        count = sum(r['query_count'] for r in values)
        return dict(queries=count,
            jaccard=math.fsum(r['jaccard_accuracy'] * r['query_count'] for r in values) / count,
            exact=math.fsum(r['ordered_exact_accuracy'] * r['query_count'] for r in values) / count,
            oracle=math.fsum(r['oracle_jaccard'] * r['query_count'] for r in values) / count)
    comparisons = {}
    for new, base in [('as_context', 'cold'), ('as_and_path', 'as_context'),
                      ('as_and_path', 'cold'), ('as_context', 'gmpi_ce'), ('as_and_path', 'gmpi_ce')]:
        a, b = aggregate(new, cfg['sources']), aggregate(base, cfg['sources'])
        deltas = [metrics[source, new]['jaccard_accuracy'] - metrics[source, base]['jaccard_accuracy']
                  for source in cfg['sources']]
        comparisons[new + '_vs_' + base] = dict(jaccard_delta=a['jaccard'] - b['jaccard'],
            exact_percentage_point_delta=100 * (a['exact'] - b['exact']),
            source_wins=sum(value > 1e-12 for value in deltas),
            source_losses=sum(value < -1e-12 for value in deltas),
            source_ties=sum(abs(value) <= 1e-12 for value in deltas))
    conclusion = dict(updated_utc=utc_now(), comparisons=comparisons,
        final_labels_used_for_tuning=False, source_specific_test_selection=False,
        as_pretraining_improves_overall_jaccard=comparisons['as_context_vs_cold']['jaccard_delta'] > 0,
        path_pretraining_adds_overall_jaccard_gain=comparisons['as_and_path_vs_as_context']['jaccard_delta'] > 0,
        production_method_replaced=False,
        interpretation='Exploratory single-seed controlled initialization experiment on an existing test set; no automatic production adoption.',
        completion_audit_sha256=sha256(run / 'COMPLETION_AUDIT.json'))
    write_json(run / 'CONCLUSION.json', conclusion)
    lines = ['<!-- representation-pretraining-derived-report -->', '',
             '统一比较（相同候选和查询，按查询数加权）：', '',
             '| 配置 | 查询数 | Jaccard | 完整匹配率 | Oracle 差距 |', '|---|---:|---:|---:|---:|']
    for name, label in [('gmpi_ce', 'GMPI CE 参考'), ('cold', '同结构冷启动'),
                        ('as_context', 'AS 预训练'), ('as_and_path', 'AS＋路径预训练')]:
        value = aggregate(name, cfg['sources'])
        lines.append('| {} | {} | {:.6f} | {:.4%} | {:.6f} |'.format(label, value['queries'], value['jaccard'],
                     value['exact'], value['oracle'] - value['jaccard']))
    lines += ['', '控制变量的总体差值：', '',
              '| 对照 | Jaccard 差值 | 完整匹配百分点差值 | 提升／下降／持平的 AS |', '|---|---:|---:|---:|']
    for name, value in comparisons.items():
        lines.append('| {} | {:+.6f} | {:+.4f} | {}／{}／{} |'.format(name, value['jaccard_delta'],
                     value['exact_percentage_point_delta'], value['source_wins'], value['source_losses'], value['source_ties']))
    lines += ['', '自监督语料规模（仅训练端点；AS 词表占比不代表测试路径覆盖率）：', '',
              '| AS | 训练路径 | 不同 AS | AS 词表占比 | 不同有向跳转 | 路径节点数中位数／最大值 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in corpus_rows:
        lines.append('| {} | {} | {} | {:.2%} | {} | {:g}／{} |'.format(row['source'], row['paths'],
            row['distinct_asns'], row['vocabulary_fraction'], row['directed_edges'], row['median_nodes'], row['maximum_nodes']))
    lines += ['', '高低 rho 分组（每组分别按查询数加权）：', '',
              '| 配置 | rho 分组 | Jaccard | 完整匹配率 |', '|---|---|---:|---:|']
    for name in ['cold', *cfg['variants'], 'gmpi_ce']:
        for label, high in [('高', True), ('低', False)]:
            group = [source for source in cfg['sources'] if (metrics[source, name]['rho'] > 1) == high]
            value = aggregate(name, group)
            lines.append('| {} | {} | {:.6f} | {:.4%} |'.format(name, label, value['jaccard'], value['exact']))
    lines += ['', '实测墙钟时间（秒）：', '',
              '| AS | 配置 | AS 预训练 | 路径预训练 | 排序微调 | 预测 | 评测 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in costs:
        lines.append('| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(row['source'], row['variant'],
            row['as_pretraining_wall_seconds'], row['path_pretraining_wall_seconds'], row['train_wall_seconds'],
            row['predict_wall_seconds'], row['evaluate_wall_seconds']))
    lines += ['', '新工作实际阶段时间合计：wall {:.2f} s，CPU {:.2f} s。AS 预训练在实际合计中只计一次；'.format(
              actual_new_wall, actual_new_cpu) + '逐配置成本比较中，两配置各计一次共享初始化成本。',
              '第一源 AS 预训练含公共环境加载；计时含 I/O、检查点与初始化，阶段合计不等于并行环境下的完整实验历时，'
              '不据此宣称纯 GPU 计算加速。逐项 wall/CPU 时间见 TIMING.csv 和 TIMING_SUMMARY.json。',
              '冷启动→AS 预训练、AS 预训练→AS＋路径预训练分别改变一个阶段。与 GMPI CE 的跨结构差异不能仅归因于预训练。',
              '这些是已有测试集上的探索性单种子结果，不按测试逐源挑模型。负结果同样保留，冻结生产方法未自动替换。']
    a, b = comparisons['as_context_vs_cold'], comparisons['as_and_path_vs_as_context']
    lines += ['', '本轮结论：AS 预训练相对同结构冷启动，{} 个 AS 提升、{} 个下降，'.format(a['source_wins'], a['source_losses'])
              + '总体 Jaccard 变化 {:+.6f}，完整匹配变化 {:+.4f} 个百分点。'.format(
                  a['jaccard_delta'], a['exact_percentage_point_delta']),
              '在同一 AS 初始化上增加路径预训练，总体 Jaccard 再变化 {:+.6f}，'.format(b['jaccard_delta'])
              + '完整匹配变化 {:+.4f} 个百分点。'.format(b['exact_percentage_point_delta'])]
    if not conclusion['as_pretraining_improves_overall_jaccard'] and not conclusion['path_pretraining_adds_overall_jaccard_gain']:
        lines += ['这两项初始化方案均未验证总体收益，保留为负结果，不替换当前方法。']
    lines += ['该结论限于本轮同一网络、每源小规模合成增强库及固定训练方案；不能据此断言所有路径预训练或序列编码均无效。'
              '本轮没有增加新的路由证据。语料规模和弱监督迁移是否限制效果，仍是待验证问题，不能由本轮结果确定因果。']
    path = run / 'REPORT.md'
    original = path.read_text().partition('<!-- representation-pretraining-derived-report -->')[0].rstrip()
    temporary = run / 'REPORT.derived.tmp.md'
    temporary.write_text(original + '\n\n' + '\n'.join(lines) + '\n'); temporary.replace(path)
    print('Published audited comparisons and measured stage timing for all 10 sources / 20 new models', flush=True)


if __name__ == '__main__':
    report()
