"""PATS libraries and frozen independent traceroute evaluation for AS63835."""
from pathlib import Path
import csv

from pathveil.experiment.contracts import freeze, require, verify_files
from pathveil.utils import read_json, sha256, utc_now, write_json


def prepare_source(run, source):
    run, source = Path(run), str(source)
    proto = read_json(run / 'PROTOCOL.json')
    require(proto['sources'] == [source], 'Traceroute runs require one explicitly registered source')
    admission_file = run / 'augmentation' / ('as' + source) / 'ADMISSION.json'
    admission = read_json(admission_file)
    upstream = read_json(run / 'UPSTREAM_BOUND.json')
    upstream_release = read_json(Path(upstream['run']) / 'upstream/RELEASE.json')
    graph = {name: upstream_release[name] for name in ('selected_neighbors_csv', 'recursive_anchors_csv')}
    outputs = {admission['path']: admission['sha256'], admission['json_path']: admission['json_sha256']}
    outputs.update({path: sha256(path) for path in graph.values()})
    verify_files(outputs)
    release = run / 'traceroute_release/RELEASE.json'
    freeze(release, dict(status='PASS', date=proto['date'], sources=[source],
        files_by_source={source: admission}, outputs_sha256=outputs, **graph,
        formal_reference_labels_read=False, training_and_PG_pool='same_R_s_hash_augmentation_only',
        generator='PATS', source_tree_sha256=sha256(run / 'query_trees' / (source + '.json'))))
    files = [release, admission_file, run / 'query_trees' / (source + '.json'),
             run / 'construction' / source / 'CONSTRUCTION_COMPLETED.json']
    value = dict(status='PASS', source=source, date=proto['date'], protocol_sha256=sha256(run / 'PROTOCOL.json'),
                 outputs_sha256={str(path): sha256(path) for path in files}, completed_utc=utc_now())
    freeze(run / 'construction' / source / 'SOURCE_READY.json', value)
    return value


def execute(run, device):
    from pathveil.traceroute.runner import run_traceroute
    from .workflow import write_progress

    run = Path(run)
    proto = read_json(run / 'PROTOCOL.json')
    require(proto['sources'] == ['63835'], 'Traceroute formal cohort changed')
    source = proto['sources'][0]
    config = proto['method']['gmpi']
    write_progress(run, 'traceroute_training_prediction_evaluation', source=source)
    result = run_traceroute(
        run, run / 'data/data_manifest.json', run / 'traceroute_release/RELEASE.json', device,
        source=source, date=proto['date'], prefix_tree_dir=run / 'query_trees',
        pathgen_workers=proto['method']['selection_policy']['pathgen_train_threads'],
        **{key: config[key] for key in ('epochs', 'batch_size', 'patience', 'validation_fraction', 'k', 'seed')})
    summary = read_json(result['summary'])
    metrics = summary['metrics']
    row = dict(source_asn=source, query_count=metrics['all_truth_pairs'],
               reference_unique_paths=metrics['all_unique_truth_paths'],
               jaccard_accuracy=metrics['pair_jaccard_accuracy'],
               ordered_exact_accuracy=metrics['pair_any_truth_exact_rate'],
               coverage=metrics['endpoint_valid_pair_coverage'])
    write_json(run / 'RESULTS.json', dict(status='COMPLETED', date=proto['date'], evaluation_mode='traceroute',
        completed_sources=1, expected_sources=1, per_source=[row], summary=result['summary'],
        summary_sha256=sha256(result['summary']), final_labels_used_for_training_or_selection=False))
    with (run / 'RESULTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (run / 'REPORT.md').write_text(
        '# ' + proto['date'] + '：PATS独立traceroute评测\n\n'
        '| 源AS | 查询 | 参考路径 | Jaccard | 完整有序匹配 | 覆盖率 |\n'
        '|---|---:|---:|---:|---:|---:|\n'
        '| {source_asn} | {query_count} | {reference_unique_paths} | {jaccard_accuracy:.4%} | '
        '{ordered_exact_accuracy:.4%} | {coverage:.4%} |\n\n'.format(**row)
        + '训练与选模只使用BGP增强路径；独立traceroute标签仅在最终评分阶段打开。\n')
    write_json(run / 'STATUS.json', dict(status='COMPLETED', stage='complete', updated_utc=utc_now()))
    write_progress(run, 'completed', sources=1, evaluation_mode='traceroute')
