#!/usr/bin/env python3
"""Independent corpus/parameter-boundary/candidate/metric audit for SSL controls."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pathInference.experiment.contracts import require, signature, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json
from audit_gmpi_confidence import candidate_membership, check_history


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def check_changes(before, after, stage):
    require(before.keys() == after.keys(), 'Parameter schema changed during pretraining')
    changed = []
    for name in before:
        require(before[name].shape == after[name].shape, 'Parameter capacity changed')
        require(bool(after[name].isfinite().all()), 'Nonfinite pretrained weights')
        if not before[name].equal(after[name]):
            changed.append(name)
    # Independent allow-list rather than importing the numerical module's list.
    allowed = {'asEmbedding'} if stage == 'as_context' else {
        'node', 'endpoints', 'position', 'reverse_position', 'transition',
        'token_norm', 'encoder', 'pool', 'path_output'}
    require(changed and all(name.split('.')[0] in allowed for name in changed),
            'Unrelated AS/path/scoring parameters changed')
    require(bool(after['asEmbedding.weight'][0].eq(0).all()), 'Padding embedding changed')
    return changed


def corpus_audit(source, binding, run, cfg):
    import numpy as np
    base = ROOT / cfg['frozen_candidate_run']
    vocabulary = read_json(base / ('as' + source) / 'VOCABULARY.json')
    admission = read_json(base / 'augmentation' / ('as' + source) / 'ADMISSION.json')
    split_path = Path(binding['binding']['split']) / 'split_manifest.json'
    split = read_json(split_path)
    train, validation = set(split['train_pairs']), set(split['validation_pairs'])
    require(train.isdisjoint(validation), 'Training/validation pair overlap')
    corpus = read_json(run / ('as' + source) / 'CORPUS.json')
    verify_files({corpus['corpus']: corpus['corpus_sha256'], corpus['library']: corpus['library_sha256']})
    require(corpus['library'] == admission['path'] and corpus['library_sha256'] == admission['sha256']
            and corpus['vocabulary_sha256'] == sha256(base / ('as' + source) / 'VOCABULARY.json')
            and corpus['split_sha256'] == sha256(split_path), 'Pretraining input identity differs')
    with Path(admission['path']).open(newline='') as stream:
        library = [tuple(row[0].split('->')) for row in csv.reader(stream)]
    expected = list(dict.fromkeys(path for path in library if path[0] + '->' + path[-1] in train))
    encoded = np.load(corpus['corpus'], allow_pickle=False)
    decoded = [tuple(vocabulary[int(node)] for node in row if node) for row in encoded]
    require(decoded == expected and signature(decoded) == corpus['training_path_set_sha256']
            and signature(library) == corpus['admitted_library_path_set_sha256'] == admission['path_set_sha256']
            == binding['binding']['admitted_library_path_set_sha256'], 'SSL corpus differs from admitted train paths')
    require(corpus['paths'] == len(decoded) and corpus['validation_pairs_excluded'] and not corpus['final_labels_read']
            and all(p[0] + '->' + p[-1] not in validation for p in decoded), 'Validation routes leaked into pretraining')
    return vocabulary, corpus


def check_aggregates(rows, summary, variants):
    fields = ('jaccard_accuracy', 'ordered_exact_accuracy', 'cold_jaccard', 'cold_exact',
              'gmpi_ce_jaccard', 'gmpi_ce_exact', 'cold_delta', 'gmpi_ce_delta', 'oracle_jaccard', 'oracle_gap')
    for variant in variants:
        subset = [r for r in rows if r['variant'] == variant]
        value = summary['variants'][variant]
        for group, recorded in [(subset, value['micro']),
                *[([r for r in subset if (r['rho'] > 1) == high], value['by_rho'][name])
                  for name, high in [('high', True), ('low', False)]]]:
            count = sum(r['query_count'] for r in group)
            require(recorded['queries'] == count and recorded['sources'] == len(group), 'Wrong aggregate denominator')
            if count:
                for field in fields:
                    expected = math.fsum(r[field] * r['query_count'] for r in group) / count
                    require(math.isclose(recorded[field], expected, abs_tol=1e-12), 'Wrong aggregate metric: ' + field)
        for field in ('jaccard_accuracy', 'ordered_exact_accuracy'):
            require(math.isclose(value['macro'][field], math.fsum(r[field] for r in subset) / len(subset),
                                 abs_tol=1e-12), 'Wrong macro metric')


def audit():
    import numpy as np
    import torch
    from pathInference.downstream.evaluation import score_source
    from pathInference.research.path_sequence_confidence import OrderedPathConfidence
    os.sched_setaffinity(0, {6, 7}); torch.set_num_threads(2)
    cfg_path = ROOT / 'research/representation_pretraining_v1/CONFIG.json'
    cfg = read_json(cfg_path)
    run = ROOT / 'runs/20260201/representation_pretraining_v1'
    snapshot = read_json(run / 'SNAPSHOT.json')
    original = read_json(ROOT / cfg['source_snapshot'])
    require(snapshot['configuration'] == cfg and snapshot['sources'] == original['sources']
            and len(cfg['sources']) == len(set(cfg['sources'])) == 10
            and cfg['sources'] == original['configuration']['sources']
            and cfg['architecture'] == original['configuration']['architecture']
            and cfg['seed'] == original['configuration']['seed']
            and cfg['label_temperature'] == original['configuration']['label_temperature']
            and cfg['variants'] == ['as_context', 'as_and_path']
            and not snapshot['candidates_regenerated'] and not snapshot['test_labels_used_for_tuning']
            and not snapshot['extra_path_evidence'], 'Controlled protocol changed')
    verify_files(snapshot['files_sha256'])
    code_paths = [ROOT / 'src/pathInference/research' / name for name in
        ('representation_pretraining.py', 'representation_experiment.py', 'path_sequence_confidence.py',
         'gmpi_confidence.py', 'gmpi_confidence_loss.py')] + [ROOT / 'research/representation_pretraining_v1/run.py']
    expected_code = {str(path): sha256(path) for path in code_paths}
    for name, stage in [('ALL_MODELS_FROZEN.json', 'train'), ('ALL_PREDICTIONS_FROZEN.json', 'predict')]:
        barrier = read_json(run / name)
        expected = {str(run / 'tasks' / (s + '_' + v + '_' + stage + '.json'))
                    for s in cfg['sources'] for v in cfg['variants']}
        require(barrier['snapshot_sha256'] == sha256(run / 'SNAPSHOT.json')
                and set(barrier['receipts_sha256']) == expected, 'Missing global freeze barrier')
        verify_files(barrier['receipts_sha256'])
    stage_times = {stage: [datetime.fromisoformat(read_json(run / 'tasks' /
        (s + '_' + v + '_' + stage + '.json'))['completed_utc']) for s in cfg['sources'] for v in cfg['variants']]
        for stage in ('train', 'predict', 'evaluate')}
    require(max(stage_times['train']) <= min(stage_times['predict'])
            and max(stage_times['predict']) <= min(stage_times['evaluate']), 'Freeze-stage ordering differs')
    runtime = read_json(run / 'RUNTIME.json')
    require(runtime['device_runtime']['effective'] == 'cuda' and runtime['deterministic_algorithms']
            and not runtime['tf32'] and not runtime['vscode_hardware_acceleration_changed'], 'Runtime contract differs')
    vectors_record = read_json(run / 'BASE_VECTORS.json')
    verify_files({vectors_record['path']: vectors_record['sha256']})
    vectors = torch.from_numpy(np.load(vectors_record['path'], allow_pickle=False))
    rows, records = [], []
    for source in cfg['sources']:
        binding = snapshot['sources'][source]
        vocabulary, corpus = corpus_audit(source, binding, run, cfg)
        require(len(vocabulary) == len(vectors), 'Shared vocabulary differs')
        torch.manual_seed(cfg['seed'])
        network = OrderedPathConfidence(vectors, **cfg['architecture'])
        before = network.state_dict()
        cold_initial = read_json(ROOT / cfg['cold_control'] / ('as' + source) / 'INITIALIZATION.json')
        require(state_hash(before) == cold_initial['state_dict_sha256'], 'Paired cold initialization differs')
        pretrains = {}
        for stage in ('as_context', 'path_context'):
            directory = run / ('as' + source) / ('pretrain_' + stage)
            frozen = read_json(directory / 'PRETRAIN_FROZEN.json')
            verify_files({frozen['model']: frozen['model_sha256']})
            state = torch.load(frozen['model'], map_location='cpu')
            actual_changes = check_changes(before, state, stage)
            if stage == 'as_context':
                observed_nodes = np.unique(np.load(corpus['corpus'], allow_pickle=False))
                untouched = torch.ones(len(vectors), dtype=torch.bool)
                untouched[torch.as_tensor(observed_nodes)] = False
                require(torch.equal(before['asEmbedding.weight'][untouched], state['asEmbedding.weight'][untouched]),
                        'AS pretraining updated nodes outside the admitted training corpus')
            identity = dict(snapshot_sha256=sha256(run / 'SNAPSHOT.json'), source=source, stage=stage,
                            corpus_sha256=corpus['corpus_sha256'])
            if stage == 'path_context':
                identity['parent_sha256'] = pretrains['as_context']['model_sha256']
            require(frozen['identity'] == identity and frozen['configuration'] == cfg['pretraining'][stage]
                    and frozen['base_vectors_sha256'] == vectors_record['sha256']
                    and frozen['changed_parameters'] == actual_changes
                    and frozen['initial_state_sha256'] == state_hash(before)
                    and frozen['final_state_sha256'] == state_hash(state)
                    and not frozen['final_labels_read'] and not frozen['validation_routes_read']
                    and frozen['auxiliary_heads_discarded'], 'Pretraining boundary/checkpoint differs')
            history = read_json(directory / 'HISTORY.json')
            require(len(history) == frozen['epochs'] == cfg['pretraining'][stage]['epochs']
                    and [r['epoch'] for r in history] == list(range(len(history)))
                    and all(math.isfinite(r['loss']) and r['loss'] >= 0 and r['examples'] > 0 for r in history),
                    'Incomplete auxiliary training history')
            timing = read_json(directory / 'TIMING_pretrain.json')
            require(all(math.isfinite(timing[key]) and timing[key] >= 0 for key in ('wall_seconds', 'cpu_seconds'))
                    and all(math.isfinite(row[key]) and row[key] >= 0 for row in history
                            for key in ('wall_seconds', 'cpu_seconds')), 'Invalid measured pretraining time')
            before, pretrains[stage] = state, frozen
        del before, state, network
        predictions = {}
        cold_dir = ROOT / cfg['cold_control'] / ('as' + source) / 'ordered_path'
        cold, gmpi = read_json(cold_dir / 'RESULTS.json'), read_json(
            ROOT / cfg['gmpi_control'] / ('as' + source) / 'query_cross_entropy/RESULTS.json')
        for variant in cfg['variants']:
            directory = run / ('as' + source) / variant
            model = read_json(directory / 'MODEL_FROZEN.json')
            parent = pretrains['as_context' if variant == 'as_context' else 'path_context']
            require(model['identity'] == dict(snapshot_sha256=sha256(run / 'SNAPSHOT.json'), source=source,
                        variant=variant, pretraining_sha256=parent['model_sha256'])
                    and model['initial_sha256'] == parent['final_state_sha256']
                    and model['configuration'] == binding['arguments'] == read_json(cold_dir / 'MODEL_FROZEN.json')['configuration']
                    and model['label_temperature'] == cfg['label_temperature']
                    and model['admitted_library_path_set_sha256'] == corpus['admitted_library_path_set_sha256']
                    and not model['final_labels_read'], 'Fine-tuning changed non-initialization factors')
            verify_files({model['model']: model['model_sha256']})
            weights = torch.load(model['model'], map_location='cpu')
            network = OrderedPathConfidence(vectors, **cfg['architecture']); network.load_state_dict(weights, strict=True)
            require(all(bool(t.isfinite().all()) for t in weights.values())
                    and bool(weights['asEmbedding.weight'][0].eq(0).all()), 'Invalid fine-tuned weights')
            del network, weights
            check_history(read_json(directory / 'TRAINING_HISTORY.json'), model, binding['arguments']['patience'])
            prediction = read_json(directory / 'PREDICTIONS_FROZEN.json')
            require(prediction['model_sha256'] == model['model_sha256']
                    and prediction['candidate_sha256'] == binding['candidate_sha256']
                    and not prediction['final_labels_read'], 'Prediction identity differs')
            verify_files({prediction['predictions']: prediction['predictions_sha256']})
            predictions[variant] = prediction['predictions']
            for stage in ('train', 'predict', 'evaluate'):
                timing = read_json(directory / ('TIMING_' + stage + '.json'))
                require(all(math.isfinite(timing[key]) and timing[key] >= 0
                            for key in ('wall_seconds', 'cpu_seconds')), 'Invalid measured stage time')
            recorded = read_json(directory / 'RESULTS.json')
            independent = score_source(recorded['reference_file'], prediction['predictions'], source, recorded['fixed_identity'])
            for field in ('query_count', 'valid_predictions', 'ordered_exact_matches', 'ordered_exact_accuracy',
                          'jaccard_accuracy', 'total_jaccard_reward'):
                require(math.isclose(recorded[field], independent[field], rel_tol=1e-12, abs_tol=1e-10),
                        'Independent score differs: ' + field)
            require(recorded['candidate_sha256'] == cold['candidate_sha256'] == gmpi['candidate_sha256']
                    and recorded['query_count'] == cold['query_count'] == gmpi['query_count']
                    and recorded['fixed_identity'] == cold['fixed_identity'] == gmpi['fixed_identity'], 'Control queries differ')
            rows.append(dict(recorded, cold_jaccard=cold['jaccard_accuracy'], cold_exact=cold['ordered_exact_accuracy'],
                gmpi_ce_jaccard=gmpi['jaccard_accuracy'], gmpi_ce_exact=gmpi['ordered_exact_accuracy'],
                cold_delta=recorded['jaccard_accuracy'] - cold['jaccard_accuracy'],
                gmpi_ce_delta=recorded['jaccard_accuracy'] - gmpi['jaccard_accuracy'],
                oracle_gap=recorded['oracle_jaccard'] - recorded['jaccard_accuracy']))
        predictions['cold_control'] = cold_dir / 'ranked_predictions.csv'
        membership = candidate_membership(binding, vocabulary, predictions, binding['arguments']['k'])
        require(membership['slot_capped_queries'] == 0, 'Capped candidates require a separate oracle')
        tasks = [run / 'tasks' / (source + '_corpus.json')]
        tasks += [run / 'tasks' / (source + '_' + stage + '_pretrain.json') for stage in ('as_context', 'path_context')]
        tasks += [run / 'tasks' / (source + '_' + v + '_' + stage + '.json')
                  for v in cfg['variants'] for stage in ('train', 'predict', 'evaluate')]
        for path in tasks:
            task = read_json(path)
            require(task['code_sha256'] == expected_code and task['config_sha256'] == sha256(cfg_path), 'Task identity differs')
            verify_files(task['outputs_sha256'])
        records.append(dict(source=source, status='PASS', corpus_paths=corpus['paths'], candidate_membership=membership,
                            pretrained_parameters_checked=True, independently_rescored_variants=cfg['variants']))
    check_aggregates(rows, read_json(run / 'SUMMARY.json'), cfg['variants'])
    # Ensure final tables contain the actual independently rescored source rows.
    published = read_json(run / 'RESULTS.json')
    require(len(published) == len(rows) == 20, 'Incomplete result table')
    for actual, expected in zip(published, rows):
        for field in ('source', 'variant', 'query_count', 'jaccard_accuracy', 'ordered_exact_accuracy',
                      'cold_delta', 'gmpi_ce_delta', 'oracle_gap'):
            require(actual[field] == expected[field], 'Published source result differs')
    with (run / 'RESULTS.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(rows), 'Incomplete CSV result table')
    for actual, expected in zip(csv_rows, rows):
        require(actual['source'] == expected['source'] and actual['variant'] == expected['variant'],
                'CSV result source/order differs')
        for field in ('query_count', 'jaccard_accuracy', 'ordered_exact_accuracy', 'cold_delta',
                      'gmpi_ce_delta', 'oracle_jaccard', 'oracle_gap'):
            require(math.isclose(float(actual[field]), expected[field], rel_tol=1e-12, abs_tol=1e-12),
                    'CSV result metric differs: ' + field)
    require((run / 'REPORT.md').is_file(), 'Missing human-readable experiment report')
    write_json(run / 'COMPLETION_AUDIT.json', dict(status='PASS', updated_utc=utc_now(),
        sources=records, queries_per_variant=sum(r['candidate_membership']['queries'] for r in records),
        variants=cfg['variants'], corpus_and_parameter_boundaries_checked=True,
        model_and_prediction_barriers_checked=True, receipt_stage_order_checked=True,
        independently_rescored=True, aggregates_checked=True, csv_results_checked=True))
    print('PASS: 10 sources, 20 frozen models/predictions, SSL boundaries, candidate membership and independent scores', flush=True)


if __name__ == '__main__':
    audit()
