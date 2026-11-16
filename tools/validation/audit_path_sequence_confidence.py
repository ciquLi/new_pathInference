#!/usr/bin/env python3
"""Independent data, checkpoint, candidate membership and score audit."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from pathInference.experiment.contracts import require, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json
from audit_gmpi_confidence import candidate_membership, check_history


def check_protocol(cfg, snapshot, original):
    require(snapshot['configuration'] == cfg and snapshot['sources'] == original['sources']
            and len(cfg['sources']) == len(set(cfg['sources'])) == 10
            and cfg['sources'] == original['configuration']['sources']
            and cfg['label_temperature'] == original['configuration']['label_temperature']
            and cfg['seed'] == original['configuration']['seed']
            and cfg['control_variant'] == 'query_cross_entropy'
            and cfg['variants'] == ['ordered_path'], 'Architecture control protocol differs')
    require(not snapshot['candidates_regenerated'] and not snapshot['test_labels_used_for_tuning'],
            'Upstream/test contract changed')


def check_training_control(model, baseline, arguments, binding, cfg):
    require(model['configuration'] == baseline['configuration'] == arguments
            and model['label_temperature'] == baseline['label_temperature'] == cfg['label_temperature']
            and model['admitted_library_path_set_sha256'] == baseline['admitted_library_path_set_sha256']
            == binding['binding']['admitted_library_path_set_sha256']
            and not model['final_labels_read'], 'Training changed another factor')


def check_summary(rows, summary):
    """Recompute every reported aggregate from independently rescored sources."""
    metrics = ('jaccard_accuracy', 'ordered_exact_accuracy', 'control_jaccard', 'control_exact',
               'jaccard_delta', 'exact_delta', 'oracle_jaccard', 'oracle_gap')
    derived = [dict(row, jaccard_delta=row['jaccard_accuracy'] - row['control_jaccard'],
                    exact_delta=row['ordered_exact_accuracy'] - row['control_exact'],
                    oracle_gap=row['oracle_jaccard'] - row['jaccard_accuracy']) for row in rows]

    def check(subset, recorded):
        count = sum(row['query_count'] for row in subset)
        require(recorded['queries'] == count, 'Aggregate query count differs')
        if not count:
            return
        require(recorded['sources'] == len(subset), 'Aggregate source count differs')
        for field in metrics:
            value = math.fsum(row[field] * row['query_count'] for row in subset) / count
            require(math.isclose(recorded[field], value, rel_tol=1e-12, abs_tol=1e-12),
                    'Aggregate metric differs: ' + field)

    require(summary['queries'] == sum(row['query_count'] for row in derived), 'Summary scope differs')
    check(derived, summary['micro'])
    for field in ('jaccard_accuracy', 'ordered_exact_accuracy', 'jaccard_delta', 'exact_delta'):
        require(math.isclose(summary['macro'][field], math.fsum(row[field] for row in derived) / len(derived),
                             rel_tol=1e-12, abs_tol=1e-12), 'Macro metric differs: ' + field)
    for name, high in [('high', True), ('low', False)]:
        check([row for row in derived if (row['rho'] > 1) == high], summary['by_rho'][name])


def audit():
    import torch
    from pathInference.downstream.evaluation import score_source
    os.sched_setaffinity(0, {6, 7})
    torch.set_num_threads(2)
    cfg_file = ROOT / 'research/path_sequence_confidence_v1/CONFIG.json'
    cfg = read_json(cfg_file)
    run = ROOT / 'runs/20260201/path_sequence_confidence_v1'
    snapshot = read_json(run / 'SNAPSHOT.json')
    control = ROOT / cfg['control_run']
    original = read_json(ROOT / cfg['source_snapshot'])
    check_protocol(cfg, snapshot, original)
    verify_files(snapshot['files_sha256'])
    for name, stage in [('ALL_MODELS_FROZEN.json', 'train'), ('ALL_PREDICTIONS_FROZEN.json', 'predict')]:
        barrier = read_json(run / name)
        expected = {str(run / 'tasks' / (source + '_ordered_path_' + stage + '.json')) for source in cfg['sources']}
        require(barrier['snapshot_sha256'] == sha256(run / 'SNAPSHOT.json')
                and set(barrier['receipts_sha256']) == expected, 'Incomplete model/prediction barrier')
        verify_files(barrier['receipts_sha256'])
    runtime = read_json(run / 'RUNTIME.json')
    require(runtime['device_runtime']['effective'] == 'cuda' and runtime['deterministic_algorithms']
            and not runtime['tf32'] and not runtime['vscode_hardware_acceleration_changed'],
            'Runtime control differs')
    results, rescored_sources = [], []
    for source in cfg['sources']:
        binding = snapshot['sources'][source]
        arguments = binding['arguments']
        split = read_json(Path(binding['binding']['split']) / 'split_manifest.json')
        require(set(split['train_pairs']).isdisjoint(split['validation_pairs']), 'Pair split overlap')
        directory = run / ('as' + source) / 'ordered_path'
        model = read_json(directory / 'MODEL_FROZEN.json')
        base_dir = control / ('as' + source) / cfg['control_variant']
        base_model = read_json(base_dir / 'MODEL_FROZEN.json')
        check_training_control(model, base_model, arguments, binding, cfg)
        require(model['identity'] == dict(snapshot_sha256=sha256(run / 'SNAPSHOT.json'),
                source=source, variant='ordered_path'), 'Model identity differs')
        initial = read_json(run / ('as' + source) / 'INITIALIZATION.json')
        require(model['initial_sha256'] == initial['state_dict_sha256'] and initial['seed'] == cfg['seed']
                and initial['snapshot_sha256'] == sha256(run / 'SNAPSHOT.json'), 'Initialization identity differs')
        check_history(read_json(directory / 'TRAINING_HISTORY.json'), model, arguments['patience'])
        verify_files({model['model']: model['model_sha256']})
        state = torch.load(model['model'], map_location='cpu')
        require(all(bool(torch.isfinite(value).all()) for value in state.values()), 'Nonfinite model weights')
        vocabulary = read_json(ROOT / cfg['frozen_candidate_run'] / ('as' + source) / 'VOCABULARY.json')
        require(state['asEmbedding.weight'].shape[0] == len(vocabulary)
                and bool(state['asEmbedding.weight'][0].eq(0).all()), 'Vocabulary/padding weights differ')
        from pathInference.research.path_sequence_confidence import OrderedPathConfidence
        network = OrderedPathConfidence(torch.zeros_like(state['asEmbedding.weight']), **cfg['architecture'])
        network.load_state_dict(state, strict=True)
        parameters = sum(value.numel() for value in network.parameters())
        del network, state
        prediction = read_json(directory / 'PREDICTIONS_FROZEN.json')
        require(prediction['model_sha256'] == model['model_sha256']
                and prediction['candidate_sha256'] == binding['candidate_sha256']
                and not prediction['final_labels_read'], 'Prediction identity differs')
        verify_files({prediction['predictions']: prediction['predictions_sha256']})
        timings = {}
        for stage in ('train', 'predict', 'evaluate'):
            task = read_json(run / 'tasks' / (source + '_ordered_path_' + stage + '.json'))
            expected_code = [ROOT / 'src/pathInference/research' / filename for filename in
                             ('path_sequence_confidence.py', 'path_sequence_experiment.py',
                              'gmpi_confidence.py', 'gmpi_confidence_loss.py')]
            expected_code += [ROOT / 'research/path_sequence_confidence_v1/run.py']
            require(task['source'] == source and task['stage'] == stage
                    and task['code_sha256'] == {str(path): sha256(path) for path in expected_code}
                    and task['config_sha256'] == sha256(cfg_file), 'Task protocol differs')
            verify_files(task['code_sha256']); verify_files(task['outputs_sha256'])
            timing = read_json(directory / ('TIMING_' + stage + '.json'))
            require(all(math.isfinite(timing[key]) and timing[key] >= 0
                        for key in ('wall_seconds', 'cpu_seconds')), 'Invalid timing')
            timings[stage] = timing
        membership = candidate_membership(binding, vocabulary,
            {'ordered_path': prediction['predictions'],
             'gmpi_query_cross_entropy': base_dir / 'ranked_predictions.csv'}, arguments['k'])
        require(membership['slot_capped_queries'] == 0, 'Compute a separate capped oracle before comparing')
        # All ten prediction receipts and the root barrier have been verified above.
        recorded = read_json(directory / 'RESULTS.json')
        independent = score_source(recorded['reference_file'], prediction['predictions'], source,
                                   recorded['fixed_identity'])
        for field in ('query_count', 'valid_predictions', 'ordered_exact_matches', 'ordered_exact_accuracy',
                      'jaccard_accuracy', 'total_jaccard_reward'):
            require(math.isclose(recorded[field], independent[field], rel_tol=1e-12, abs_tol=1e-10),
                    'Independent rescore differs: ' + field)
        base = read_json(base_dir / 'RESULTS.json')
        require(recorded['candidate_sha256'] == base['candidate_sha256']
                and recorded['query_count'] == base['query_count']
                and recorded['fixed_identity'] == base['fixed_identity']
                and recorded['oracle_jaccard'] == base['oracle_jaccard'], 'Comparison identity differs')
        rescored_sources.append(dict(independent, source=source, rho=recorded['rho'],
            control_jaccard=base['jaccard_accuracy'], control_exact=base['ordered_exact_accuracy'],
            oracle_jaccard=base['oracle_jaccard']))
        results.append(dict(source=source, status='PASS', parameters=parameters,
            candidate_membership=membership, independent_rescore=True, measured_timing=timings))
    total = sum(row['candidate_membership']['queries'] for row in results)
    summary = read_json(run / 'SUMMARY.json')
    require(summary['queries'] == summary['micro']['queries'] == total, 'Aggregate query count differs')
    check_summary(rescored_sources, summary)
    completion = dict(status='PASS', checked_utc=utc_now(), sources=results, pending=[],
                      queries=total, final_labels_read=True, independent_rescore=True,
                      changed_factor=cfg['changed_factor'])
    write_json(run / 'COMPLETION_AUDIT.json', completion)
    print('PASS: ordered-path architecture, ten sources, ' + str(total) + ' queries', flush=True)


if __name__ == '__main__':
    audit()
