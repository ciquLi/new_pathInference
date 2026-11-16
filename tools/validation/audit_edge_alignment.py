#!/usr/bin/env python3
"""Independent architecture, input, checkpoint, membership and metric checks."""
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
from pathInference.experiment.contracts import require, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json
from audit_gmpi_confidence import candidate_membership, check_history
from audit_path_sequence_confidence import check_summary, check_training_control


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), array.shape]).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def check_architecture(scope, model, initial, control_initial, variant):
    require(scope['use_base'] == (variant == 'gmpi_edge_residual')
            and scope['input_semantics_version'] == 2 and scope['final_correction_head_zero']
            and not scope['pretrained_checkpoint_loaded']
            and scope['full_initial_sha256'] == initial['state_dict_sha256'] == model['initial_sha256']
            and scope['base_initial_sha256'] == control_initial['state_dict_sha256'],
            'Architecture or paired cold initialization differs')
    inactive = ['pr', 'pm'] if variant == 'edge_only' else []
    require(scope['dormant_modules'] == inactive, 'Undeclared dormant parameters')
    require(all(name.split('.')[0] not in inactive for name in scope['trainable_parameter_names']),
            'Standalone scorer activates the GMPI branch')


def audit():
    import numpy as np
    import torch
    from pathInference.downstream.evaluation import score_source
    from pathInference.research.edge_alignment_confidence import EdgeAlignmentConfidence
    os.sched_setaffinity(0, {6, 7}); torch.set_num_threads(2)
    cfg_file = ROOT / 'research/edge_alignment_v1/CONFIG.json'
    cfg = read_json(cfg_file); run = ROOT / 'runs/20260201/edge_alignment_v1'
    snapshot = read_json(run / 'SNAPSHOT.json'); original = read_json(ROOT / cfg['source_snapshot'])
    require(snapshot['configuration'] == cfg and snapshot['sources'] == original['sources']
            and len(cfg['sources']) == len(set(cfg['sources'])) == 10
            and cfg['sources'] == original['configuration']['sources']
            and cfg['date'] == original['configuration']['date']
            and cfg['seed'] == original['configuration']['seed']
            and cfg['label_temperature'] == original['configuration']['label_temperature']
            and cfg['variants'] == ['edge_only', 'gmpi_edge_residual']
            and not cfg['pretraining'] and not cfg['additional_evidence'] and not cfg['candidate_set_self_attention']
            and not snapshot['additional_evidence'] and not snapshot['candidates_regenerated']
            and not snapshot['test_labels_used_for_tuning'], 'Frozen architecture experiment contract differs')
    verify_files(snapshot['files_sha256'])
    for name, stage in [('ALL_MODELS_FROZEN.json', 'train'), ('ALL_PREDICTIONS_FROZEN.json', 'predict')]:
        barrier = read_json(run / name)
        expected = {str(run / 'tasks' / (s + '_' + v + '_' + stage + '.json'))
                    for s in cfg['sources'] for v in cfg['variants']}
        require(barrier['snapshot_sha256'] == sha256(run / 'SNAPSHOT.json')
                and set(barrier['receipts_sha256']) == expected, 'Incomplete global freeze barrier')
        verify_files(barrier['receipts_sha256'])
    stage_times = {stage: [datetime.fromisoformat(read_json(run / 'tasks' /
        (s + '_' + v + '_' + stage + '.json'))['completed_utc']) for s in cfg['sources'] for v in cfg['variants']]
        for stage in ('train', 'predict', 'evaluate')}
    require(max(stage_times['train']) <= min(stage_times['predict'])
            and max(stage_times['predict']) <= min(stage_times['evaluate']), 'Freeze stage ordering differs')
    runtime = read_json(run / 'RUNTIME.json')
    require(runtime['device_runtime']['effective'] == 'cuda' and runtime['device_runtime']['intraop_threads'] == 2
            and runtime['deterministic_algorithms'] and not runtime['tf32']
            and not runtime['vscode_hardware_acceleration_changed'], 'Runtime resource contract differs')
    code_paths = [ROOT / 'src/pathInference/research' / name for name in
                 ('edge_alignment_confidence.py', 'edge_alignment_experiment.py', 'gmpi_confidence.py', 'gmpi_confidence_loss.py')]
    code_paths += [ROOT / 'src/pathInference/core' / name for name in
                  ('GMPI_pathLikelihoodEst.py', 'myDataset.py', 'PathInference_trainer.py', 'environment.py')]
    code_paths += [ROOT / 'research/edge_alignment_v1/run.py']
    expected_code = {str(p): sha256(p) for p in code_paths}
    vectors = torch.from_numpy(np.load(str(ROOT / cfg['raw_vector_cache']), allow_pickle=False))
    rescored, sources = [], []
    for source in cfg['sources']:
        binding = snapshot['sources'][source]; arguments = binding['arguments']
        split = read_json(Path(binding['binding']['split']) / 'split_manifest.json')
        require(set(split['train_pairs']).isdisjoint(split['validation_pairs']), 'Pair-disjoint split overlap')
        control_dir = ROOT / cfg['control_run'] / ('as' + source)
        control_initial = read_json(control_dir / 'INITIALIZATION.json')
        initial = read_json(run / ('as' + source) / 'INITIALIZATION.json')
        require(initial['seed'] == cfg['seed'] and initial['snapshot_sha256'] == sha256(run / 'SNAPSHOT.json'),
                'Initial state identity differs')
        base_model = read_json(control_dir / cfg['control_variant'] / 'MODEL_FROZEN.json')
        base_result = read_json(control_dir / cfg['control_variant'] / 'RESULTS.json')
        predictions = {}; trained = []; alignment_initials = []
        for variant in cfg['variants']:
            directory = run / ('as' + source) / variant
            model = read_json(directory / 'MODEL_FROZEN.json'); scope = read_json(directory / 'ARCHITECTURE.json')
            check_training_control(model, base_model, arguments, binding, cfg)
            check_architecture(scope, model, initial, control_initial, variant)
            require(model['identity'] == dict(snapshot_sha256=sha256(run / 'SNAPSHOT.json'), source=source, variant=variant),
                    'Model identity differs')
            check_history(read_json(directory / 'TRAINING_HISTORY.json'), model, arguments['patience'])
            torch.manual_seed(cfg['seed'])
            network = EdgeAlignmentConfidence(vectors, hidden_size=arguments['hidden_size'],
                use_base=variant == 'gmpi_edge_residual', **cfg['architecture'])
            fresh = network.state_dict()
            require(state_hash(fresh) == scope['full_initial_sha256']
                    and state_hash({k:v for k,v in fresh.items() if not k.startswith('alignment.')}) == scope['base_initial_sha256']
                    and state_hash(network.alignment.state_dict()) == scope['alignment_initial_sha256'],
                    'Independent cold reconstruction differs')
            names = [name for name, p in network.named_parameters() if p.requires_grad]
            require(names == scope['trainable_parameter_names']
                    and sum(p.numel() for p in network.parameters()) == scope['total_parameters']
                    and sum(p.numel() for p in network.parameters() if p.requires_grad) == scope['trainable_parameters'],
                    'Effective architecture capacity differs')
            verify_files({model['model']: model['model_sha256']})
            weights = torch.load(model['model'], map_location='cpu')
            require(all(bool(t.isfinite().all()) for t in weights.values())
                    and bool(weights['asEmbedding.weight'][0].eq(0).all()), 'Invalid model weights/padding')
            if variant == 'edge_only':
                require(all(torch.equal(weights[name], value) for name, value in fresh.items()
                            if name.startswith(('pr.', 'pm.'))), 'Dormant original GMPI weights were trained')
            network.load_state_dict(weights, strict=True)
            alignment_initials.append(scope['alignment_initial_sha256'])
            trained.append(dict(variant=variant, total_parameters=scope['total_parameters'],
                trainable_parameters=scope['trainable_parameters'], dormant_modules=scope['dormant_modules'],
                selected_checkpoint_checked=True, paired_cold_initialization_checked=True,
                correction_head_changed=not bool(weights['alignment.output.3.weight'].eq(0).all())))
            del network, fresh, weights
            prediction = read_json(directory / 'PREDICTIONS_FROZEN.json')
            require(prediction['model_sha256'] == model['model_sha256']
                    and prediction['candidate_sha256'] == binding['candidate_sha256'] and not prediction['final_labels_read'],
                    'Prediction identity differs')
            verify_files({prediction['predictions']: prediction['predictions_sha256']})
            predictions[variant] = prediction['predictions']
            for stage in ('train', 'predict', 'evaluate'):
                task = read_json(run / 'tasks' / (source + '_' + variant + '_' + stage + '.json'))
                require(task['source'] == source and task['variant'] == variant and task['stage'] == stage
                        and task['config_sha256'] == sha256(cfg_file) and task['code_sha256'] == expected_code,
                        'Task numerical identity differs')
                verify_files(task['outputs_sha256'])
                timing = read_json(directory / ('TIMING_' + stage + '.json'))
                require(all(math.isfinite(timing[k]) and timing[k] >= 0 for k in ('wall_seconds', 'cpu_seconds')), 'Invalid timing')
            result = read_json(directory / 'RESULTS.json')
            independent = score_source(result['reference_file'], prediction['predictions'], source, result['fixed_identity'])
            for field in ('query_count', 'valid_predictions', 'ordered_exact_matches', 'ordered_exact_accuracy',
                          'jaccard_accuracy', 'total_jaccard_reward'):
                require(math.isclose(result[field], independent[field], rel_tol=1e-12, abs_tol=1e-10), 'Independent score differs: ' + field)
            require(result['candidate_sha256'] == base_result['candidate_sha256']
                    and result['query_count'] == base_result['query_count'] and result['fixed_identity'] == base_result['fixed_identity']
                    and result['oracle_jaccard'] == base_result['oracle_jaccard'], 'Control evaluated different queries')
            rescored.append(dict(independent, source=source, variant=variant, rho=result['rho'],
                oracle_jaccard=result['oracle_jaccard'], control_jaccard=base_result['jaccard_accuracy'],
                control_exact=base_result['ordered_exact_accuracy']))
        require(len(set(alignment_initials)) == 1, 'Unpaired alignment initialization')
        predictions['gmpi_ce'] = control_dir / cfg['control_variant'] / 'ranked_predictions.csv'
        vocabulary = read_json(ROOT / cfg['frozen_candidate_run'] / ('as' + source) / 'VOCABULARY.json')
        require(len(vocabulary) == len(vectors), 'AS vocabulary capacity differs')
        membership = candidate_membership(binding, vocabulary, predictions, arguments['k'])
        require(membership['slot_capped_queries'] == 0, 'A capped candidate oracle requires separate computation')
        sources.append(dict(source=source, status='PASS', candidate_membership=membership, models=trained,
                            independently_rescored_variants=cfg['variants']))
    summary = read_json(run / 'SUMMARY.json')
    for variant in cfg['variants']:
        check_summary([r for r in rescored if r['variant'] == variant], summary['variants'][variant])
    published = read_json(run / 'RESULTS.json')
    with (run / 'RESULTS.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(published) == len(csv_rows) == len(rescored) == 20, 'Incomplete source result table')
    for actual, line, expected in zip(published, csv_rows, rescored):
        require(actual['source'] == line['source'] == expected['source']
                and actual['variant'] == line['variant'] == expected['variant'], 'Source table order differs')
        for field in ('query_count', 'jaccard_accuracy', 'ordered_exact_accuracy'):
            require(math.isclose(actual[field], expected[field], rel_tol=1e-12, abs_tol=1e-10)
                    and math.isclose(float(line[field]), expected[field], rel_tol=1e-12, abs_tol=1e-10), 'Published source metric differs')
    write_json(run / 'COMPLETION_AUDIT.json', dict(status='PASS', checked_utc=utc_now(), sources=sources,
        variants=cfg['variants'], queries_per_variant=sum(r['candidate_membership']['queries'] for r in sources),
        model_and_prediction_barriers_checked=True, receipt_stage_order_checked=True,
        paired_initialization_and_effective_parameter_scope_checked=True, independent_rescore=True,
        candidate_membership_checked=True, aggregates_checked=True, csv_results_checked=True))
    print('PASS: both network architectures, all ten sources, paired initialization, frozen candidates and independently rescored metrics', flush=True)


if __name__ == '__main__':
    audit()
