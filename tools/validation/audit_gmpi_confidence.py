#!/usr/bin/env python3
"""Audit paired GMPI controls independently of their numerical dispatcher."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from pathInference.experiment.contracts import require, signature, verify_files
from pathInference.utils import read_json, sha256, utc_now, write_json


def check_history(history, model, patience):
    require(history and [r['epoch'] for r in history] == list(range(len(history))),
            'Epoch history is incomplete')
    best, stale, selected = -float('inf'), 0, None
    for row in history:
        require(stale < patience, 'Training continued beyond frozen patience')
        for key in ('training_jaccard', 'validation_jaccard'):
            require(math.isfinite(row[key]) and 0 <= row[key] <= 1.000001,
                    'Invalid weak-label reward')
        for key in ('training_loss', 'validation_original_smooth_l1_loss', 'wall_seconds', 'cpu_seconds'):
            require(math.isfinite(row[key]) and row[key] >= 0, 'Invalid epoch loss/time')
        improved = row['validation_jaccard'] > best
        require(improved == row['checkpoint_improved'], 'Best epoch decision differs')
        if improved:
            best, stale, selected = row['validation_jaccard'], 0, row['epoch']
        else:
            stale += 1
    require(model['epochs'] == len(history) and model['selected_epoch'] == selected
            and model['validation_jaccard'] == best, 'Selected checkpoint differs')


def usable_candidates(row, vocabulary, k):
    """Independently reproduce endpoint filtering, stable deduplication and slot cap."""
    require(len(row) == 5 and set(row[3].split('|')) == {'0.0'},
            'Final candidate CSV contains labels or malformed rows')
    pair = tuple(row[4].split('->'))
    paths, seen = [], set()
    for encoded in row[0].split('|'):
        path = tuple(str(vocabulary[int(index)]) for index in encoded.split('->'))
        require(2 <= len(path) <= 20 and all(n.isascii() and n.isdecimal() and int(n) > 0 for n in path),
                'Unusable candidate sequence')
        if (path[0], path[-1]) != pair or path in seen:
            continue
        seen.add(path)
        paths.append('->'.join(path))
    require(paths, 'No usable candidates')
    return paths if k is None else paths[:k]


def candidate_membership(binding, vocabulary, prediction_files, k):
    """One streaming candidate pass checks both losses without reading test routes."""
    inventory = read_json(binding['inventory'])['pairs']
    with Path(binding['dispatch']).open() as stream:
        dispatch_rows = [json.loads(line) for line in stream]
    states = {row['pair']: row['status'] for row in dispatch_rows}
    require(len(states) == len(dispatch_rows) == len(inventory)
            and set(states) == set(inventory), 'Dispatch inventory differs')
    predictions = {}
    for variant, file in prediction_files.items():
        with Path(file).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require([r['pair'] for r in rows] == inventory, 'Prediction order/inventory differs')
        predictions[variant] = {r['pair']: r for r in rows}
    seen, maximum, capped = set(), 0, 0
    with Path(binding['candidate']).open(newline='') as stream:
        for row in csv.reader(stream):
            pair = row[4]
            require(pair not in seen and states.get(pair) == 'model_candidates', 'Duplicate/extra model query')
            seen.add(pair)
            all_paths = usable_candidates(row, vocabulary, None)
            maximum = max(maximum, len(all_paths))
            capped += int(k is not None and len(all_paths) > k)
            allowed = set(all_paths[:k])
            for variant, rows in predictions.items():
                require(rows[pair]['status'] == 'predicted' and rows[pair]['predicted_path'] in allowed,
                        'Prediction outside frozen candidates: ' + variant + '/' + pair)
    require(seen == {p for p, s in states.items() if s == 'model_candidates'}, 'Missing model query')
    for pair, state in states.items():
        if state == 'model_candidates':
            continue
        require(state in ('direct_graph_edge_candidate', 'same_as_query', 'no_usable_candidate'),
                'Unexpected direct dispatch')
        expected = pair if state == 'direct_graph_edge_candidate' else ''
        expected_status = 'direct_graph_edge' if expected else state
        for rows in predictions.values():
            require(rows[pair]['predicted_path'] == expected and rows[pair]['status'] == expected_status,
                    'Non-model prediction differs from frozen dispatch')
    return dict(queries=len(inventory), model_queries=len(seen), variants=list(predictions), status='PASS',
                maximum_usable_candidates=maximum, slot_capped_queries=capped, candidate_slots=k)


def audit(run, rescore=False, check_candidates=False, check_weights=False):
    cfg_file = ROOT / 'research/gmpi_confidence_v1/CONFIG.json'
    cfg, snap = read_json(cfg_file), read_json(run / 'SNAPSHOT.json')
    os.sched_setaffinity(0, {6, 7})
    require(snap['configuration'] == cfg and not snap['candidates_regenerated']
            and not snap['test_labels_used_for_tuning'], 'Frozen protocol differs')
    verify_files(snap['files_sha256'])
    sources, variants = cfg['sources'], cfg['variants']
    require(len(set(sources)) == len(sources) == 10
            and variants == ['smooth_l1', 'query_cross_entropy'], 'Control scope differs')
    source_snapshot = read_json(ROOT / cfg['source_snapshot'])
    first_vocabulary = None
    for source in sources:
        expected = source_snapshot['sources'][source]
        require({k: v for k, v in snap['sources'][source].items() if k != 'arguments'} == expected,
                'Upstream snapshot changed')
        vocabulary = read_json(ROOT / cfg['frozen_candidate_run'] / ('as' + source) / 'VOCABULARY.json')
        require(first_vocabulary is None or vocabulary == first_vocabulary,
                'Shared GMPI environment has source-dependent vocabularies')
        first_vocabulary = vocabulary
    for barrier_name, stage in (('ALL_MODELS_FROZEN.json', 'train'), ('ALL_PREDICTIONS_FROZEN.json', 'predict')):
        path = run / barrier_name
        if not path.exists():
            continue
        barrier = read_json(path)
        expected_files = {str(run / 'tasks' / ('_'.join((s, v, stage)) + '.json'))
                          for s in sources for v in variants}
        require(barrier['snapshot_sha256'] == sha256(run / 'SNAPSHOT.json')
                and set(barrier['receipts_sha256']) == expected_files, 'Barrier scope differs')
        verify_files(barrier['receipts_sha256'])
    runtime_file = run / 'RUNTIME.json'
    if runtime_file.exists():
        runtime = read_json(runtime_file)
        require(runtime['device_runtime']['effective'] == 'cuda' and runtime['deterministic_algorithms']
                and not runtime['tf32'] and not runtime['vscode_hardware_acceleration_changed'],
                'Runtime device/determinism differs')
    pending, models, results, membership = [], [], [], []
    for source in sources:
        binding = snap['sources'][source]
        split = read_json(Path(binding['binding']['split']) / 'split_manifest.json')
        require(set(split['train_pairs']).isdisjoint(split['validation_pairs']), 'Pair split overlaps')
        admission = read_json(ROOT / cfg['frozen_candidate_run'] / 'augmentation' / ('as' + source) / 'ADMISSION.json')
        with Path(admission['path']).open(newline='') as stream:
            library = [tuple(row[0].split('->')) for row in csv.reader(stream)]
        prepared = read_json(binding['binding']['prepared'])
        require(signature(library) == admission['path_set_sha256']
                == binding['binding']['admitted_library_path_set_sha256']
                == prepared['admitted_library_path_set_sha256'][source], 'Admitted library differs')
        initial_file = run / ('as' + source) / 'INITIALIZATION.json'
        paired, weight_schema, predictions = [], None, {}
        for variant in variants:
            directory = run / ('as' + source) / variant
            model_file = directory / 'MODEL_FROZEN.json'
            if not model_file.exists():
                pending.append(dict(source=source, variant=variant, stage='train')); continue
            model, initial = read_json(model_file), read_json(initial_file)
            require(model['identity'] == dict(snapshot_sha256=sha256(run / 'SNAPSHOT.json'), source=source, variant=variant)
                    and model['configuration'] == binding['arguments'] and not model['final_labels_read']
                    and model['label_temperature'] == cfg['label_temperature'], 'Model protocol differs')
            require(model['initial_sha256'] == initial['state_dict_sha256']
                    and initial['seed'] == cfg['seed']
                    and initial['snapshot_sha256'] == sha256(run / 'SNAPSHOT.json'), 'Initialization differs')
            verify_files({model['model']: model['model_sha256']})
            history = read_json(directory / 'TRAINING_HISTORY.json')
            check_history(history, model, binding['arguments']['patience'])
            if check_weights:
                import torch
                state = torch.load(model['model'], map_location='cpu')
                schema = {key: (tuple(value.shape), str(value.dtype)) for key, value in state.items()}
                require(all(bool(torch.isfinite(v).all()) for v in state.values()), 'Nonfinite model weights')
                require(weight_schema is None or weight_schema == schema, 'Paired architecture differs')
                weight_schema = schema
                del state
            paired.append(model)
            for stage in ('train', 'predict', 'evaluate'):
                path = run / 'tasks' / ('_'.join((source, variant, stage)) + '.json')
                if not path.exists():
                    continue
                receipt = read_json(path)
                require(receipt['config_sha256'] == sha256(cfg_file) and receipt['source'] == source
                        and receipt['variant'] == variant and receipt['stage'] == stage, 'Task identity differs')
                verify_files(receipt['code_sha256']); verify_files(receipt['outputs_sha256'])
                timing = read_json(directory / ('TIMING_' + stage + '.json'))
                require(all(math.isfinite(timing[k]) and timing[k] >= 0 for k in ('wall_seconds', 'cpu_seconds')),
                        'Invalid measured stage time')
            models.append(dict(source=source, variant=variant, epochs=model['epochs'], selected_epoch=model['selected_epoch'],
                validation_jaccard=model['validation_jaccard'],
                epoch_wall_seconds=math.fsum(r['wall_seconds'] for r in history),
                epoch_cpu_seconds=math.fsum(r['cpu_seconds'] for r in history), status='PASS'))
            prediction_file = directory / 'PREDICTIONS_FROZEN.json'
            if not prediction_file.exists():
                pending.append(dict(source=source, variant=variant, stage='predict')); continue
            require((run / 'ALL_MODELS_FROZEN.json').is_file(), 'Premature prediction')
            prediction = read_json(prediction_file)
            require(prediction['model_sha256'] == model['model_sha256']
                    and prediction['candidate_sha256'] == binding['candidate_sha256']
                    and not prediction['final_labels_read'], 'Prediction identity differs')
            verify_files({prediction['predictions']: prediction['predictions_sha256']})
            predictions[variant] = prediction['predictions']
            if not (directory / 'RESULTS.json').exists():
                pending.append(dict(source=source, variant=variant, stage='evaluate')); continue
            require((run / 'ALL_PREDICTIONS_FROZEN.json').is_file(), 'Premature final evaluation')
            result = read_json(directory / 'RESULTS.json')
            original = read_json(ROOT / cfg['frozen_candidate_run'] / ('as' + source) / 'RESULTS.json')
            require(result['query_count'] == original['query_count'] == prediction['queries']
                    and result['fixed_identity'] == original['fixed_identity']
                    and result['prediction_sha256'] == prediction['predictions_sha256']
                    and result['candidate_sha256'] == binding['candidate_sha256']
                    and result['model_sha256'] == model['model_sha256']
                    and not result['final_labels_used_for_tuning'], 'Final evaluation identity differs')
            require(result['missing_predictions'] == result['invalid_nonempty_predictions'] == 0
                    and result['jaccard_accuracy'] <= original['oracle_jaccard'] + 1e-10
                    and result['ordered_exact_accuracy'] <= original['candidate_exact_available'] + 1e-10,
                    'Final result exceeds candidates or loses queries')
            if rescore:
                from pathInference.downstream.evaluation import score_source
                independent = score_source(original['reference_file'], prediction['predictions'], source, original['fixed_identity'])
                for key in ('query_count', 'ordered_exact_matches', 'total_jaccard_reward', 'jaccard_accuracy'):
                    require(abs(independent[key] - result[key]) < 1e-10, 'Independent rescore differs')
            results.append(dict(source=source, variant=variant, status='PASS', independent_rescore=rescore))
        if len(paired) == 2:
            require(paired[0]['initial_sha256'] == paired[1]['initial_sha256'], 'Paired initial weights differ')
        if check_candidates and len(predictions) == len(variants):
            vocabulary = read_json(ROOT / cfg['frozen_candidate_run'] / ('as' + source) / 'VOCABULARY.json')
            membership.append(dict(source=source, **candidate_membership(binding, vocabulary, predictions, binding['arguments']['k'])))
    if not pending:
        rows, summary = read_json(run / 'RESULTS.json'), read_json(run / 'SUMMARY.json')
        require(len(rows) == len(sources) * len(variants), 'Aggregate scope differs')
        require({(r['source'], r['variant']) for r in rows} == {(s, v) for s in sources for v in variants}, 'Aggregate duplicates')
        for row in rows:
            require(row == read_json(run / ('as' + row['source']) / row['variant'] / 'RESULTS.json'), 'Aggregate row differs')
        for variant in variants:
            subset = [r for r in rows if r['variant'] == variant]
            count = sum(r['query_count'] for r in subset)
            require(summary['variants'][variant]['queries'] == count, 'Aggregate denominator differs')
            for key in ('jaccard_accuracy', 'ordered_exact_accuracy', 'oracle_jaccard'):
                expected = math.fsum(r[key] * r['query_count'] for r in subset) / count
                require(abs(summary['variants'][variant][key] - expected) < 1e-12, 'Aggregate metric differs')
    return dict(status='INCOMPLETE' if pending else 'PASS', checked_utc=utc_now(), models=models,
        final_results=results, pending=pending, candidate_membership=membership,
        weights_checked=check_weights, final_labels_read=bool(rescore and results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/20260201/gmpi_confidence_v1')
    parser.add_argument('--rescore', action='store_true')
    parser.add_argument('--check-candidates', action='store_true')
    parser.add_argument('--check-weights', action='store_true')
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.rescore, args.check_candidates, args.check_weights)
    filename = 'COMPLETION_AUDIT.json' if result['status'] == 'PASS' else 'PRELIMINARY_AUDIT.json'
    write_json(args.run_dir / filename, result)
    print(json.dumps(dict(status=result['status'], models_checked=len(result['models']),
                         results_checked=len(result['final_results']), pending=len(result['pending'])), ensure_ascii=False))
