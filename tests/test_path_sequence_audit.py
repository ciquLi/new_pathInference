"""Reject objective, split and training changes hidden in an architecture run."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / 'tools/validation'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('path_sequence_audit', DIRECTORY / 'audit_path_sequence_confidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def protocol():
    cfg = dict(sources=list(map(str, range(10))), label_temperature=.1, seed=1,
               control_variant='query_cross_entropy', variants=['ordered_path'])
    original = dict(configuration=copy.deepcopy(cfg), sources={s: {'split': 'fixed'} for s in cfg['sources']})
    snapshot = dict(configuration=copy.deepcopy(cfg), sources=copy.deepcopy(original['sources']),
                    candidates_regenerated=False, test_labels_used_for_tuning=False)
    return cfg, snapshot, original


@pytest.mark.parametrize('change', ['temperature', 'split', 'scope', 'test'])
def test_architecture_audit_rejects_extra_factors(change):
    cfg, snapshot, original = protocol()
    audit.check_protocol(cfg, snapshot, original)
    if change == 'temperature':
        cfg['label_temperature'] = .2
        snapshot['configuration']['label_temperature'] = .2
    elif change == 'split':
        snapshot['sources']['0']['split'] = 'other'
    elif change == 'scope':
        cfg['sources'] = cfg['sources'][:2]
        snapshot['configuration']['sources'] = cfg['sources']
    else:
        snapshot['test_labels_used_for_tuning'] = True
    with pytest.raises(ValueError):
        audit.check_protocol(cfg, snapshot, original)


@pytest.mark.parametrize('change', ['lr', 'library', 'labels'])
def test_training_control_rejects_retuning_and_other_libraries(change):
    args = dict(lr=.001, patience=3)
    model = dict(configuration=copy.deepcopy(args), label_temperature=.1,
                 admitted_library_path_set_sha256='fixed', final_labels_read=False)
    baseline = copy.deepcopy(model)
    binding = dict(binding=dict(admitted_library_path_set_sha256='fixed'))
    audit.check_training_control(model, baseline, args, binding, dict(label_temperature=.1))
    if change == 'lr':
        model['configuration']['lr'] = .005
    elif change == 'library':
        model['admitted_library_path_set_sha256'] = 'other'
    else:
        model['final_labels_read'] = True
    with pytest.raises(ValueError, match='another factor'):
        audit.check_training_control(model, baseline, args, binding, dict(label_temperature=.1))


@pytest.mark.parametrize('change', ['micro', 'macro', 'rho'])
def test_summary_audit_rejects_wrong_means_or_groups(change):
    row = dict(query_count=3, rho=2, jaccard_accuracy=.6, ordered_exact_accuracy=.2,
               control_jaccard=.5, control_exact=.1, oracle_jaccard=.9)
    metrics = dict(queries=3, sources=1, jaccard_accuracy=.6, ordered_exact_accuracy=.2,
                   control_jaccard=.5, control_exact=.1, oracle_jaccard=.9,
                   jaccard_delta=.1, exact_delta=.1, oracle_gap=.3)
    summary = dict(queries=3, micro=copy.deepcopy(metrics), macro=dict(jaccard_accuracy=.6,
                   ordered_exact_accuracy=.2, jaccard_delta=.1, exact_delta=.1),
                   by_rho=dict(high=copy.deepcopy(metrics), low=dict(queries=0)))
    audit.check_summary([row], summary)
    if change == 'micro':
        summary['micro']['jaccard_accuracy'] = .8
    elif change == 'macro':
        summary['macro']['exact_delta'] = .7
    else:
        summary['by_rho']['high']['queries'] = 0
    with pytest.raises(ValueError):
        audit.check_summary([row], summary)
