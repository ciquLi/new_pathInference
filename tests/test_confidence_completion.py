"""Completion must wait for all controls and stages, not an early summary."""
import importlib.util
import json
from pathlib import Path

_FILE = Path(__file__).resolve().parents[1] / 'research/candidate_confidence_v2/finish.py'
_SPEC = importlib.util.spec_from_file_location('confidence_completion', _FILE)
completion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(completion)


def test_completion_requires_all_receipts_and_barriers(tmp_path):
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    for name in ('ALL_MODELS_FROZEN.json', 'ALL_PREDICTIONS_FROZEN.json', 'SUMMARY.json', 'REPORT.md'):
        (tmp_path / name).write_text('{}')
    for stage in ('train', 'predict', 'evaluate'):
        for source in ('1', '2'):
            (tasks / (source + '_' + stage + '.json')).write_text('{}')
    result = completion.state(tmp_path, ['1', '2'], [''])
    assert result['ready_for_audit']
    (tasks / '2_evaluate.json').unlink()
    assert not completion.state(tmp_path, ['1', '2'], [''])['ready_for_audit']
    (tmp_path / 'STATUS.json').write_text(json.dumps(dict(error='failed', updated_utc='now')))
    assert completion.state(tmp_path, ['1', '2'], [''])['error'] == 'failed'


def test_paired_controls_are_both_required(tmp_path):
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    for stage in ('train', 'predict', 'evaluate'):
        (tasks / ('1_smooth_l1_' + stage + '.json')).write_text('{}')
    result = completion.state(tmp_path, ['1'], ['smooth_l1', 'query_cross_entropy'])
    assert result['expected'] == 2
    assert result['counts'] == dict(train=1, predict=1, evaluate=1)
    assert not result['ready_for_audit']
