import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools/validation' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cleanup_preserves_external_evidence(tmp_path):
    backfill = module('backfill_pathradar_details')
    control = tmp_path / 'control'
    control.mkdir()
    external = tmp_path / 'external'
    external.mkdir()
    marker = external / 'original.txt'
    marker.write_text('keep')
    (control / 'evidence').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError): backfill.remove_temporary_evidence(control)
    assert marker.read_text() == 'keep'
    (control / 'evidence').unlink()
    assert backfill.remove_temporary_evidence(control) is False
    (control / 'evidence').mkdir()
    assert backfill.remove_temporary_evidence(control) is True
    assert marker.exists()


def test_independent_audit_keeps_uncovered_queries():
    audit = module('audit_pathradar_pair_details')
    rows = [dict(src='1', dst='2', predicted_path='1->2', valid_prediction='1', exact_match='1', jaccard='1'),
            dict(src='1', dst='3', predicted_path='', valid_prediction='0', exact_match='0', jaccard='0')]
    summary = dict(requested_pairs=2, coverage_count=1, exact_match_count=1, mean_jaccard_over_requested=.5)
    expected = {('1', '2'), ('1', '3')}
    assert audit.audit_rows(rows, expected, summary)['records'] == 2
    with pytest.raises(ValueError): audit.audit_rows(rows[:1], expected, summary)
    with pytest.raises(ValueError): audit.audit_rows([rows[0], rows[0]], expected, summary)
