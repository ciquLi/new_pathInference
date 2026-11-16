"""Micro metrics and missing timing must not be confused in confidence reports."""
import importlib.util
import copy
from pathlib import Path

import pytest

_FILE = Path(__file__).resolve().parents[1] / 'tools/validation/report_candidate_confidence.py'
_SPEC = importlib.util.spec_from_file_location('confidence_report', _FILE)
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def observation(source, queries, value, seconds):
    return dict(source=source, queries=queries, jaccard=value, exact=value,
                oracle_jaccard=1., jaccard_oracle_gap=1.-value,
                exact_available=1., exact_oracle_gap=1.-value,
                train_wall_seconds=seconds, train_cpu_seconds=seconds,
                predict_wall_seconds=seconds, predict_cpu_seconds=seconds)


def test_report_weights_queries_and_preserves_missing_historical_times():
    rows = [observation('1', 1, .5, None), observation('2', 3, 1., 2.)]
    result = report.summarize(rows)
    assert result['micro']['jaccard'] == pytest.approx(.875)
    assert result['macro']['jaccard'] == pytest.approx(.75)
    assert result['micro']['jaccard_oracle_gap'] == pytest.approx(.125)
    assert result['train_wall_seconds'] == dict(measured_sources=1, total_sources=2, measured_sum=2.)
    result = report.summarize([observation('1', 1, .5, None)])
    assert result['train_cpu_seconds']['measured_sum'] is None


def test_report_rejects_duplicate_source_weighting():
    with pytest.raises(ValueError, match='Duplicate'):
        report.summarize([observation('1', 1, .5, 2.), observation('1', 3, 1., 2.)])


def test_report_does_not_aggregate_different_hardware_as_one_method():
    a = dict(observation('1', 1, .5, 2.), device='CPU')
    b = dict(observation('2', 1, .6, 3.), device='CUDA')
    with pytest.raises(ValueError, match='hardware'):
        report.summarize([a, b])


def test_report_rejects_accuracy_above_candidate_oracle(tmp_path):
    metric = dict(rho=.5, query_count=1, jaccard_accuracy=.95, ordered_exact_accuracy=.8)
    with pytest.raises(ValueError, match='upper bound'):
        report.row('1', 'method', metric, dict(oracle_jaccard=.9, candidate_exact_available=.9), tmp_path)


@pytest.mark.parametrize('change', ['missing_source', 'capped', 'no_rescore', 'denominator'])
def test_optional_network_requires_complete_same_pool_audit(change):
    audit = dict(status='PASS', final_labels_read=True, independent_rescore=True, pending=[], queries=3,
                 sources=[dict(source='1', status='PASS', independent_rescore=True,
                               candidate_membership=dict(status='PASS', slot_capped_queries=0, queries=3))])
    report.check_sequence_audit(audit, ['1'])
    if change == 'missing_source':
        audit['sources'] = []
    elif change == 'capped':
        audit['sources'][0]['candidate_membership']['slot_capped_queries'] = 1
    elif change == 'no_rescore':
        audit['sources'][0]['independent_rescore'] = False
    else:
        audit['queries'] = 10
    with pytest.raises(ValueError):
        report.check_sequence_audit(audit, ['1'])


def test_report_keeps_negative_network_result_and_distinguishes_architecture(tmp_path):
    row = dict(observation('1', 3, .7, 2.), rho=.5, method='ordered_path_network')
    losses = dict(sources=[dict(source='1', rho=.5, queries=3, jaccard_change=.01, exact_change=.01)])
    network = dict(sources=[dict(source='1', rho=.5, queries=3, jaccard_change=-.02, exact_change=-.03)],
                   jaccard_wins=0, jaccard_losses=1)
    result = dict(rows=[row], inputs_sha256={},
                  aggregates={'ordered_path_network': {'all': report.summarize([row])}},
                  comparisons=dict(LightGBM=copy.deepcopy(losses), GMPI=copy.deepcopy(losses), Network=network))
    report.write_report(result, tmp_path)
    text = (tmp_path / 'REPORT.md').read_text()
    assert '-0.020000' in text and '提升 0 源、下降 1 源' in text
    assert '不改变损失或输入' in text and '不等于整体实验历时' in text
