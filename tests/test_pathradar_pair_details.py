import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('pair_details', Path(__file__).resolve().parents[1] /
    'research/baselines/pathradar/performance/pair_details.py')
details = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(details)


def fixture():
    return dict(include_pair_details=True, requested_pairs=2, pairs_with_ground_truth=2,
        pairs_missing_ground_truth=0, coverage_count=1, exact_match_count=1,
        mean_jaccard_over_requested=.5, pair_details=[
            dict(src='1', dst='2', predicted_path=['1','2'], covered=True,
                 exact_match=True, jaccard=1.),
            dict(src='1', dst='3', predicted_path=None, covered=False,
                 exact_match=False, jaccard=0.)])


def test_persists_uncovered_queries(tmp_path):
    result = fixture()
    output = tmp_path / 'evaluation.json'
    output.write_text('{}')
    receipt = details.persist(output, result, {('1','2'), ('1','3')})
    assert receipt['records'] == receipt['requested_pairs'] == 2
    assert output.with_suffix('.pairs.csv').read_text().splitlines()[-1] == '1,3,,0.0,0,0'


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'inventory', 'invalid', 'score'])
def test_rejects_incomplete_or_inconsistent_details(change):
    result = fixture()
    expected = {('1','2'), ('1','3')}
    if change == 'missing': result['pair_details'].pop()
    if change == 'duplicate': result['pair_details'][1] = result['pair_details'][0]
    if change == 'inventory': expected = {('1','2'), ('1','4')}
    if change == 'invalid': result['pair_details'][1]['predicted_path'] = ['1','3']
    if change == 'score': result['mean_jaccard_over_requested'] = .6
    with pytest.raises(ValueError): details.validate(result, expected)
