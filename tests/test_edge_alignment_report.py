"""Do not publish conclusions from incomplete or inconsistent final results."""
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools/validation/report_edge_alignment.py'
SPEC = importlib.util.spec_from_file_location('edge_alignment_report', PATH)
report = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(report)


def fixture():
    cfg = dict(sources=['a', 'b'], variants=['edge_only', 'gmpi_edge_residual'])
    audit = dict(status='PASS', independent_rescore=True, candidate_membership_checked=True,
                 aggregates_checked=True, csv_results_checked=True, receipt_stage_order_checked=True,
                 paired_initialization_and_effective_parameter_scope_checked=True,
                 variants=cfg['variants'], sources=[dict(source=s) for s in cfg['sources']], queries_per_variant=5)
    rows = [dict(source=s, variant=v, query_count=n) for s, n in [('a', 2), ('b', 3)] for v in cfg['variants']]
    return cfg, audit, rows


@pytest.mark.parametrize('failure', ['unrescored', 'source_order', 'missing_variant', 'duplicate', 'denominator'])
def test_refuses_incomplete_or_mixed_results(failure):
    cfg, audit, rows = copy.deepcopy(fixture())
    report.check_scope(audit, cfg, rows)
    if failure == 'unrescored': audit['independent_rescore'] = False
    elif failure == 'source_order': audit['sources'].reverse()
    elif failure == 'missing_variant': rows.pop()
    elif failure == 'duplicate': rows[-1] = rows[0]
    else: rows[-1]['query_count'] += 1
    with pytest.raises(ValueError):
        report.check_scope(audit, cfg, rows)


def test_aggregation_weights_queries_instead_of_sources():
    rows = [dict(query_count=1, jaccard_accuracy=0., ordered_exact_accuracy=0., oracle_jaccard=1.),
            dict(query_count=3, jaccard_accuracy=1., ordered_exact_accuracy=1., oracle_jaccard=1.)]
    assert report.aggregate(rows) == dict(queries=4, jaccard_accuracy=.75,
                                          ordered_exact_accuracy=.75, oracle_jaccard=1.)
