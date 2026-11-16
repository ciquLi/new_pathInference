"""Ranking derivatives, query isolation and durable prediction recovery."""
import sqlite3

import numpy as np
import pytest

from pathInference.research.candidate_confidence import (
    commit_batch, grouped_softmax, listwise_derivatives, open_journal,
)


def test_softmax_groups_do_not_mix_and_large_scores_are_stable():
    values = np.array([10000., 10000., -10000., -9999., 2.])
    probabilities = grouped_softmax(values, [2, 2, 1])
    assert probabilities[:2].tolist() == [0.5, 0.5]
    assert sum(probabilities[2:4]) == pytest.approx(1.)
    assert probabilities[-1] == 1.
    assert np.isfinite(probabilities).all()
    with pytest.raises(ValueError, match="groups"):
        grouped_softmax(values, [2, 2])
    with pytest.raises(ValueError, match="temperature"):
        grouped_softmax(values, [5], 0.)


def test_listwise_gradient_and_diagonal_hessian_match_finite_differences():
    scores = np.array([0.2, -0.4, 0.5, 1., -2.])
    labels = np.array([0.9, 0.6, 1., 0.2, 0.4])
    groups = [3, 2]
    gradient, hessian = listwise_derivatives(scores, labels, groups, .1)
    targets = grouped_softmax(labels, groups, .1)
    def loss(x):
        return -np.sum(targets * np.log(grouped_softmax(x, groups)))
    for i in range(len(scores)):
        delta = np.zeros_like(scores); delta[i] = 1e-4
        numeric_gradient = (loss(scores + delta) - loss(scores - delta)) / 2e-4
        numeric_hessian = (loss(scores + delta) - 2*loss(scores) + loss(scores-delta)) / 1e-8
        assert gradient[i] == pytest.approx(numeric_gradient, abs=1e-7)
        assert hessian[i] == pytest.approx(numeric_hessian, abs=1e-6)
    assert sum(gradient[:3]) == pytest.approx(0.)
    assert sum(gradient[3:]) == pytest.approx(0.)


def test_single_candidate_has_no_sorting_signal_and_query_shift_does_not_change_loss():
    scores, labels, groups = np.array([2., 3., 9.]), np.array([1., .4, .8]), [1, 2]
    g, h = listwise_derivatives(scores, labels, groups, .1)
    assert g[0] == h[0] == 0.
    shifted = scores + np.array([50., -7., -7.])
    g2, h2 = listwise_derivatives(shifted, labels, groups, .1)
    assert np.allclose(g, g2) and np.allclose(h, h2)


def test_journal_resumes_only_matching_model_and_candidate_identity(tmp_path):
    path = tmp_path / "prediction.sqlite"
    identity = {"model": "a", "candidate": "b"}
    database = open_journal(path, identity)
    commit_batch(database, [("1->2", "1->3->2", "predicted")], 100)
    database.close()
    database = open_journal(path, identity)
    assert database.execute("SELECT value FROM metadata WHERE key='offset'").fetchone()[0] == "100"
    assert database.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        commit_batch(database, [("1->4", "1->4", "predicted"),
                                ("1->2", "1->2", "predicted")], 200)
    assert database.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
    assert database.execute("SELECT value FROM metadata WHERE key='offset'").fetchone()[0] == "100"
    database.close()
    with pytest.raises(ValueError, match="identity"):
        open_journal(path, {"model": "different", "candidate": "b"})
