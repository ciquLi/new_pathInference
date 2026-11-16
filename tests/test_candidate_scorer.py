"""Meaningful invariants for the independent final-candidate scorer."""
import numpy as np
import pytest

from pathInference.research.candidate_scorer import decode_candidates, query_top1, Features
from pathInference.experiment.contracts import forbid_formal_references


def test_query_reward_keeps_pair_groups_and_deterministic_ties():
    labels = np.array([0.4, 1., 0.2, 0.8, 0.3])
    scores = np.array([0., 0., 9., 1., 2.])
    assert query_top1(labels, scores, [2, 3]) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="offsets"):
        query_top1(labels, scores, [3])


def test_decode_checks_endpoints_and_does_not_consume_test_labels():
    vocabulary = ["<pad>", "19529", "6939", "174", "1"]
    row = ["1->2->4|1->3->4", "", "", "forbidden label", "19529->1"]
    assert decode_candidates(row, vocabulary, "19529") == [("19529", "6939", "1"), ("19529", "174", "1")]
    with pytest.raises(ValueError, match="decoded"):
        decode_candidates(["1->2->3", "", "", "", "19529->1"], vocabulary, "19529")
    with pytest.raises(ValueError, match="duplicated"):
        decode_candidates(["1->2->4|1->2->4", "", "", "", "19529->1"], vocabulary, "19529")


def test_training_guard_rejects_heldout_route_access(tmp_path):
    path = tmp_path / "path_without_prefix/test/19529.csv"
    path.parent.mkdir(parents=True)
    path.write_text("19529->6939->1\n")
    with forbid_formal_references():
        with pytest.raises(PermissionError):
            path.read_text()
    assert path.read_text() == "19529->6939->1\n"


def test_feature_adapter_preserves_production_row_index_contract():
    from types import SimpleNamespace
    from collections import Counter
    from pathInference.upstream.implementation.destination_features import candidate_queries
    extractor = Features.__new__(Features)
    extractor.source = "19529"
    extractor.runtime = SimpleNamespace(safe={"index": {}}, public=SimpleNamespace(relations={}))
    extractor.prefixes = {("19529", "6939")}
    extractor.plan = {"prefix_edge_provenance": {}}
    extractor.counts = Counter()
    extractor.builder = SimpleNamespace(feature_names=["x"], features=lambda row: {"x": len(row["path"])})
    def append(source, rows):
        for i, row in enumerate(rows):
            candidate_queries(row, source, i)
        return np.zeros((len(rows), 254))
    extractor.store = SimpleNamespace(append=append)
    paths = [("19529", "6939", "1"), ("19529", "6939", "174", "1")]
    assert extractor.matrix(paths).shape == (2, 255)
