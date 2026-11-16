"""Behavioral guarantees required before the architecture sees frozen data."""
import copy

import pytest
import torch
from torch.nn import functional as F

from pathInference.research.path_sequence_confidence import OrderedPathConfidence
from pathInference.research.gmpi_confidence_loss import query_cross_entropy


def inputs(empty_history=False):
    paths = torch.tensor([[[1, 2, 3, 4, 0], [1, 3, 2, 4, 0], [0, 0, 0, 0, 0]]])
    pairs = torch.tensor([[[1, 4], [1, 4], [0, 0]]])
    hops = torch.tensor([[[3], [2], [0]]])
    history = torch.tensor([[[5, 2, 3, 4, 0], [6, 7, 4, 0, 0], [0, 0, 0, 0, 0]]])
    hpairs = torch.tensor([[[5, 4], [6, 4], [0, 0]]])
    hhops = torch.tensor([[[3], [7], [0]]])
    if empty_history:
        history.zero_(); hpairs.zero_(); hhops.zero_()
    return dict(kpathsEmbedding=paths, k_lastSecHops=hops, kcurASPair=pairs,
                hcurASPair=pairs.clone(), hispathEmbedding=history,
                his_lastSecHops=hhops, hisASPairsEmbedding=hpairs)


def model():
    torch.manual_seed(27)
    return OrderedPathConfidence(torch.randn(12, 8), width=8, heads=2,
                                 layers=2, feedforward=16, dropout=0).eval()


def test_order_changes_representation_even_with_identical_nodes_and_endpoints():
    net, item = model(), inputs()
    # Hold the explicit penultimate field equal too, isolating full path order.
    representation, _ = net.encode(item['kpathsEmbedding'], item['kcurASPair'],
                                   torch.tensor([[[3], [3], [0]]]))
    assert not torch.allclose(representation[:, 0], representation[:, 1], atol=1e-5)
    scores = net(**item)
    assert not torch.allclose(scores[:, 0], scores[:, 1], atol=1e-5)


@pytest.mark.parametrize('empty', [False, True])
def test_padding_does_not_change_scores_and_empty_history_has_finite_gradients(empty):
    net, item = model(), inputs(empty)
    expected = net(**item)
    padded = {}
    for name, value in item.items():
        value = F.pad(value, (0, 3)) if name in ('kpathsEmbedding', 'hispathEmbedding') else value
        padded[name] = F.pad(value, (0, 0, 0, 4))
    actual = net(**padded)
    torch.testing.assert_allclose(actual[:, :3], expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(actual[:, 2:]) == 0
    torch.testing.assert_allclose(actual.sum(1), torch.ones(1, 1))
    loss = query_cross_entropy(actual, F.pad(torch.tensor([[[.9], [.3], [0.]]]), (0, 0, 0, 4)), [2])
    loss.backward()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert net.encoder[0].attention.query.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(net.asEmbedding.weight.grad[0]) == 0


def test_candidate_permutation_is_equivariant_history_permutation_is_invariant():
    net, item = model(), inputs()
    expected = net(**item)
    order = torch.tensor([2, 1, 0])
    permuted = copy.deepcopy(item)
    for key in ('kpathsEmbedding', 'k_lastSecHops', 'kcurASPair'):
        permuted[key] = permuted[key][:, order]
    # Query endpoint lookup must not assume slot zero is occupied.
    actual = net(**permuted)
    torch.testing.assert_allclose(actual, expected[:, order], atol=1e-6, rtol=1e-6)
    permuted = copy.deepcopy(item)
    for key in ('hispathEmbedding', 'his_lastSecHops', 'hisASPairsEmbedding', 'hcurASPair'):
        permuted[key] = permuted[key][:, order]
    torch.testing.assert_allclose(net(**permuted), expected, atol=1e-6, rtol=1e-6)


def test_history_is_used_and_checkpoint_roundtrip_preserves_scores(tmp_path):
    net, item = model(), inputs()
    assert not torch.allclose(net(**item), net(**inputs(True)), atol=1e-5)
    path = tmp_path / 'weights.pt'
    torch.save(net.state_dict(), path)
    restored = model()
    restored.load_state_dict(torch.load(path, map_location='cpu'))
    torch.testing.assert_allclose(restored(**item), net(**item), atol=0, rtol=0)


def test_engine_isolation_preserves_completed_control_globals():
    from pathInference.research import gmpi_confidence as control
    from pathInference.research.path_sequence_experiment import make_engine
    original = control.RUN, control.CONFIG, control.make_model
    engine = make_engine()
    assert engine is not control and engine.RUN != control.RUN
    assert (control.RUN, control.CONFIG, control.make_model) == original
