"""Architecture behavior, initialization controls and sequence support semantics."""
import copy

import pytest
import torch
from torch.nn import functional as F

from pathInference.core.GMPI_pathLikelihoodEst import attentionBasedScoreNetwork
from pathInference.research.edge_alignment_confidence import (
    EdgeAlignmentConfidence, directed_hashes, history_contains, longest_history_suffix, local_convolution)
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


def model(use_base=True, active_head=True):
    torch.manual_seed(27)
    net = EdgeAlignmentConfidence(torch.randn(12, 8), hidden_size=16, use_base=use_base,
                                  width=8, hidden=16, max_length=20, candidate_chunk=1)
    if active_head:
        torch.nn.init.normal_(net.alignment.output[-1].weight, std=.1)
    return net.eval()


def test_new_model_preserves_original_cold_gmpi_weights_rng_and_initial_scores():
    torch.manual_seed(3); vectors = torch.randn(12, 8)
    torch.manual_seed(27)
    base = attentionBasedScoreNetwork(vectors, 24, 8, 32, 16, 20).eval()
    after_base = torch.get_rng_state()
    torch.manual_seed(27)
    residual = EdgeAlignmentConfidence(vectors, hidden_size=16, use_base=True, width=8, hidden=16).eval()
    assert torch.equal(torch.get_rng_state(), after_base)
    state = residual.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in base.state_dict().items())
    for empty in [False, True]:
        torch.testing.assert_allclose(residual(**inputs(empty)), base(**inputs(empty)), atol=1e-6, rtol=1e-6)


def test_both_variants_have_identical_initial_tensors_but_different_active_graphs():
    residual, standalone = model(True, False), model(False, False)
    assert all(torch.equal(p, standalone.state_dict()[name]) for name, p in residual.state_dict().items())
    assert all(not p.requires_grad for p in standalone.pr.parameters())
    assert all(not p.requires_grad for p in standalone.pm.parameters())
    assert standalone.asEmbedding.weight.requires_grad
    actual = standalone(**inputs())
    torch.testing.assert_allclose(actual, torch.tensor([[[.5], [.5], [0.]]]), atol=0, rtol=0)


def test_directed_edges_triples_and_ordered_suffix_are_not_node_bags():
    paths = torch.tensor([[[1, 2, 3, 4, 0], [1, 3, 2, 4, 0]]])
    history = torch.tensor([[[9, 2, 3, 4, 0], [0, 0, 0, 0, 0]]])
    edges, triples = directed_hashes(paths, 12)
    he, ht = directed_hashes(history, 12)
    assert history_contains(edges, he).tolist() == [[[False, True, True, False], [False, False, False, False]]]
    assert history_contains(triples, ht).tolist() == [[[False, False, True, False], [False, False, False, False]]]
    assert longest_history_suffix(paths, history).tolist() == [[3., 1.]]
    assert not history_contains(torch.zeros_like(edges), he).any()


@pytest.mark.parametrize('dilation', [1, 2])
def test_window_linear_formula_matches_convolution_values_and_gradients(dilation):
    torch.manual_seed(42)
    conv = torch.nn.Conv1d(8, 8, 3, dilation=dilation, padding=dilation)
    x = torch.randn(2, 3, 5, 8, requires_grad=True)
    expected = conv(x.reshape(6, 5, 8).transpose(1, 2)).transpose(1, 2).reshape_as(x)
    actual = local_convolution(x, conv)
    torch.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    probe = torch.randn_like(x)
    a = torch.autograd.grad((actual * probe).sum(), (x, conv.weight, conv.bias), retain_graph=True)
    b = torch.autograd.grad((expected * probe).sum(), (x, conv.weight, conv.bias))
    for left, right in zip(a, b):
        torch.testing.assert_allclose(left, right, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize('use_base', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_padding_and_empty_history_are_finite_and_do_not_change_scores(use_base, empty):
    net, item = model(use_base), inputs(empty)
    expected = net(**item)
    padded = {}
    for name, value in item.items():
        if name in ('kpathsEmbedding', 'hispathEmbedding'):
            value = F.pad(value, (0, 3))
        padded[name] = F.pad(value, (0, 0, 0, 4))
    actual = net(**padded)
    torch.testing.assert_allclose(actual[:, :3], expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(actual[:, 2:]) == 0 and torch.isfinite(actual).all()
    loss = query_cross_entropy(actual, F.pad(torch.tensor([[[.9], [.3], [0.]]]), (0, 0, 0, 4)), [2])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert torch.count_nonzero(net.asEmbedding.weight.grad[0]) == 0


@pytest.mark.parametrize('use_base', [False, True])
def test_candidate_and_history_permutations_preserve_corresponding_scores(use_base):
    net, item = model(use_base), inputs()
    expected, order = net(**item), torch.tensor([2, 1, 0])
    permuted = copy.deepcopy(item)
    for name in ['kpathsEmbedding', 'k_lastSecHops', 'kcurASPair']:
        permuted[name] = permuted[name][:, order]
    torch.testing.assert_allclose(net(**permuted), expected[:, order], atol=1e-6, rtol=1e-6)
    permuted = copy.deepcopy(item)
    for name in ['hispathEmbedding', 'his_lastSecHops', 'hisASPairsEmbedding', 'hcurASPair']:
        permuted[name] = permuted[name][:, order]
    torch.testing.assert_allclose(net(**permuted), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('use_base', [False, True])
def test_competing_candidate_count_does_not_change_existing_pairwise_log_odds(use_base):
    net, item = model(use_base), inputs()
    scores = net(**item).squeeze(-1)
    expanded = copy.deepcopy(item)
    expanded['kpathsEmbedding'][:, 2] = torch.tensor([1, 7, 4, 0, 0])
    expanded['k_lastSecHops'][:, 2] = torch.tensor([7])
    expanded['kcurASPair'][:, 2] = torch.tensor([1, 4])
    actual = net(**expanded).squeeze(-1)
    torch.testing.assert_allclose(scores[:, 0].log() - scores[:, 1].log(),
                                 actual[:, 0].log() - actual[:, 1].log(), atol=1e-5, rtol=1e-5)


def test_late_features_respond_to_history_and_internal_path_order():
    net, item = model(False), inputs()
    features = net.alignment.features(item['kpathsEmbedding'], item['kcurASPair'],
        item['k_lastSecHops'], item['hispathEmbedding'], net.asEmbedding)
    assert not torch.allclose(features[:, 0], features[:, 1], atol=1e-5)
    empty = net.alignment.features(item['kpathsEmbedding'], item['kcurASPair'], item['k_lastSecHops'],
                                    torch.zeros_like(item['hispathEmbedding']), net.asEmbedding)
    assert not torch.allclose(features[:, :2], empty[:, :2], atol=1e-5)
    standalone = model(False, False).train()
    optimizer = torch.optim.Adam([p for p in standalone.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad(); output = standalone(**item)
        query_cross_entropy(output, torch.tensor([[[.9], [.3], [0.]]]), [2]).backward(); optimizer.step()
    assert standalone.alignment.convolutions[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in standalone.pr.parameters())


def test_similarity_chunks_are_computational_only_and_checkpoint_roundtrip(tmp_path):
    net, item = model(False), inputs()
    expected = net(**item)
    net.alignment.candidate_chunk = 8
    torch.testing.assert_allclose(net(**item), expected, atol=1e-6, rtol=1e-6)
    path = tmp_path / 'model.pt'; torch.save(net.state_dict(), path)
    restored = model(False); restored.load_state_dict(torch.load(path, map_location='cpu'))
    torch.testing.assert_allclose(restored(**item), expected, atol=1e-6, rtol=1e-6)


def test_engine_is_private_and_does_not_modify_completed_controls():
    from pathInference.research import gmpi_confidence as old
    from pathInference.research.edge_alignment_experiment import make_engine
    before = old.CONFIG, old.RUN, old.make_model
    engine = make_engine()
    assert engine is not old and (old.CONFIG, old.RUN, old.make_model) == before
