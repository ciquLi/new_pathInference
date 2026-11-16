"""Research loss preserves network probabilities and excludes padded candidates."""
import pytest
import torch

from pathInference.research.gmpi_confidence_loss import query_cross_entropy, replace_training_loss


def test_normalized_output_loss_matches_cross_entropy_and_has_no_padding_gradient():
    logits = torch.tensor([[.3, -.2, 0.], [.8, 0., 0.]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    probabilities = logits.masked_fill(~mask, -float('inf')).softmax(dim=1)
    labels = torch.tensor([[.9, .6, 0.], [1., 0., 0.]], dtype=torch.float64)
    loss = query_cross_entropy(probabilities.unsqueeze(-1), labels.unsqueeze(-1), [2, 1])
    expected = -(labels[0, :2].div(.1).softmax(0) * probabilities[0, :2].log()).sum() / 2
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    expected_grad = (probabilities[0, :2] - labels[0, :2].div(.1).softmax(0)) / 2
    assert torch.allclose(logits.grad[0, :2], expected_grad)
    assert logits.grad[0, 2].item() == 0.
    assert torch.all(logits.grad[1] == 0.)


def test_loss_rejects_logits_and_invalid_candidate_mass():
    labels = torch.tensor([[[.9], [.3]]])
    with pytest.raises(ValueError, match='probabilities'):
        query_cross_entropy(torch.tensor([[[2.], [-1.]]]), labels, [2])
    with pytest.raises(ValueError, match='mass'):
        query_cross_entropy(torch.tensor([[[.6], [.2]]]), labels, [2])


def frozen_example(item):
    pathScore, similarity = item[0], item[1]
    loss = F.smooth_l1_loss(pathScore, similarity.detach())
    return loss


def no_loss_example(item):
    return item[0]


def test_adapter_changes_only_loss_and_preserves_original_function():
    import torch.nn.functional as F
    frozen_example.__globals__['F'] = F
    scores = torch.tensor([[[.6], [.4]]], requires_grad=True)
    labels = torch.tensor([[[1.], [.5]]])
    item = [scores, labels, None, None, None, None, None, None, None, torch.tensor([2])]
    original = frozen_example(item)
    adapted = replace_training_loss(frozen_example)
    assert adapted(item).item() == pytest.approx(query_cross_entropy(scores, labels, [2]).item())
    assert frozen_example(item).item() == original.item()
    assert original.item() != adapted(item).item()
    with pytest.raises(ValueError, match='exactly one'):
        replace_training_loss(no_loss_example)
