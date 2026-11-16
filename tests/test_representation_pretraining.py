"""Leakage, parameter boundaries and auxiliary sequence learning guarantees."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pathInference.research.path_sequence_confidence import OrderedPathConfidence, masked_softmax
from pathInference.research import representation_pretraining as ssl
from pathInference.research.representation_experiment import verify_stage_changes


def network():
    torch.manual_seed(9)
    return OrderedPathConfidence(torch.randn(16, 8), width=8, heads=2,
                                 layers=1, feedforward=16, dropout=.1)


def test_corpus_uses_only_admitted_whole_train_pairs(tmp_path):
    library, split = tmp_path / 'paths.csv', tmp_path / 'split.json'
    with library.open('w', newline='') as stream:
        csv.writer(stream).writerows([['1->2->3'], ['1->4'], ['1->2->3'], ['1->5']])
    split.write_text(json.dumps(dict(train_pairs=['1->3'], validation_pairs=['1->4'])))
    vocab = ['<pad>', '<unk>', '<eos>', '<sos>', '<mask>', '1', '2', '3', '4', '5']
    encoded, manifest = ssl.training_corpus(library, split, vocab, '1', 5)
    assert encoded.tolist() == [[5, 6, 7, 0, 0]]
    assert manifest['paths'] == 1 and manifest['excluded'] == dict(validation=1, other=1)
    assert manifest['training_pairs'] == ['1->3'] and not manifest['final_labels_read']
    # Cannot silently map unknown nodes, truncate long routes or overlap splits.
    with pytest.raises(ValueError, match='truncate'):
        ssl.training_corpus(library, split, vocab, '1', 2)
    split.write_text(json.dumps(dict(train_pairs=['1->3'], validation_pairs=['1->3'])))
    with pytest.raises(ValueError, match='overlap'):
        ssl.training_corpus(library, split, vocab, '1', 5)


def test_pretraining_token_pool_exactly_matches_deployed_encoder():
    model = network().eval()
    paths = torch.tensor([[5, 6, 7, 0], [5, 8, 9, 10]])
    tokens = ssl.contextual_tokens(model, paths)
    weights = masked_softmax(model.pool(tokens).squeeze(-1), paths.ne(0))
    pooled = (tokens * weights[..., None]).sum(1)
    endpoints = torch.tensor([[5, 7], [5, 10]])
    hops = torch.tensor([6, 9])
    expected = model.path_output(torch.cat((pooled, model.endpoints(model.asEmbedding(endpoints).flatten(1)),
                                           model.node(model.asEmbedding(hops))), -1))
    torch.testing.assert_allclose(ssl.path_vectors(model, paths), expected, atol=1e-6, rtol=1e-6)
    padded = F.pad(paths, (0, 3))
    torch.testing.assert_allclose(ssl.contextual_tokens(model, padded)[:, :4], tokens, atol=1e-6, rtol=1e-6)


def test_masking_removes_target_and_preserves_length_and_other_tokens():
    paths = torch.tensor([[5, 6, 7, 8, 0], [5, 9, 0, 0, 0]])
    corrupted, positions, targets = ssl.mask_interior(paths)
    assert positions[0].item() in (1, 2) and positions[1].item() == 1
    assert torch.equal(paths.ne(0), corrupted.ne(0))
    restored = corrupted.clone(); restored[torch.arange(2), positions] = targets
    assert torch.equal(restored, paths)
    assert torch.equal(corrupted[torch.arange(2), positions], torch.tensor([4, 4]))


def test_observed_alternative_next_hops_are_never_negatives():
    embedding = torch.nn.Embedding(16, 8)
    pool = torch.tensor([5, 6, 7, 8])
    targets, sources = torch.tensor([6]), torch.tensor([5])
    observed = torch.tensor([5 * 16 + 6, 5 * 16 + 7])
    torch.manual_seed(2)
    negatives = pool[torch.randint(4, (1, 128))]
    # Reset with a fixed query to keep the random draw identical.
    query = torch.ones(1, 8)
    torch.manual_seed(2)
    logits = ssl.sampled_logits(query, embedding, targets, pool, 128, .1, sources, observed)
    excluded = negatives.eq(6) | negatives.eq(7)
    assert bool(logits[:, 1:][excluded].eq(-1e4).all())
    assert bool(logits[:, 1:][~excluded].gt(-1e4).all())


@pytest.mark.parametrize('stage', ['as_context', 'path_context'])
def test_auxiliary_training_changes_only_intended_transferred_parameters(stage):
    model = network()
    paths = torch.tensor([[5, 6, 7, 0], [5, 8, 9, 0], [5, 10, 11, 12]])
    before = {name: value.clone() for name, value in model.state_dict().items()}
    ssl.configure_stage(model, stage)
    head = torch.nn.Linear(8, 8)
    optimizer = torch.optim.Adam([p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad], lr=.001)
    loss, count = ssl.stage_epoch(model, paths, stage, head, optimizer,
        dict(batch_size=3, negatives=8, temperature=.1, gradient_clip=1., contrastive_weight=.1))
    assert np.isfinite(loss) and count > 0
    after = model.state_dict()
    changed = [name for name in before if not before[name].equal(after[name])]
    verify_stage_changes(stage, changed)
    assert model.asEmbedding.weight[0].eq(0).all()
    if stage == 'path_context':
        assert before['asEmbedding.weight'].equal(after['asEmbedding.weight'])
        assert any(name.startswith('encoder.') for name in changed)
    else:
        assert changed == ['asEmbedding.weight']


def test_pair_equivalent_paths_are_not_contrastive_false_negatives():
    first, second = torch.randn(2, 8, requires_grad=True), torch.randn(2, 8, requires_grad=True)
    loss = ssl.contrastive_loss(first, second, torch.tensor([[5, 7], [5, 7]]), .1)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(first.grad).all()


def test_isolated_engine_does_not_change_existing_experiment():
    from pathInference.research import gmpi_confidence as control
    from pathInference.research.representation_experiment import make_engine
    original = control.RUN, control.CONFIG, control.make_model
    engine = make_engine()
    assert engine is not control and engine.RUN != control.RUN
    assert (control.RUN, control.CONFIG, control.make_model) == original


def test_auxiliary_epoch_resume_preserves_optimizer_and_dropout_sequence():
    model = network()
    ssl.configure_stage(model, 'path_context')
    head = torch.nn.Linear(8, 8)
    optimizer = torch.optim.Adam([p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad], lr=.001)
    paths = torch.tensor([[5, 6, 7, 0], [5, 8, 9, 0], [5, 10, 11, 12]])
    cfg = dict(batch_size=2, negatives=8, temperature=.1, gradient_clip=1., contrastive_weight=.1)
    ssl.stage_epoch(model, paths, 'path_context', head, optimizer, cfg)
    checkpoint = copy.deepcopy(dict(model=model.state_dict(), head=head.state_dict(),
                                    optimizer=optimizer.state_dict(), rng=torch.get_rng_state()))
    expected_loss, expected_count = ssl.stage_epoch(model, paths, 'path_context', head, optimizer, cfg)
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint['model']); head.load_state_dict(checkpoint['head'])
    optimizer.load_state_dict(checkpoint['optimizer']); torch.set_rng_state(checkpoint['rng'])
    actual_loss, actual_count = ssl.stage_epoch(model, paths, 'path_context', head, optimizer, cfg)
    assert actual_loss == expected_loss and actual_count == expected_count
    assert all(expected[key].equal(value) for key, value in model.state_dict().items())


def test_independent_audit_rejects_path_pretraining_as_or_scorer_changes():
    folder = Path(__file__).resolve().parents[1] / 'tools/validation'
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('representation_audit', folder / 'audit_representation_pretraining.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    before = network().state_dict()
    after = copy.deepcopy(before)
    after['node.weight'][0, 0] += 1
    audit.check_changes(before, after, 'path_context')
    after['asEmbedding.weight'][5, 0] += 1
    with pytest.raises(ValueError, match='Unrelated'):
        audit.check_changes(before, after, 'path_context')
    with pytest.raises(ValueError, match='Unrelated'):
        audit.check_changes(before, after, 'as_context')
