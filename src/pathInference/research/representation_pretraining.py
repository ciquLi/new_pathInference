"""Self-supervised initialization, never route generation or final supervision.

Only the training-pair portion of each frozen admitted library is consumed.
The deployed network is exactly OrderedPathConfidence; auxiliary heads are
discarded before the unchanged query-level ranking fine-tuning stage.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathInference.experiment.contracts import require, signature
from pathInference.utils import read_json


PATH_MODULES = ('node', 'endpoints', 'position', 'reverse_position', 'transition',
                'token_norm', 'encoder', 'pool', 'path_output')


def training_corpus(library, split_file, vocabulary, source, max_length=20):
    """Exact whole paths only; no new subpaths/labels or validation-pair routes."""
    split = read_json(split_file)
    train, validation = set(split['train_pairs']), set(split['validation_pairs'])
    require(train.isdisjoint(validation), 'Pair-disjoint split overlap')
    stoi = {str(node): index for index, node in enumerate(vocabulary)}
    require(vocabulary[:5] == ['<pad>', '<unk>', '<eos>', '<sos>', '<mask>'],
            'Unexpected frozen vocabulary special tokens')
    all_paths, retained, excluded = [], [], dict(validation=0, other=0)
    with Path(library).open(newline='') as stream:
        for row in csv.reader(stream):
            require(len(row) == 1, 'Malformed admitted path library')
            path = tuple(row[0].split('->'))
            require(2 <= len(path) <= max_length and path[0] == source
                    and all(node.isascii() and node.isdecimal() and int(node) > 0 for node in path),
                    'Invalid admitted path; never truncate or invent nodes')
            all_paths.append(path)
            pair = path[0] + '->' + path[-1]
            if pair not in train:
                excluded['validation' if pair in validation else 'other'] += 1
                continue
            require(all(node in stoi for node in path), 'Unseen AS must not silently map to padding')
            retained.append(path)
    # Stable order plus deduplication prevents occurrence-count changes to weights.
    retained = list(dict.fromkeys(retained))
    require(retained, 'No admitted training paths for self supervision')
    encoded = np.zeros((len(retained), max_length), dtype=np.int64)
    for index, path in enumerate(retained):
        encoded[index, :len(path)] = [stoi[node] for node in path]
    return encoded, dict(paths=len(retained), excluded=excluded,
        admitted_library_path_set_sha256=signature(all_paths),
        training_path_set_sha256=signature(retained),
        training_pairs=sorted({p[0] + '->' + p[-1] for p in retained}),
        validation_pairs_excluded=True, final_labels_read=False,
        corpus_policy='whole admitted paths whose endpoints belong to frozen train_pairs only')


def contextual_tokens(model, paths):
    """Same path token computation as the frozen encoder, with no endpoint bypass."""
    mask = paths.ne(0)
    nodes = model.node(model.asEmbedding(paths)).masked_fill(~mask[..., None], 0)
    previous = F.pad(nodes[:, :-1], (0, 0, 1, 0))
    edges = model.transition(torch.cat((previous, nodes, nodes - previous, nodes * previous), -1))
    edges = edges.masked_fill(~(mask & F.pad(mask[:, :-1], (1, 0), value=False))[..., None], 0)
    position = torch.arange(paths.shape[1], device=paths.device)
    reverse = (mask.sum(-1, keepdim=True) - 1 - position).clamp(min=0)
    tokens = model.token_norm(nodes + edges + model.position(position) + model.reverse_position(reverse))
    tokens = tokens.masked_fill(~mask[..., None], 0)
    for block in model.encoder:
        tokens = block(tokens, mask)
    return tokens


def path_vectors(model, paths):
    lengths = paths.ne(0).sum(-1)
    pairs = torch.stack((paths[:, 0], paths.gather(1, (lengths - 1)[:, None]).squeeze(1)), -1)
    hops = paths.gather(1, (lengths - 2)[:, None])
    return model.encode(paths[:, None], pairs[:, None], hops[:, None])[0][:, 0]


def mask_interior(paths, mask_id=4):
    """Mask one actual interior AS; for two-hop paths mask the destination."""
    lengths = paths.ne(0).sum(-1)
    require(bool(lengths.ge(2).all()), 'Pretraining requires actual paths')
    widths = (lengths - 2).clamp(min=1)
    positions = (torch.rand(len(paths), device=paths.device) * widths).long() + 1
    rows = torch.arange(len(paths), device=paths.device)
    labels = paths[rows, positions].clone()
    corrupted = paths.clone()
    corrupted[rows, positions] = mask_id
    return corrupted, positions, labels


def sampled_logits(queries, embedding, targets, pool, negatives, temperature,
                   sources=None, observed_keys=None):
    """Sampled identification task; alternatives are not labelled illegal routes."""
    indices = torch.randint(len(pool), (len(targets), negatives), device=targets.device)
    negative = pool[indices]
    candidates = torch.cat((targets[:, None], negative), 1)
    vectors = F.normalize(embedding(candidates), dim=-1)
    logits = (F.normalize(queries, dim=-1)[:, None] * vectors).sum(-1) / temperature
    # Repeated true target samples must never be negative examples.
    excluded = negative.eq(targets[:, None])
    if observed_keys is not None:
        keys = sources[:, None] * embedding.num_embeddings + negative
        positions = torch.searchsorted(observed_keys, keys.reshape(-1)).reshape_as(keys)
        excluded |= observed_keys[positions.clamp(max=len(observed_keys) - 1)].eq(keys)
    logits[:, 1:] = logits[:, 1:].masked_fill(excluded, -1e4)
    return logits


def contrastive_loss(first, second, endpoints, temperature):
    if len(first) < 2:
        return (first.sum() + second.sum()) * 0
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / temperature
    same_pair = (endpoints[:, None] == endpoints[None, :]).all(-1)
    identity = torch.eye(len(first), dtype=torch.bool, device=first.device)
    logits = logits.masked_fill(same_pair & ~identity, -1e4)
    targets = torch.arange(len(first), device=first.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def edge_examples(paths):
    edges = [(int(a), int(b)) for path in paths for a, b in zip(path[:-1], path[1:]) if a and b]
    require(edges, 'No directed training transitions')
    return np.asarray(edges, dtype=np.int64)


def configure_stage(model, stage):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (name.startswith('asEmbedding.') if stage == 'as_context'
                                   else name.split('.')[0] in PATH_MODULES)


def stage_epoch(model, paths, stage, head, optimizer, cfg):
    """Fixed-epoch auxiliary training; no validation/test routes are read."""
    model.train()
    device = model.asEmbedding.weight.device
    pool = torch.unique(paths[paths.ne(0)])
    require(bool(pool.ge(5).all()), 'Special tokens must not be reconstruction targets')
    dataset = torch.as_tensor(edge_examples(paths.cpu().numpy()), device=device) if stage == 'as_context' else paths
    observed = (torch.unique(dataset[:, 0] * model.asEmbedding.num_embeddings + dataset[:, 1], sorted=True)
                if stage == 'as_context' else None)
    order = torch.randperm(len(dataset), device=device)
    total, examples = 0., 0
    for offset in range(0, len(order), cfg['batch_size']):
        batch = dataset[order[offset:offset + cfg['batch_size']]]
        optimizer.zero_grad()
        if stage == 'as_context':
            queries = head(model.asEmbedding(batch[:, 0]))
            logits = sampled_logits(queries, model.asEmbedding, batch[:, 1], pool,
                                    cfg['negatives'], cfg['temperature'], batch[:, 0], observed)
            # Multiple observed next hops are positives, never artificial negatives.
            loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
        else:
            corrupted, positions, targets = mask_interior(batch)
            tokens = contextual_tokens(model, corrupted)
            queries = head(tokens[torch.arange(len(batch), device=device), positions])
            logits = sampled_logits(queries, model.asEmbedding, targets, pool,
                                    cfg['negatives'], cfg['temperature'])
            reconstruction = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
            first, second = path_vectors(model, batch), path_vectors(model, batch)
            lengths = batch.ne(0).sum(-1)
            endpoints = torch.stack((batch[:, 0], batch.gather(1, (lengths - 1)[:, None]).squeeze(1)), -1)
            loss = reconstruction + cfg['contrastive_weight'] * contrastive_loss(
                first, second, endpoints, cfg['temperature'])
        require(bool(torch.isfinite(loss)), 'Nonfinite pretraining loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters()) + list(head.parameters())
                                       if p.requires_grad], cfg['gradient_clip'])
        optimizer.step()
        total += float(loss.detach()) * len(batch)
        examples += len(batch)
    return total / examples, examples
