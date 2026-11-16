"""Ordered, candidate-specific confidence network over the frozen GMPI inputs.

No topology labels or extra routes are loaded here. Directed transitions are
learned from the existing ordered AS embeddings, not inferred AS relationships.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(logits, mask, dim=-1):
    mask = mask.to(dtype=torch.bool, device=logits.device)
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=dim)
    weights = weights.masked_fill(~mask, 0)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


class Attention(nn.Module):
    """Explicit attention handles absent history without all-masked NaNs."""

    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('Attention width must be divisible by heads')
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, queries, memory, memory_mask):
        batch, count, width = queries.shape

        def heads(value):
            return value.reshape(batch, -1, self.heads, self.head_width).transpose(1, 2)

        q, k, v = heads(self.query(queries)), heads(self.key(memory)), heads(self.value(memory))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_width)
        weights = masked_softmax(scores, memory_mask[:, None, None, :])
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, count, width)
        output = self.output(output)
        return output * memory_mask.any(dim=-1)[:, None, None].to(output.dtype)


class ContextBlock(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.attention = Attention(width, heads)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(feedforward, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask):
        values = self.norm1(values + self.dropout(self.attention(values, values, mask)))
        values = self.norm2(values + self.dropout(self.feedforward(values)))
        return values.masked_fill(~mask[..., None], 0)


class OrderedPathConfidence(nn.Module):
    input_semantics_version = 2

    def __init__(self, vectors, width=64, heads=4, layers=2, feedforward=128,
                 dropout=0.1, max_length=20):
        super().__init__()
        if vectors.ndim != 2 or max_length < 2:
            raise ValueError('Expected AS vectors and a path capacity of at least two')
        self.max_length, self.width = max_length, width
        vectors = vectors.detach().clone().float()
        vectors[0].zero_()
        self.asEmbedding = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)
        self.node = nn.Linear(vectors.shape[1], width)
        self.endpoints = nn.Linear(2 * vectors.shape[1], width)
        self.position = nn.Embedding(max_length, width)
        self.reverse_position = nn.Embedding(max_length, width)
        # Both order and direction survive: [previous, current, delta, product].
        self.transition = nn.Linear(4 * width, width)
        self.token_norm = nn.LayerNorm(width)
        self.encoder = nn.ModuleList([ContextBlock(width, heads, feedforward, dropout)
                                     for _ in range(layers)])
        self.pool = nn.Linear(width, 1)
        self.path_output = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.history_attention = Attention(width, heads)
        self.candidate_context = ContextBlock(width, heads, feedforward, dropout)
        self.context_norm = nn.LayerNorm(width)
        self.scorer = nn.Sequential(nn.Linear(5 * width, feedforward), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(feedforward, 1))

    def encode(self, paths, pairs, penultimate):
        batch, slots, length = paths.shape
        if not 2 <= length <= self.max_length:
            raise ValueError('Path length exceeds frozen capacity; paths must never be truncated')
        if pairs.shape != (batch, slots, 2) or penultimate.shape != (batch, slots, 1):
            raise ValueError('Invalid path, endpoint or penultimate input shape')
        mask = paths.ne(0)
        # Padding has no influence even after learned projections acquire biases.
        nodes = self.node(self.asEmbedding(paths)).masked_fill(~mask[..., None], 0)
        previous = F.pad(nodes[:, :, :-1, :], (0, 0, 1, 0))
        edges = self.transition(torch.cat((previous, nodes, nodes - previous, nodes * previous), dim=-1))
        edge_mask = mask & F.pad(mask[:, :, :-1], (1, 0), value=False)
        edges = edges.masked_fill(~edge_mask[..., None], 0)
        positions = torch.arange(length, device=paths.device)
        reverse = (mask.sum(-1, keepdim=True) - 1 - positions).clamp(min=0)
        tokens = self.token_norm(nodes + edges + self.position(positions) + self.reverse_position(reverse))
        tokens = tokens.masked_fill(~mask[..., None], 0).reshape(batch * slots, length, self.width)
        flat_mask = mask.reshape(batch * slots, length)
        for block in self.encoder:
            tokens = block(tokens, flat_mask)
        weights = masked_softmax(self.pool(tokens).squeeze(-1), flat_mask)
        pooled = (tokens * weights[..., None]).sum(dim=1).reshape(batch, slots, self.width)
        endpoints = self.endpoints(self.asEmbedding(pairs).reshape(batch, slots, -1))
        hops = self.node(self.asEmbedding(penultimate).squeeze(-2))
        representation = self.path_output(torch.cat((pooled, endpoints, hops), dim=-1))
        return representation.masked_fill(~mask.any(-1)[..., None], 0), mask.any(-1)

    def forward(self, kpathsEmbedding, k_lastSecHops, kcurASPair, hcurASPair,
                hispathEmbedding, his_lastSecHops, hisASPairsEmbedding):
        original_slots = kpathsEmbedding.shape[1]

        def trim(paths, pairs, hops):
            # Remove padding work only, never a real path or hop. Using the last
            # occupied position also permits noncontiguous slots in invariance tests.
            mask = paths.ne(0)
            slots = max(1, int((mask.any(-1) * torch.arange(1, paths.shape[1] + 1,
                                                           device=paths.device)).max().item()))
            length = max(2, int((mask * torch.arange(1, paths.shape[2] + 1,
                                                    device=paths.device)).max().item()))
            return paths[:, :slots, :length], pairs[:, :slots], hops[:, :slots]

        if hcurASPair.shape != hisASPairsEmbedding.shape:
            raise ValueError('Invalid query/history endpoint shape')
        kpathsEmbedding, kcurASPair, k_lastSecHops = trim(kpathsEmbedding, kcurASPair, k_lastSecHops)
        hispathEmbedding, hisASPairsEmbedding, his_lastSecHops = trim(
            hispathEmbedding, hisASPairsEmbedding, his_lastSecHops)
        candidates, valid = self.encode(kpathsEmbedding, kcurASPair, k_lastSecHops)
        history, history_valid = self.encode(hispathEmbedding, hisASPairsEmbedding, his_lastSecHops)
        # Candidate-specific history support; absent history contributes exactly zero.
        query_pairs = self.endpoints(self.asEmbedding(kcurASPair).reshape(
            candidates.shape[0], candidates.shape[1], -1))
        query = (query_pairs * valid[..., None]).sum(1, keepdim=True) / valid.sum(1)[:, None, None].clamp_min(1)
        support = self.history_attention(candidates + query, history, history_valid)
        contextual = self.context_norm(candidates + support + query)
        contextual = self.candidate_context(contextual, valid)
        features = torch.cat((contextual, support, contextual * support,
                              torch.abs(contextual - support), query.expand_as(contextual)), dim=-1)
        logits = self.scorer(features).squeeze(-1)
        probabilities = masked_softmax(logits, valid)[..., None]
        return F.pad(probabilities, (0, 0, 0, original_slots - probabilities.shape[1]))
