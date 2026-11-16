"""Directed-fragment late interaction over the unchanged seven GMPI inputs.

The two architecture variants share identical initial tensors. One replaces the
GMPI scorer; the other adds its learned fragment score to the original GMPI
logits. No route generation, additional evidence or pretraining lives here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pathInference.core.GMPI_pathLikelihoodEst import attentionBasedScoreNetwork, masked_softmax


def trim_paths(paths, pairs, hops):
    occupied = paths.ne(0)
    slots = max(1, int((occupied.any(-1) * torch.arange(1, paths.shape[1] + 1,
                                                     device=paths.device)).max().item()))
    length = max(2, int((occupied * torch.arange(1, paths.shape[2] + 1,
                                               device=paths.device)).max().item()))
    return paths[:, :slots, :length], pairs[:, :slots], hops[:, :slots]


def first_query_pair(pairs, valid):
    index = valid.long().argmax(-1)
    return pairs[torch.arange(len(pairs), device=pairs.device), index]


def local_convolution(values, convolution):
    """Exact Conv1d window formula using GEMM, avoiding old CuDNN tiny-path kernels."""
    length = values.shape[-2]
    dilation = convolution.dilation[0]
    padding = convolution.padding[0]
    padded = F.pad(values, (0, 0, padding, padding))
    window = torch.cat([padded[..., offset * dilation:offset * dilation + length, :]
                        for offset in range(convolution.kernel_size[0])], -1)
    weight = convolution.weight.permute(0, 2, 1).reshape(convolution.out_channels, -1)
    return F.linear(window, weight, convolution.bias)


def masked_statistics(values, mask):
    count = mask.sum(-1).clamp_min(1)
    any_valid = mask.any(-1)
    mean = values.masked_fill(~mask, 0).sum(-1) / count
    low = values.masked_fill(~mask, 1e4).min(-1)[0].masked_fill(~any_valid, 0)
    high = values.masked_fill(~mask, -1e4).max(-1)[0].masked_fill(~any_valid, 0)
    first = values[..., 0].masked_fill(~any_valid, 0)
    # torch 1.8 CUDA gather backward uses nondeterministic scatter_add. An exact
    # one-hot reduction preserves the same selected value with deterministic grad.
    last_mask = torch.arange(values.shape[-1], device=values.device).eq((count - 1)[..., None])
    last = (values * last_mask).sum(-1).masked_fill(~any_valid, 0)
    return [mean, low, high, first, last]


def directed_hashes(paths, vocabulary_size):
    """Collision-free bigram/trigram codes; zero denotes padding, not a route."""
    u, v = paths[..., :-1], paths[..., 1:]
    valid = u.ne(0) & v.ne(0)
    edges = (u * vocabulary_size + v).masked_fill(~valid, 0)
    if paths.shape[-1] > 2:
        a, b, c = paths[..., :-2], paths[..., 1:-1], paths[..., 2:]
        triples = ((a * vocabulary_size + b) * vocabulary_size + c).masked_fill(
            ~(a.ne(0) & b.ne(0) & c.ne(0)), 0)
        triples = F.pad(triples, (1, 0))
    else:
        triples = torch.zeros_like(edges)
    return edges, triples


def history_contains(query_codes, history_codes):
    boundaries = history_codes.flatten(1).sort(-1)[0].contiguous()
    values = query_codes.flatten(1).contiguous()
    positions = torch.searchsorted(boundaries, values).clamp(max=boundaries.shape[-1] - 1)
    matched = boundaries.gather(1, positions).eq(values) & values.ne(0)
    return matched.reshape_as(query_codes)


def longest_history_suffix(candidates, history):
    """Exact ordered suffix support, with bounded B*K*H rather than B*K*H*L storage."""
    def reversed_nodes(paths):
        count = paths.ne(0).sum(-1)
        positions = torch.arange(paths.shape[-1], device=paths.device)
        valid = positions < count[..., None]
        indices = (count[..., None] - 1 - positions).clamp(min=0)
        return paths.gather(-1, indices), valid
    c, cmask = reversed_nodes(candidates)
    h, hmask = reversed_nodes(history)
    alive = cmask[..., :1] & hmask[:, None, :, 0]
    count = candidates.new_zeros(candidates.shape[:2] + (history.shape[1],), dtype=torch.float32)
    for offset in range(min(candidates.shape[-1], history.shape[-1])):
        alive = alive & cmask[..., offset, None] & hmask[:, None, :, offset]
        alive = alive & c[..., offset, None].eq(h[:, None, :, offset])
        count = count + alive.to(count.dtype)
    return count.max(-1)[0]


class DirectedFragmentAlignment(nn.Module):
    def __init__(self, embedding_size, width=48, hidden=64, max_length=20, candidate_chunk=8):
        super().__init__()
        self.width, self.max_length, self.candidate_chunk = width, max_length, candidate_chunk
        self.source = nn.Linear(embedding_size, width)
        self.destination = nn.Linear(embedding_size, width)
        self.edge = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.position, self.reverse_position = nn.Embedding(max_length, width), nn.Embedding(max_length, width)
        nn.init.normal_(self.position.weight, std=.02)
        nn.init.normal_(self.reverse_position.weight, std=.02)
        self.convolutions = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1),
                                          nn.Conv1d(width, width, 3, dilation=2, padding=2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width), nn.LayerNorm(width)])
        self.query = nn.Sequential(nn.Linear(2 * width, width), nn.GELU())
        self.reliability = nn.Sequential(nn.Linear(width + 6, width), nn.GELU(), nn.Linear(width, 1))
        self.gate = nn.Sequential(nn.Linear(3 * width + 2, width), nn.GELU(), nn.Linear(width, 1))
        self.output = nn.Sequential(nn.Linear(4 * width + 20, hidden), nn.GELU(),
                                    nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        # Residual variant begins as the paired cold GMPI; standalone begins uniform.
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def encode_edges(self, paths, embedding):
        mask = paths[..., :-1].ne(0) & paths[..., 1:].ne(0)
        u, v = torch.tanh(self.source(embedding(paths[..., :-1]))), torch.tanh(self.destination(embedding(paths[..., 1:])))
        values = self.edge(torch.cat((u, v, u - v, u * v), -1))
        positions = torch.arange(values.shape[-2], device=paths.device)
        reverse = (mask.sum(-1, keepdim=True) - 1 - positions).clamp(min=0)
        values = values + self.position(positions) + self.reverse_position(reverse)
        values = values.masked_fill(~mask[..., None], 0)
        for conv, norm in zip(self.convolutions, self.norms):
            update = local_convolution(values, conv)
            values = norm(values + F.gelu(update)).masked_fill(~mask[..., None], 0)
        return values, mask

    def max_similarity(self, candidates, history, history_mask):
        batch, slots, length, width = candidates.shape
        memory = F.normalize(history, dim=-1).reshape(batch, -1, width).transpose(1, 2)
        valid_memory = history_mask.flatten(1)
        available = valid_memory.any(-1)
        pieces = []
        # No real path/token is removed; chunking bounds temporary similarity storage.
        for offset in range(0, slots, self.candidate_chunk):
            query = F.normalize(candidates[:, offset:offset + self.candidate_chunk], dim=-1)
            size = query.shape[1]
            scores = torch.bmm(query.reshape(batch, size * length, width), memory)
            best = scores.masked_fill(~valid_memory[:, None, :], -1e4).max(-1)[0]
            pieces.append(best.reshape(batch, size, length) * available[:, None, None].to(best.dtype))
        return torch.cat(pieces, 1)

    def features(self, paths, pairs, hops, history, embedding):
        if not 2 <= paths.shape[-1] <= self.max_length or not 2 <= history.shape[-1] <= self.max_length:
            raise ValueError('Path exceeds frozen capacity; never truncate a real route')
        candidate, mask = self.encode_edges(paths, embedding)
        observed, observed_mask = self.encode_edges(history, embedding)
        similarity = self.max_similarity(candidate, observed, observed_mask).masked_fill(~mask, 0)
        edge_codes, triple_codes = directed_hashes(paths, embedding.num_embeddings)
        hist_edges, hist_triples = directed_hashes(history, embedding.num_embeddings)
        edge_match = history_contains(edge_codes, hist_edges).to(candidate.dtype)
        triple_match = history_contains(triple_codes, hist_triples).to(candidate.dtype)
        src = torch.tanh(self.source(embedding(pairs[..., 0])))
        dst = torch.tanh(self.destination(embedding(pairs[..., 1])))
        penultimate = torch.tanh(self.source(embedding(hops.squeeze(-1))))
        query = self.query(torch.cat((src, dst), -1))[:, :, None, :]
        position = torch.arange(mask.shape[-1], device=paths.device, dtype=candidate.dtype)
        length = mask.sum(-1).clamp_min(1)
        forward = position.expand_as(mask) / length[..., None]
        reverse = (length[..., None] - 1 - position).clamp(min=0) / length[..., None]
        agreement = (F.normalize(candidate, dim=-1) * F.normalize(query, dim=-1)).sum(-1)
        reliability = self.reliability(torch.cat((candidate, similarity[..., None], edge_match[..., None],
            triple_match[..., None], agreement[..., None], forward[..., None], reverse[..., None]), -1)).squeeze(-1)
        gates = self.gate(torch.cat((candidate, candidate * query, (candidate - query).abs(),
                                    forward[..., None], reverse[..., None]), -1)).squeeze(-1)
        attention = masked_softmax(gates, mask, -1)
        statistics = masked_statistics(reliability, mask) + [(attention * reliability).sum(-1)]
        statistics += masked_statistics(similarity, mask) + [(attention * similarity).sum(-1)]
        statistics += [(edge_match * mask).sum(-1) / length,
                       triple_match.sum(-1) / (length - 1).clamp_min(1),
                       edge_match[..., 0], edge_match.gather(-1, (length - 1)[..., None]).squeeze(-1)]
        suffix = longest_history_suffix(paths, history)
        statistics += [length.to(candidate.dtype) / (self.max_length - 1), suffix / (length + 1),
                       suffix / self.max_length, observed_mask.flatten(1).any(-1)[:, None].expand_as(length).to(candidate.dtype)]
        pooled = (attention[..., None] * candidate).sum(-2)
        features = torch.cat((src, dst, penultimate, pooled, torch.stack(statistics, -1)), -1)
        return features.masked_fill(~mask.any(-1)[..., None], 0)

    def forward(self, paths, pairs, hops, history, embedding):
        return self.output(self.features(paths, pairs, hops, history, embedding))


class EdgeAlignmentConfidence(attentionBasedScoreNetwork):
    """Standalone fragment scorer or GMPI plus learned late-interaction residual."""
    def __init__(self, vectors, hidden_size=512, use_base=True, width=48, hidden=64,
                 max_length=20, candidate_chunk=8):
        dimension = vectors.shape[-1]
        super().__init__(vectors, dimension * 3, dimension, dimension * 4, hidden_size, max_length)
        self.use_base = use_base
        # Preserve the CPU random state after the paired original GMPI constructor.
        state = torch.get_rng_state()
        self.alignment = DirectedFragmentAlignment(dimension, width, hidden, max_length, candidate_chunk)
        torch.set_rng_state(state)
        if not use_base:
            for module in (self.pr, self.pm):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def base_logits(self, paths, hops, pairs, query_history_pairs, history, hist_hops, hist_pairs):
        candidate_mask, history_mask = paths.ne(0), history.ne(0)
        query_pairs = self.asEmbedding(pairs).flatten(-2)
        candidates = self.pr(self.asEmbedding(paths), query_pairs, self.asEmbedding(hops), candidate_mask)
        observed = self.pr(self.asEmbedding(history), self.asEmbedding(hist_pairs).flatten(-2),
                           self.asEmbedding(hist_hops), history_mask)
        context = self.pm(torch.cat((self.asEmbedding(query_history_pairs).flatten(-2),
                                    self.asEmbedding(hist_pairs).flatten(-2)), -1), observed, history_mask.any(-1))
        actual_query = self.asEmbedding(first_query_pair(pairs, candidate_mask.any(-1))).flatten(-2)
        fallback = torch.cat((actual_query.new_zeros(len(paths), 1, candidates.shape[-1] - query_pairs.shape[-1]),
                              actual_query[:, None]), -1)
        context = torch.where(history_mask.any(-1).any(-1)[:, None, None], context, fallback)
        return torch.bmm(candidates, context.transpose(-1, -2))

    def forward(self, kpathsEmbedding, k_lastSecHops, kcurASPair, hcurASPair,
                hispathEmbedding, his_lastSecHops, hisASPairsEmbedding):
        if hcurASPair.shape != hisASPairsEmbedding.shape:
            raise ValueError('History/query endpoint shapes differ')
        valid = kpathsEmbedding.ne(0).any(-1)
        base = self.base_logits(kpathsEmbedding, k_lastSecHops, kcurASPair, hcurASPair,
                                hispathEmbedding, his_lastSecHops, hisASPairsEmbedding) if self.use_base else None
        paths, pairs, hops = trim_paths(kpathsEmbedding, kcurASPair, k_lastSecHops)
        history, _, _ = trim_paths(hispathEmbedding, hisASPairsEmbedding, his_lastSecHops)
        correction = self.alignment(paths, pairs, hops, history, self.asEmbedding)
        correction = F.pad(correction, (0, 0, 0, kpathsEmbedding.shape[1] - paths.shape[1]))
        logits = correction if base is None else base + correction
        return masked_softmax(logits, valid[..., None], -2)
