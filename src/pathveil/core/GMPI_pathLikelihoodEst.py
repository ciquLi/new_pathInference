# 有batch_size后增加了一个维度。目前的数据没有对齐，没法设置batch_size大于一
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def masked_softmax(logits, mask, dim):
    """Normalize valid entries; an entirely masked group has zero weight."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    values = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = F.softmax(values, dim=dim).masked_fill(~mask, 0.0)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.unsqueeze(0)  # for batch_szie
        self.register_buffer("pe", pe)

    def forward(self, x, masked=False):
        if masked:
            if x.size(-2) > self.pe.size(-2):
                raise ValueError("Path exceeds positional-encoding capacity")
            positions = self.pe[:, :, : x.size(-2), :]
        else:
            # Preserve the execution semantics of previously saved models.
            positions = self.pe[:, : x.size(-2)]
        x = x + positions
        return self.dropout(x)


class selfAttentionNetwork(nn.Module):
    def __init__(self, nodeEmbeddingLen, hiddedNum, maxLen):
        super(selfAttentionNetwork, self).__init__()
        self.pe = PositionalEncoding(nodeEmbeddingLen, 0.1, maxLen)
        self.w1 = nn.Linear(hiddedNum, 1)
        self.w2 = nn.Linear(nodeEmbeddingLen, hiddedNum)
        self.w3 = nn.Linear(nodeEmbeddingLen, hiddedNum)
        self.w4 = nn.Linear(nodeEmbeddingLen, hiddedNum)
        self.w5 = nn.Linear(nodeEmbeddingLen, hiddedNum)
        # self.dropout = nn.Dropout(p=0.2)

    # path batch_size*k*max_pathlen*nodeEmbeddingLen
    def forward(self, paths, lastSecHops, valid_mask=None):
        paths = self.pe(paths, masked=valid_mask is not None)
        if valid_mask is not None:
            paths = paths.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            length = valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
            mean_path = paths.sum(dim=-2, keepdim=True) / length.clamp_min(1)
        else:
            mean_path = torch.mean(paths, dim=-2, keepdim=True)
        h = (
            self.w2(paths[:, :, 1:2, :])
            + self.w3(lastSecHops)
            + self.w4(mean_path)
            + self.w5(paths)
        )
        # h = self.w2(paths[:,:,1:2,:]) + self.w4(torch.mean(paths,dim=-2,keepdim=True)) + self.w5(paths)
        h = self.w1(h)
        if valid_mask is None:
            e = torch.exp(h)
            s = torch.sum(e, dim=-2, keepdim=True)
            att = e / s
        else:
            att = masked_softmax(h, valid_mask.unsqueeze(-1), dim=-2)
        rpaths = torch.sum(att * paths, dim=-2, keepdim=False)
        # rpaths = self.dropout(rpaths)
        return rpaths  # k x nodeEmbeddingLen


class pathRepresentationNetwork(nn.Module):
    def __init__(self, pathEmbeddingLength, nodeEmbeddingLen, hiddedNum, maxLen):
        super(pathRepresentationNetwork, self).__init__()
        self.att = selfAttentionNetwork(nodeEmbeddingLen, hiddedNum, maxLen)
        # feed-forward network
        self.ffn1 = nn.Linear(pathEmbeddingLength, pathEmbeddingLength)
        self.ffn2 = nn.Linear(pathEmbeddingLength, pathEmbeddingLength)

    # paths: k x maxLen x nodeEmbeddingLen
    # aspairs: k x (nodeEmbeddingLen*2)
    def forward(self, paths, aspairs, lastSecHops, valid_mask=None):
        rpaths = self.att(paths, lastSecHops, valid_mask)
        h = torch.cat([rpaths, aspairs], dim=-1)
        x = F.relu(self.ffn1(h))
        x = F.relu(self.ffn2(x))
        x = x + h
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.any(dim=-1, keepdim=True), 0.0)
        return x


class pathMergingNetwork(nn.Module):
    def __init__(self, pairEmbeddingLen, hiddedNum):
        super(pathMergingNetwork, self).__init__()
        self.w1 = nn.Linear(hiddedNum, 1)
        self.w2 = nn.Linear(pairEmbeddingLen, hiddedNum)

    # m x (pairEmbeddingLen)
    def forward(self, pairs, paths, valid_mask=None):
        h = F.relu(self.w2(pairs))
        h = self.w1(h)
        if valid_mask is None:
            e = torch.exp(h)
            s = torch.sum(e, dim=-2, keepdim=True)
            att = e / s
            att = att / torch.sum(att, dim=-2, keepdim=True)
        else:
            att = masked_softmax(h, valid_mask.unsqueeze(-1), dim=-2)
        state = torch.sum(att * paths, dim=-2, keepdim=True)
        return state


class ASEmbedding(nn.Embedding):
    def __init__(self, nodeNum, nodeEmbeddingLen, ASVectors):
        super().__init__(nodeNum, nodeEmbeddingLen, padding_idx=0)
        self.weight.data = ASVectors.clone()
        self.weight.data[0].zero_()
        self.weight.requires_grad = True


class attentionBasedScoreNetwork(nn.Module):
    def __init__(
        self, ASVectors, pathEmbeddingLength, nodeEmbeddingLen, pairEmbeddingLen, hiddedNum, maxLen
    ):
        super(attentionBasedScoreNetwork, self).__init__()
        self.input_semantics_version = 2
        self.pr = pathRepresentationNetwork(
            pathEmbeddingLength, nodeEmbeddingLen, hiddedNum, maxLen
        )
        self.pm = pathMergingNetwork(pairEmbeddingLen, hiddedNum)
        self.asEmbedding = ASEmbedding(ASVectors.shape[0], nodeEmbeddingLen, ASVectors)

    def forward(
        self,
        kpathsEmbedding,
        k_lastSecHops,
        kcurASPair,
        hcurASPair,
        hispathEmbedding,
        his_lastSecHops,
        hisASPairsEmbedding,
    ):
        masked = getattr(self, "input_semantics_version", 1) >= 2
        if masked:
            for name, paths, hops in (
                ("candidate", kpathsEmbedding, k_lastSecHops),
                ("history", hispathEmbedding, his_lastSecHops),
            ):
                if paths.dim() != 3 or paths.size(-1) < 2 or hops.shape != paths.shape[:2] + (1,):
                    raise ValueError("Invalid %s path/hop shapes; check argument order" % name)
            candidate_mask = kpathsEmbedding.ne(0)
            history_mask = hispathEmbedding.ne(0)
        else:
            candidate_mask = history_mask = None
        # # GMPI
        kpathsEmbedding = self.asEmbedding(kpathsEmbedding)
        kcurASPair = self.asEmbedding(kcurASPair)
        kcurASPair = kcurASPair.reshape(kcurASPair.size(0), kcurASPair.size(1), -1)

        hcurASPair = self.asEmbedding(hcurASPair)
        hcurASPair = hcurASPair.reshape(hcurASPair.size(0), hcurASPair.size(1), -1)
        hispathEmbedding = self.asEmbedding(hispathEmbedding)
        hisASPairsEmbedding = self.asEmbedding(hisASPairsEmbedding)
        hisASPairsEmbedding = hisASPairsEmbedding.reshape(
            hisASPairsEmbedding.size(0), hisASPairsEmbedding.size(1), -1
        )

        # path representation of GMPI
        kPathsRepresenation = self.pr(
            kpathsEmbedding, kcurASPair, self.asEmbedding(k_lastSecHops), candidate_mask
        )

        hisPathsRepresenation = self.pr(
            hispathEmbedding, hisASPairsEmbedding, self.asEmbedding(his_lastSecHops), history_mask
        )

        # observed positive paths merging of GMPI
        pairs = torch.cat((hcurASPair, hisASPairsEmbedding), axis=-1)
        history_slots = history_mask.any(dim=-1) if masked else None
        stateRepresentation = self.pm(pairs, hisPathsRepresenation, history_slots)
        if masked:
            # With no observed paths, condition on the actual query endpoints.
            # This is a query feature, not a fabricated historical route.
            path_dim = kPathsRepresenation.size(-1) - kcurASPair.size(-1)
            query_state = torch.cat(
                (kcurASPair.new_zeros(kcurASPair.size(0), 1, path_dim), kcurASPair[:, :1, :]),
                dim=-1,
            )
            stateRepresentation = torch.where(
                history_slots.any(dim=-1).view(-1, 1, 1), stateRepresentation, query_state
            )
        w = torch.transpose(stateRepresentation, -2, -1)
        pathScore = torch.bmm(kPathsRepresenation, w)
        if masked:
            pathScore = masked_softmax(pathScore, candidate_mask.any(dim=-1).unsqueeze(-1), dim=-2)
        else:
            pathScore = F.softmax(pathScore, dim=-2)
        return pathScore
