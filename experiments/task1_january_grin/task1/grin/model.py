"""Independent PyTorch implementation of GRIN's core architecture.

Reference: https://github.com/Graph-Machine-Learning-Group/grin
Two directional GRIL modules: temporal first estimate, self-excluding spatial
second estimate, diffusion-GRU state update, followed by bidirectional MLP merge.
One recurrent layer, diffusion order 2, spatial decoder order 1, no global
attention/node output embedding. Time covariates and training loss are adapted
for TrafficFlowBench; this is not an exact reproduction of the paper's benchmark.
"""
from __future__ import annotations

import torch
from torch import nn


class DiffusionLinear(nn.Module):
    def __init__(self, inputs, outputs, order=2, include_self=True):
        super().__init__()
        self.order, self.include_self = order, include_self
        self.project = nn.Linear(inputs * (2 * order + int(include_self)), outputs)

    def forward(self, x, supports):
        parts = [x] if self.include_self else []
        for support in supports:
            propagated = x
            for _ in range(self.order):
                propagated = torch.matmul(support, propagated)
                parts.append(propagated)
        return self.project(torch.cat(parts, dim=-1))


class DiffusionGRU(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.gates = DiffusionLinear(inputs + hidden, 2 * hidden)
        self.candidate = DiffusionLinear(inputs + hidden, hidden)

    def forward(self, x, previous, supports):
        reset, keep = torch.sigmoid(self.gates(torch.cat([x, previous], -1), supports)).chunk(2, -1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * previous], -1), supports))
        return keep * previous + (1 - keep) * candidate


class SpatialDecoder(nn.Module):
    def __init__(self, channels, hidden, covariates):
        super().__init__()
        self.encode = nn.Linear(2 * channels + hidden + covariates, hidden)
        self.neighbors = DiffusionLinear(hidden, hidden, order=1, include_self=False)
        self.refine = nn.Linear(2 * hidden, hidden)
        self.activation = nn.PReLU()
        self.readout = nn.Linear(2 * hidden, channels)

    def forward(self, filled, mask, previous, covariates, supports):
        encoded = self.encode(torch.cat([filled, mask, previous, covariates], -1))
        spatial = self.neighbors(encoded, supports)
        spatial = self.activation(self.refine(torch.cat([spatial, previous], -1)))
        representation = torch.cat([spatial, previous], -1)
        return self.readout(representation), representation


class GRIL(nn.Module):
    def __init__(self, nodes, channels, hidden, covariates):
        super().__init__()
        self.initial_state = nn.Parameter(torch.randn(nodes, hidden) / hidden ** .5)
        self.temporal_readout = nn.Linear(hidden, channels)
        self.spatial = SpatialDecoder(channels, hidden, covariates)
        self.recurrent = DiffusionGRU(2 * channels + covariates, hidden)

    def forward(self, x, mask, covariates, supports):
        state = self.initial_state.unsqueeze(0).expand(x.shape[0], -1, -1)
        first, second, representations = [], [], []
        for t in range(x.shape[1]):
            observed, valid, u = x[:, t], mask[:, t], covariates[:, t]
            temporal = self.temporal_readout(state)
            filled = torch.where(valid, observed, temporal)
            spatial, representation = self.spatial(filled, valid.to(x.dtype), state, u, supports)
            filled = torch.where(valid, observed, spatial)
            state = self.recurrent(torch.cat([filled, valid.to(x.dtype), u], -1), state, supports)
            first.append(temporal); second.append(spatial); representations.append(representation)
        return (torch.stack(first, 1), torch.stack(second, 1), torch.stack(representations, 1))


class GRIN(nn.Module):
    def __init__(self, adjacency, hidden=32, channels=2, covariates=4, dropout=.1):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32).clone()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError('Adjacency must be square')
        if not torch.isfinite(adjacency).all() or (adjacency < 0).any() or adjacency.diag().any():
            raise ValueError('Adjacency must be finite, nonnegative and have no self loops')
        for name, matrix in [('incoming', adjacency), ('outgoing', adjacency.T)]:
            support = matrix / matrix.sum(1, keepdim=True).clamp_min(1e-12)
            self.register_buffer(name, support)
        self.forward_gril = GRIL(len(adjacency), channels, hidden, covariates)
        self.backward_gril = GRIL(len(adjacency), channels, hidden, covariates)
        self.merge = nn.Sequential(nn.Linear(4 * hidden + channels, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, channels))

    def forward(self, x, mask, covariates):
        # All tensors are [batch,time,node,channel]; masked numeric payload is ignored.
        mask = mask.bool()
        x = torch.where(mask, x, torch.zeros_like(x))
        supports = [self.incoming, self.outgoing]
        f1, f2, fr = self.forward_gril(x, mask, covariates, supports)
        b1, b2, br = self.backward_gril(x.flip(1), mask.flip(1), covariates.flip(1), supports)
        b1, b2, br = b1.flip(1), b2.flip(1), br.flip(1)
        imputation = self.merge(torch.cat([fr, br, mask.to(x.dtype)], -1))
        return imputation, (f1, f2, b1, b2)


def imputation_loss(prediction, auxiliary, target, supervision, channel_scale, aux_weight=.1):
    """Squared error on selected hidden labels and artificially removed observations.

    channel_scale converts normalized errors to speed/25 and per-lane flow/600;
    channel weights follow the scoring contract. Auxiliary losses train both
    directional first/second stages. Return a differentiable zero for no labels.
    """
    weights = prediction.new_tensor([.54, .46])
    mask = supervision.to(prediction.dtype)
    denominator = (mask * weights).sum().clamp_min(1)
    def loss(value):
        error = (value - target) * channel_scale
        return (error.square() * mask * weights).sum() / denominator
    main = loss(prediction)
    return main + aux_weight * sum(loss(a) for a in auxiliary) / len(auxiliary)
