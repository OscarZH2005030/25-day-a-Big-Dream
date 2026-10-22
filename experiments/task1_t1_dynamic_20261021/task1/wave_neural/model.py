"""Small independent bidirectional graph/temporal residual imputer."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUTS = 31


class GraphTemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.graph = nn.Linear(hidden * 3, hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, up, down):
        spatial = self.graph(torch.cat([x, torch.matmul(up, x), torch.matmul(down, x)], -1))
        b, t, n, h = spatial.shape
        temporal = self.temporal(F.silu(spatial).permute(0, 2, 3, 1).reshape(b*n, h, t))
        temporal = temporal.reshape(b, n, h, t).permute(0, 3, 1, 2)
        return self.norm(x + F.silu(temporal))


class WaveNeural(nn.Module):
    def __init__(self, adjacency, hidden=32, layers=2):
        super().__init__()
        a = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not torch.isfinite(a).all() or (a < 0).any():
            raise ValueError('Invalid graph adjacency')
        for name, matrix in [('up', a), ('down', a.T)]:
            self.register_buffer(name, matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-8))
        self.encode = nn.Linear(INPUTS, hidden)
        self.blocks = nn.ModuleList([GraphTemporalBlock(hidden, 2**i) for i in range(layers)])
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, prior, observed, mask):
        if features.shape[-1] != INPUTS or features.shape[:-1] != prior.shape[:-1] or prior.shape != observed.shape or mask.shape != prior.shape:
            raise ValueError('Input tensor shape differs')
        x = F.silu(self.encode(features))
        for block in self.blocks:
            x = block(x, self.up, self.down)
        scale = prior.new_tensor([25., 600.])
        positive = (prior / scale).clamp_min(1e-4)
        inverse = positive + torch.log(-torch.expm1(-positive))
        prediction = F.softplus(inverse + self.head(x)) * scale
        return torch.where(mask, observed, prediction)
