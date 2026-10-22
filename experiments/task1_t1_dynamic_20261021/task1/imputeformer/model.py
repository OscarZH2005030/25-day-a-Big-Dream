"""Compact ImputeFormer adaptation; see THIRD_PARTY_NOTICE.txt for MIT attribution.

Core structure follows Tong Nie et al.: projected temporal attention, embedded
linear spatial attention and a learned [window,node,embedding] representation.
Local adaptations: two output channels, explicit availability masks and four
time covariates, compact dimensions, score-scaled supervision, no Fourier loss.
This is not a reproduction of the paper's original training recipe.
"""
from __future__ import annotations

import torch
from torch import nn

UPSTREAM = 'https://github.com/tongnie/ImputeFormer'
UPSTREAM_COMMIT = '1c3868cf499951ac43b4fc7280593fbb745f58c7'


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('Attention width must be divisible by heads')
        self.heads, self.head_dim = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query, key, value):
        def split(x):
            return x.reshape(*x.shape[:-1], self.heads, self.head_dim).transpose(-3, -2)
        q, k, v = split(self.query(query)), split(self.key(key)), split(self.value(value))
        weights = torch.softmax((q @ k.transpose(-1, -2)) / self.head_dim ** .5, dim=-1)
        result = (weights @ v).transpose(-3, -2).flatten(-2)
        return self.output(result)


class ProjectedTemporalAttention(nn.Module):
    def __init__(self, width, heads, rank, dropout):
        super().__init__()
        self.projector = nn.Parameter(torch.randn(rank, width))
        self.collect = Attention(width, heads)
        self.distribute = Attention(width, heads)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x):
        # [B,N,T,D] -> rank temporal summaries -> every timestep. Not causal.
        projector = self.projector.expand(*x.shape[:-2], -1, -1)
        summaries = self.collect(projector, x, x)
        update = self.distribute(x, projector, summaries)
        x = self.norm1(x + self.drop(update))
        return self.norm2(x + self.drop(self.feed_forward(x)))


class EmbeddedSpatialAttention(nn.Module):
    def __init__(self, width, embedding_dim, feed_forward, dropout):
        super().__init__()
        self.query_key = nn.Linear(embedding_dim, width)
        self.value = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, feed_forward), nn.ReLU(), nn.Linear(feed_forward, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, embedding):
        # [B,T,N,D]; query normalizes features, transposed key normalizes nodes.
        # Associativity avoids constructing a dense [N,N] attention matrix.
        logits = self.query_key(embedding)
        query = torch.softmax(logits, dim=-1)
        key = torch.softmax(logits.transpose(-1, -2), dim=-1)
        update = query @ (key @ self.value(x))
        x = self.norm1(x + self.drop(update))
        return self.norm2(x + self.drop(self.feed_forward(x)))


class ImputeFormer(nn.Module):
    def __init__(self, nodes, window=48, hidden=64, embedding_dim=32, heads=4,
                 layers=2, rank=8, feed_forward=128, dropout=.1):
        super().__init__()
        if min(nodes, window, hidden, embedding_dim, heads, layers, rank, feed_forward) <= 0:
            raise ValueError('Model dimensions must be positive')
        if embedding_dim >= hidden or hidden % heads or rank > window or not 0 <= dropout < 1:
            raise ValueError('Invalid embedding/head/rank/dropout configuration')
        self.nodes, self.window = nodes, window
        self.input = nn.Linear(8, hidden - embedding_dim)  # speed, flow, masks, 4 covariates
        self.embedding = nn.Parameter(torch.empty(window, nodes, embedding_dim))
        nn.init.xavier_uniform_(self.embedding)
        self.temporal = nn.ModuleList([ProjectedTemporalAttention(hidden, heads, rank, dropout) for _ in range(layers)])
        self.spatial = nn.ModuleList([EmbeddedSpatialAttention(hidden, embedding_dim, feed_forward, dropout) for _ in range(layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x, mask, covariates):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.window, self.nodes, 2):
            raise ValueError('Input must match configured [batch,window,nodes,2]')
        if mask.shape != x.shape or covariates.shape != (*x.shape[:-1], 4):
            raise ValueError('Mask/covariate shape mismatch')
        mask = mask.bool()
        clean = torch.where(mask, x, torch.zeros_like(x))
        values = self.input(torch.cat([clean, mask.to(x.dtype), covariates], dim=-1))
        values = torch.cat([values, self.embedding.expand(x.shape[0], -1, -1, -1)], dim=-1)
        for temporal, spatial in zip(self.temporal, self.spatial, strict=True):
            values = temporal(values.transpose(1, 2)).transpose(1, 2)
            values = spatial(values, self.embedding)
        # Same inference interface as GRIN, with no directional auxiliary heads.
        return self.readout(values), ()


def reconstruction_loss(prediction, target, supervision, channel_scale):
    # Same scoring scales and channel weights as the existing GRIN main loss.
    weights = prediction.new_tensor([.54, .46])
    active = supervision.bool()
    error = torch.where(active, prediction - target, torch.zeros_like(prediction)) * channel_scale
    denominator = (active * weights).sum().clamp_min(1)
    return (error.square() * weights).sum() / denominator
