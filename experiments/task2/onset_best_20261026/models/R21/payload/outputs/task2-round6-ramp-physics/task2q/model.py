"""Small-check Task 2 network; no fitting or data reads happen on import.

This is a new, factorized temporal Transformer with a sparse, delayed, directed
graph branch.  It is inspired by propagation-delay modelling, not an
implementation or reproduction of PDFormer.  All temporal attention is confined
to the supplied history (the caller must guarantee every frame is at or before T).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _delayed_history(states: Tensor, nodes: Tensor, lag: Tensor) -> Tensor:
    """Interpolate source histories at t-lag, repeating first available frame.

    states: B,T,N,D; nodes/lag: E.  Fractional lag preserves short-link delays
    which rounding to five-minute integer frames would erase.  Missing history
    before the first frame is held at that frame, never read from the future.
    """
    steps = states.shape[1]
    positions = torch.arange(steps, device=states.device, dtype=states.dtype)[:, None]
    positions = (positions - lag[None]).clamp(0, steps - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp_max(steps - 1)
    alpha = (positions - lower).unsqueeze(0).unsqueeze(-1)
    return states[:, lower, nodes[None], :] * (1 - alpha) + states[:, upper, nodes[None], :] * alpha


class DelayedDirectedGraph(nn.Module):
    """Sparse neighbor attention with distinct flow and queue-wave directions.

    edges[0] -> edges[1] means upstream -> downstream.  Forward messages use a
    60 km/h prior; reverse messages use a 15 km/h queue-wave prior.  Distances
    are approximated by half the length of each adjacent link.  These are
    explicit, testable priors, not calibrated physical parameters.  The branch
    yields exactly zero for nodes without incoming messages of either type.
    """

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.queries = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.keys = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.values = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.outputs = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.type_gate = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(dropout)

    def _direction(self, states: Tensor, source: Tensor, destination: Tensor,
                   lag: Tensor, kind: int) -> Tensor:
        batch, steps, nodes, width = states.shape
        source_states = _delayed_history(states, source, lag)
        q = self.queries[kind](states[:, :, destination]).reshape(batch, steps, -1, self.heads, self.head_width)
        k = self.keys[kind](source_states).reshape(batch, steps, -1, self.heads, self.head_width)
        values = self.values[kind](source_states).reshape_as(k)
        score = (q * k).sum(-1) / self.head_width ** 0.5
        index = destination[None, None, :, None].expand_as(score)
        maxima = score.new_full((batch, steps, nodes, self.heads), -torch.inf)
        maxima.scatter_reduce_(2, index, score, reduce="amax", include_self=True)
        # Detaching the softmax stabilizer avoids gradients through equal maxima.
        exponent = (score - maxima.gather(2, index).detach()).exp()
        denominator = score.new_zeros((batch, steps, nodes, self.heads))
        denominator.scatter_add_(2, index, exponent)
        attention = exponent / denominator.gather(2, index).clamp_min(1e-8)
        messages = values * self.dropout(attention).unsqueeze(-1)
        output = states.new_zeros((batch, steps, nodes, self.heads, self.head_width))
        output.scatter_add_(2, index.unsqueeze(-1).expand_as(messages), messages)
        return self.outputs[kind](output.reshape(batch, steps, nodes, width))

    def forward(self, states: Tensor, edges: Tensor, length_km: Tensor | None) -> Tensor:
        if edges.numel() == 0:
            return torch.zeros_like(states)
        source, destination = edges
        if length_km is None:
            distance = states.new_zeros(source.shape)
        else:
            distance = (length_km[source] + length_km[destination]) * 0.5
        flow_lag = (distance / (60.0 * 5.0 / 60.0)).clamp(0, states.shape[1] - 1)
        wave_lag = (distance / (15.0 * 5.0 / 60.0)).clamp(0, states.shape[1] - 1)
        flow = self._direction(states, source, destination, flow_lag, 0)
        wave = self._direction(states, destination, source, wave_lag, 1)
        gates = self.type_gate.sigmoid()
        return gates[0] * flow + gates[1] * wave


class _SpatioTemporalBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, use_graph: bool) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.graph_norm = nn.LayerNorm(width)
        self.graph = DelayedDirectedGraph(width, heads, dropout) if use_graph else None
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor, edges: Tensor, length_km: Tensor | None) -> Tensor:
        batch, steps, nodes, width = states.shape
        history = self.temporal_norm(states).permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        update, _ = self.temporal(history, history, history, need_weights=False)
        states = states + self.dropout(update.reshape(batch, nodes, steps, width).permute(0, 2, 1, 3))
        if self.graph is not None:
            states = states + self.dropout(self.graph(self.graph_norm(states), edges, length_km))
        return states + self.dropout(self.ff(self.ff_norm(states)))


class QueueForecaster(nn.Module):
    """Shared onset/ongoing forecaster returning six future logits per link.

    x: B,13,N,F, or fewer valid history frames (up to 13); static: N,S or B,N,S;
    edges: long[2,E] upstream->downstream node indices; condition: long[B],
    0=onset and 1=ongoing; length_km: optional nonnegative physical link lengths.
    Missing observations must already be filled with finite causal values and
    represented by explicit input mask features.  No future eligibility enters
    this network.  use_graph=False is the time-only architecture ablation.
    """

    def __init__(self, input_dim: int, static_dim: int = 6, width: int = 128,
                 layers: int = 4, heads: int = 4, dropout: float = 0.1,
                 horizon: int = 6, use_graph: bool = True) -> None:
        super().__init__()
        if min(input_dim, static_dim, width, layers, heads, horizon) < 1 or width % heads:
            raise ValueError("Dimensions must be positive and width must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.input_dim, self.static_dim = input_dim, static_dim
        self.width, self.horizon, self.use_graph = width, horizon, use_graph
        self.input_projection = nn.Linear(input_dim, width)
        self.static_projection = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.Linear(width, width))
        self.history_position = nn.Parameter(torch.empty(13, width))
        self.future_queries = nn.Parameter(torch.empty(horizon, width))
        self.condition_embedding = nn.Embedding(2, width)
        self.blocks = nn.ModuleList(_SpatioTemporalBlock(width, heads, dropout, use_graph) for _ in range(layers))
        self.history_norm = nn.LayerNorm(width)
        self.decoder = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.decoder_norm = nn.LayerNorm(width)
        self.queue_heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(2))
        self.speed_head = nn.Linear(width, 1)
        nn.init.normal_(self.history_position, std=0.02)
        nn.init.normal_(self.future_queries, std=0.02)

    def forward(self, x: Tensor, static: Tensor, edges: Tensor, condition: Tensor,
                length_km: Tensor | None = None) -> dict[str, Tensor]:
        if x.ndim != 4 or x.shape[-1] != self.input_dim or not 1 <= x.shape[1] <= 13:
            raise ValueError("x must be B,T,N,input_dim with 1 <= T <= 13")
        batch, steps, nodes, _ = x.shape
        if static.shape not in ((nodes, self.static_dim), (batch, nodes, self.static_dim)):
            raise ValueError("static must be N,static_dim or B,N,static_dim")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
            raise ValueError("edges must be a long tensor with shape 2,E")
        if condition.shape != (batch,) or condition.dtype != torch.long or ((condition < 0) | (condition > 1)).any():
            raise ValueError("condition must be long[B], 0=onset or 1=ongoing")
        if edges.numel() and ((edges < 0).any() or (edges >= nodes).any()):
            raise ValueError("edges contain out-of-range node indices")
        if length_km is not None:
            if length_km.shape != (nodes,) or not torch.isfinite(length_km).all() or (length_km < 0).any():
                raise ValueError("length_km must contain N finite, nonnegative physical lengths")
            length_km = length_km.to(device=x.device, dtype=x.dtype)
        static_state = self.static_projection(static)
        if static.ndim == 2:
            static_state = static_state.unsqueeze(0)
        states = self.input_projection(x) + static_state[:, None] + self.history_position[-steps:][None, :, None]
        for block in self.blocks:
            states = block(states, edges, length_km)
        history = self.history_norm(states).permute(0, 2, 1, 3).reshape(batch * nodes, steps, self.width)
        queries = self.future_queries[None, None] + states[:, -1, :, None]
        queries = queries + self.condition_embedding(condition)[:, None, None]
        queries = queries.reshape(batch * nodes, self.horizon, self.width)
        decoded, _ = self.decoder(queries, history, history, need_weights=False)
        decoded = self.decoder_norm(queries + decoded).reshape(batch, nodes, self.horizon, self.width)
        all_logits = torch.cat([head(decoded) for head in self.queue_heads], dim=-1)
        choice = condition[:, None, None, None].expand(batch, nodes, self.horizon, 1)
        logits = all_logits.gather(-1, choice).squeeze(-1).transpose(1, 2)
        speed = self.speed_head(decoded).squeeze(-1).transpose(1, 2)
        return {"logits": logits, "speed": speed}


def masked_queue_loss(logits: Tensor, labels: Tensor, label_mask: Tensor,
                      confidence: Tensor | None = None, iou_weight: float = 0.3) -> dict[str, Tensor]:
    """Confidence-weighted BCE + per-window soft Jaccard, never micro pooled.

    Each nonempty supervised window contributes equally regardless of node count
    or queue area.  Unknown/NaN labels and zero-confidence cells are excluded.
    Windows with no supervision are omitted (not credited as perfect IoU); an
    entirely empty batch returns differentiable zero.  Reported official/proxy
    hard IoU must be computed separately on the original hard-label mask.
    """
    if logits.ndim != 3 or labels.shape != logits.shape or label_mask.shape != logits.shape:
        raise ValueError("logits, labels and label_mask must have identical B,H,N shapes")
    if not 0 <= iou_weight <= 1:
        raise ValueError("iou_weight must be in [0,1]")
    valid = label_mask.bool() & torch.isfinite(labels)
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels)).to(logits.dtype)
    if ((safe_labels < 0) | (safe_labels > 1)).any():
        raise ValueError("valid labels must be probabilities in [0,1]")
    weights = valid.to(logits.dtype)
    if confidence is not None:
        if confidence.shape != logits.shape:
            raise ValueError("confidence must have the same shape as logits")
        weights = weights * torch.nan_to_num(confidence.to(logits.dtype), nan=0, posinf=0, neginf=0).clamp(0, 1)
    count = weights.sum((1, 2))
    active = count > 0
    bce_per_window = (F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none") * weights).sum((1, 2)) / count.clamp_min(1e-8)
    probability = logits.sigmoid()
    intersection = (weights * probability * safe_labels).sum((1, 2))
    union = (weights * (probability + safe_labels - probability * safe_labels)).sum((1, 2))
    # For valid all-negative windows BCE handles false positives; an empty
    # supervision window is still excluded below rather than called correct.
    soft_iou = (intersection + 1e-6) / (union + 1e-6)
    divisor = active.sum().clamp_min(1)
    bce = (bce_per_window * active).sum() / divisor
    soft_iou_loss = ((1 - soft_iou) * active).sum() / divisor
    return {"total": bce + iou_weight * soft_iou_loss, "bce": bce,
            "soft_iou_loss": soft_iou_loss, "valid_windows": active.sum()}
