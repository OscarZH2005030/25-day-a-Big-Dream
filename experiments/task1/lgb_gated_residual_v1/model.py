"""Original graph/time attention with LightGBM skip and independent channel gates."""
import math
import torch
from torch import nn


class GraphTimeBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.d = heads, width//heads
        self.tnorm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.snorm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.direction_bias = nn.Embedding(3, heads)
        self.project = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4*width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4*width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, g):
        b, t, n, w = x.shape
        z = self.tnorm(x).permute(0, 2, 1, 3).reshape(b*n, t, w)
        z = self.temporal(z, z, z, need_weights=False)[0]
        x = x+self.drop(z.reshape(b, n, t, w).permute(0, 2, 1, 3))
        q, k, v = self.qkv(self.snorm(x)).chunk(3, dim=-1)
        q = q.reshape(b, t, n, self.heads, self.d)
        k = k.reshape(b, t, n, self.heads, self.d)[:, :, g['neighbors']]
        v = v.reshape(b, t, n, self.heads, self.d)[:, :, g['neighbors']]
        logits = (q[:, :, :, None]*k).sum(-1)/math.sqrt(self.d)
        logits = logits+self.direction_bias(g['direction'])[None, None]
        logits = logits.masked_fill(~g['neighbor_valid'][None, None, :, :, None], -1e4)
        attention = torch.softmax(logits, dim=3)
        message = (attention[..., None]*v).sum(dim=3).reshape(b, t, n, w)
        x = x+self.drop(self.project(message))
        return x+self.drop(self.ff(x))


class GatedResidual(nn.Module):
    def __init__(self, inputs=106, width=64, heads=4, layers=3, window=24, dropout=.1):
        super().__init__()
        self.input = nn.Linear(inputs, width)
        self.position = nn.Parameter(torch.randn(1, window, 1, width)*.01)
        self.blocks = nn.ModuleList([GraphTimeBlock(width, heads, dropout) for _ in range(layers)])
        self.residual_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width//2), nn.GELU(), nn.Linear(width//2, 1)) for _ in range(2)])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(inputs+width, width//2), nn.GELU(), nn.Linear(width//2, 1)) for _ in range(2)])
        for head, gate in zip(self.residual_heads, self.gates):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
            nn.init.zeros_(gate[-1].weight); nn.init.constant_(gate[-1].bias, -2.)

    def forward(self, batch, graph):
        raw = batch['features']
        x = self.input(raw)+self.position[:, :raw.shape[1]]
        for block in self.blocks: x = block(x, graph)
        delta = torch.cat([head(x) for head in self.residual_heads], dim=-1)
        gate_input = torch.cat([raw, x], dim=-1)
        gate = torch.sigmoid(torch.cat([g(gate_input) for g in self.gates], dim=-1))
        # Global graph isolation, not accidental isolation caused by a training patch.
        enabled = (~graph['isolated'])[None, None, :, None] & ~batch['visible'].bool()
        correction = gate*delta*enabled
        prediction = (batch['base']+correction).clamp_min(0.)
        return dict(prediction=prediction, correction=prediction-batch['base'], gate=gate,
                    raw_correction=correction)


def objective(output, batch, graph, shrink):
    mask = batch['target'].bool()
    if not mask.any(): raise ValueError('No template supervision in batch')
    # residual labels were generated out of fold, already in speed/25, flow/lane/600.
    error = output['correction'][mask]-batch['residual'][mask]
    rmse = torch.sqrt((error**2).mean(dim=0)+1e-8)
    reconstruction = .54*rmse[0]+.46*rmse[1]
    eligible = mask & ~graph['isolated'][None, None, :]
    regularizer = (output['correction'][eligible]**2).mean() if eligible.any() else output['correction'].sum()*0
    loss = reconstruction+shrink*regularizer
    return loss, dict(reconstruction=float(reconstruction.detach()), shrink_penalty=float(regularizer.detach()),
        speed_rmse_kmh=float(rmse[0].detach())*25, flow_per_lane_rmse=float(rmse[1].detach())*600,
        gate_speed=float(output['gate'][mask][:, 0].mean().detach()),
        gate_flow=float(output['gate'][mask][:, 1].mean().detach()))
