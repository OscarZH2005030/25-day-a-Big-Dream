from __future__ import annotations

import numpy as np
import torch

from ..wave_neural.model import WaveNeural

DILATIONS = [1, 2, 4, 8, 16, 32]


class ContextNeural(WaveNeural):
    def __init__(self, adjacency):
        super().__init__(adjacency, hidden=32, layers=len(DILATIONS))


def supervised_loss(values, target, supervise):
    losses = []
    for ch, scale, weight in [(0, 25., .54), (1, 600., .46)]:
        selected = supervise[..., ch]
        difference = (values[..., ch][selected]-target[..., ch][selected])/scale
        losses.append(weight*difference.square().mean() if selected.any() else values.sum()*0)
    return sum(losses)


def receptive_field_check(adjacency, device='cpu', window=48):
    """Separate nonzero-head probe: never changes the training model or RNG state."""
    device = torch.device(device)
    devices = [device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(7621)
        model = ContextNeural(adjacency).to(device).eval()
        with torch.no_grad():
            model.head.weight.normal_(0, .05)
        n = len(adjacency)
        features = torch.randn(1, window, n, 31, device=device, requires_grad=True)
        prior = features.new_tensor([80., 600.]).expand(1, window, n, 2)
        outputs = model(features, prior, torch.zeros_like(prior), torch.zeros_like(prior, dtype=torch.bool))
        rows = []
        for t in [0, window//2, window-1]:
            gradient = torch.autograd.grad(outputs[0, t, n//2, 0], features, retain_graph=True)[0]
            support = gradient.abs().sum((0, 2, 3)).detach().cpu().numpy()
            active = np.flatnonzero(support>0).tolist()
            if active != list(range(window)) or not np.isfinite(support).all():
                raise ValueError(f'Temporal gradient support incomplete at output {t}: {active}')
            rows.append(dict(output_slot=t, first_input_slot=active[0], last_input_slot=active[-1],
                             input_slots=len(active), minimum_gradient_l1=float(support.min())))
    return dict(passed=True, kernel=3, dilations=DILATIONS, theoretical_slots=127,
                actual_window_slots=window, nodes=len(adjacency), probe='independent model, nonzero readout head',
                limitation='48 actual input slots; no raw features outside each window; padding is not extra data', outputs=rows)
