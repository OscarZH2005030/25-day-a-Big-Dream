"""Architecture compatible with the validated Task 1 checkpoint."""
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(64, 64, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x) * .1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(15, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[Block() for _ in range(5)])
        self.head = nn.Conv2d(64, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("base_scale", torch.tensor([4., 2000/600]).reshape(1, 2, 1, 1))

    def forward(self, x):
        base = x[:, :2].float() * self.base_scale
        residual = self.head(self.blocks(torch.nn.functional.silu(self.encoder(x)))).float()
        return (base + residual).clamp_min(0)
