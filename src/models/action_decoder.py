"""Action Decoder (SPEC §8).

Small MLP head mapping a clean action latent [B, 8, 768] -> waypoints [B, 8, 2].
Trained JOINTLY with the DiT (NOT a separate frozen phase). At training time its
input is the one-step clean estimate a_hat = a_t + (1 - t) * v_a_pred (SPEC §8.2),
so the waypoint loss gradient flows back through v_a_pred into the DiT.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionDecoder(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 256, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, a_latent: torch.Tensor) -> torch.Tensor:
        # [B, 8, 768] -> [B, 8, 2]
        return self.net(a_latent)
