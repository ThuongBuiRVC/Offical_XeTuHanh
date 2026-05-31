"""Input embedders (SPEC §3, §4).

- Nav / Ego / Action waypoints have input dim < 768, so they are lifted to 768
  through small MLPs:  Linear(in, 384) -> GELU -> LayerNorm(384) -> Linear(384, 768).
- V-JEPA latents are NEVER projected (SPEC §2.1 / §3 convention) — handled in full_model.
- (t, d) conditioning is turned into a single cond vector [B, 768] (SPEC §4) and
  injected via AdaLN-Zero (NOT concatenated to the sequence).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _lift_mlp(in_dim: int, hidden: int = 384, out_dim: int = 768) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, out_dim),
    )


class InputEmbedder(nn.Module):
    """Lift a low-dim input [B, N, in_dim] to [B, N, 768]."""

    def __init__(self, in_dim: int, dim: int = 768, hidden: int = 384):
        super().__init__()
        self.net = _lift_mlp(in_dim, hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sinusoidal_embed(x: torch.Tensor, dim: int = 768) -> torch.Tensor:
    """Scalar -> sinusoidal embedding. x: [B] -> [B, dim] (SPEC §4)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    args = x[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class CondEmbedder(nn.Module):
    """Turn (t, d) into a single cond vector [B, 768] via additive sinusoidal MLPs (SPEC §4)."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.step_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # t, d: [B] -> cond: [B, 768]
        t_emb = self.time_mlp(sinusoidal_embed(t, self.dim))
        d_emb = self.step_mlp(sinusoidal_embed(d, self.dim))
        return t_emb + d_emb
