"""DiT backbone (SPEC §6).

8 AdaLN-Zero blocks, each: RMSNorm -> modulate -> Self-Attn,
                          RMSNorm -> Cross-Attn (NO modulation),
                          RMSNorm -> modulate -> FFN.
Then a zero-init FinalLayer. (t, d) come in as a single cond vector [B, 768].

Attention uses 12 heads (head_dim 64), qk-norm (RMSNorm per head), and 1D RoPE
along the sequence axis for both Q and KV.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# ----------------------------- RoPE 1D ----------------------------- #
def build_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    """Return (cos, sin) each [seq_len, head_dim] for rotary embedding."""
    half = head_dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(base, device=device)) * torch.arange(half, device=device) / half
    )
    pos = torch.arange(seq_len, device=device).float()
    ang = torch.outer(pos, freqs)                       # [L, half]
    emb = torch.cat([ang, ang], dim=-1)                 # [L, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, L, Dh]; cos/sin: [L, Dh]."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ----------------------------- Attention ----------------------------- #
class Attention(nn.Module):
    """Multi-head attention supporting self- and cross-attention with qk-norm + RoPE."""

    def __init__(self, dim: int, num_heads: int = 12, qk_norm: bool = True, rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,Dh]

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        ctx = x if context is None else context
        q = self._split(self.q_proj(x))      # [B,H,Lq,Dh]
        k = self._split(self.k_proj(ctx))    # [B,H,Lk,Dh]
        v = self._split(self.v_proj(ctx))

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rope:
            Lq, Lk = q.shape[2], k.shape[2]
            cos_q, sin_q = build_rope_cache(Lq, self.head_dim, q.device, q.dtype)
            q = apply_rope(q, cos_q, sin_q)
            if context is None:
                cos_k, sin_k = cos_q, sin_q
            else:
                cos_k, sin_k = build_rope_cache(Lk, self.head_dim, k.device, k.dtype)
            k = apply_rope(k, cos_k, sin_k)

        out = F.scaled_dot_product_attention(q, k, v)    # [B,H,Lq,Dh]
        B, H, Lq, Dh = out.shape
        out = out.transpose(1, 2).reshape(B, Lq, H * Dh)
        return self.out_proj(out)


# ----------------------------- DiT Block ----------------------------- #
class DiTBlock(nn.Module):
    """AdaLN-Zero block (SPEC §6.2). cross-attn is NOT modulated."""

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0,
                 qk_norm: bool = True, rope: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm, rope=rope)
        self.norm2 = RMSNorm(dim)
        self.cross = Attention(dim, num_heads, qk_norm=qk_norm, rope=rope)
        self.norm3 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )
        # AdaLN-Zero modulator: cond -> 6 vectors of dim D. Init 0 -> identity at start.
        self.ada_mod = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada_mod.weight)
        nn.init.zeros_(self.ada_mod.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, sh1, g1, s2, sh2, g2 = self.ada_mod(cond).chunk(6, dim=-1)
        s1, sh1, g1 = (v.unsqueeze(1) for v in (s1, sh1, g1))
        s2, sh2, g2 = (v.unsqueeze(1) for v in (s2, sh2, g2))

        # 1) self-attn (modulated)
        h = self.norm1(x) * (1 + s1) + sh1
        x = x + g1 * self.attn(h)
        # 2) cross-attn to past context (NOT modulated)
        x = x + self.cross(self.norm2(x), ctx)
        # 3) FFN (modulated)
        h = self.norm3(x) * (1 + s2) + sh2
        x = x + g2 * self.ffn(h)
        return x


class FinalLayer(nn.Module):
    """Zero-init output head (SPEC §6.3) -> velocity ~ 0 at init."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.ada_mod = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.ada_mod.weight)
        nn.init.zeros_(self.ada_mod.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s, sh = self.ada_mod(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + s.unsqueeze(1)) + sh.unsqueeze(1)
        return self.linear(h)


class DiT(nn.Module):
    """8-layer DiT (SPEC §6). forward(noisy_seq, past_ctx, cond) -> velocity [B, L, D]."""

    def __init__(self, dim: int = 768, num_layers: int = 8, num_heads: int = 12,
                 mlp_ratio: float = 4.0, qk_norm: bool = True, rope: bool = True):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, qk_norm=qk_norm, rope=rope)
            for _ in range(num_layers)
        ])
        self.final = FinalLayer(dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, ctx, cond)
        return self.final(x, cond)
