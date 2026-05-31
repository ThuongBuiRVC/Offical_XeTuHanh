"""Flow Matching + Shortcut utilities (SPEC §7).

Convention: t=0 is pure noise, t=1 is clean; linear interpolation path.
    x_t = (1 - t) * noise + t * clean
    v   = clean - noise            (constant velocity along the straight path)

Shortcut (Frans et al. 2024): the model also takes step-size d. Self-consistency
target = average of two consecutive d/2 predictions (computed without gradient).
"""
from __future__ import annotations

from typing import Callable

import torch


def noise(x_clean: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation noising. t: [B] broadcast to x's rank (SPEC §7.1)."""
    t = t.view(-1, *([1] * (x_clean.dim() - 1)))
    return (1 - t) * x_noise + t * x_clean


def velocity_target(x_clean: torch.Tensor, x_noise: torch.Tensor) -> torch.Tensor:
    return x_clean - x_noise


# Default shortcut step sizes: 1/128 .. 1/2 (SPEC §7.2)
DEFAULT_D_VALUES = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]


def sample_t_d(
    batch_size: int,
    device,
    d_values: list[float] | None = None,
    shortcut_frac: float = 0.25,
):
    """Sample (t, d) per SPEC §7.2.

    Returns:
        t        : [B] in [0, 1)
        d        : [B] (0 for flow-only samples, else a value from d_values)
        mask_sc  : [B] bool, True where d > 0 (self-consistency samples)

    For shortcut samples, t is constrained so that t + d <= 1 (the two half-steps
    used to build the self-consistency target stay in the valid range).
    """
    if d_values is None:
        d_values = DEFAULT_D_VALUES
    d_values_t = torch.tensor(d_values, device=device, dtype=torch.float32)

    t = torch.rand(batch_size, device=device)
    d = torch.zeros(batch_size, device=device)
    mask_sc = torch.rand(batch_size, device=device) < shortcut_frac

    n_sc = int(mask_sc.sum().item())
    if n_sc > 0:
        idx = torch.randint(0, len(d_values), (n_sc,), device=device)
        d_sel = d_values_t[idx]
        d[mask_sc] = d_sel
        # keep t + d <= 1 so t+d/2 is valid for the consistency target
        t[mask_sc] = torch.rand(n_sc, device=device) * (1 - d_sel)
    return t, d, mask_sc


@torch.no_grad()
def compute_shortcut_targets(
    core_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    t: torch.Tensor,
    d: torch.Tensor,
):
    """Self-consistency target (SPEC §7.2), no gradient.

    Args:
        core_fn: (noisy_seq, t, d) -> velocity_full [B, L, D] (DiT, before decoder).
        x_t:     concatenated noisy core [B, L, D] (= cat([z_t, a_t])).
        t, d:    [B] timestep / step-size for the *shortcut* (full step d).

    Returns:
        v_target [B, L, D] = (v1 + v2) / 2.
    """
    half = (d / 2).view(-1, *([1] * (x_t.dim() - 1)))
    v1 = core_fn(x_t, t, d / 2)
    x_mid = x_t + half * v1
    v2 = core_fn(x_mid, t + d / 2, d / 2)
    return 0.5 * (v1 + v2)
