"""ODE solvers for shortcut flow matching (SPEC §10).

Integrates the learned velocity field from t=0 (noise) to t=1 (clean) over a
small number of shortcut steps. Euler (default) or Heun.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def solve(model, past_ctx, steps: int = 4, solver: str = "euler", generator=None):
    """Roll out z and a from pure noise to clean latents.

    Args:
        model: FullModel (uses .core, .n_v, .n_act, .dim).
        past_ctx: [B, L_ctx, D] context tokens (K,V).
        steps: number of integration steps (shortcut step size d = 1/steps).
        solver: "euler" | "heun".

    Returns:
        z [B, N_v, D], a [B, n_act, D] (clean latent estimates at t=1).
    """
    B = past_ctx.shape[0]
    dev = past_ctx.device
    n_v, n_act, dim = model.n_v, model.n_act, model.dim

    z = torch.randn(B, n_v, dim, device=dev, generator=generator)
    a = torch.randn(B, n_act, dim, device=dev, generator=generator)

    d_step = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * d_step, device=dev)
        d = torch.full((B,), d_step, device=dev)
        seq = torch.cat([z, a], dim=1)
        v_z, v_a = model.core(seq, past_ctx, t, d)

        if solver == "heun":
            z1 = z + d_step * v_z
            a1 = a + d_step * v_a
            t2 = torch.full((B,), (i + 1) * d_step, device=dev)
            seq2 = torch.cat([z1, a1], dim=1)
            v_z2, v_a2 = model.core(seq2, past_ctx, t2, d)
            z = z + d_step * 0.5 * (v_z + v_z2)
            a = a + d_step * 0.5 * (v_a + v_a2)
        else:
            z = z + d_step * v_z
            a = a + d_step * v_a

    return z, a
