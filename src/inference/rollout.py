"""Closed-loop rollout for NuPlan / nav2sim (SPEC §10).

Given raw past observations, encodes context, solves the shortcut ODE, and
decodes future waypoints. Only `waypoint` is fed to the planner; `z` is for viz.
"""
from __future__ import annotations

import torch

from .ode_solver import solve


@torch.no_grad()
def rollout(model, past_cam, route, ego, steps: int = 4, solver: str = "euler"):
    """Returns (waypoint [B, 8, 2], z [B, N_v, 768])."""
    model.eval()
    past_ctx = model.encode_context(past_cam, route, ego)
    z, a = solve(model, past_ctx, steps=steps, solver=solver)
    waypoint = model.action_decoder(a)
    return waypoint, z
