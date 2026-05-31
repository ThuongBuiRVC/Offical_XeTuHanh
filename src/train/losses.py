"""Loss functions (SPEC §9.3, §9.4).

L_total = L_flow_z + lambda_a*L_flow_a + lambda_wp*L_wp
        + lambda_smooth*L_smooth + lambda_sc*L_sc
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_a: float = 1.0
    lambda_wp: float = 5.0
    lambda_smooth: float = 0.1
    lambda_sc: float = 1.0
    huber_beta: float = 0.5

    @classmethod
    def from_config(cls, cfg) -> "LossWeights":
        l = cfg.loss
        return cls(l.lambda_a, l.lambda_wp, l.lambda_smooth, l.lambda_sc, l.huber_beta)


def smoothness_loss(wp: torch.Tensor) -> torch.Tensor:
    """2nd-order temporal difference penalty (SPEC §9.3). wp: [B, 8, 2]."""
    diff2 = wp[:, 2:] - 2 * wp[:, 1:-1] + wp[:, :-2]   # [B, 6, 2]
    return (diff2 ** 2).mean()


def compute_losses(
    out: dict,
    wp_gt: torch.Tensor,
    weights: LossWeights,
    mask_sc: torch.Tensor | None = None,
    sc_target_z: torch.Tensor | None = None,
    sc_target_a: torch.Tensor | None = None,
) -> dict:
    """Compute all losses and the weighted total (SPEC §9.3/§9.4).

    Args:
        out: dict from FullModel.forward (v_z_pred, v_a_pred, *targets, waypoint_pred).
        wp_gt: ground-truth waypoints [B, 8, 2].
        mask_sc: [B] bool of shortcut samples (d>0). If None/empty -> L_sc = 0.
        sc_target_z / sc_target_a: self-consistency targets [B, *, D] (no grad).
    """
    v_z_pred, v_a_pred = out["v_z_pred"], out["v_a_pred"]
    wp_pred = out["waypoint_pred"]

    L_flow_z = F.mse_loss(v_z_pred, out["v_z_target"])
    L_flow_a = F.mse_loss(v_a_pred, out["v_a_target"])
    L_wp = F.smooth_l1_loss(wp_pred, wp_gt, beta=weights.huber_beta)
    L_smooth = smoothness_loss(wp_pred)

    if mask_sc is not None and mask_sc.any() and sc_target_z is not None:
        m = mask_sc
        L_sc = F.mse_loss(v_z_pred[m], sc_target_z[m]) + F.mse_loss(v_a_pred[m], sc_target_a[m])
    else:
        L_sc = v_z_pred.new_zeros(())

    L_total = (
        L_flow_z
        + weights.lambda_a * L_flow_a
        + weights.lambda_wp * L_wp
        + weights.lambda_smooth * L_smooth
        + weights.lambda_sc * L_sc
    )

    return {
        "total": L_total,
        "flow_z": L_flow_z.detach(),
        "flow_a": L_flow_a.detach(),
        "wp": L_wp.detach(),
        "smooth": L_smooth.detach(),
        "sc": L_sc.detach() if torch.is_tensor(L_sc) else L_sc,
    }
