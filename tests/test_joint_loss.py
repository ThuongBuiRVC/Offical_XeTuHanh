"""Core joint-training test: waypoint-loss gradient reaches the DiT (SPEC §13.5).

Note on zero-init: the FinalLayer linear is zero-initialized (SPEC §6.3), so at
exact init the velocity is identically 0 and *no* gradient can reach the DiT
(d out / d x = W = 0). That is expected. To verify there is NO detach in the
wiring (the real point of this test), we give the final linear a small non-zero
weight first, mimicking a model a few steps into training, then check DiT grads.
"""
import torch
import torch.nn.functional as F

from src.train.losses import LossWeights, compute_losses
from src.models.flow_matching import compute_shortcut_targets, sample_t_d
from tests._helpers import build_model, dummy_batch


def test_waypoint_grad_reaches_dit():
    torch.manual_seed(0)
    B = 2
    model = build_model(img_size=96)
    # Break the zero-init of the output head so connectivity is observable.
    torch.nn.init.normal_(model.dit.final.linear.weight, std=0.02)

    b = dummy_batch(B, img_size=96)
    out = model(b["past_cam"], b["fut_cam"], b["route"], b["ego"], b["wp_gt"],
                t=torch.rand(B), d=torch.zeros(B))

    L = F.smooth_l1_loss(out["waypoint_pred"], b["wp_gt"])
    model.zero_grad()
    L.backward()

    # gradient must reach DiT block params (through v_a_pred -> a_hat -> decoder)
    grads = [p.grad for p in model.dit.blocks.parameters() if p.grad is not None]
    assert len(grads) > 0
    total = sum(g.abs().sum().item() for g in grads)
    assert total > 0, "waypoint loss did not reach DiT -> joint training broken"


def test_full_loss_backward():
    torch.manual_seed(0)
    B = 2
    model = build_model(img_size=96)
    weights = LossWeights()
    b = dummy_batch(B, img_size=96)

    t, d, mask = sample_t_d(B, "cpu", shortcut_frac=1.0)  # force shortcut path
    out = model(b["past_cam"], b["fut_cam"], b["route"], b["ego"], b["wp_gt"], t, d)

    past_ctx = out["past_ctx"]
    x_t = torch.cat([out["z_t"], out["a_t"]], dim=1)
    v_sc = compute_shortcut_targets(
        lambda seq, tt, dd: model.core_velocity(seq, past_ctx, tt, dd), x_t, t, d
    )
    sc_z = v_sc[:, : model.n_v]
    sc_a = v_sc[:, model.n_v : model.n_v + model.n_act]

    losses = compute_losses(out, b["wp_gt"], weights, mask, sc_z, sc_a)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None for p in model.dit.parameters())
