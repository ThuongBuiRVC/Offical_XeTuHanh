"""Acceptance: shapes, N_v, frozen/trainable, AdaLN identity (SPEC §13.1/.2/.4/.6)."""
import torch

from tests._helpers import build_model, dummy_batch


def test_shapes_and_nv():
    B = 2
    model = build_model(img_size=384)
    # 13.4 N_v consistency: 384/16=24 -> 24^2 * (8/2) = 2304
    assert model.n_v == 2304, model.n_v

    b = dummy_batch(B, img_size=384)
    out = model(b["past_cam"], b["fut_cam"], b["route"], b["ego"], b["wp_gt"],
                t=torch.rand(B), d=torch.zeros(B))
    # 13.1 shape sanity
    assert out["v_z_pred"].shape == (B, model.n_v, 768)
    assert out["v_a_pred"].shape == (B, 8, 768)
    assert out["waypoint_pred"].shape == (B, 8, 2)


def test_frozen_trainable():
    model = build_model(img_size=96)
    # 13.2
    assert all(not p.requires_grad for p in model.vjepa.parameters())
    assert any(p.requires_grad for p in model.dit.parameters())
    assert any(p.requires_grad for p in model.action_decoder.parameters())


def test_adaln_identity_at_init():
    B = 2
    model = build_model(img_size=96)
    b = dummy_batch(B, img_size=96)
    out = model(b["past_cam"], b["fut_cam"], b["route"], b["ego"], b["wp_gt"],
                t=torch.rand(B), d=torch.zeros(B))
    # 13.6: zero-init final -> velocity ~ 0 at init
    assert out["v_z_pred"].abs().mean() < 1e-5
    assert out["v_a_pred"].abs().mean() < 1e-5
