"""Flow-matching noising + shortcut sampling (SPEC §7)."""
import torch

from src.models.flow_matching import (
    DEFAULT_D_VALUES,
    compute_shortcut_targets,
    noise,
    sample_t_d,
    velocity_target,
)


def test_noise_endpoints():
    x_clean = torch.randn(4, 8, 768)
    x_noise = torch.randn(4, 8, 768)
    # t=1 -> clean, t=0 -> noise
    assert torch.allclose(noise(x_clean, x_noise, torch.ones(4)), x_clean, atol=1e-5)
    assert torch.allclose(noise(x_clean, x_noise, torch.zeros(4)), x_noise, atol=1e-5)


def test_velocity_target():
    x_clean = torch.randn(4, 8, 768)
    x_noise = torch.randn(4, 8, 768)
    assert torch.allclose(velocity_target(x_clean, x_noise), x_clean - x_noise)


def test_sample_t_d():
    torch.manual_seed(0)
    t, d, mask = sample_t_d(2000, "cpu", d_values=DEFAULT_D_VALUES, shortcut_frac=0.25)
    assert t.shape == (2000,) and d.shape == (2000,)
    # ~25% shortcut samples
    frac = mask.float().mean().item()
    assert 0.18 < frac < 0.32, frac
    # flow-only samples have d==0; shortcut samples have d>0 and t+d<=1
    assert (d[~mask] == 0).all()
    assert (d[mask] > 0).all()
    assert (t[mask] + d[mask] <= 1.0 + 1e-5).all()


def test_shortcut_target_average():
    # core_fn returns constant velocity -> self-consistency target == that constant
    const = torch.randn(3, 10, 768)

    def core_fn(seq, t, d):
        return const

    x_t = torch.randn(3, 10, 768)
    t = torch.rand(3)
    d = torch.full((3,), 0.25)
    vt = compute_shortcut_targets(core_fn, x_t, t, d)
    assert torch.allclose(vt, const, atol=1e-5)
