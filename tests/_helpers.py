"""Shared test helpers."""
from __future__ import annotations

import torch

from src.models.full_model import FullModel


def build_model(img_size: int = 384, num_layers: int = 8) -> FullModel:
    """FullModel with the offline V-JEPA stub (no network) at a given resolution."""
    return FullModel(
        ego_dim=7,
        vjepa_kwargs=dict(img_size=img_size, num_frames=8, force_fallback=True),
        dit_kwargs=dict(num_layers=num_layers),
    )


def dummy_batch(B: int, img_size: int = 384, ego_dim: int = 7):
    return dict(
        past_cam=torch.randn(B, 8, 3, img_size, img_size),
        fut_cam=torch.randn(B, 8, 3, img_size, img_size),
        route=torch.randn(B, 20, 2),
        ego=torch.randn(B, 8, ego_dim),
        wp_gt=torch.randn(B, 8, 2),
    )
