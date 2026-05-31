"""Image / sequence transforms (SPEC §12).

STATUS: PLACEHOLDER. Normalization stats and augmentation will be finalized with
the real NuPlan/nav2sim pipeline. V-JEPA 2 expects ImageNet-style normalization.
"""
from __future__ import annotations

import torch

# ImageNet normalization (V-JEPA default). Confirm against the official preprocessing.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_clip(clip: torch.Tensor) -> torch.Tensor:
    """clip: [T, 3, H, W] in [0,1] -> normalized."""
    return (clip - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_clip(clip: torch.Tensor) -> torch.Tensor:
    return clip * IMAGENET_STD + IMAGENET_MEAN
