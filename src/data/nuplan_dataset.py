"""Cached NuPlan dataset for the joint world-action model.

The training loop expects data to be prepared ahead of time with
``scripts/prepare_nuplan_cache.py``.  Each sample is a small ``.pt`` file with
preprocessed tensors, so training avoids image decoding, map queries, and pose
frame transforms in worker processes.

Required tensor keys per cached sample:
    past_cam : [8, 3, 384, 384] float16/float32, ImageNet-normalized
    fut_cam  : [8, 3, 384, 384] float16/float32, ImageNet-normalized
    route    : [20, 2]          float16/float32, ego-frame route waypoints
    ego      : [8, ego_dim]     float16/float32, ego history in current ego frame
    wp_gt    : [8, 2]           float16/float32, future expert waypoints in meters

``PlaceholderDataset`` remains available for plumbing tests and smoke runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = ("past_cam", "fut_cam", "route", "ego", "wp_gt")


class CachedNuPlanDataset(Dataset):
    """Lazy loader for prepared per-sample cache files."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        t_past: int = 8,
        t_fut: int = 8,
        img_size: int = 384,
        n_route: int = 20,
        n_ego: int = 8,
        n_act: int = 8,
        ego_dim: int = 7,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"cache split directory not found: {self.split_dir}")

        self.files = sorted(self.split_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no .pt cache files found in {self.split_dir}")

        self.expected_shapes = {
            "past_cam": (t_past, 3, img_size, img_size),
            "fut_cam": (t_fut, 3, img_size, img_size),
            "route": (n_route, 2),
            "ego": (n_ego, ego_dim),
            "wp_gt": (n_act, 2),
        }

    def __len__(self) -> int:
        return len(self.files)

    def _validate(self, sample: dict[str, Any], path: Path) -> None:
        for key in REQUIRED_KEYS:
            if key not in sample:
                raise KeyError(f"{path}: missing required key {key!r}")
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{path}: key {key!r} is {type(value).__name__}, expected Tensor")
            expected = self.expected_shapes[key]
            if tuple(value.shape) != expected:
                raise ValueError(f"{path}: key {key!r} shape {tuple(value.shape)} != {expected}")
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"{path}: key {key!r} contains NaN or Inf")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.files[idx]
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"{path}: expected a dict, got {type(sample).__name__}")
        self._validate(sample, path)
        out: dict[str, torch.Tensor | str] = {key: sample[key].float() for key in REQUIRED_KEYS}
        meta = sample.get("meta", {})
        out["token"] = str(meta.get("token", path.stem)) if isinstance(meta, dict) else path.stem
        return out


class PlaceholderDataset(Dataset):
    """Random-tensor dataset matching the real I/O contract for smoke tests."""

    def __init__(
        self,
        length: int = 256,
        t_past: int = 8,
        t_fut: int = 8,
        img_size: int = 384,
        n_route: int = 20,
        n_ego: int = 8,
        n_act: int = 8,
        ego_dim: int = 7,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.t_past, self.t_fut, self.img_size = t_past, t_fut, img_size
        self.n_route, self.n_ego, self.n_act, self.ego_dim = n_route, n_ego, n_act, ego_dim
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + idx)
        size = self.img_size
        return {
            "past_cam": torch.randn(self.t_past, 3, size, size, generator=g),
            "fut_cam": torch.randn(self.t_fut, 3, size, size, generator=g),
            "route": torch.randn(self.n_route, 2, generator=g),
            "ego": torch.randn(self.n_ego, self.ego_dim, generator=g),
            "wp_gt": torch.randn(self.n_act, 2, generator=g),
        }


def build_dataset(cfg, split: str = "train") -> Dataset:
    """Build either the prepared cache dataset or the placeholder smoke dataset."""
    common = dict(
        t_past=cfg.dims.t_past,
        t_fut=cfg.dims.t_fut,
        img_size=cfg.vjepa.img_size,
        n_route=cfg.dims.n_route,
        n_ego=cfg.dims.n_ego,
        n_act=cfg.dims.n_act,
        ego_dim=cfg.dims.ego_dim,
    )
    if getattr(cfg.data, "use_placeholder", False):
        length = int(getattr(cfg.data, "placeholder_train_size", 256 if split == "train" else 32))
        if split != "train":
            length = int(getattr(cfg.data, "placeholder_val_size", 32))
        return PlaceholderDataset(length=length, seed=0 if split == "train" else 10000, **common)

    cache_root = getattr(cfg.data, "cache_root", None)
    if cache_root is None:
        raise ValueError("data.cache_root must be set when data.use_placeholder=false")
    return CachedNuPlanDataset(cache_root, split=split, **common)
