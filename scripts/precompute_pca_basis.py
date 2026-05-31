"""Precompute a fixed PCA basis for latent visualization (SPEC §11).

Fits PCA on ~N V-JEPA future latents and caches (basis, mean) to disk so that
viz colors stay consistent across batches/checkpoints.

Usage:
    python -m scripts.precompute_pca_basis --config configs/train.yaml --num 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.nuplan_dataset import build_dataset
from src.models.vjepa_wrapper import VJEPAWrapper
from src.viz.pca_latent import fit_pca_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--num", type=int, default=1000, help="number of latent samples")
    ap.add_argument("--out", default="logs/pca_basis.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    v = cfg.vjepa
    vjepa = VJEPAWrapper(
        model_name=v.model_name, img_size=v.img_size, num_frames=v.num_frames,
        tubelet_size=v.tubelet_size, patch_size=v.patch_size,
        embed_dim=cfg.dims.embed_dim, pretrained=v.pretrained, hub_repo=v.hub_repo,
    ).to(device)

    ds = build_dataset(cfg, split="train")
    dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=2)

    collected = []
    n = 0
    for batch in dl:
        fut = batch["fut_cam"].to(device)
        feats = vjepa(fut)                      # [B, N_v, 768]
        collected.append(feats.reshape(-1, feats.shape[-1]).cpu())
        n += feats.shape[0]
        if n >= args.num:
            break

    latents = torch.cat(collected, dim=0)
    basis, mean = fit_pca_basis(latents)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"basis": basis, "mean": mean}, args.out)
    print(f"[precompute_pca_basis] saved basis {tuple(basis.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
