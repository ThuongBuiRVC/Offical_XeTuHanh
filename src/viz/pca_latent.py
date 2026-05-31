"""PCA latent visualization (SPEC §11) — debug tool, NOT a pixel decoder.

Projects V-JEPA latent tokens [B, N_v, 768] down to 3 channels and lays them out
as a [B, 8, 384, 384, 3] video for qualitative inspection of the world model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def visualize_latent_pca(
    z: torch.Tensor,
    t_fut: int = 8,
    tubelet: int = 2,
    patch_grid: int = 24,
    out_size: int = 384,
    basis: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """z: [B, N_v, 768] -> [B, 8, out_size, out_size, 3] in [0, 1].

    If `basis` ([768,3]) and `mean` ([768]) are given (precomputed offline), they
    are used for a consistent colormap across batches (SPEC §11 recommendation);
    otherwise PCA is fit on the current batch.
    """
    B, N_v, Dn = z.shape
    t_tube = t_fut // tubelet
    assert N_v == t_tube * patch_grid * patch_grid, (
        f"{N_v} != {t_tube * patch_grid * patch_grid} (t_tube*grid^2)"
    )
    flat = z.reshape(B * N_v, Dn).float()

    if basis is not None and mean is not None:
        flat = flat - mean.to(flat.device)
        proj = flat @ basis.to(flat.device)            # [B*N_v, 3]
    else:
        flat = flat - flat.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(flat, q=3, niter=4)
        proj = flat @ V[:, :3]

    # global (whole-video) normalization to avoid flicker (SPEC §11)
    pmin = proj.min(0, keepdim=True).values
    pmax = proj.max(0, keepdim=True).values
    proj = (proj - pmin) / (pmax - pmin + 1e-8)

    vis = proj.reshape(B, t_tube, patch_grid, patch_grid, 3)
    vis = vis.repeat_interleave(tubelet, dim=1)                      # [B, 8, 24, 24, 3]
    vis = vis.permute(0, 1, 4, 2, 3).reshape(B * t_fut, 3, patch_grid, patch_grid)
    vis = F.interpolate(vis, size=(out_size, out_size), mode="bilinear", align_corners=False)
    vis = vis.reshape(B, t_fut, 3, out_size, out_size).permute(0, 1, 3, 4, 2)
    return vis.clamp(0, 1)


@torch.no_grad()
def fit_pca_basis(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a fixed PCA basis from stacked latents [M, 768]. Returns (basis [768,3], mean [768])."""
    mean = latents.float().mean(0)
    centered = latents.float() - mean
    _, _, V = torch.pca_lowrank(centered, q=3, niter=4)
    return V[:, :3].contiguous(), mean.contiguous()
