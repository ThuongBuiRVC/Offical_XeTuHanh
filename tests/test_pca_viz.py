"""PCA viz smoke test (SPEC §13.7)."""
import torch

from src.viz.pca_latent import fit_pca_basis, visualize_latent_pca


def test_pca_viz_shape():
    z = torch.randn(2, 2304, 768)
    vis = visualize_latent_pca(z)
    assert vis.shape == (2, 8, 384, 384, 3)
    assert vis.min() >= 0 and vis.max() <= 1
    assert torch.isfinite(vis).all()


def test_pca_viz_with_fixed_basis():
    latents = torch.randn(5000, 768)
    basis, mean = fit_pca_basis(latents)
    assert basis.shape == (768, 3) and mean.shape == (768,)
    z = torch.randn(2, 2304, 768)
    vis = visualize_latent_pca(z, basis=basis, mean=mean)
    assert vis.shape == (2, 8, 384, 384, 3)
    assert torch.isfinite(vis).all()
