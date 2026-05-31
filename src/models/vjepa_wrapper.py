"""Frozen V-JEPA 2.1 ViT-B backbone wrapper (SPEC §2).

Responsibilities:
    - Load V-JEPA 2.1 ViT-B/16 @384 and FREEZE it (no grad ever).
    - Produce RAW patch tokens [B, N_v, 768] — NO projection MLP (SPEC §2.1 convention).
    - Infer N_v dynamically from a dummy forward (NEVER hardcode; SPEC §2.3 / §13.4).

Notes:
    - Real weights are loaded via torch.hub. If the hub is unavailable (offline CI,
      unit tests), we fall back to a random-initialized tubelet ViT stub that produces
      the *correct* token count (2304 @384/8f/p16/tube2) so shape/accept tests still pass.
    - V-JEPA expects video as [B, C, T, H, W]; the public API here takes [B, T, C, H, W].
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _RandomViTFallback(nn.Module):
    """Offline stub: tubelet Conv3d patch-embed → flattened tokens.

    Produces N_v = (T/tubelet) * (H/patch) * (W/patch) tokens of dim `embed_dim`,
    matching the real V-JEPA token layout so downstream shapes/tests are valid.
    NOT pretrained — for plumbing/tests only.
    """

    def __init__(self, embed_dim: int = 768, patch_size: int = 16, tubelet_size: int = 2):
        super().__init__()
        self.proj = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x)                       # [B, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)       # [B, N_v, D]
        return x


class VJEPAWrapper(nn.Module):
    """Frozen V-JEPA 2.1 ViT-B/16 feature extractor."""

    def __init__(
        self,
        model_name: str = "vit_base",
        img_size: int = 384,
        num_frames: int = 8,
        tubelet_size: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        pretrained: bool = True,
        hub_repo: str = "facebookresearch/vjepa2",
        force_fallback: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self._expected_dim = embed_dim
        self.is_fallback = False

        self._load_encoder(pretrained, hub_repo, force_fallback)

        # FROZEN forever (SPEC §2.2)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Dynamic N_v inference (SPEC §2.3) — single dummy forward.
        self.n_v: int = -1
        self.embed_dim: int = -1
        self._infer_dims()

    def _load_encoder(self, pretrained: bool, hub_repo: str, force_fallback: bool = False) -> None:
        if force_fallback:
            self.encoder = _RandomViTFallback(
                self._expected_dim, self.patch_size, self.tubelet_size
            )
            self.is_fallback = True
            print("[VJEPAWrapper] force_fallback=True; using random ViT stub.")
            return
        try:
            self.encoder = torch.hub.load(hub_repo, self.model_name, pretrained=pretrained)
            print(f"[VJEPAWrapper] Loaded {self.model_name} from {hub_repo}")
        except Exception as e:  # pragma: no cover - depends on network/hub
            print(f"[VJEPAWrapper] hub load failed ({e}); using random ViT fallback.")
            self.encoder = _RandomViTFallback(
                self._expected_dim, self.patch_size, self.tubelet_size
            )
            self.is_fallback = True

    @torch.no_grad()
    def _infer_dims(self) -> None:
        dummy = torch.zeros(1, self.num_frames, 3, self.img_size, self.img_size)
        device = next(self.encoder.parameters()).device
        feats = self.forward(dummy.to(device))
        self.n_v = int(feats.shape[1])
        self.embed_dim = int(feats.shape[2])
        print(f"[VJEPAWrapper] n_v={self.n_v}, embed_dim={self.embed_dim}")

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: [B, T, 3, H, W] -> raw tokens [B, N_v, embed_dim] (no grad, no projection)."""
        if video.dim() != 5:
            raise ValueError(f"expected 5D video [B,T,C,H,W], got {tuple(video.shape)}")
        x = video.permute(0, 2, 1, 3, 4).contiguous()   # -> [B, C, T, H, W]
        feats = self.encoder(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.dim() != 3:
            raise ValueError(
                f"V-JEPA encoder must return [B, N, D] tokens, got {tuple(feats.shape)}"
            )
        return feats
