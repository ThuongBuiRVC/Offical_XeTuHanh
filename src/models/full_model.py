"""End-to-end orchestrator (SPEC §5, §9.2).

Wires: frozen V-JEPA -> input embedders -> DiT (AdaLN-Zero, shortcut FM) ->
slice velocities -> one-step clean action estimate -> action decoder.

Single-stage joint model: everything except V-JEPA is trainable, one optimizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .action_decoder import ActionDecoder
from .dit import DiT
from .flow_matching import noise, velocity_target
from .input_embedders import CondEmbedder, InputEmbedder
from .vjepa_wrapper import VJEPAWrapper


class FullModel(nn.Module):
    def __init__(
        self,
        ego_dim: int = 6,
        dim: int = 768,
        n_act: int = 8,
        route_dim: int = 2,
        action_dim: int = 2,
        vjepa_kwargs: dict | None = None,
        dit_kwargs: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_act = n_act

        # Frozen V-JEPA backbone -> dynamic N_v (SPEC §2)
        self.vjepa = VJEPAWrapper(**(vjepa_kwargs or {}))
        self.n_v = self.vjepa.n_v
        assert self.vjepa.embed_dim == dim, (
            f"V-JEPA embed_dim {self.vjepa.embed_dim} != model dim {dim}"
        )

        # Input embedders (SPEC §3) — NO MLP on V-JEPA latents.
        self.nav_mlp = InputEmbedder(route_dim, dim)
        self.ego_mlp = InputEmbedder(ego_dim, dim)
        self.action_mlp = InputEmbedder(action_dim, dim)
        self.cond_embedder = CondEmbedder(dim)

        # DiT backbone (SPEC §6)
        self.dit = DiT(dim=dim, **(dit_kwargs or {}))

        # Action decoder (SPEC §8) — trained jointly.
        self.action_decoder = ActionDecoder(dim)

    @classmethod
    def from_config(cls, cfg) -> "FullModel":
        v = cfg.vjepa
        return cls(
            ego_dim=cfg.dims.ego_dim,
            dim=cfg.dims.embed_dim,
            n_act=cfg.dims.n_act,
            vjepa_kwargs=dict(
                model_name=v.model_name, img_size=v.img_size, num_frames=v.num_frames,
                tubelet_size=v.tubelet_size, patch_size=v.patch_size,
                embed_dim=cfg.dims.embed_dim, pretrained=v.pretrained, hub_repo=v.hub_repo,
                force_fallback=getattr(v, "force_fallback", False),
            ),
            dit_kwargs=dict(
                num_layers=cfg.dit.num_layers, num_heads=cfg.dit.num_heads,
                mlp_ratio=cfg.dit.mlp_ratio, qk_norm=cfg.dit.qk_norm, rope=cfg.dit.rope,
            ),
        )

    # --------------------------------------------------------------- #
    def cond(self, t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.cond_embedder(t, d)

    def encode_context(self, past_cam, route, ego) -> torch.Tensor:
        """past_ctx [B, N_v + N_route + N_ego, D] for cross-attn K,V (SPEC §5)."""
        camera_past = self.vjepa(past_cam)            # [B, N_v, D] (no grad, no projection)
        route_tok = self.nav_mlp(route)               # [B, 20, D]
        ego_tok = self.ego_mlp(ego)                   # [B, 8, D]
        return torch.cat([camera_past, route_tok, ego_tok], dim=1)

    def core(self, noisy_seq, past_ctx, t, d):
        """DiT forward -> (v_z, v_a). Used for training core + shortcut + rollout."""
        out = self.dit(noisy_seq, past_ctx, self.cond(t, d))
        v_z = out[:, : self.n_v, :]
        v_a = out[:, self.n_v : self.n_v + self.n_act, :]
        return v_z, v_a

    def core_velocity(self, noisy_seq, past_ctx, t, d) -> torch.Tensor:
        """Full velocity [B, L, D] (concatenated z|a). For shortcut target closure."""
        return self.dit(noisy_seq, past_ctx, self.cond(t, d))

    # --------------------------------------------------------------- #
    def forward(self, past_cam, fut_cam, route, ego, wp_gt, t, d):
        """Single joint forward (SPEC §9.2).

        Returns a dict with velocities, waypoint prediction, FM targets, and the
        intermediates (past_ctx, z_t, a_t) needed to build the shortcut target.
        """
        # --- Encode (V-JEPA frozen / no grad inside wrapper) ---
        camera_past = self.vjepa(past_cam)            # [B, N_v, D]
        z_clean = self.vjepa(fut_cam)                 # [B, N_v, D]  (target)
        route_tok = self.nav_mlp(route)
        ego_tok = self.ego_mlp(ego)
        a_clean = self.action_mlp(wp_gt)              # [B, 8, D]

        # --- Noising (SPEC §7.1), independent noise for z and a, shared t ---
        z_noise = torch.randn_like(z_clean)
        a_noise = torch.randn_like(a_clean)
        z_t = noise(z_clean, z_noise, t)
        a_t = noise(a_clean, a_noise, t)
        v_z_target = velocity_target(z_clean, z_noise)
        v_a_target = velocity_target(a_clean, a_noise)

        # --- DiT forward ---
        past_ctx = torch.cat([camera_past, route_tok, ego_tok], dim=1)
        noisy_seq = torch.cat([z_t, a_t], dim=1)
        out = self.dit(noisy_seq, past_ctx, self.cond(t, d))
        v_z_pred = out[:, : self.n_v, :]
        v_a_pred = out[:, self.n_v : self.n_v + self.n_act, :]

        # --- One-step clean action estimate -> decode (SPEC §8.2). NO detach. ---
        a_hat_clean = a_t + (1 - t).view(-1, 1, 1) * v_a_pred
        waypoint_pred = self.action_decoder(a_hat_clean)

        return {
            "v_z_pred": v_z_pred,
            "v_a_pred": v_a_pred,
            "v_z_target": v_z_target,
            "v_a_target": v_a_target,
            "waypoint_pred": waypoint_pred,
            "z_t": z_t,
            "a_t": a_t,
            "past_ctx": past_ctx,
        }
