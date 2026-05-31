"""Exponential Moving Average of model weights (SPEC §9.5, decay 0.9999).

Used for eval/inference. Only tracks trainable params + buffers of the wrapped
model; the frozen V-JEPA is excluded automatically (its params don't change, and
we skip params with requires_grad=False to save memory).
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow[name]
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    class swap:
        """Context manager: temporarily load EMA weights into the model."""

        def __init__(self, ema: "EMA", model: nn.Module):
            self.ema, self.model = ema, model

        def __enter__(self):
            self._backup = {
                name: p.detach().clone()
                for name, p in self.model.named_parameters()
                if name in self.ema.shadow
            }
            self.ema.copy_to(self.model)
            return self.model

        def __exit__(self, *exc):
            for name, p in self.model.named_parameters():
                if name in self._backup:
                    p.data.copy_(self._backup[name])
