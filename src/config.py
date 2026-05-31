"""Config loading utilities.

Loads `configs/train.yaml` into a nested attribute-accessible namespace.
Keeps the YAML as the single source of truth (SPEC §12: one config).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict that also allows attribute access and nested dotted defaults."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def default_config() -> Config:
    """Repo-default config (configs/train.yaml)."""
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "train.yaml")
