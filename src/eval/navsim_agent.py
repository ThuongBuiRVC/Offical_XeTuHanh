"""NAVSIM agent adapter for the joint world-action model.

NAVSIM calls ``compute_trajectory`` once per token. With ``requires_scene=True``
the call signature is ``compute_trajectory(agent_input, scene)`` (see
navsim/planning/script/utils.py), so the agent receives the full ``Scene`` and can
build the route from the map exactly like training.

Train/test parity is enforced by routing ALL feature extraction through
``src.data.navsim_features`` — the same module the cache-prep script uses.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import Config, load_config
from src.data.navsim_features import FeatureSpec, build_inference_inputs
from src.inference.rollout import rollout
from src.models.full_model import FullModel
from src.utils.ema import EMA


def _as_config(raw) -> Config:
    return raw if isinstance(raw, Config) else Config(raw)


def _trajectory_heading(xy: np.ndarray) -> np.ndarray:
    """Derive per-step heading from finite differences (carry last when stationary)."""
    headings = np.zeros((xy.shape[0],), dtype=np.float32)
    prev = np.zeros((2,), dtype=np.float32)
    last_heading = 0.0
    for idx, point in enumerate(xy.astype(np.float32)):
        delta = point - prev
        if float(np.linalg.norm(delta)) > 1e-4:
            last_heading = math.atan2(float(delta[1]), float(delta[0]))
        headings[idx] = last_heading
        prev = point
    return headings


class WorldModelNavsimAgent:
    """Hydra-instantiable NAVSIM agent for the joint DiT + Shortcut-FM model."""

    # We need the Scene (map + roadblock ids) to build the route -> requires_scene.
    requires_scene = True

    def __init__(
        self,
        ckpt_path: str,
        config_path: str = "configs/train.yaml",
        device: str = "cuda",
        image_size: Optional[int] = None,
        trajectory_sampling=None,
    ) -> None:
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        self.ckpt_path = Path(ckpt_path)
        self.config_path = config_path
        self.device_name = device
        self.image_size_override = image_size
        self.trajectory_sampling = trajectory_sampling or TrajectorySampling(
            time_horizon=4, interval_length=0.5
        )
        self.device = torch.device("cpu")
        self.model: Optional[FullModel] = None
        self.spec: Optional[FeatureSpec] = None
        self.ode_steps = 4
        self.ode_solver = "euler"

    def name(self) -> str:
        return "WorldModelNavsimAgent"

    def get_sensor_config(self):
        from navsim.common.dataclasses import SensorConfig

        return SensorConfig(
            cam_f0=True,
            cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False,
            cam_b0=False, lidar_pc=False,
        )

    def initialize(self) -> None:
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.ckpt_path}")
        self.device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        cfg = _as_config(ckpt.get("cfg") or load_config(self.config_path))
        if self.image_size_override is not None:
            cfg.vjepa.img_size = int(self.image_size_override)

        self.spec = FeatureSpec.from_config(cfg)
        self.ode_steps = int(cfg.inference.ode_steps)
        self.ode_solver = str(cfg.inference.solver)

        self.model = FullModel.from_config(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        # Use EMA weights for eval/inference (SPEC §9.5).
        if "ema" in ckpt:
            ema = EMA(self.model, decay=float(cfg.optim.ema_decay))
            ema.load_state_dict(ckpt["ema"])
            ema.copy_to(self.model)
        self.model.eval()

    @torch.no_grad()
    def compute_trajectory(self, agent_input, scene=None):
        from navsim.common.dataclasses import Trajectory

        if self.model is None or self.spec is None:
            raise RuntimeError("agent was not initialized")

        past_cam, ego, route = build_inference_inputs(agent_input, scene, self.spec)
        past_cam = past_cam.unsqueeze(0).to(self.device)
        ego = ego.unsqueeze(0).to(self.device)
        route = route.unsqueeze(0).to(self.device)

        waypoint, _ = rollout(
            self.model, past_cam, route, ego,
            steps=self.ode_steps, solver=self.ode_solver,
        )
        xy = waypoint[0].float().cpu().numpy().astype(np.float32)
        headings = _trajectory_heading(xy)
        poses = np.concatenate([xy, headings[:, None]], axis=1).astype(np.float32)
        return Trajectory(poses=poses, trajectory_sampling=self.trajectory_sampling)
