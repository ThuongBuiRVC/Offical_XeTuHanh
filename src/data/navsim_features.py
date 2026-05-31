"""Shared NAVSIM feature extraction — SINGLE SOURCE OF TRUTH for train & test.

The whole point of this module is to guarantee **train/test parity**: the exact
same functions turn a NAVSIM ``Scene`` / ``AgentInput`` into model tensors, so the
network never sees "train one way, test another way".

Who calls what:
    - ``scripts/prepare_nuplan_cache.py`` (TRAIN): builds cached samples from
      ``Scene`` objects loaded via NAVSIM ``SceneLoader``. Uses
      :func:`build_training_sample` (past_cam + ego + route + fut_cam + wp_gt).
    - ``src/eval/navsim_agent.py`` (TEST): at every token NAVSIM hands the agent an
      ``AgentInput`` plus the ``Scene`` (``requires_scene=True``). Uses
      :func:`build_inference_inputs` (past_cam + ego + route) — the *same* builders.

Frame conventions (kept identical on both paths):
    - ego_pose comes from ``AgentInput.ego_statuses`` and is already in the current
      ego (rear-axle) local frame — NAVSIM computed it with
      ``convert_absolute_to_relative_se2_array``. We consume it verbatim.
    - velocity/acceleration are taken verbatim from ``EgoStatus`` (body frame).
    - route + future waypoints are transformed into the SAME current-ego local frame
      with the SAME NAVSIM helper, so every spatial tensor shares one frame.

The model-input contract (matches SPEC §3 and ``CachedNuPlanDataset``):
    past_cam [T_past, 3, H, W]   fut_cam [T_fut, 3, H, W]
    route    [N_route, 2]        ego     [N_ego, ego_dim=7]   wp_gt [N_act, 2]

NAVSIM history-frame note:
    navtest fixes num_history_frames=4 (1.5s @2Hz) but SPEC wants t_past=8. The
    builders resample the available frames to the model dims (``_resample_to_n``),
    so 4 history frames -> 8 tokens by nearest-index duplication. This is applied
    IDENTICALLY on train and test, so parity holds; only the past temporal
    resolution is limited by what NAVSIM provides. Future frames (target) come
    from the privileged future frames, so t_fut=8 is real (needs num_future_frames>=8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch


# ImageNet normalization (V-JEPA expects this). Keep identical to training prep.
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class FeatureSpec:
    """Dimensions/timing for feature extraction (mirror cfg.dims / cfg.vjepa)."""

    n_past: int = 8
    n_fut: int = 8
    n_route: int = 20
    n_ego: int = 8
    n_act: int = 8
    ego_dim: int = 7
    image_size: int = 384
    route_lookahead_m: float = 80.0

    @classmethod
    def from_config(cls, cfg) -> "FeatureSpec":
        return cls(
            n_past=cfg.dims.t_past,
            n_fut=cfg.dims.t_fut,
            n_route=cfg.dims.n_route,
            n_ego=cfg.dims.n_ego,
            n_act=cfg.dims.n_act,
            ego_dim=cfg.dims.ego_dim,
            image_size=cfg.vjepa.img_size,
            route_lookahead_m=float(getattr(cfg.data, "route_lookahead_m", 80.0)),
        )


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def preprocess_camera(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    """RGB uint8/float [H,W,3] -> normalized [3, image_size, image_size] float32.

    Crop the top 35% (sky/hood-free framing), center-crop to a square, resize, then
    ImageNet-normalize. This exact transform is shared by train and test.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "opencv-python is required for NAVSIM image preprocessing. Install with "
            "`uv pip install --python .venv/bin/python opencv-python`."
        ) from exc

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image.shape}")

    height, width = image.shape[:2]
    crop_y = int(round(height * 0.35))
    cropped = image[crop_y:, :, :]
    if cropped.shape[0] < 64:
        cropped = image

    crop_h, crop_w = cropped.shape[:2]
    if crop_w > crop_h:
        offset = (crop_w - crop_h) // 2
        cropped = cropped[:, offset : offset + crop_h, :]
    elif crop_h > crop_w:
        offset = (crop_h - crop_w) // 2
        cropped = cropped[offset : offset + crop_w, :, :]

    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr)


def _cam_f0_image(cameras_obj: Any) -> np.ndarray:
    image = getattr(cameras_obj.cam_f0, "image", None)
    if image is None:
        raise ValueError("CAM_F0 image not loaded; check sensor config (cam_f0 must be on)")
    return image


def _resample_to_n(items: list, n: int) -> list:
    """Nearest-index resample a list to exactly n elements (keeps endpoints)."""
    if len(items) == n:
        return items
    if not items:
        raise ValueError("empty frame list")
    idx = np.linspace(0, len(items) - 1, n)
    return [items[int(round(i))] for i in idx]


def build_camera_tensor(cameras_list: Sequence[Any], image_size: int, n_frames: int,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """List of NAVSIM ``Cameras`` -> [n_frames, 3, H, W]."""
    cameras_list = _resample_to_n(list(cameras_list), n_frames)
    frames = [preprocess_camera(_cam_f0_image(c), image_size).to(dtype) for c in cameras_list]
    return torch.stack(frames, dim=0)


# --------------------------------------------------------------------------- #
# Ego
# --------------------------------------------------------------------------- #
def build_ego_tensor(ego_statuses: Sequence[Any], n_ego: int, ego_dim: int = 7,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """List of NAVSIM ``EgoStatus`` -> [n_ego, ego_dim].

    Feature layout (ego_dim=7): [x, y, yaw, vx, vy, ax, ay].
    ego_pose is already local (current ego frame); vel/acc are taken verbatim.
    """
    statuses = _resample_to_n(list(ego_statuses), n_ego)
    rows = []
    for status in statuses:
        pose = np.asarray(status.ego_pose, dtype=np.float32)         # [x, y, yaw] local
        vel = np.asarray(status.ego_velocity, dtype=np.float32)      # [vx, vy]
        acc = np.asarray(status.ego_acceleration, dtype=np.float32)  # [ax, ay]
        row = [pose[0], pose[1], pose[2], vel[0], vel[1], acc[0], acc[1]]
        rows.append(row[:ego_dim])
    ego = np.asarray(rows, dtype=np.float32)
    # Last (current) frame is the origin of the local frame -> zero its pose.
    ego[-1, 0:3] = 0.0
    return torch.from_numpy(ego).to(dtype)


# --------------------------------------------------------------------------- #
# Route (from map + roadblock ids, transformed into current-ego local frame)
# --------------------------------------------------------------------------- #
def _se2_to_local(points_world_se2: np.ndarray, ego_pose_global: np.ndarray) -> np.ndarray:
    """Transform world [N,3] (x,y,heading) to local frame of ego_pose_global [x,y,yaw].

    Uses the SAME NAVSIM helper as ego/waypoint transforms so all frames match.
    """
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    origin = StateSE2(float(ego_pose_global[0]), float(ego_pose_global[1]), float(ego_pose_global[2]))
    return convert_absolute_to_relative_se2_array(origin, np.asarray(points_world_se2, dtype=np.float64))


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _roadblock_polyline_world(map_api: Any, roadblock_ids: Sequence[str],
                              ego_xy: np.ndarray) -> np.ndarray:
    """Stitch roadblock baseline paths into a world polyline [M,3] (x,y,heading)."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    chunks: list[np.ndarray] = []
    last_anchor_xy = np.asarray(ego_xy, dtype=np.float64)
    for roadblock_id in roadblock_ids:
        roadblock = None
        for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            try:
                roadblock = map_api.get_map_object(str(roadblock_id), layer)
            except Exception:
                roadblock = None
            if roadblock is not None:
                break
        if roadblock is None:
            continue

        best_lane, best_dist = None, math.inf
        for lane in roadblock.interior_edges:
            path = lane.baseline_path.discrete_path
            if not path:
                continue
            head = path[0]
            dist = (head.x - last_anchor_xy[0]) ** 2 + (head.y - last_anchor_xy[1]) ** 2
            if dist < best_dist:
                best_dist, best_lane = dist, lane
        if best_lane is None:
            continue

        path = best_lane.baseline_path.discrete_path
        chunk = np.asarray([(p.x, p.y, p.heading) for p in path], dtype=np.float64)
        if chunks and len(chunk) > 1:
            chunk = chunk[1:]  # avoid duplicating the joint point
        if len(chunk) > 0:
            chunks.append(chunk)
            last_anchor_xy = chunk[-1, :2]

    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _walk_forward(polyline_world: np.ndarray, ego_xy: np.ndarray):
    diffs = polyline_world[:, :2] - np.asarray(ego_xy)[None, :]
    closest = int(np.argmin((diffs * diffs).sum(axis=1)))
    forward = polyline_world[closest:]
    if len(forward) < 2:
        return forward, np.zeros((len(forward),), dtype=np.float64)
    seg = np.linalg.norm(np.diff(forward[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    return forward, cumulative


def _interp_at(forward: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    if distance >= cumulative[-1]:
        return forward[-1].copy()
    hi = int(np.searchsorted(cumulative, distance, side="right"))
    lo = max(0, hi - 1)
    denom = max(cumulative[hi] - cumulative[lo], 1e-6)
    alpha = float((distance - cumulative[lo]) / denom)
    xy = forward[lo, :2] * (1.0 - alpha) + forward[hi, :2] * alpha
    yaw = _wrap_to_pi(forward[lo, 2] + alpha * _wrap_to_pi(forward[hi, 2] - forward[lo, 2]))
    return np.asarray([xy[0], xy[1], yaw], dtype=np.float64)


def build_route_tensor(map_api: Optional[Any], roadblock_ids: Sequence[str],
                       ego_pose_global: np.ndarray, n_route: int,
                       lookahead_m: float = 80.0,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a [n_route, 2] route polyline in the current-ego local frame.

    Returns zeros if the map/roadblocks are unavailable — but this should be rare:
    both train and test feed the same scene, so both get the same (non-zero) route.
    """
    zeros = torch.zeros((n_route, 2), dtype=dtype)
    if map_api is None or not roadblock_ids:
        return zeros

    ego_xy = np.asarray(ego_pose_global, dtype=np.float64)[:2]
    polyline = _roadblock_polyline_world(map_api, roadblock_ids, ego_xy)
    if len(polyline) < 2:
        return zeros

    forward, cumulative = _walk_forward(polyline, ego_xy)
    if len(forward) < 2 or cumulative[-1] < 1.0:
        return zeros

    distances = np.linspace(lookahead_m / n_route, lookahead_m, n_route)
    route_world = np.stack([_interp_at(forward, cumulative, float(d)) for d in distances])
    route_local = _se2_to_local(route_world, ego_pose_global)[:, :2].astype(np.float32)
    return torch.from_numpy(route_local).to(dtype)


# --------------------------------------------------------------------------- #
# Future waypoints (training target)
# --------------------------------------------------------------------------- #
def build_future_waypoints(scene: Any, n_act: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Expert future xy in the current-ego local frame -> [n_act, 2]."""
    traj = scene.get_future_trajectory(num_trajectory_frames=n_act)
    poses = np.asarray(traj.poses, dtype=np.float32)  # [n_act, 3] (x, y, heading)
    if poses.shape[0] < n_act:
        raise ValueError(f"scene has {poses.shape[0]} future poses, need {n_act}")
    return torch.from_numpy(poses[:n_act, :2]).to(dtype)


# --------------------------------------------------------------------------- #
# Scene helpers (current-ego context)
# --------------------------------------------------------------------------- #
def _current_frame_index(scene: Any) -> int:
    return scene.scene_metadata.num_history_frames - 1


def _current_ego_global(scene: Any) -> np.ndarray:
    """Global [x, y, yaw] of the current ego (last history frame)."""
    status = scene.frames[_current_frame_index(scene)].ego_status
    return np.asarray(status.ego_pose, dtype=np.float64)


def _current_roadblock_ids(scene: Any) -> list[str]:
    return list(scene.frames[_current_frame_index(scene)].roadblock_ids or [])


# --------------------------------------------------------------------------- #
# Top-level builders (the ONLY entry points train & test should use)
# --------------------------------------------------------------------------- #
def build_inference_inputs(agent_input: Any, scene: Optional[Any], spec: FeatureSpec):
    """TEST path. Returns (past_cam [T,3,H,W], ego [N_ego,7], route [N_route,2]) — UNBATCHED.

    ``scene`` may be None (route falls back to zeros), but with requires_scene=True
    NAVSIM always provides it, matching the training route exactly.
    """
    past_cam = build_camera_tensor(agent_input.cameras, spec.image_size, spec.n_past)
    ego = build_ego_tensor(agent_input.ego_statuses, spec.n_ego, spec.ego_dim)
    if scene is not None:
        route = build_route_tensor(
            scene.map_api, _current_roadblock_ids(scene), _current_ego_global(scene),
            spec.n_route, spec.route_lookahead_m,
        )
    else:
        route = torch.zeros((spec.n_route, 2), dtype=torch.float32)
    return past_cam, ego, route


def build_training_sample(scene: Any, spec: FeatureSpec, store_dtype: torch.dtype = torch.float16) -> dict:
    """TRAIN path. Build one cached sample dict from a NAVSIM ``Scene``.

    past_cam/ego/route are produced by the SAME builders as :func:`build_inference_inputs`
    (via ``scene.get_agent_input()``), so the cached training distribution matches the
    test-time distribution by construction.
    """
    agent_input = scene.get_agent_input()
    past_cam, ego, route = build_inference_inputs(agent_input, scene, spec)

    # Future camera frames (world-model target) come from privileged scene frames.
    n_hist = scene.scene_metadata.num_history_frames
    future_cameras = [scene.frames[n_hist + k].cameras for k in range(spec.n_fut)]
    fut_cam = build_camera_tensor(future_cameras, spec.image_size, spec.n_fut)

    wp_gt = build_future_waypoints(scene, spec.n_act)

    sample = {
        "past_cam": past_cam.to(store_dtype),
        "fut_cam": fut_cam.to(store_dtype),
        "route": route.to(store_dtype),
        "ego": ego.to(store_dtype),
        "wp_gt": wp_gt.to(store_dtype),
        "meta": {
            "token": scene.scene_metadata.initial_token,
            "log_name": scene.scene_metadata.log_name,
            "map_name": scene.scene_metadata.map_name,
            "route_nonzero": bool(torch.count_nonzero(route).item() > 0),
        },
    }
    return sample
