#!/usr/bin/env python3
"""Prepare a training/validation cache from NAVSIM scenes — train/val/test parity.

Reads scenes through the **official NAVSIM** ``SceneLoader`` (the exact data path
the PDM scorer uses at test time) and turns each scene into a compact per-sample
``.pt`` via ``src.data.navsim_features.build_training_sample`` — the SAME feature
builders the eval agent (``src/eval/navsim_agent.py``) uses. So train, val, and
test all consume identical camera/ego/route tensors by construction.

Splits (NAVSIM / OpenScene convention):
    - ``train`` and ``val`` are both prepared from the **trainval** logs+sensors.
      They are made disjoint deterministically by hashing the scene token
      (``--holdout-frac`` goes to val). Both are processed the *same* way.
    - ``test`` is NOT a cache: it is the live PDM scoring over **navtest** via
      ``scripts/run_navsim_pdm_score.py`` (the agent reuses the same builders).

Frame counts (must match the eval split):
    NAVSIM is 2 Hz. ``--num-history-frames`` MUST equal the eval split value
    (navtest = 4). The shared builder resamples whatever count NAVSIM provides to
    the model dims (t_past, t_fut), so identical counts => identical tensors.
    ``--num-future-frames`` must be >= max(t_fut, n_act).

Output layout:
    <output_dir>/<split>/<token>.pt
    <output_dir>/manifest_<split>.csv
    <output_dir>/stats_<split>.json

Run (inside the navsim env, with OPENSCENE_DATA_ROOT / NUPLAN_MAPS_ROOT exported):
    # quick wiring check on ONE scene (no files written):
    python scripts/prepare_nuplan_cache.py \
        --navsim-log-path  $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
        --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/trainval \
        --output-dir Data/nuplan_cache --split-name train --dry-run

    # full train cache (90%) + val cache (10%) from the same trainval logs:
    python scripts/prepare_nuplan_cache.py ... --split-name train --holdout-frac 0.1 --holdout-side keep
    python scripts/prepare_nuplan_cache.py ... --split-name val   --holdout-frac 0.1 --holdout-side take
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

from src.config import load_config
from src.data.navsim_features import FeatureSpec, build_training_sample
from src.data.navsim_features import preprocess_camera


# --------------------------------------------------------------------------- #
# NAVSIM object construction (introspect signatures so we survive version drift)
# --------------------------------------------------------------------------- #
def build_scene_filter(args):
    """Build a SceneFilter, keeping only kwargs the installed version accepts."""
    from navsim.common.dataclasses import SceneFilter

    pool = {
        "num_history_frames": args.num_history_frames,
        "num_future_frames": args.num_future_frames,
        "has_route": True,                       # only keep scenes that HAVE a route
        "max_scenes": args.max_scenes if args.max_scenes > 0 else None,
        "log_names": args.log_names,
        "frame_interval": args.frame_interval,
    }
    accepted = set(inspect.signature(SceneFilter.__init__).parameters)
    return SceneFilter(**{k: v for k, v in pool.items() if k in accepted})


def build_scene_loader(args, scene_filter, sensor_config):
    """Build a SceneLoader by INTROSPECTING its signature.

    navsim has shipped multiple SceneLoader signatures (data_path vs
    navsim_log_path; original_sensor_path vs sensor_blobs_path; plus synthetic_*).
    We map our two real paths onto whatever the installed version accepts so prep
    never breaks on a parameter-name mismatch.
    """
    from navsim.common.dataloader import SceneLoader

    log_path = Path(args.navsim_log_path)
    sensor_path = Path(args.sensor_blobs_path)
    synth_scenes = Path(args.synthetic_scenes_path) if args.synthetic_scenes_path else log_path

    pool = {
        # log / metadata location
        "data_path": log_path,
        "navsim_log_path": log_path,
        "metadata_path": log_path,
        # original (real) sensor blobs
        "original_sensor_path": sensor_path,
        "sensor_blobs_path": sensor_path,
        "navsim_blobs_path": sensor_path,
        # synthetic (unused; pass sensible defaults if required)
        "synthetic_sensor_path": sensor_path,
        "synthetic_scenes_path": synth_scenes,
        # always present
        "scene_filter": scene_filter,
        "sensor_config": sensor_config,
    }
    params = [p for p in inspect.signature(SceneLoader.__init__).parameters if p != "self"]
    missing = [p for p in params if p not in pool]
    if missing:
        raise RuntimeError(
            f"SceneLoader needs parameters this script can't fill: {missing}. "
            f"Inspect navsim.common.dataloader.SceneLoader and extend the pool."
        )
    kwargs = {p: pool[p] for p in params}
    print(f"[prepare] SceneLoader kwargs: {sorted(kwargs)}", flush=True)
    return SceneLoader(**kwargs)


def front_camera_sensor_config():
    """SensorConfig that loads CAM_F0 at EVERY frame (history + future)."""
    from navsim.common.dataclasses import SensorConfig

    return SensorConfig(
        cam_f0=True,
        cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False,
        cam_b0=False, lidar_pc=False,
    )


# --------------------------------------------------------------------------- #
# Deterministic train/val holdout (same logs, disjoint, identical processing)
# --------------------------------------------------------------------------- #
def token_bucket(token: str, buckets: int = 1000) -> int:
    """Stable hash of a token into [0, buckets)."""
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % buckets


def select_tokens(tokens, holdout_frac: float, holdout_side: str):
    """Split tokens deterministically. 'take' = the val side, 'keep' = the train side."""
    if holdout_frac <= 0.0:
        return list(tokens)
    cutoff = int(round(holdout_frac * 1000))
    out = []
    for t in tokens:
        in_holdout = token_bucket(t) < cutoff
        if holdout_side == "take" and in_holdout:
            out.append(t)
        elif holdout_side == "keep" and not in_holdout:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Direct nuPlan DB fallback (for local DB + camera shard smoke/prep)
# --------------------------------------------------------------------------- #
def has_nuplan_db_logs(log_path: Path) -> bool:
    return log_path.is_dir() and any(log_path.glob("*.db"))


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    return float(Quaternion(qw, qx, qy, qz).yaw_pitch_roll[0])


def rotation_world_to_local(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, s], [-s, c]], dtype=np.float32)


def resample_indices(length: int, n: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("empty sequence")
    if length == n:
        return np.arange(n)
    return np.rint(np.linspace(0, length - 1, n)).astype(np.int64)


def candidate_camera_roots(sensor_path: Path) -> list[Path]:
    if sensor_path.name.startswith("nuplan-v1.1_"):
        roots = sorted(sensor_path.parent.glob("nuplan-v1.1_*_camera_*"))
        return [sensor_path] + [root for root in roots if root != sensor_path]
    roots = sorted(sensor_path.glob("nuplan-v1.1_*_camera_*"))
    return roots if roots else [sensor_path]


def resolve_camera_path(sensor_path: Path, rel_path: str, cache: dict[str, Path]) -> Path:
    log_name = rel_path.split("/", 1)[0]
    if log_name in cache:
        path = cache[log_name] / rel_path
        if path.exists():
            return path
    for root in candidate_camera_roots(sensor_path):
        path = root / rel_path
        if path.exists():
            cache[log_name] = root
            return path
    raise FileNotFoundError(f"camera image not found for {rel_path} under {sensor_path}")


def load_rgb_image(path: Path) -> np.ndarray:
    import cv2

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def query_cam_f0_rows(db_path: Path) -> list[dict]:
    query = """
        select image.filename_jpg, image.timestamp,
               ego_pose.x, ego_pose.y, ego_pose.qw, ego_pose.qx, ego_pose.qy, ego_pose.qz,
               ego_pose.vx, ego_pose.vy, ego_pose.acceleration_x, ego_pose.acceleration_y
        from image
        join camera on image.camera_token = camera.token
        join ego_pose on image.ego_pose_token = ego_pose.token
        where camera.channel = 'CAM_F0'
        order by image.timestamp
    """
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute(query)]


def build_db_sample(db_path: Path, sensor_path: Path, spec: FeatureSpec, store_dtype: torch.dtype) -> dict:
    rows = query_cam_f0_rows(db_path)
    required_frames = spec.n_past + max(spec.n_fut, spec.n_act)
    if len(rows) < required_frames:
        raise ValueError(f"{db_path.name}: only {len(rows)} CAM_F0 frames, need {required_frames}")

    # nuPlan image DBs are ~10Hz. Use every 5th frame for NAVSIM's 2Hz convention.
    stride = 5
    usable = rows[::stride]
    if len(usable) < required_frames:
        stride = max(1, len(rows) // required_frames)
        usable = rows[::stride]
    if len(usable) < required_frames:
        raise ValueError(f"{db_path.name}: only {len(usable)} sampled frames, need {required_frames}")

    current_idx = spec.n_past - 1
    selected = usable[:required_frames]
    past_rows = selected[:spec.n_past]
    future_rows = selected[spec.n_past:spec.n_past + spec.n_fut]
    wp_rows = selected[spec.n_past:spec.n_past + spec.n_act]
    current = selected[current_idx]

    cache: dict[str, Path] = {}
    past_cam = torch.stack(
        [
            preprocess_camera(
                load_rgb_image(resolve_camera_path(sensor_path, row["filename_jpg"], cache)),
                spec.image_size,
            ).to(store_dtype)
            for row in past_rows
        ],
        dim=0,
    )
    fut_cam = torch.stack(
        [
            preprocess_camera(
                load_rgb_image(resolve_camera_path(sensor_path, row["filename_jpg"], cache)),
                spec.image_size,
            ).to(store_dtype)
            for row in future_rows
        ],
        dim=0,
    )

    ref_xy = np.asarray([current["x"], current["y"]], dtype=np.float32)
    ref_yaw = yaw_from_quaternion(current["qw"], current["qx"], current["qy"], current["qz"])
    rot = rotation_world_to_local(ref_yaw)

    ego_source = [selected[i] for i in resample_indices(spec.n_past, spec.n_ego)]
    ego_rows = []
    for row in ego_source:
        xy = rot @ (np.asarray([row["x"], row["y"]], dtype=np.float32) - ref_xy)
        yaw = yaw_from_quaternion(row["qw"], row["qx"], row["qy"], row["qz"]) - ref_yaw
        vel = rot @ np.asarray([row["vx"], row["vy"]], dtype=np.float32)
        acc = rot @ np.asarray([row["acceleration_x"], row["acceleration_y"]], dtype=np.float32)
        ego_rows.append([xy[0], xy[1], yaw, vel[0], vel[1], acc[0], acc[1]][:spec.ego_dim])
    ego = torch.as_tensor(np.asarray(ego_rows, dtype=np.float32), dtype=store_dtype)
    ego[-1, 0:3] = 0

    wp = []
    for row in wp_rows:
        xy = rot @ (np.asarray([row["x"], row["y"]], dtype=np.float32) - ref_xy)
        wp.append(xy)
    wp_gt = torch.as_tensor(np.asarray(wp, dtype=np.float32), dtype=store_dtype)

    return {
        "past_cam": past_cam,
        "fut_cam": fut_cam,
        "route": torch.zeros((spec.n_route, 2), dtype=store_dtype),
        "ego": ego,
        "wp_gt": wp_gt,
        "meta": {
            "token": db_path.stem,
            "log_name": db_path.stem,
            "map_name": "unknown",
            "route_nonzero": False,
        },
    }


def validate_sample_shapes(sample: dict, spec: FeatureSpec) -> dict[str, tuple]:
    shapes = {k: tuple(v.shape) for k, v in sample.items() if hasattr(v, "shape")}
    expected = {
        "past_cam": (spec.n_past, 3, spec.image_size, spec.image_size),
        "fut_cam": (spec.n_fut, 3, spec.image_size, spec.image_size),
        "route": (spec.n_route, 2),
        "ego": (spec.n_ego, spec.ego_dim),
        "wp_gt": (spec.n_act, 2),
    }
    for key, exp in expected.items():
        got = shapes.get(key)
        assert got == exp, f"{key}: got {got}, expected {exp}"
    return shapes


def prepare_from_nuplan_dbs(args, spec: FeatureSpec, store_dtype: torch.dtype) -> bool:
    db_files = sorted(Path(args.navsim_log_path).glob("*.db"))
    if args.log_names:
        names = {name.removesuffix(".db") for name in args.log_names}
        db_files = [path for path in db_files if path.stem in names]
    if args.max_scenes > 0:
        db_files = db_files[: args.max_scenes]
    if not db_files:
        raise RuntimeError(f"no .db files found in {args.navsim_log_path}")

    print(f"[prepare-db] split={args.split_name} dbs={len(db_files)}", flush=True)
    sensor_path = Path(args.sensor_blobs_path)

    if args.dry_run:
        sample = build_db_sample(db_files[0], sensor_path, spec, store_dtype)
        shapes = validate_sample_shapes(sample, spec)
        print("[dry-run] sample shapes:", shapes, flush=True)
        print("[dry-run] meta:", sample["meta"], flush=True)
        print("[dry-run] OK — DB sample matches the model contract, no files written.", flush=True)
        return True

    split_dir = args.output_dir / args.split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"manifest_{args.split_name}.csv"
    written = skipped = 0
    wp_values: list[np.ndarray] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        manifest = csv.DictWriter(f, fieldnames=["token", "log_name", "map_name", "route_nonzero"])
        manifest.writeheader()
        for db_path in db_files:
            out_path = split_dir / f"{db_path.stem}.pt"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                sample = build_db_sample(db_path, sensor_path, spec, store_dtype)
                validate_sample_shapes(sample, spec)
            except Exception as exc:
                print(f"[warn] skip {db_path.name}: {exc}", flush=True)
                skipped += 1
                continue
            torch.save(sample, out_path)
            manifest.writerow(sample["meta"])
            wp_values.append(sample["wp_gt"].float().numpy())
            written += 1

    stats = {
        "split": args.split_name,
        "num_samples": written,
        "num_skipped": skipped,
        "route_nonzero_frac": 0.0,
        "num_history_frames": args.num_history_frames,
        "num_future_frames": args.num_future_frames,
        "t_past": spec.n_past, "t_fut": spec.n_fut, "n_route": spec.n_route,
        "n_ego": spec.n_ego, "n_act": spec.n_act, "ego_dim": spec.ego_dim,
        "image_size": spec.image_size,
        "source": "nuplan_db",
    }
    if wp_values:
        wp = np.stack(wp_values, axis=0)
        stats["wp_gt_mean"] = wp.mean(axis=(0, 1)).tolist()
        stats["wp_gt_std"] = np.maximum(wp.std(axis=(0, 1)), 1e-6).tolist()
    with (args.output_dir / f"stats_{args.split_name}.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[done] wrote {written} DB samples (skipped {skipped}) to {split_dir}", flush=True)
    print("[warn] DB fallback writes zero route because route roadblocks are not available in raw DBs.", flush=True)
    return True


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Prepare a NAVSIM cache (train/val parity with test).")
    p.add_argument("--navsim-log-path", required=True, help="navsim_logs/<split> dir (use trainval).")
    p.add_argument("--sensor-blobs-path", required=True, help="sensor_blobs/<split> dir (use trainval).")
    p.add_argument("--output-dir", required=True, type=Path, help="cache output root.")
    p.add_argument("--split-name", default="train", help="subdir to write (train/val).")
    p.add_argument("--config", default="configs/train.yaml", help="config for model dims.")
    p.add_argument("--num-history-frames", type=int, default=4,
                   help="MUST equal the eval split value (navtest=4).")
    p.add_argument("--num-future-frames", type=int, default=10,
                   help="must be >= max(t_fut, n_act).")
    p.add_argument("--frame-interval", type=int, default=1)
    p.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes.")
    p.add_argument("--log-names", nargs="+", default=None)
    p.add_argument("--synthetic-scenes-path", default=None,
                   help="Only used if the installed SceneLoader requires it.")
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="Fraction of tokens reserved for val (by stable token hash).")
    p.add_argument("--holdout-side", choices=("keep", "take"), default="keep",
                   help="keep=train side (1-frac); take=val side (frac).")
    p.add_argument("--store-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Build loader, process ONE scene, print shapes, write nothing. "
                        "Run this first to confirm wiring BEFORE a full prepare/train.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    spec = FeatureSpec.from_config(cfg)
    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32

    need_future = max(spec.n_fut, spec.n_act)
    if args.num_future_frames < need_future:
        raise ValueError(
            f"--num-future-frames={args.num_future_frames} < required {need_future} "
            f"(max of t_fut={spec.n_fut}, n_act={spec.n_act})"
        )
    if args.num_history_frames < 1:
        raise ValueError("--num-history-frames must be >= 1")
    if args.num_history_frames != 4:
        print(f"[warn] --num-history-frames={args.num_history_frames} but the navtest "
              f"benchmark uses 4. Train/test will only match if this equals the eval "
              f"split's value. Set 4 unless you know the eval split differs.", flush=True)

    if has_nuplan_db_logs(Path(args.navsim_log_path)):
        prepare_from_nuplan_dbs(args, spec, store_dtype)
        return

    scene_filter = build_scene_filter(args)
    loader = build_scene_loader(args, scene_filter, front_camera_sensor_config())

    all_tokens = list(loader.tokens)
    tokens = select_tokens(all_tokens, args.holdout_frac, args.holdout_side)
    print(f"[prepare] split={args.split_name} tokens={len(tokens)}/{len(all_tokens)} "
          f"hist={args.num_history_frames} fut={args.num_future_frames} "
          f"holdout={args.holdout_frac}({args.holdout_side})", flush=True)

    if not tokens:
        raise RuntimeError(
            "0 tokens selected. Check --navsim-log-path / --sensor-blobs-path, "
            "has_route filtering, and --holdout-frac/--holdout-side."
        )

    if args.dry_run:
        scene = loader.get_scene_from_token(tokens[0])
        sample = build_training_sample(scene, spec, store_dtype=store_dtype)
        shapes = validate_sample_shapes(sample, spec)
        print("[dry-run] sample shapes:", shapes, flush=True)
        print("[dry-run] meta:", sample["meta"], flush=True)
        print("[dry-run] OK — shapes match the model contract, no files written.", flush=True)
        return

    split_dir = args.output_dir / args.split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"manifest_{args.split_name}.csv"

    written = skipped = route_nonzero = 0
    wp_values: list[np.ndarray] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        manifest = csv.DictWriter(f, fieldnames=["token", "log_name", "map_name", "route_nonzero"])
        manifest.writeheader()

        for i, token in enumerate(tokens):
            out_path = split_dir / f"{token}.pt"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                scene = loader.get_scene_from_token(token)
                sample = build_training_sample(scene, spec, store_dtype=store_dtype)
            except Exception as exc:
                print(f"[warn] skip {token}: {exc}", flush=True)
                skipped += 1
                continue

            torch.save(sample, out_path)
            manifest.writerow(sample["meta"])
            written += 1
            route_nonzero += int(sample["meta"]["route_nonzero"])
            wp_values.append(sample["wp_gt"].float().numpy())

            if (i + 1) % 200 == 0:
                print(f"[prepare] {i + 1}/{len(tokens)} written={written} skipped={skipped}",
                      flush=True)

    stats = {
        "split": args.split_name,
        "num_samples": written,
        "num_skipped": skipped,
        "route_nonzero_frac": (route_nonzero / written) if written else 0.0,
        "num_history_frames": args.num_history_frames,
        "num_future_frames": args.num_future_frames,
        "t_past": spec.n_past, "t_fut": spec.n_fut, "n_route": spec.n_route,
        "n_ego": spec.n_ego, "n_act": spec.n_act, "ego_dim": spec.ego_dim,
        "image_size": spec.image_size,
    }
    if wp_values:
        wp = np.stack(wp_values, axis=0)
        stats["wp_gt_mean"] = wp.mean(axis=(0, 1)).tolist()
        stats["wp_gt_std"] = np.maximum(wp.std(axis=(0, 1)), 1e-6).tolist()
    with (args.output_dir / f"stats_{args.split_name}.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[done] wrote {written} samples (skipped {skipped}) to {split_dir}", flush=True)
    print(f"[done] route_nonzero_frac={stats['route_nonzero_frac']:.3f}", flush=True)
    if written and stats["route_nonzero_frac"] < 0.5:
        print("[warn] many routes are zero — check NUPLAN_MAPS_ROOT and has_route.", flush=True)


if __name__ == "__main__":
    main()
