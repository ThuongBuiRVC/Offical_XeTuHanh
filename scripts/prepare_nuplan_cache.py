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
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.navsim_features import FeatureSpec, build_training_sample


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
        shapes = {k: tuple(v.shape) for k, v in sample.items() if hasattr(v, "shape")}
        print("[dry-run] sample shapes:", shapes, flush=True)
        print("[dry-run] meta:", sample["meta"], flush=True)
        expected = {
            "past_cam": (spec.n_past, 3, spec.image_size, spec.image_size),
            "fut_cam": (spec.n_fut, 3, spec.image_size, spec.image_size),
            "route": (spec.n_route, 2),
            "ego": (spec.n_ego, spec.ego_dim),
            "wp_gt": (spec.n_act, 2),
        }
        for k, exp in expected.items():
            got = shapes.get(k)
            assert got == exp, f"{k}: got {got}, expected {exp}"
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
