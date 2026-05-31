#!/usr/bin/env python3
"""Run official NAVSIM PDMS/EPDMS for a trained checkpoint.

This is a thin Python wrapper around NAVSIM's own scripts:
  1. navsim/planning/script/run_metric_caching.py
  2. navsim/planning/script/run_pdm_score_one_stage.py or run_pdm_score.py

The wrapper keeps repo-specific paths, Hydra overrides, and checkpoint agent
configuration in one place without reimplementing the official scorer.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NAVSIM_ROOT = Path("/home/thuongbui/Project/Diffusionaction_jepa/Tools/navsim")
DEFAULT_NAVSIM_DATA = Path("/home/thuongbui/Project/Diffusionaction_jepa/Data/navsim")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def command_to_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def require_paths(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required path(s):\n{lines}")


def run_command(command: Sequence[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"[cmd] {command_to_text(command)}")
    if dry_run:
        return
    subprocess.run(list(command), env=env, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NAVSIM PDMS/EPDMS for this repo's checkpoint.")
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python interpreter used to run NAVSIM scripts.",
    )
    parser.add_argument(
        "--navsim-devkit-root",
        type=Path,
        default=env_path("NAVSIM_DEVKIT_ROOT", DEFAULT_NAVSIM_ROOT),
        help="Path to the official NAVSIM devkit root.",
    )
    parser.add_argument(
        "--openscene-data-root",
        type=Path,
        default=env_path("OPENSCENE_DATA_ROOT", DEFAULT_NAVSIM_DATA),
        help="Root containing navsim_logs/ and sensor_blobs/.",
    )
    parser.add_argument(
        "--navsim-exp-root",
        type=Path,
        default=env_path("NAVSIM_EXP_ROOT", ROOT_DIR / "logs/navsim_eval"),
        help="Directory where NAVSIM writes metric cache and score CSVs.",
    )
    parser.add_argument(
        "--nuplan-maps-root",
        type=Path,
        default=env_path("NUPLAN_MAPS_ROOT", ROOT_DIR / "Data/nuplan-maps-v1.0/maps"),
        help="Path to nuplan-maps-v1.0/maps.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=env_path("CKPT_PATH", ROOT_DIR / "logs/ckpt_best.pt"),
        help="Training checkpoint to evaluate.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=env_path("CONFIG_PATH", ROOT_DIR / "configs/train.yaml"),
        help="Training config used as fallback when checkpoint has no cfg.",
    )
    parser.add_argument(
        "--train-test-split",
        default=os.environ.get("TRAIN_TEST_SPLIT", "navtest"),
        help="NAVSIM train_test_split config name.",
    )
    parser.add_argument(
        "--data-split",
        default=os.environ.get("DATA_SPLIT", "test"),
        help="Subdirectory under navsim_logs/ and sensor_blobs/.",
    )
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        default=None,
        help="MetricCache output/input directory. Defaults to navsim-exp-root/metric_cache_<split>.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("EXPERIMENT_NAME"),
        help="NAVSIM experiment output name.",
    )
    parser.add_argument(
        "--stage",
        choices=("one_stage", "two_stage"),
        default=os.environ.get("NAVSIM_STAGE", "one_stage"),
        help="one_stage for NAVSIM v1 PDMS, two_stage for newer EPDMS configs.",
    )
    parser.add_argument(
        "--worker",
        default=os.environ.get("WORKER", "sequential"),
        help="NAVSIM worker config override.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=int(os.environ.get("MAX_SCENES", "0")),
        help="Limit scene count for debugging. 0 means full split.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=int(os.environ.get("IMAGE_SIZE", "384")),
        help="Image size passed to the model agent adapter.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        help="Device passed to the model agent adapter.",
    )
    parser.add_argument(
        "--build-cache",
        action=argparse.BooleanOptionalAction,
        default=env_bool("BUILD_CACHE", True),
        help="Run NAVSIM metric caching before scoring.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate required files/directories before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and environment, but do not execute NAVSIM scripts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    navsim_root = resolve(args.navsim_devkit_root)
    data_root = resolve(args.openscene_data_root)
    exp_root = resolve(args.navsim_exp_root)
    maps_root = resolve(args.nuplan_maps_root)
    ckpt_path = resolve(args.ckpt_path)
    config_path = resolve(args.config_path)
    metric_cache_path = resolve(args.metric_cache_path or exp_root / f"metric_cache_{args.train_test_split}")
    experiment_name = args.experiment_name or f"world_model_{args.train_test_split}"

    metric_script = navsim_root / "navsim/planning/script/run_metric_caching.py"
    one_stage_script = navsim_root / "navsim/planning/script/run_pdm_score_one_stage.py"
    two_stage_script = navsim_root / "navsim/planning/script/run_pdm_score.py"
    score_script = one_stage_script if args.stage == "one_stage" else two_stage_script

    required = [
        score_script,
        data_root / "navsim_logs" / args.data_split,
        data_root / "sensor_blobs" / args.data_split,
        ckpt_path,
        config_path,
    ]
    if args.build_cache:
        required.append(metric_script)
    if args.validate:
        require_paths(required)

    exp_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENSCENE_DATA_ROOT"] = str(data_root)
    env["NAVSIM_EXP_ROOT"] = str(exp_root)
    env["NUPLAN_MAPS_ROOT"] = str(maps_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT_DIR), str(navsim_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    common_args = [
        f"train_test_split={args.train_test_split}",
        f"worker={args.worker}",
        f"metric_cache_path={metric_cache_path}",
    ]
    if args.max_scenes > 0:
        common_args.append(f"train_test_split.scene_filter.max_scenes={args.max_scenes}")

    if args.dry_run:
        print(f"[env] OPENSCENE_DATA_ROOT={env['OPENSCENE_DATA_ROOT']}")
        print(f"[env] NAVSIM_EXP_ROOT={env['NAVSIM_EXP_ROOT']}")
        print(f"[env] NUPLAN_MAPS_ROOT={env['NUPLAN_MAPS_ROOT']}")
        print(f"[env] PYTHONPATH={env['PYTHONPATH']}")

    if args.build_cache:
        run_command([args.python_bin, str(metric_script), *common_args], env=env, dry_run=args.dry_run)

    score_args = [
        *common_args,
        "agent._target_=src.eval.navsim_agent.WorldModelNavsimAgent",
        f"+agent.ckpt_path={ckpt_path}",
        f"+agent.config_path={config_path}",
        f"+agent.image_size={args.image_size}",
        f"+agent.device={args.device}",
        f"experiment_name={experiment_name}",
    ]
    run_command([args.python_bin, str(score_script), *score_args], env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
