"""Single-stage joint training loop with checkpoint/resume and validation metrics."""
from __future__ import annotations

import argparse
import copy
import math
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.data.nuplan_dataset import build_dataset
from src.inference.rollout import rollout
from src.models.flow_matching import compute_shortcut_targets, sample_t_d
from src.models.full_model import FullModel
from src.train.losses import LossWeights, compute_losses
from src.utils.ema import EMA
from src.utils.logging import Logger


def cfg_to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: cfg_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cfg_to_plain(v) for v in value]
    return value


def set_nested(cfg: Config, dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for key in parts[:-1]:
        cur = getattr(cur, key)
    cur[parts[-1]] = value


def build_optimizer(model: FullModel, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )


def lr_lambda(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def move_batch(batch: dict, device: str) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def amp_dtype_of(cfg):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.optim.amp_dtype]


def forward_and_loss(model: FullModel, batch: dict, weights: LossWeights, cfg, device: str):
    batch_size = batch["past_cam"].shape[0]
    t, d, mask_sc = sample_t_d(
        batch_size, device, d_values=cfg.flow.d_values, shortcut_frac=cfg.flow.shortcut_frac
    )
    out = model(
        batch["past_cam"],
        batch["fut_cam"],
        batch["route"],
        batch["ego"],
        batch["wp_gt"],
        t,
        d,
    )

    sc_target_z = sc_target_a = None
    if mask_sc.any():
        past_ctx = out["past_ctx"]
        x_t = torch.cat([out["z_t"], out["a_t"]], dim=1)
        v_sc = compute_shortcut_targets(
            lambda seq, tt, dd: model.core_velocity(seq, past_ctx, tt, dd), x_t, t, d
        )
        sc_target_z = v_sc[:, : model.n_v, :]
        sc_target_a = v_sc[:, model.n_v : model.n_v + model.n_act, :]

    return compute_losses(out, batch["wp_gt"], weights, mask_sc, sc_target_z, sc_target_a)


def ade_fde(pred: torch.Tensor, gt: torch.Tensor, horizons=(1.0, 2.0, 4.0), dt: float = 0.5) -> dict:
    dist = torch.linalg.norm(pred - gt, dim=-1)
    metrics = {
        "ade": float(dist.mean()),
        "fde": float(dist[:, -1].mean()),
    }
    for horizon in horizons:
        k = min(max(1, int(round(horizon / dt))), dist.shape[1])
        metrics[f"ade@{horizon:g}s"] = float(dist[:, :k].mean())
    return metrics


@torch.no_grad()
def evaluate_open_loop(model: FullModel, val_dl, cfg, device: str, max_batches: int) -> dict:
    model.eval()
    sums: dict[str, float] = {}
    seen = 0
    for batch_idx, batch in enumerate(val_dl):
        if batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        pred, _ = rollout(
            model,
            batch["past_cam"],
            batch["route"],
            batch["ego"],
            steps=cfg.inference.ode_steps,
            solver=cfg.inference.solver,
        )
        metrics = ade_fde(pred.float().cpu(), batch["wp_gt"].float().cpu())
        bs = pred.shape[0]
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * bs
        seen += bs
    return {key: value / max(1, seen) for key, value in sums.items()}


@torch.no_grad()
def validate(model, ema, val_dl, weights, cfg, device: str, logger, step: int) -> dict:
    model.eval()
    val_batches = int(getattr(cfg.log, "val_batches", 8))
    agg: dict[str, float] = {}
    n = 0
    with EMA.swap(ema, model):
        for batch in val_dl:
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=amp_dtype_of(cfg),
                enabled=(device != "cpu"),
            ):
                losses = forward_and_loss(model, batch, weights, cfg, device)
            for key, value in losses.items():
                agg[key] = agg.get(key, 0.0) + float(value)
            n += 1
            if n >= val_batches:
                break
        agg = {key: value / max(1, n) for key, value in agg.items()}
        if int(getattr(cfg.log, "open_loop_batches", 0)) > 0:
            ol = evaluate_open_loop(model, val_dl, cfg, device, int(cfg.log.open_loop_batches))
            agg.update({f"ol_{key}": value for key, value in ol.items()})
    logger.log(step, agg, prefix="val")
    model.train()
    return agg


def save_ckpt(out_dir, name, step, model, ema, optim, sched, cfg, metrics):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "cfg": cfg_to_plain(cfg),
        "metrics": metrics,
    }
    path = out_dir / name
    torch.save(ckpt, path)
    print(f"[ckpt] saved {path}")


def load_ckpt(path, model, ema, optim=None, sched=None, device="cpu") -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if sched is not None and "sched" in ckpt:
        sched.load_state_dict(ckpt["sched"])
    step = int(ckpt.get("step", 0))
    print(f"[resume] loaded {path} at step={step}")
    return step


def load_scheduler_state(path, sched, device="cpu") -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "sched" in ckpt:
        sched.load_state_dict(ckpt["sched"])


def train(cfg, resume: str | None = None, epochs: int | None = None, max_steps_override: int | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)

    model = FullModel.from_config(cfg).to(device)
    model.train()
    print(
        f"[train] n_v={model.n_v} trainable="
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M"
    )

    weights = LossWeights.from_config(cfg)
    optim = build_optimizer(model, cfg)

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        pin_memory=(device != "cpu"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        num_workers=max(0, min(2, int(cfg.data.num_workers))),
        pin_memory=(device != "cpu"),
    )
    if len(train_dl) == 0:
        raise RuntimeError("empty train dataloader")

    ema = EMA(model, decay=cfg.optim.ema_decay)

    start_step = 0
    if resume:
        start_step = load_ckpt(resume, model, ema, optim, None, device)

    steps_per_epoch = max(1, math.ceil(len(train_dl) / max(1, int(cfg.optim.grad_accum))))
    max_steps = int(max_steps_override or cfg.optim.max_steps)
    if epochs is not None:
        max_steps = start_step + int(epochs) * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_lambda(s, cfg.optim.warmup_steps, max_steps)
    )
    if resume:
        load_scheduler_state(resume, sched, device)

    data_iter = cycle(train_dl)
    accum = int(cfg.optim.grad_accum)
    dev_type = device.split(":")[0]
    best_val = math.inf

    for step in range(start_step + 1, max_steps + 1):
        optim.zero_grad(set_to_none=True)
        step_losses: dict[str, float] = {}
        for _ in range(accum):
            batch = move_batch(next(data_iter), device)
            with torch.autocast(device_type=dev_type, dtype=amp_dtype_of(cfg), enabled=(device != "cpu")):
                losses = forward_and_loss(model, batch, weights, cfg, device)
            (losses["total"] / accum).backward()
            for key, value in losses.items():
                step_losses[key] = step_losses.get(key, 0.0) + float(value) / accum

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.optim.grad_clip
        )
        optim.step()
        sched.step()
        ema.update(model)

        if step % cfg.log.log_every == 0 or step == 1:
            step_losses["lr"] = sched.get_last_lr()[0]
            logger.log(step, step_losses, prefix="train")

        end_of_epoch = (step % steps_per_epoch) == 0
        should_validate = step % cfg.log.val_every == 0 or end_of_epoch or step == max_steps
        latest_val = {}
        if should_validate:
            latest_val = validate(model, ema, val_dl, weights, cfg, device, logger, step)
            if latest_val.get("total", math.inf) < best_val:
                best_val = latest_val["total"]
                save_ckpt(out_dir, "ckpt_best.pt", step, model, ema, optim, sched, cfg, latest_val)
        if step % cfg.log.ckpt_every == 0 or end_of_epoch or step == max_steps:
            save_ckpt(out_dir, "ckpt_last.pt", step, model, ema, optim, sched, cfg, latest_val)
            save_ckpt(out_dir, f"ckpt_{step:07d}.pt", step, model, ema, optim, sched, cfg, latest_val)

    logger.close()


def make_smoke_cfg(cfg: Config) -> Config:
    cfg = Config(copy.deepcopy(cfg_to_plain(cfg)))
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.out_dir = "logs/smoke"
    cfg.data.use_placeholder = True
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.data.placeholder_train_size = 4
    cfg.data.placeholder_val_size = 2
    cfg.vjepa.force_fallback = True
    cfg.vjepa.pretrained = False
    cfg.vjepa.img_size = 96
    cfg.dit.num_layers = 1
    cfg.optim.grad_accum = 1
    cfg.optim.max_steps = 2
    cfg.log.log_every = 1
    cfg.log.val_every = 1
    cfg.log.ckpt_every = 2
    cfg.log.val_batches = 1
    cfg.log.open_loop_batches = 1
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None, help="Path to ckpt_last.pt/ckpt_*.pt to continue training.")
    parser.add_argument("--epochs", type=int, default=None, help="Train for this many epochs in this run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override optim.max_steps for this run.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny placeholder-data train/val smoke test.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = make_smoke_cfg(cfg)
        args.epochs = None
        args.max_steps = 2
    train(cfg, resume=args.resume, epochs=args.epochs, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
