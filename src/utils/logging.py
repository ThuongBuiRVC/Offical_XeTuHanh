"""Lightweight logging: console + JSONL, optional TensorBoard if available."""
from __future__ import annotations

import json
import time
from pathlib import Path


class Logger:
    def __init__(self, out_dir: str | Path, use_tb: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = (self.out_dir / "metrics.jsonl").open("a")
        self.t0 = time.time()
        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.out_dir / "tb"))
            except Exception as e:  # pragma: no cover
                print(f"[Logger] TensorBoard unavailable ({e}); JSONL only.")

    def log(self, step: int, metrics: dict, prefix: str = "train") -> None:
        flat = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
        record = {"step": step, "elapsed": round(time.time() - self.t0, 1), **flat}
        self.jsonl.write(json.dumps(record) + "\n")
        self.jsonl.flush()
        if self.tb:
            for k, v in flat.items():
                self.tb.add_scalar(k, v, step)
        msg = "  ".join(f"{k}={v:.4f}" for k, v in flat.items())
        print(f"[{step:>7}] ({prefix}) {msg}")

    def close(self) -> None:
        self.jsonl.close()
        if self.tb:
            self.tb.close()
