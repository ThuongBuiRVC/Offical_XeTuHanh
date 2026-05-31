# Driving World Model

Single-stage joint training for a driving world-action model: frozen V-JEPA
encodes past and future video, a DiT with Shortcut Flow Matching predicts future
V-JEPA latents and action latents, and a jointly trained action decoder produces
future driving waypoints.

The implementation follows [`SPEC.md`](SPEC.md): one model, one optimizer, one
training phase. V-JEPA is frozen; the DiT, low-dimensional input embedders,
conditioning embedder, and action decoder are trained together.

## Repository Layout

| Path | Purpose |
|---|---|
| `scripts/prepare_nuplan_cache.py` | Build the training cache from NAVSIM scenes (shared features with eval). |
| `scripts/run_navsim_pdm_score.py` | Run official NAVSIM PDM scoring with the trained checkpoint. |
| `src/data/navsim_features.py` | **Shared** camera/ego/route feature builders — single source of truth for train & test. |
| `src/data/nuplan_dataset.py` | Load prepared cache files into the training contract. |
| `src/models/` | V-JEPA wrapper, input embedders, DiT, flow matching, action decoder, full model. |
| `src/train/` | Joint training loop and losses. |
| `src/eval/navsim_agent.py` | NAVSIM-compatible agent adapter for PDMS/EPDMS evaluation. |
| `src/inference/` | Shortcut ODE rollout for waypoint inference. |
| `src/viz/` | PCA visualization utilities for latent video debugging. |
| `tests/` | Shape, noising, joint-loss, and visualization checks. |

## Data Contract

Training reads prepared `.pt` samples, not raw NuPlan files. Each sample contains:

| Key | Shape | Description |
|---|---:|---|
| `past_cam` | `[8, 3, 384, 384]` | Past front-camera frames, ImageNet-normalized. |
| `fut_cam` | `[8, 3, 384, 384]` | Future front-camera frames for V-JEPA latent targets. |
| `route` | `[20, 2]` | Route lookahead waypoints in the current ego frame. |
| `ego` | `[8, 7]` | `[x, y, yaw, vx, vy, ax, ay]` history in the current ego frame. |
| `wp_gt` | `[8, 2]` | Expert future waypoints at `0.5s..4.0s`, meters, current ego frame. |

## Setup

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python -e .
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python opencv-python
```

Data prep and the eval agent both need the **NAVSIM devkit** importable
(`navsim`, `nuplan`). Export the standard NAVSIM env vars first:

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim   # has navsim_logs/ and sensor_blobs/
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps-v1.0/maps
```

## Prepare Data

Train, val, and test all read scenes through the **official NAVSIM `SceneLoader`**
(the exact path the PDM scorer uses) and the **same** feature builders in
[`src/data/navsim_features.py`](src/data/navsim_features.py), so the three splits
are processed identically by construction. Route comes from the map + roadblock
ids in the current-ego frame (NOT zeros), matching the agent.

- **train** and **val** are both prepared from the **trainval** logs, made
  disjoint by a stable token-hash holdout (`--holdout-frac`).
- **test** is NOT a cache — it is live PDM scoring over **navtest** (see below).

**Step 0 — verify wiring on ONE scene before a long run (`--dry-run`):**

```bash
.venv/bin/python scripts/prepare_nuplan_cache.py \
  --navsim-log-path  $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
  --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/trainval \
  --output-dir Data/nuplan_cache --split-name train --dry-run
```

It builds one sample end-to-end and asserts the shapes match the model contract.
No files are written. Fix any path/route issue here, not mid-training.

**Step 1 — full train (90%) and val (10%) from the SAME trainval logs:**

```bash
# train side (everything NOT in the 10% holdout)
.venv/bin/python scripts/prepare_nuplan_cache.py \
  --navsim-log-path  $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
  --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/trainval \
  --output-dir Data/nuplan_cache --split-name train \
  --num-history-frames 4 --num-future-frames 10 \
  --holdout-frac 0.1 --holdout-side keep

# val side (the 10% holdout) — same flags, just --holdout-side take
.venv/bin/python scripts/prepare_nuplan_cache.py \
  --navsim-log-path  $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
  --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/trainval \
  --output-dir Data/nuplan_cache --split-name val \
  --num-history-frames 4 --num-future-frames 10 \
  --holdout-frac 0.1 --holdout-side take
```

> **Parity rule:** `--num-history-frames` MUST equal the eval split's value
> (navtest = 4; the script warns otherwise). NAVSIM is 2 Hz; the shared builder
> resamples the loaded history/future to the model dims (`t_past=8`, `t_fut=8`).
> `--num-future-frames` must be ≥ max(`t_fut`, `n_act`).

The cache is written as:

```text
Data/nuplan_cache/
├── train/<token>.pt
├── val/<token>.pt
├── manifest_train.csv  manifest_val.csv
└── stats_train.json    stats_val.json   # includes route_nonzero_frac
```

> No NAVSIM data on this machine? The code paths are still exercised by the test
> suite (offline) and by `--smoke` (placeholder tensors). You only need real
> `navsim_logs`/`sensor_blobs` + maps when actually preparing/scoring.

## Train

Set `data.cache_root` in [`configs/train.yaml`](configs/train.yaml) to the cache
root, then run:

```bash
.venv/bin/python -m src.train.train --config configs/train.yaml
```

Validation runs during training and logs:

| Metric | Meaning |
|---|---|
| `val/total`, `val/flow_z`, `val/flow_a`, `val/wp`, `val/smooth`, `val/sc` | Training-objective validation losses. |
| `val/ol_ade`, `val/ol_fde`, `val/ol_ade@1s`, `val/ol_ade@2s`, `val/ol_ade@4s` | Open-loop waypoint error on the prepared validation cache. |

Checkpoints are written to `out_dir`:

| File | Use |
|---|---|
| `ckpt_last.pt` | Latest train checkpoint, includes model, EMA, optimizer, scheduler, config, and metrics. |
| `ckpt_best.pt` | Best checkpoint by validation total loss. |
| `ckpt_XXXXXXX.pt` | Periodic snapshots. |

To stop after a fixed number of epochs in the current run:

```bash
.venv/bin/python -m src.train.train --config configs/train.yaml --epochs 10
```

To resume training later:

```bash
.venv/bin/python -m src.train.train \
  --config configs/train.yaml \
  --resume logs/ckpt_last.pt \
  --epochs 10
```

`--epochs` means “train this many additional epochs in this run”. You can also
use `--max-steps N` for step-based runs.

For a code-only smoke run without NuPlan data:

```bash
PYTHONPATH= .venv/bin/python -m src.train.train --config configs/train.yaml --smoke
```

## NAVSIM PDMS Test

The prepared training cache is not enough for official PDMS. NAVSIM scoring
requires official `navsim_logs`, `sensor_blobs`, maps, and `MetricCache`.

For NAVSIM v1-style PDMS, run one-stage scoring:

```bash
.venv/bin/python scripts/run_navsim_pdm_score.py \
  --ckpt-path logs/ckpt_best.pt \
  --stage one_stage \
  --train-test-split navtest \
  --data-split test \
  --build-cache
```

Useful debug run before full test:

```bash
.venv/bin/python scripts/run_navsim_pdm_score.py \
  --ckpt-path logs/ckpt_best.pt \
  --stage one_stage \
  --max-scenes 100 \
  --build-cache
```

The script calls NAVSIM's `run_metric_caching.py` when `BUILD_CACHE=1`, then
calls `run_pdm_score_one_stage.py`. For newer NAVSIM two-stage EPDMS configs,
set `NAVSIM_STAGE=two_stage`.

> **Which interpreter:** `run_navsim_pdm_score.py` itself is pure stdlib, but it
> dispatches the actual NAVSIM scripts via `--python-bin` (default: the running
> interpreter). Our `.venv` does NOT have `navsim`/`nuplan`, so point it at the
> env that does, e.g. `--python-bin /path/to/navsim-env/bin/python`. The wrapper
> already injects this repo's root into `PYTHONPATH` so NAVSIM can import the
> agent at `src.eval.navsim_agent.WorldModelNavsimAgent`.

Each NAVSIM token calls the agent once. The model outputs 8 waypoints at 2 Hz
for a 4 second horizon; the adapter derives heading and returns a NAVSIM
`Trajectory [8, 3]`. NAVSIM interpolates this trajectory to the scorer frequency
and computes the PDM metrics.

## Smoke And Tests

```bash
PYTHONPATH= .venv/bin/python -m pytest tests -q
```

Tests use an offline V-JEPA stub and do not require downloading pretrained
weights.

Additional smoke checks used by this repo:

```bash
PYTHONPATH= .venv/bin/python -m py_compile \
  scripts/prepare_nuplan_cache.py src/train/train.py src/eval/navsim_agent.py

PYTHONPATH= .venv/bin/python -m src.train.train --config configs/train.yaml --smoke
```

## Notes

- `N_v` is inferred dynamically by the V-JEPA wrapper; it is not hardcoded.
- V-JEPA latents are passed directly into the DiT without projection.
- Route, ego, and waypoint tensors are low-dimensional inputs and are lifted to
  the DiT dimension with MLPs.
- Waypoint loss is applied through the decoded one-step clean action estimate,
  so gradients flow through the action velocity prediction into the DiT.
