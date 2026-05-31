"""Tests for the NAVSIM cache prep helpers (no navsim install required).

Covers the two things most likely to silently break train/val/test parity:
  1. SceneLoader signature introspection (navsim ships several signatures).
  2. Deterministic, disjoint train/val holdout split.
"""
import importlib.util
import sys
import types
from pathlib import Path

PREP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_nuplan_cache.py"


def _load_prep():
    spec = importlib.util.spec_from_file_location("prep_mod", PREP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    navsim_log_path = "/logs/trainval"
    sensor_blobs_path = "/blobs/trainval"
    synthetic_scenes_path = None
    num_history_frames = 4
    num_future_frames = 10
    max_scenes = 0
    log_names = None
    frame_interval = 1


SIGNATURES = {
    "orig": ["data_path", "original_sensor_path", "scene_filter",
             "synthetic_sensor_path", "synthetic_scenes_path", "sensor_config"],
    "blobs": ["sensor_blobs_path", "navsim_blobs_path", "data_path",
              "synthetic_scenes_path", "scene_filter", "sensor_config"],
    "logpath": ["navsim_log_path", "metadata_path", "scene_filter",
                "sensor_blobs_path", "sensor_config"],
    "minimal": ["data_path", "original_sensor_path", "scene_filter", "sensor_config"],
}


def test_sceneloader_introspection_all_signatures():
    prep = _load_prep()
    try:
        for params in SIGNATURES.values():
            src = ("def __init__(self, " + ", ".join(params) + "):\n"
                   "    self.kw = {" + ",".join(f"'{p}': {p}" for p in params) + "}\n")
            ns: dict = {}
            exec(src, ns)
            Fake = type("SceneLoader", (), {"__init__": ns["__init__"]})
            fake_mod = types.ModuleType("navsim.common.dataloader")
            fake_mod.SceneLoader = Fake
            sys.modules["navsim.common.dataloader"] = fake_mod

            obj = prep.build_scene_loader(_Args(), scene_filter="SF", sensor_config="SC")
            assert obj.kw["scene_filter"] == "SF"
            assert obj.kw["sensor_config"] == "SC"
            # every param the signature declared must be filled
            assert set(obj.kw) == set(params)
    finally:
        sys.modules.pop("navsim.common.dataloader", None)


def test_holdout_split_disjoint_and_deterministic():
    prep = _load_prep()
    tokens = [f"tok_{i:05d}" for i in range(10000)]
    train = set(prep.select_tokens(tokens, 0.1, "keep"))
    val = set(prep.select_tokens(tokens, 0.1, "take"))
    assert train.isdisjoint(val)
    assert train | val == set(tokens)
    assert 0.08 < len(val) / len(tokens) < 0.12
    # deterministic
    assert set(prep.select_tokens(tokens, 0.1, "take")) == val
    # no holdout -> keep everything
    assert len(prep.select_tokens(tokens, 0.0, "keep")) == len(tokens)


def test_token_bucket_stable():
    prep = _load_prep()
    assert prep.token_bucket("abc") == prep.token_bucket("abc")
    assert 0 <= prep.token_bucket("abc") < 1000
