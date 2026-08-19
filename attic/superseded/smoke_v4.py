#!/usr/bin/env python3
"""
smoke_v4.py -- verify the v4 (pixel-free, 15x15 patch, 11 RAM feats) env reads
all 14 stages correctly BEFORE launching the run.

Checks per stage: obs has exactly keys {ram, grid} (NO pixels), ram is length 11,
grid is 15x15x4, blue count sane, features finite. Any BAD = do not launch.
"""
import os, inspect, importlib
import numpy as np
os.environ.setdefault("SS_DROP_POSITION", "1")

mod = importlib.import_module("special_stage_env_v4")
import gymnasium as gym
EnvClass = mod.SpecialStageEnv
import special_stage_env_v4 as sse
if hasattr(sse, "RESTART_PROB"):
    sse.RESTART_PROB = 0.0

env = EnvClass()
if hasattr(env, "_log_episode"):
    env._log_episode = lambda *a, **k: None

# confirm observation space shape first
sp = env.observation_space
print("obs space keys:", sorted(sp.spaces.keys()))
assert "pixels" not in sp.spaces, "PIXELS STILL PRESENT — v4 drop failed"
assert sp.spaces["ram"].shape == (11,), f"ram shape {sp.spaces['ram'].shape} != (11,)"
assert sp.spaces["grid"].shape == (15, 15, 4), f"grid shape {sp.spaces['grid'].shape} != (15,15,4)"
print("obs space OK: ram(11), grid(15x15x4), no pixels\n")

print(f"{'stg':<5}{'blue':<7}{'total':<8}{'ram_len':<9}{'grid_shape':<14}{'gsum':<9}verdict")
print("-" * 66)
bad = []
for sid in range(1, 15):
    sf = os.path.join("states", f"curriculum_stage_{sid}.state")
    if not os.path.exists(sf):
        print(f"{sid:<5}MISSING"); bad.append(sid); continue
    env.state_path = sf
    if hasattr(sse, "STAGE_FILES"):
        sse.STAGE_FILES = [sf]
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    info = {}
    for _ in range(4):
        obs, _, term, trunc, info = env.step(0)
    keys_ok = set(obs.keys()) == {"ram", "grid"}
    r = obs["ram"]; g = obs["grid"]
    blue = info.get("blue_spheres_remaining")
    total = int(getattr(env, "_stage_total", 0))
    gsum = float(g.sum())
    ok = (keys_ok and r.shape == (11,) and g.shape == (15,15,4)
          and isinstance(blue,(int,float)) and blue and blue > 0
          and total > 0 and gsum > 0 and np.all(np.isfinite(r)))
    if not ok: bad.append(sid)
    print(f"{sid:<5}{str(blue):<7}{total:<8}{r.shape[0]:<9}{str(g.shape):<14}{gsum:<9.1f}"
          + ("OK" if ok else "*** BAD ***"))
env.close()
print()
print(">>> ALL 14 READ CORRECTLY. Safe to launch v4." if not bad
      else f">>> PROBLEM stages: {bad} — do NOT launch until resolved.")
