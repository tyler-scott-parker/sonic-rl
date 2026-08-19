#!/usr/bin/env python3
"""
smoke_all14.py -- verify special_stage_env reads ALL 14 stages correctly.

The back-7 (8-14) are S&K Super Emerald stages ($FFBB=1); the front-7 are
Sonic 3 Chaos stages. Same Blue Spheres engine, so the RAM layout SHOULD be
identical -- but verify before burning a long run.

For each stage: load it, reset, step a few frames, and report:
  blue    - starting blue_spheres_remaining (should match check_stages.py)
  total   - env's captured _stage_total (per-stage normalization)
  grid    - total grid-patch content + blue-channel content (must be non-zero)
  ram0/8  - normalized features 0 and 8 (must be sane, not NaN/negative)
  swap    - _ram_swapped flag

A stage is BAD if: blue is 0/None, grid is empty, or ram features are wrong.
"""
import os, sys, inspect, importlib
import numpy as np

os.environ.setdefault("SS_DROP_POSITION", "1")   # match training config

mod = importlib.import_module("special_stage_env")
import gymnasium as gym
EnvClass = next(o for _, o in vars(mod).items()
                if inspect.isclass(o) and issubclass(o, (gym.Env, gym.Wrapper))
                and o.__module__ == mod.__name__)

import special_stage_env as sse
if hasattr(sse, "RESTART_PROB"):
    sse.RESTART_PROB = 0.0

env = EnvClass()
if hasattr(env, "_log_episode"):
    env._log_episode = lambda *a, **k: None

print(f"{'stg':<5}{'blue':<7}{'total':<8}{'grid_sum':<10}{'grid_blue':<11}"
      f"{'ram0':<8}{'ram8':<8}{'swap':<6}verdict")
print("-" * 78)

bad = []
for sid in range(1, 15):
    sf = os.path.join("states", f"curriculum_stage_{sid}.state")
    if not os.path.exists(sf):
        print(f"{sid:<5}MISSING FILE")
        bad.append(sid)
        continue
    env.state_path = sf
    if hasattr(sse, "STAGE_FILES"):
        sse.STAGE_FILES = [sf]          # force this single stage on reset
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    info = {}
    for _ in range(4):
        obs, _, term, trunc, info = env.step(0)

    blue = info.get("blue_spheres_remaining")
    total = int(getattr(env, "_stage_total", 0))
    g = obs["grid"] if isinstance(obs, dict) else None
    gsum = float(g.sum()) if g is not None else -1
    gblue = float(g[:, :, 0].sum()) if g is not None else -1
    r = obs["ram"] if isinstance(obs, dict) else None
    ram0 = float(r[0]) if r is not None else float("nan")
    ram8 = float(r[8]) if r is not None else float("nan")
    swap = getattr(env, "_ram_swapped", None)

    ok = (isinstance(blue, (int, float)) and blue and blue > 0
          and total > 0 and gsum > 0
          and np.isfinite(ram0) and np.isfinite(ram8)
          and 0.0 <= ram0 <= 1.5 and -0.1 <= ram8 <= 1.5)
    verdict = "OK" if ok else "*** BAD ***"
    if not ok:
        bad.append(sid)
    print(f"{sid:<5}{str(blue):<7}{total:<8}{gsum:<10.1f}{gblue:<11.1f}"
          f"{ram0:<8.3f}{ram8:<8.3f}{str(swap):<6}{verdict}")

env.close()
print()
if bad:
    print(f">>> PROBLEM stages: {bad} -- do NOT launch the 14-board run until resolved.")
else:
    print(">>> ALL 14 READ CORRECTLY. Safe to launch the 14-board run.")
