#!/usr/bin/env python3
"""
transfer_probe.py -- zero-shot transfer test for the Stage-1 policy.

Drops the trained policy into other special stages FROM SPHERE ZERO (fresh
starts, no restart scaffolding, NO training) and reports how far it gets.
This is the apples-to-apples comparison against the Stage-1 fresh-start
numbers (best 96, recent avg ~45).

Reading the result:
  - near Stage-1 levels  -> skill is local/reactive, generalizes; multi-stage cheap
  - much lower (~0-15)   -> skill is layout-MEMORIZED; needs arch work or many
                            stages in the mix before multi-stage training pays
  - in between (~20-40)  -> partial transfer; drip-feed likely works

Safe: CPU inference (no VRAM contention), no log writes, no training, touches
no checkpoints. Run while training is stopped, or alongside it (CPU only).

Usage:
  python3 transfer_probe.py                       # 60 eps each on stages 1,2,3
  python3 transfer_probe.py 100                    # 100 eps each
  python3 transfer_probe.py 100 2 3                # 100 eps each, only stages 2,3
"""
import sys, os, inspect, importlib
import numpy as np

MODEL   = "sonic_ss_model_v2"
VECNORM = "sonic_ss_vecnormalize_v2.pkl"
STATEDIR = "states"

args = [a for a in sys.argv[1:]]
n_eps = int(args[0]) if args and args[0].isdigit() else 60
stage_ids = [int(a) for a in args[1:] if a.isdigit()] or [1, 2, 3]

# ---- import the real env class -------------------------------------------
mod = importlib.import_module("special_stage_env")
import gymnasium as gym
EnvClass = next(o for _, o in vars(mod).items()
                if inspect.isclass(o) and issubclass(o, (gym.Env, gym.Wrapper))
                and o.__module__ == mod.__name__)
env = EnvClass()
if hasattr(env, "_log_episode"):
    env._log_episode = lambda *a, **k: None      # never touch the training log

# disable restart scaffolding entirely: we want honest fresh starts
import special_stage_env as sse
if hasattr(sse, "RESTART_PROB"):
    sse.RESTART_PROB = 0.0

# ---- load policy on CPU --------------------------------------------------
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load(MODEL, device="cpu")

normalize_obs = None
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vn = VecNormalize.load(VECNORM, DummyVecEnv([lambda: EnvClass()]))
    vn.training = False
    if getattr(vn, "norm_obs", False):
        normalize_obs = vn.normalize_obs
except Exception:
    pass
prep = (lambda o: normalize_obs(o)) if normalize_obs else (lambda o: o)

print(f"policy: {MODEL} (gamma={model.gamma})  obs_norm={normalize_obs is not None}")
print(f"{n_eps} fresh episodes per stage, stochastic, no training\n")

def run_stage(state_file, n):
    """Point the env's fresh-start state at this stage; run n fresh episodes."""
    env.state_path = state_file
    results = []
    for _ in range(n):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        lstm, ep_start = None, np.ones((1,), dtype=bool)
        done, best = False, 0
        steps = 0
        while not done and steps < 6000:
            action, lstm = model.predict(prep(obs), state=lstm,
                                         episode_start=ep_start,
                                         deterministic=False)
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, info = env.step(int(action))
            steps += 1
            rem = info.get("blue_spheres_remaining")
            if isinstance(rem, (int, float)):
                # collected = (start total) - remaining; track via spheres counter
                pass
            done = term or trunc
        # read this-episode collected straight off the env's counter
        results.append(int(getattr(env, "_spheres_collected", 0)))
    return results

def summarize(name, res):
    if not res:
        print(f"{name}: no episodes")
        return
    arr = np.array(res)
    buckets = [0,10,20,30,40,50,60,70,80,90,103]
    hist = [int(((arr>=lo)&(arr<hi)).sum()) for lo,hi in zip(buckets, buckets[1:])]
    print(f"{name}:  avg={arr.mean():5.1f}  median={int(np.median(arr)):3d}  "
          f"max={arr.max():3d}  p90={int(np.percentile(arr,90)):3d}")
    labels = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90+"]
    line = "  ".join(f"{l}:{c}" for l, c in zip(labels, hist) if c)
    print(f"        dist: {line}\n")

for sid in stage_ids:
    sf = os.path.join(STATEDIR, f"curriculum_stage_{sid}.state")
    if not os.path.exists(sf):
        print(f"stage {sid}: {sf} not found, skipping\n")
        continue
    res = run_stage(sf, n_eps)
    summarize(f"STAGE {sid}", res)

print(">>> Compare STAGE 2/3 against STAGE 1. Similar = generalizes. "
      "Much lower = memorized Stage 1's layout.")
