#!/usr/bin/env python3
"""
zeroshot_test.py -- Step-2 zero-shot transfer evaluation.

Runs the multistage-v2 policy on the HELD-OUT stages (5,6 — never trained) and,
for comparison, on the TRAINED stages, all from sphere zero with no training.
The question: does the skill reach boards it has NEVER seen?

Read:
  - trained-stage avg = the ceiling this policy reaches on boards it practiced
  - held-out avg as a FRACTION of the trained avg = the zero-shot transfer.
    ~1.0 => the skill is fully general (unseen boards play as well as trained ones)
    ~0.1 => it only learned the specific trained boards (memorized the set)

This is a STRONGER test than Step 1: Step 1 measured generalization AMONG trained
stages; this measures generalization to stages that were held out entirely.

Safe: CPU inference, no training, no log writes, touches no checkpoints.

Usage:
  python3 zeroshot_test.py                 # 40 eps each; trained=1,2,3,4,7 held=5,6
  python3 zeroshot_test.py 80              # 80 eps each
  python3 zeroshot_test.py 40 v1           # evaluate the v1 (2-stage) model instead
"""
import sys, os, inspect, importlib
import numpy as np

args = [a for a in sys.argv[1:]]
n_eps = int(args[0]) if args and args[0].isdigit() else 40
which = "v1" if "v1" in args else "v2"

which = "v3" if "v3" in args else which
MODEL   = f"sonic_ss_multistage_{which}"
VECNORM = f"{MODEL}_vecnormalize.pkl"
STATEDIR = "states"
# holdout is 5,6 for BOTH v2 and v3 on purpose -> directly comparable zero-shot
if which == "v3":
    TRAINED = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14]
else:
    TRAINED = [1, 2, 3, 4, 7]
HELDOUT = [5, 6]

# position must be dropped to match the trained observation
os.environ.setdefault("SS_DROP_POSITION", "1")

mod = importlib.import_module("special_stage_env")
import gymnasium as gym
EnvClass = next(o for _, o in vars(mod).items()
                if inspect.isclass(o) and issubclass(o, (gym.Env, gym.Wrapper))
                and o.__module__ == mod.__name__)
env = EnvClass()
if hasattr(env, "_log_episode"):
    env._log_episode = lambda *a, **k: None

import special_stage_env as sse
if hasattr(sse, "RESTART_PROB"):
    sse.RESTART_PROB = 0.0

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

print(f"model: {MODEL}  (SS_DROP_POSITION={os.environ.get('SS_DROP_POSITION')})")
print(f"{n_eps} fresh episodes per stage, stochastic, no training\n")

def run_stage(sid, n):
    sf = os.path.join(STATEDIR, f"curriculum_stage_{sid}.state")
    if not os.path.exists(sf):
        return None
    env.state_path = sf
    if hasattr(sse, "STAGE_FILES"):
        sse.STAGE_FILES = [sf]   # force this single stage on fresh resets
    res = []
    for _ in range(n):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        lstm, ep_start, done, steps = None, np.ones((1,), bool), False, 0
        while not done and steps < 6000:
            a, lstm = model.predict(prep(obs), state=lstm,
                                    episode_start=ep_start, deterministic=False)
            ep_start = np.zeros((1,), bool)
            obs, _, term, trunc, info = env.step(int(a))
            steps += 1
            done = term or trunc
        res.append(int(getattr(env, "_spheres_collected", 0)))
    tot = int(getattr(env, "_stage_total", 0)) or 1
    arr = np.array(res)
    return arr.mean(), int(np.median(arr)), arr.max(), tot

def block(title, ids):
    print(title)
    avgs = []
    for sid in ids:
        r = run_stage(sid, n_eps)
        if r is None:
            print(f"  stage {sid}: state not found")
            continue
        avg, med, mx, tot = r
        avgs.append(avg / tot)
        print(f"  stage {sid}:  avg={avg:5.1f}  median={med:3d}  max={mx:3d}  "
              f"({100*avg/tot:4.1f}% of {tot})")
    return np.mean(avgs) if avgs else 0.0

tr = block(f"TRAINED stages {TRAINED}:", TRAINED)
print()
ho = block("HELD-OUT stages (5,6 — never trained):", HELDOUT)

print("\n" + "=" * 52)
print(f"  trained avg:  {100*tr:4.1f}% of board")
print(f"  held-out avg: {100*ho:4.1f}% of board")
if tr > 0:
    print(f"  ZERO-SHOT TRANSFER (held/trained): {ho/tr:.2f}")
    print("   ~1.0 = fully general (unseen boards ≈ trained)")
    print("   ~0.1 = memorized the trained set, doesn't reach new boards")
