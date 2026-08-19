#!/usr/bin/env python3
"""
eval_win.py  --  read-only win-frame tracer

Purpose: confirm whether the board-clear is OBSERVABLE in the info stream,
i.e. whether a frame with blue_spheres_remaining == 0 appears (while still
in-stage) BEFORE special_stage flips to 0. The success-detection fix depends
on that frame being visible.

Safe by construction:
  - imports the REAL SpecialStageEnv class (reuses your exact obs builder)
  - runs the policy on CPU (no VRAM contention with the live trainer)
  - monkeypatches _log_episode to a no-op (cannot pollute ss_log.jsonl)
  - loads a deep s100 restart state (~2 spheres from clearing) so a decent
    policy closes it out quickly
  - touches no training files, writes nothing

Usage:
  python3 eval_win.py            # 40 attempts from the deepest pool bucket
  python3 eval_win.py 100        # 100 attempts
  python3 eval_win.py 100 s97    # 100 attempts, force the s97 bucket
"""
import sys, glob, inspect, collections
import numpy as np

MODEL   = "sonic_ss_model_v2"
VECNORM = "sonic_ss_vecnormalize_v2.pkl"
POOLDIR = "states/curriculum_pool"

attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 40
bucket   = sys.argv[2] if len(sys.argv) > 2 else None

# ----------------------------------------------------------------------
# 1. import the real env module + discover the env class
# ----------------------------------------------------------------------
import importlib
mod = importlib.import_module("special_stage_env")

import gymnasium as gym
candidates = []
for name, obj in vars(mod).items():
    if inspect.isclass(obj) and issubclass(obj, (gym.Env, gym.Wrapper)) \
       and obj.__module__ == mod.__name__:
        candidates.append((name, obj))
if not candidates:
    print("No gym.Env/Wrapper subclass found in special_stage_env.py.")
    print("Members:", [n for n in vars(mod) if not n.startswith('_')])
    sys.exit(1)
EnvName, EnvClass = candidates[0]
print("env class:", EnvName)
try:
    print("constructor signature:", str(inspect.signature(EnvClass.__init__)))
except (TypeError, ValueError):
    print("constructor signature: <unavailable>")

# ----------------------------------------------------------------------
# 2. instantiate (try no-arg; if it fails, report the signature)
# ----------------------------------------------------------------------
try:
    env = EnvClass()
except Exception as e:
    print("\nNo-arg construction failed:", repr(e))
    print("Tell me the required constructor args and I'll adjust.")
    sys.exit(1)

# never write to the training log from this eval
if hasattr(env, "_log_episode"):
    env._log_episode = lambda *a, **k: None

print("obs space type:", type(env.observation_space).__name__)
if isinstance(env.observation_space, gym.spaces.Dict):
    print("obs keys:", {k: tuple(v.shape) for k, v in env.observation_space.spaces.items()})

# ----------------------------------------------------------------------
# 3. find the underlying emulator (the thing with set_state/get_ram)
# ----------------------------------------------------------------------
def find_emulator(e):
    seen, stack = set(), [e]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        em = getattr(o, "em", None)
        if em is not None and hasattr(em, "set_state") and hasattr(em, "get_state"):
            return em
        if hasattr(o, "set_state") and hasattr(o, "get_state"):
            return o
        for attr in ("base_env", "env", "unwrapped", "venv", "gym_env", "_env"):
            child = getattr(o, attr, None)
            if child is not None and child is not o:
                stack.append(child)
    return None

emulator = find_emulator(env)
if emulator is None:
    print("\nCould not locate the emulator handle (set_state/get_ram).")
    print("Attributes on env:", [a for a in dir(env) if not a.startswith('__')][:40])
    sys.exit(1)
print("emulator handle found:", type(emulator).__name__)

# ----------------------------------------------------------------------
# 4. pick a deep restart state
# ----------------------------------------------------------------------
def pool(b):
    return sorted(p for p in glob.glob(f"{POOLDIR}/{b}_*") if p.endswith(".state"))

state_files = []
if bucket:
    state_files = pool(bucket)
else:
    for b in ("s100", "s97", "s94"):
        state_files = pool(b)
        if state_files:
            bucket = b
            break
if not state_files:
    buckets = sorted({p.split('/')[-1].split('_')[0] for p in glob.glob(f'{POOLDIR}/*')})
    print(f"\nNo states in requested bucket. Available buckets: {buckets}")
    sys.exit(1)
state_path = state_files[0]
print(f"using state: {state_path}  (bucket {bucket})\n")
state_blob = open(state_path, "rb").read()

# ----------------------------------------------------------------------
# 5. load the policy (CPU, to avoid VRAM contention with the live trainer)
# ----------------------------------------------------------------------
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load(MODEL, device="cpu")
print("model loaded on CPU. gamma=%s  lr(read)=%s\n" % (model.gamma, model.learning_rate))

# optional VecNormalize for obs (best-effort; skipped if norm_obs is off)
normalize_obs = None
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    tmp = DummyVecEnv([lambda: EnvClass()])
    vn = VecNormalize.load(VECNORM, tmp)
    vn.training = False
    if getattr(vn, "norm_obs", False):
        normalize_obs = vn.normalize_obs
        print("VecNormalize: norm_obs=True -> normalizing policy inputs")
    else:
        print("VecNormalize: norm_obs=False -> raw obs to policy")
except Exception as e:
    print("VecNormalize not applied (%s) -> raw obs to policy" % type(e).__name__)
print()

def prep(obs):
    return normalize_obs(obs) if normalize_obs is not None else obs

# ----------------------------------------------------------------------
# 6. drive the env's OWN restart path (forced to our deep state) so the
#    injected board actually owns the episode: reset() does set_state +
#    _reset_state + _rebuild_converted + correct obs. We just force
#    _weighted_pick -> our file and force the restart roll to fire.
# ----------------------------------------------------------------------
import special_stage_env as sse

# force the restart branch in reset() to always fire and always pick our state
env._weighted_pick = lambda files: state_path
_orig_restart_prob = sse.RESTART_PROB
sse.RESTART_PROB = 1.0
# reset() also requires self._original_grid is not None to consider a restart;
# a first plain reset populates it.
if getattr(env, "_original_grid", None) is None:
    env.reset()

def reset_restart():
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return obs

# gather several deep states to try (the logged clear may have been a
# different one than the first file, and stochastic play varies per state)
deep_states = []
for b in ([bucket] if bucket else ["s100", "s97", "s94"]):
    deep_states += pool(b)
deep_states = deep_states[:8] or [state_path]
print("trying %d deep state(s), stochastic policy:" % len(deep_states))
for sp_ in deep_states:
    print("   ", sp_.split('/')[-1])
print()

caught = False
DETERMINISTIC = False        # match training: sampled actions, not argmax
per_state = max(5, attempts // max(1, len(deep_states)))

for sp_path in deep_states:
    env._weighted_pick = lambda files, _p=sp_path: _p
    for attempt in range(1, per_state + 1):
        obs = reset_restart()
        lstm_states = None
        ep_start = np.ones((1,), dtype=bool)
        tail = collections.deque(maxlen=12)
        saw_zero = False
        start_blue = None
        restart_from = getattr(env, "_restart_from", None)
        done = False
        steps = 0
        min_blue = 999
        while not done and steps < 4000:
            action, lstm_states = model.predict(
                prep(obs), state=lstm_states, episode_start=ep_start,
                deterministic=DETERMINISTIC)
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, info = env.step(int(action))
            steps += 1
            blue = info.get("blue_spheres_remaining")
            sp   = info.get("special_stage")
            if start_blue is None:
                start_blue = blue
            if isinstance(blue, (int, float)):
                min_blue = min(min_blue, blue)
            tail.append((steps, blue, sp, term, trunc))
            if blue == 0 and not saw_zero:
                saw_zero = True
            done = term or trunc

        end_blue = tail[-1][1] if tail else None
        if saw_zero or (end_blue == 0):
            caught = True
            print(f"[{sp_path.split('/')[-1]}] attempt {attempt}: restart_from={restart_from} "
                  f"start_blue={start_blue} -> CLEAR (saw blue==0: {saw_zero})")
            print("  tail (step, blue, special, term, trunc):")
            for row in tail:
                print("   ", row)
            in_stage_zero = any(b == 0 and s not in (0, None) for (_, b, s, _, _) in tail)
            same_frame    = any(b == 0 and s in (0,)        for (_, b, s, _, _) in tail)
            print(f"  blue==0 seen WHILE in-stage (special!=0): {in_stage_zero}")
            print(f"  blue==0 only AS/AFTER special flipped to 0: {same_frame}")
            break
        else:
            print(f"[{sp_path.split('/')[-1]}] attempt {attempt}: start_blue={start_blue} "
                  f"min_blue={min_blue} end_blue={end_blue} steps={steps} (no clear)")
    if caught:
        break

sse.RESTART_PROB = _orig_restart_prob

if not caught:
    print(f"\nNo clear caught in {attempts} attempts from {bucket}. "
          f"Try more attempts or a deeper bucket (e.g. s100).")
else:
    print("\n>>> Interpretation:")
    print("  If 'blue==0 WHILE in-stage' is True  -> the win frame IS observable;")
    print("     keying success on blue_spheres_remaining==0 will work.")
    print("  If it's False and 'only AS/AFTER special flipped' is True -> the board")
    print("     empties on the same frame the stage exits; we capture the clear by")
    print("     latching 'spheres_collected reached total' rather than a live blue==0.")
