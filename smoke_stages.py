#!/usr/bin/env python3
"""
smoke_stages.py -- verify the env reads every stage correctly BEFORE launching.

Replaces smoke_all14.py + smoke_v4.py, which were the same test hardcoded to two
different observation shapes. The shape is no longer hardcoded: it is asserted
against whatever the env actually built from the preset, so this one script
covers every preset, current and future.

Usage:
  python3 smoke_stages.py v5            # check the run's preset (from RUNS)
  python3 smoke_stages.py v5 --stages 1,2,3
  SS_USE_PIXELS=1 SS_PATCH_R=5 SS_DROP_POSITION=0 python3 smoke_stages.py

Any *** BAD *** = do NOT launch.
"""
import os
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('run', nargs='?', help='run name from train_ss_multistage.RUNS; '
                                       'omit to use the ambient SS_* env vars')
ap.add_argument('--stages', default=None,
                help='comma list to check (default: the run\'s stages, or 1..14)')
args = ap.parse_args()

# Preset must be applied before the env module is imported — it reads its
# observation shape at import time.
if args.run:
    # the trainer parses argv at import, so read RUNS out of its source instead
    src = open('train_ss_multistage.py').read()
    ns = {}
    exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
    if args.run not in ns['RUNS']:
        sys.exit(f'unknown run {args.run!r}. Known: {", ".join(ns["RUNS"])}')
    cfg = ns['RUNS'][args.run]
    os.environ.update(cfg['obs'])
    default_stages = cfg['stages']
else:
    default_stages = list(range(1, 15))

stages = ([int(s) for s in args.stages.split(',')] if args.stages
          else default_stages)

import numpy as np                                    # noqa: E402
import special_stage_env as sse                       # noqa: E402

sse.RESTART_PROB = 0.0
env = sse.SpecialStageEnv()
env._log_episode = lambda *a, **k: None               # no log writes from a smoke test

# ── the observation space the preset actually produced ───────────────────────
sp = env.observation_space.spaces
want_keys = {'ram', 'grid'} | ({'pixels'} if sse.USE_PIXELS else set())
want_ram = (sse.NUM_RAM_FEATURES,)
want_grid = (sse.PATCH_SIZE, sse.PATCH_SIZE, sse.GRID_CHANNELS)

print(f'preset: pixels={sse.USE_PIXELS} patch={sse.PATCH_SIZE}x{sse.PATCH_SIZE} '
      f'ram_feats={sse.NUM_RAM_FEATURES} drop_position={sse.DROP_POSITION}')
assert set(sp) == want_keys, f'obs keys {sorted(sp)} != {sorted(want_keys)}'
assert sp['ram'].shape == want_ram, f"ram {sp['ram'].shape} != {want_ram}"
assert sp['grid'].shape == want_grid, f"grid {sp['grid'].shape} != {want_grid}"
print(f'obs space OK: {sorted(want_keys)}, ram{want_ram}, grid{want_grid}\n')

print(f"{'stg':<5}{'blue':<7}{'total':<8}{'grid_sum':<10}{'grid_blue':<11}"
      f"{'ram0':<8}{'prog':<8}{'swap':<6}verdict")
print('-' * 78)

bad = []
for sid in stages:
    sf = os.path.expanduser(f'~/sonic-rl/states/curriculum_stage_{sid}.state')
    if not os.path.exists(sf):
        print(f'{sid:<5}MISSING FILE')
        bad.append(sid)
        continue
    env.state_path = sf
    sse.STAGE_FILES = [sf]              # force this single stage on reset
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    info = {}
    for _ in range(4):
        obs, _, term, trunc, info = env.step(0)

    blue = info.get('blue_spheres_remaining')
    total = int(getattr(env, '_stage_total', 0))
    g, r = obs['grid'], obs['ram']
    gsum, gblue = float(g.sum()), float(g[:, :, 0].sum())
    # index 4 is total_progress under both presets (position, when kept, sits at
    # 2-5 and shifts it to 8) — resolve it the same way the env lays it out.
    prog = float(r[4 if sse.DROP_POSITION else 8])
    swap = getattr(env, '_ram_swapped', None)

    ok = (set(obs) == want_keys and r.shape == want_ram and g.shape == want_grid
          and isinstance(blue, (int, float)) and blue and blue > 0
          and total > 0 and gsum > 0
          and np.all(np.isfinite(r)) and 0.0 <= float(r[0]) <= 1.5
          and -0.1 <= prog <= 1.5)
    if not ok:
        bad.append(sid)
    print(f'{sid:<5}{str(blue):<7}{total:<8}{gsum:<10.1f}{gblue:<11.1f}'
          f'{float(r[0]):<8.3f}{prog:<8.3f}{str(swap):<6}'
          + ('OK' if ok else '*** BAD ***'))

env.close()
print()
if bad:
    print(f'>>> PROBLEM stages: {bad} -- do NOT launch until resolved.')
    sys.exit(1)
print(f'>>> ALL {len(stages)} READ CORRECTLY. Safe to launch.')
