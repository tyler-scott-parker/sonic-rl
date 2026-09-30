#!/usr/bin/env python3
"""
smoke_stages.py -- verify the env reads every stage correctly BEFORE launching.

Replaces smoke_all14.py + smoke_v4.py, which were the same test hardcoded to two
different observation shapes. The shape is no longer hardcoded: it is asserted
against whatever the env actually built from the preset, so this one script
covers every preset, current and future.

Env v2 presets also get two checks that would have caught the 2026-09-29 bugs:
  board  -- blue cells in the grid the env reads == the game's own counter.
            (v1 read $F000 and missed board rows 24-31 on every stage.)
  ends   -- running straight, the episode ends on the very frame the exit
            animation ($E433) starts. (v1 ran ~670 more frames after a death.)

Usage:
  python3 smoke_stages.py v6            # check the run's preset (from RUNS)
  python3 smoke_stages.py v6 --stages 1,2,3
  SS_USE_PIXELS=1 SS_PATCH_R=5 SS_DROP_POSITION=0 SS_ENV_VERSION=1 python3 smoke_stages.py

Any *** BAD *** = do NOT launch.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

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
    src = open(os.path.join(HERE, 'train_ss_multistage.py')).read()
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

print(f'preset: env=v{sse.ENV_VERSION} pixels={sse.USE_PIXELS} '
      f'patch={sse.PATCH_SIZE}x{sse.PATCH_SIZE}x{sse.GRID_CHANNELS} '
      f'ram_feats={sse.NUM_RAM_FEATURES} drop_position={sse.DROP_POSITION}')
assert set(sp) == want_keys, f'obs keys {sorted(sp)} != {sorted(want_keys)}'
assert sp['ram'].shape == want_ram, f"ram {sp['ram'].shape} != {want_ram}"
assert sp['grid'].shape == want_grid, f"grid {sp['grid'].shape} != {want_grid}"
print(f'obs space OK: {sorted(want_keys)}, ram{want_ram}, grid{want_grid}\n')

print(f"{'stg':<5}{'blue':<7}{'board':<7}{'total':<8}{'grid_sum':<10}{'grid_blue':<11}"
      f"{'ram0':<8}{'swap':<6}{'ends':<9}verdict")
print('-' * 80)

bad = []
for sid in stages:
    sf = sse.stage_file(sid)
    if not os.path.exists(sf):
        print(f'{sid:<5}MISSING FILE {sf}')
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
    swap = getattr(env, '_ram_swapped', None)
    board = int((env._read_grid() == 2).sum()) if swap is not None else -1
    exiting = swap is not None and int(env._ram()[sse.ADDR_EXIT_TIMER]) != 0

    ok = (set(obs) == want_keys and r.shape == want_ram and g.shape == want_grid
          and isinstance(blue, (int, float)) and blue and blue > 0
          and total > 0 and gsum > 0 and not exiting
          and np.all(np.isfinite(r)) and 0.0 <= float(r[0]) <= 1.5)
    if not sse.V2:
        # total_progress (v1 only) is index 4, or 8 when position sits at 2-5
        prog = float(r[4 if sse.DROP_POSITION else 8])
        ok = ok and -0.1 <= prog <= 1.5

    ends = '-'
    if sse.V2:
        ok = ok and board == blue
        ends = 'no-end'                 # survived 4000 frames: fine, not checked
        for _ in range(4000):
            obs, _, term, trunc, info = env.step(0)
            timer = int(env._ram()[sse.ADDR_EXIT_TIMER])
            if term or trunc:
                stall = getattr(env, '_stalled', False)     # v3: no sphere in 10 s
                ends = (f"{env.steps}f{'s' if stall else ''}"
                        if term and (env._board_cleared or timer or stall) else 'WRONG')
                break
            if timer:
                ends = 'LATE'           # exit animation running, episode still on
                break
        ok = ok and ends not in ('WRONG', 'LATE')

    if not ok:
        bad.append(sid)
    board_s = f'{board}' + ('' if sse.V2 else '*')
    print(f'{sid:<5}{str(blue):<7}{board_s:<7}{total:<8}{gsum:<10.1f}{gblue:<11.1f}'
          f'{float(r[0]):<8.3f}{str(swap):<6}{ends:<9}'
          + ('OK' if ok else '*** BAD ***'))

env.close()
print()
if not sse.V2:
    print('* env v1 reads $F000 and misses board rows 24-31, so its board count is '
          'short by design;\n  kept only for the checkpoints trained on it.\n')
if bad:
    print(f'>>> PROBLEM stages: {bad} -- do NOT launch until resolved.')
    sys.exit(1)
if sse.V2:
    print(f'>>> ALL {len(stages)} READ CORRECTLY. Safe to launch.')
else:
    print(f'>>> ALL {len(stages)} read as env v1 expects (legacy: for v1 checkpoints '
          f'only, not new runs).')
