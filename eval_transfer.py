#!/usr/bin/env python3
"""
eval_transfer.py -- how far does a policy get on each board, from sphere zero?

Replaces transfer_probe.py + zeroshot_test.py, which were the same rollout loop
wrapped in two different hardcoded scenarios. The scenario is now arguments.

  transfer     = does the skill reach boards it PRACTICED but wasn't specialised
                 on?  (specialist on stages 2,3 vs its own stage 1)
  zero-shot    = does it reach boards it has NEVER trained on?  (--holdout)
                 The stronger test: ~1.0 of the trained average means a general
                 skill, ~0.1 means it memorised the training SET.

Results so far, on this metric: specialist 0.06 · v1 (2 boards) 0.75 ·
v2 (5 boards, held out 5,6) 0.09 — position-drop forces generalization at 2
boards but not at 5.

Safe: CPU inference (no VRAM contention), no training, no log writes, touches no
checkpoints. Run it while training is stopped, or alongside it.

Every checkpoint runs on the env version it was trained on (the run's preset,
or --preset): the specialist and v1-v5 on env v1, v6 onward on env v2.

Usage:
  python3 eval_transfer.py v5                      # v5's preset+stages, 40 eps
  python3 eval_transfer.py v5 --holdout 5,6 -n 80  # zero-shot on held-out boards
  python3 eval_transfer.py --model sonic_ss_model_v2 --preset specialist \\
                           --stages 1,2,3          # the specialist probe
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('run', nargs='?',
                help='run name from train_ss_multistage.RUNS — supplies the '
                     'checkpoint, the obs preset and the trained stage list')
ap.add_argument('--model', help='checkpoint basename (overrides the run\'s)')
ap.add_argument('--preset', choices=['specialist', 'multistage'],
                help='obs preset when not using a run name (specialist = '
                     'pixels + 11x11 patch + position; multistage = the v4/v5 '
                     'obs; both env v1). For v6+ pass the run name.')
ap.add_argument('--stages', help='comma list of stages to evaluate')
ap.add_argument('--holdout', help='comma list of NEVER-TRAINED stages; reported '
                                  'separately as the zero-shot number')
ap.add_argument('-n', '--episodes', type=int, default=40,
                help='fresh episodes per stage (default 40)')
ap.add_argument('--max-steps', type=int, default=None,
                help="per-episode cap (default: the env's own, 18000). The old "
                     "default of 6000 cut off 45%% of the specialist's "
                     "60+-sphere runs and understated its average.")
args = ap.parse_args()

PRESETS = {
    'specialist': dict(SS_ENV_VERSION='1',
                       SS_USE_PIXELS='1', SS_PATCH_R='5', SS_DROP_POSITION='0'),
    'multistage': dict(SS_ENV_VERSION='1',
                       SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
}

# The obs preset must be set before special_stage_env is imported.
MODEL = args.model
trained = [int(s) for s in args.stages.split(',')] if args.stages else None

if args.run:
    src = open(os.path.join(HERE, 'train_ss_multistage.py')).read()
    ns = {}
    exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
    if args.run not in ns['RUNS']:
        sys.exit(f'unknown run {args.run!r}. Known: {", ".join(ns["RUNS"])}')
    cfg = ns['RUNS'][args.run]
    os.environ.update(cfg['obs'])
    MODEL = MODEL or f'sonic_ss_multistage_{args.run}'
    trained = trained or cfg['stages']
elif args.preset:
    os.environ.update(PRESETS[args.preset])

if not MODEL:
    sys.exit('need a run name or --model')
trained = trained or [1, 2, 3]
holdout = [int(s) for s in args.holdout.split(',')] if args.holdout else []
VECNORM = os.path.join(HERE, f'{MODEL}_vecnormalize.pkl'
                       if MODEL.startswith('sonic_ss_multistage')
                       else 'sonic_ss_vecnormalize_v2.pkl')

import pickle                                              # noqa: E402
import numpy as np                                          # noqa: E402
import special_stage_env as sse                             # noqa: E402
from sb3_contrib import RecurrentPPO                        # noqa: E402

MAX_STEPS = args.max_steps or sse.SS_MAX_STEPS
sse.RESTART_PROB = 0.0          # honest fresh starts, no restart scaffolding
env = sse.SpecialStageEnv()
env._log_episode = lambda *a, **k: None      # never touch the training log

model = RecurrentPPO.load(os.path.join(HERE, MODEL), device='cpu')


def load_obs_normalizer(path):
    """Obs-normalizing fn from a saved VecNormalize, or identity.

    Read straight from the pickle rather than VecNormalize.load(), which wants a
    live venv — and stable_retro allows only ONE emulator per process, so
    building a second one here raises. The old probes wrapped that in a bare
    except and silently ran unnormalized. Every trainer sets norm_obs=False
    (reward-only normalization), so identity is in fact correct — but say so
    instead of swallowing an error that would also hide a real problem."""
    ident = (lambda o: o), 'identity'
    if not os.path.exists(path):
        return ident[0], f'none ({path} not found)'
    with open(path, 'rb') as f:
        vn = pickle.load(f)
    if not getattr(vn, 'norm_obs', False):
        return ident[0], 'not needed (norm_obs=False)'
    vn.training = False
    return vn.normalize_obs, 'applied from ' + path


prep, norm_note = load_obs_normalizer(VECNORM)

print(f'policy: {MODEL}  gamma={model.gamma}  obs_norm={norm_note}')
print(f'preset: env=v{sse.ENV_VERSION} pixels={sse.USE_PIXELS} '
      f'patch={sse.PATCH_SIZE}x{sse.PATCH_SIZE}x{sse.GRID_CHANNELS} '
      f'ram_feats={sse.NUM_RAM_FEATURES}')
print(f'{args.episodes} fresh episodes per stage, stochastic, no training, '
      f'cap {MAX_STEPS} steps\n')


def run_stage(sid, n):
    """n honest fresh episodes on one board. Returns (collected[], board_total)."""
    sf = sse.stage_file(sid)
    if not os.path.exists(sf):
        return None, 0
    env.state_path = sf
    sse.STAGE_FILES = [sf]      # pin it: otherwise a set SS_STAGES random-picks
    res = []
    for _ in range(n):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        lstm, ep_start, done, steps = None, np.ones((1,), bool), False, 0
        while not done and steps < MAX_STEPS:
            a, lstm = model.predict(prep(obs), state=lstm,
                                    episode_start=ep_start, deterministic=False)
            ep_start = np.zeros((1,), bool)
            obs, _, term, trunc, _info = env.step(int(a))
            steps += 1
            done = term or trunc
        res.append(int(getattr(env, '_spheres_collected', 0)))
    return res, int(getattr(env, '_stage_total', 0)) or 1


def block(title, ids):
    """Per-board rows + the mean %-of-board. % is the only comparable unit:
    boards range 37..155 spheres, so raw counts across boards are meaningless."""
    print(title)
    fracs = []
    for sid in ids:
        res, total = run_stage(sid, args.episodes)
        if res is None:
            print(f'  stage {sid:>2}: state not found')
            continue
        arr = np.array(res)
        fracs.append(arr.mean() / total)
        print(f'  stage {sid:>2}:  avg={arr.mean():5.1f}  '
              f'median={int(np.median(arr)):3d}  max={arr.max():3d}  '
              f'({100 * arr.mean() / total:4.1f}% of {total})')
    return float(np.mean(fracs)) if fracs else 0.0


tr = block(f'TRAINED stages {trained}:', trained)
ho = None
if holdout:
    print()
    ho = block(f'HELD-OUT stages {holdout} (never trained):', holdout)

print('\n' + '=' * 52)
print(f'  trained avg:  {100 * tr:4.1f}% of board')
if ho is not None:
    print(f'  held-out avg: {100 * ho:4.1f}% of board')
    if tr > 0:
        print(f'  ZERO-SHOT TRANSFER (held/trained): {ho / tr:.2f}')
        print('   ~1.0 = fully general (unseen boards play like trained ones)')
        print('   ~0.1 = memorized the trained set, does not reach new boards')
else:
    print('  (pass --holdout to get the zero-shot transfer number)')
env.close()
