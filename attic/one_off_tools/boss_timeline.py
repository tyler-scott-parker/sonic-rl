#!/usr/bin/env python3
"""boss_timeline.py — where does the time go in a boss fight?

AIZ2's start point 156 (mid-fight, 506 of Tyler's frames before the last hit)
cleared ~3/4 of the time on 2026-10-15 but kept pace with him less and less
(13 of 54 at 11:05-12:10, 3 of 56 after 12:48), clears taking ~1,360 frames.
This plays the fight from one start point with several checkpoints (same env as
training, the policy sampling its actions as in training) and times every boss
hit and every hit taken, against Tyler's own run from the same point (his
recorded inputs through the same env and the same hit counter, so the
yardstick is checked, not assumed).

Per step it logs Sonic's x/y, rings, routine (4 = hurt), the boss's x/y and
hits left (the slot the hit counter tracks), and the hit count; raw episodes
go to logs/boss_timeline_<act>_<k>.json.

    python3 attic/one_off_tools/boss_timeline.py mg2 AIZ2 156 --eps 24 \\
        checkpoints_main_mg2_keep/main_mg2_50757048_steps.zip ...
(CPU only, one thread: fine beside training.)
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('run')
ap.add_argument('act')
ap.add_argument('k', type=int, help='start point index')
ap.add_argument('models', nargs='+', help='checkpoint zips')
ap.add_argument('--eps', type=int, default=24)
args = ap.parse_args()

import train_main                                             # noqa: E402
train_main.configure_env(args.run)
os.environ['MG_LOG'] = os.devnull
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
torch.set_num_threads(1)
import coordinator as C                                       # noqa: E402
import demos                                                  # noqa: E402
from stable_baselines3 import PPO                             # noqa: E402


def boss_slot(env):
    """(x, y, hits left) of the slot the hit counter treats as a boss, or None."""
    bt = env._bt
    if bt is None or bt.prev is None:
        return None
    cand = np.flatnonzero((bt.maxseen >= C.BOSS_MIN) & (bt.maxseen <= C.BOSS_MAX) & bt.prev[:, :4].any(1))
    if not len(cand):
        return None
    row = bt.prev[cand[0]]
    word = lambda o: int(row[o]) << 8 | int(row[o + 1])
    return word(0x10), word(0x14), int(row[0x29])


def episode(env, key, k, choose, after_reset=None):
    """One episode from start point k; choose(obs) -> action. Returns a dict."""
    obs, _ = env.reset(options={'key': key, 'ckpt': k})
    if after_reset:
        after_reset()
    steps, hits_at, prev_hits = [], [], 0
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(choose(obs))
        done = term or trunc
        if env._boss_hits > prev_hits:
            hits_at += [env.frames] * (env._boss_hits - prev_hits)
            prev_hits = env._boss_hits
        b = boss_slot(env)
        steps.append([env.frames, int(info['x']), int(info['y']), int(info['rings']),
                      int(info['mg_routine'])] + (list(b) if b else [None, None, None]))
    res = info['mg_result']
    return dict(outcome=res['outcome'], frames=res['frames'], target=res['target'],
                success=bool(res['success']), hits_at=hits_at,
                hurt_at=[h[0] for h in env._hurt], steps=steps)


cfg = train_main.RUNS[args.run]
env = C.SonicCoordinator(acts=cfg['acts'], demo_prob=0.0)
segs = env.demo_info()
seg = next(s for s in env._segs if s['name'] == args.act)
key = seg['key']
print(f"{args.act} start point {args.k} (demo frame {seg['ckpts'][args.k]['frame']}), "
      f"{args.eps} episodes per checkpoint", flush=True)

# Tyler's run from the same point: his recorded inputs through the same env
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
table = meta['action_table']
orig, at = C.emu_frame, {'f': 0}


def his_input(e, _mask):
    at['f'] += 1
    return orig(e, table[actions[at['f']]])


C.emu_frame = his_input
try:
    his = episode(env, key, args.k, lambda o: 0,
                  after_reset=lambda: at.update(f=seg['ckpts'][args.k]['frame']))
finally:
    C.emu_frame = orig
print(f"Tyler: {his['outcome']} at {his['frames']} (target {his['target']}), hits at {his['hits_at']}, "
      f"hurt at {his['hurt_at']}", flush=True)

out = dict(act=args.act, k=args.k, his=his, models={})
fn = f'logs/boss_timeline_{args.act}_{args.k}.json'
for m in args.models:
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    model = PPO.load(m, device='cpu')
    model.policy.set_training_mode(False)
    name = f'{model.num_timesteps:,}'
    t0 = time.time()
    eps = [episode(env, key, args.k, lambda o: int(model.predict(o, deterministic=False)[0]))
           for _ in range(args.eps)]
    out['models'][name] = eps
    with open(fn, 'w') as f:
        json.dump(out, f)
    cl = [e for e in eps if e['outcome'] == 'clear']
    med = lambda v: int(np.median(v)) if len(v) else None
    nh = max([len(e['hits_at']) for e in cl] + [0])
    kth = [med([e['hits_at'][i] for e in cl if len(e['hits_at']) > i]) for i in range(nh)]
    oc = {o: sum(e['outcome'] == o for e in eps) for o in sorted({e['outcome'] for e in eps})}
    print(f"{name}: {oc}, kept pace {sum(e['success'] for e in eps)}/{len(eps)}, "
          f"clear frames median {med([e['frames'] for e in cl])}; hit k at (median, clears) {kth}; "
          f"hits taken per episode median {med([len(e['hurt_at']) for e in eps])} "
          f"({time.time() - t0:.0f} s)", flush=True)
print(f'raw: {fn}')
