#!/usr/bin/env python3
"""hcz2_chase_eval.py — HCZ2's wall chase (his frames ~560-2,000; doc 2026-10-17 ~09:00): N sampled tries of a
checkpoint from a start point on his route, scored by how far into the chase they get (Tyler 2026-10-17: "yes,
build it" to the front ahead; this is its pre-committed read).

    python3 attic/one_off_tools/hcz2_chase_eval.py CKPT.zip [--k 9] [--n 24]

k 9 = his t 540, just before the curve (the chase starts as he rolls up it); k 13 = t 780, in the basin.
Stages by the furthest x: curve (< 1,650), basin (< 1,930), yellow pipe (< 2,197), on to the chase's end
(< 3,061), past it (>= 3,061, where standing still is no longer crushed). CPU only.
"""
import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('model')
ap.add_argument('--k', type=int, default=9)
ap.add_argument('--n', type=int, default=24)
args = ap.parse_args()

import train_main                                             # noqa: E402
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
torch.set_num_threads(1)
import coordinator as C                                       # noqa: E402
from stable_baselines3 import PPO                             # noqa: E402

STAGES = [(1650, 'curve'), (1930, 'basin'), (2197, 'yellow pipe'), (3061, 'rest of the chase'), (10 ** 9, 'PAST THE CHASE')]
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
model = PPO.load(args.model, device='cpu')
model.policy.set_training_mode(False)
print(f"{os.path.basename(args.model)}: HCZ2 start point {args.k} (his t {seg['ckpts'][args.k]['t']}), {args.n} tries", flush=True)
got = collections.Counter()
for i in range(args.n):
    random.seed(i), np.random.seed(i), torch.manual_seed(i)
    obs, _ = env.reset(options={'key': seg['key'], 'ckpt': args.k})
    done, xmax = False, 0
    while not done:
        obs, _, term, trunc, info = env.step(int(model.predict(obs, deterministic=False)[0]))
        done = term or trunc
        xmax = max(xmax, int(info['x']))
    stage = next(name for lim, name in STAGES if xmax < lim)
    got[stage] += 1
    print(f"  try {i:2d}: {info['mg_result']['outcome']:8s} x max {xmax:5d} -> {stage}", flush=True)
print('reached (furthest stage): ' + ', '.join(f'{name} {got[name]}' for _, name in STAGES)
      + f"; past the chase {got['PAST THE CHASE']}/{args.n}")
