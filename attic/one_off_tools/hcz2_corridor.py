#!/usr/bin/env python3
"""hcz2_corridor.py — HCZ2's corridor before the curve (Tyler 2026-10-17: "Let's start with option 1"): fresh tries
stall on the curve (85%) while tries from his state at start point 9 (t 540, rolling at his speed) climb it ~23/24.
Where in the corridor does the agent's own play stop being enough?

    python3 attic/one_off_tools/hcz2_corridor.py CKPT.zip [--n 12] [--ks 0,3,5,6,7,8,9]

1. his run from the start to t 640: every 20 frames x, y, speed (px/frame), rolling/air, his action; his spin dashes
2. the agent from his state at each start point k: tries up the curve (x >= 1,650), its speed and rolling as it
   reaches x 1,440-1,500 (the curve's foot), and its spin-dash-ish presses (down / jump+down) in the corridor
CPU only.
"""
import argparse
import collections
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('model')
ap.add_argument('--n', type=int, default=12)
ap.add_argument('--ks', default='0,3,5,6,7,8,9')
args = ap.parse_args()

import train_main                                             # noqa: E402
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
torch.set_num_threads(1)
import coordinator as C, demos                                # noqa: E402
from stable_baselines3 import PPO                             # noqa: E402

act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
table = meta['action_table']
orig = C.emu_frame
st = lambda v: ('roll' if v & 4 else 'run') + ('/air' if v & 2 else '')
names = C.ACTION_NAMES

print('1. his run, start -> t 640 (every 20 frames: t x,y speed state action)')
at = {'f': seg['ckpts'][0]['frame']}
f0 = at['f']


def his_input(e, _m):
    at['f'] += 1
    return orig(e, table[actions[at['f']]])


C.emu_frame = his_input
env.reset(options={'key': seg['key'], 'ckpt': 0})
prev, rows = None, []
while env.frames < 640:
    _, _, term, trunc, info = env.step(0)
    x = int(info['x'])
    rows.append((env.frames, x, int(info['y']), (x - prev) / 4 if prev is not None else 0.0,
                 st(int(info['mg_status'])), names[int(actions[at['f']])]))
    prev = x
C.emu_frame = orig
line = []
for r in rows:
    if r[0] % 20 == 0:
        line.append(f'{r[0]} {r[1]},{r[2]} {r[3]:+.1f} {r[4]} {r[5]}')
for i in range(0, len(line), 4):
    print('   ' + ' | '.join(line[i:i + 4]))
his_acts = [int(actions[f0 + 1 + i]) for i in range(560)]
dash = [i for i in range(1, 560) if names[his_acts[i]] == 'jump+down' and names[his_acts[i - 1]] != 'jump+down']
print(f"   his jump+down presses (spin-dash revs) before t 560: {dash}; 'down' frames {sum(names[a] == 'down' for a in his_acts)}")
foot = [r for r in rows if 1440 <= r[1] <= 1500]
if foot:
    print(f'   at the curve foot (x 1,440-1,500): t {foot[0][0]}, speed {foot[0][3]:+.1f} px/frame, {foot[0][4]}')

print(f'\n2. the agent from his state at each start point ({args.n} sampled tries each)')
model = PPO.load(args.model, device='cpu')
model.policy.set_training_mode(False)
for k in [int(v) for v in args.ks.split(',')]:
    ups, feet, presses = 0, [], collections.Counter()
    for i in range(args.n):
        random.seed(i), np.random.seed(i), torch.manual_seed(i)
        obs, _ = env.reset(options={'key': seg['key'], 'ckpt': k})
        done, xmax, prev, foot_seen = False, 0, None, None
        while not done:
            a = int(model.predict(obs, deterministic=False)[0])
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            x = int(info['x'])
            if x < 1500:
                presses[names[a]] += 1
            if foot_seen is None and 1440 <= x <= 1500 and prev is not None:
                foot_seen = ((x - prev) / 4, st(int(info['mg_status'])))
            prev, xmax = x, max(xmax, x)
            if xmax >= 1650 and x >= 1650:
                break
        ups += xmax >= 1650
        if foot_seen:
            feet.append(foot_seen)
    tot = sum(presses.values()) or 1
    sp = [f[0] for f in feet]
    print(f"   k {k:2d} (his t {seg['ckpts'][k]['t']:4d}): up the curve {ups}/{args.n}; at the foot: speed median "
          f"{statistics.median(sp) if sp else float('nan'):+.1f} px/frame, rolling {sum(f[1].startswith('roll') for f in feet)}/{len(feet)}; "
          f"corridor presses: down {presses['down'] / tot:.0%}, jump+down {presses['jump+down'] / tot:.0%}, "
          f"jump(any) {sum(presses[n] for n in ('jump', 'jump+left', 'jump+right', 'jump+down')) / tot:.0%}", flush=True)
