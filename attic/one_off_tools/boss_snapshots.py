#!/usr/bin/env python3
"""boss_snapshots.py — screenshots of a fight: Tyler's run from start point K (his recorded inputs through the
training env, as boss_timeline.py) and one sampled episode of a checkpoint, saved at the frames asked for, as
one contact sheet each (Tyler 2026-10-17: "dig into MGZ2"; the boss_timeline numbers needed a picture of the arena).

    python3 attic/one_off_tools/boss_snapshots.py mg2 MGZ2 270 --frames 100,640,1760 [--model ckpt.zip --seed 3]

Writes logs/boss_frames/<act>_<k>/{tyler,agent}_sheet.png (each tile: frame, Sonic x/y, rings, boss x/y, hits
left). CPU only, one thread.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('run')
ap.add_argument('act')
ap.add_argument('k', type=int)
ap.add_argument('--frames', required=True, help='comma-separated frames from the start point')
ap.add_argument('--model', help='checkpoint zip for an agent episode')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

import train_main                                             # noqa: E402
train_main.configure_env(args.run)
os.environ['MG_LOG'] = os.devnull
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import random                                                 # noqa: E402
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from PIL import Image, ImageDraw                              # noqa: E402
torch.set_num_threads(1)
import coordinator as C                                       # noqa: E402
import demos                                                  # noqa: E402


def boss_slot(env):
    """boss_timeline.py's (x, y, hits left) of the boss slot, or None (copied: importing it runs its argparse)."""
    bt = env._bt
    if bt is None or bt.prev is None:
        return None
    cand = np.flatnonzero((bt.maxseen >= C.BOSS_MIN) & (bt.maxseen <= C.BOSS_MAX) & bt.prev[:, :4].any(1))
    if not len(cand):
        return None
    row = bt.prev[cand[0]]
    word = lambda o: int(row[o]) << 8 | int(row[o + 1])
    return word(0x10), word(0x14), int(row[0x29])


want = sorted(int(f) for f in args.frames.split(','))
out_dir = f'logs/boss_frames/{args.act}_{args.k}'
os.makedirs(out_dir, exist_ok=True)


def shoot(env, key, choose, after_reset=None):
    """Run one episode from start point k; return [(frame, label, screen)] at the wanted frames."""
    env.reset(options={'key': key, 'ckpt': args.k})
    if after_reset:
        after_reset()
    todo, shots, done = list(want), [], False
    while not done and todo:
        _, _, term, trunc, info = env.step(choose())
        done = term or trunc
        while todo and env.frames >= todo[0]:
            b = boss_slot(env)
            label = (f"f{env.frames} sonic ({info['x']},{info['y']}) r{info['rings']}"
                     + (f" boss ({b[0]},{b[1]}) left {b[2]}" if b else ' boss -'))
            shots.append((todo.pop(0), label, np.array(env.render())))
    return shots, (info['mg_result']['outcome'] if done else 'running')


def sheet(shots, name):
    h, w = shots[0][2].shape[:2]
    cols = 4
    rows = -(-len(shots) // cols)
    img = Image.new('RGB', (cols * w, rows * (h + 14)), 'black')
    d = ImageDraw.Draw(img)
    for i, (_, label, scr) in enumerate(shots):
        x, y = (i % cols) * w, (i // cols) * (h + 14)
        img.paste(Image.fromarray(scr), (x, y + 14))
        d.text((x + 2, y + 1), label, fill='white')
    path = f'{out_dir}/{name}_sheet.png'
    img.save(path)
    return path


cfg = train_main.RUNS[args.run]
env = C.SonicCoordinator(acts=cfg['acts'], demo_prob=0.0)
env.demo_info()
seg = next(s for s in env._segs if s['name'] == args.act)
key = seg['key']
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
table = meta['action_table']
orig, at = C.emu_frame, {'f': 0}


def his_input(e, _mask):
    at['f'] += 1
    return orig(e, table[actions[at['f']]])


C.emu_frame = his_input
try:
    shots, oc = shoot(env, key, lambda: 0, after_reset=lambda: at.update(f=seg['ckpts'][args.k]['frame']))
finally:
    C.emu_frame = orig
print('tyler', oc, sheet(shots, 'tyler'), flush=True)
for f, label, _ in shots:
    print('  ', label)

if args.model:
    from stable_baselines3 import PPO
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    model = PPO.load(args.model, device='cpu')
    model.policy.set_training_mode(False)
    box = {'obs': None}                               # the latest observation, from reset or step
    orig_step, orig_reset = env.step, env.reset

    def step(a):
        r = orig_step(a)
        box['obs'] = r[0]
        return r

    def reset(**kw):
        r = orig_reset(**kw)
        box['obs'] = r[0]
        return r
    env.step, env.reset = step, reset
    shots, oc = shoot(env, key, lambda: int(model.predict(box['obs'], deterministic=False)[0]))
    print('agent', oc, sheet(shots, 'agent'), flush=True)
    for f, label, _ in shots:
        print('  ', label)
