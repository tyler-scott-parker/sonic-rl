#!/usr/bin/env python3
"""hud_reading_test.py — does the base-game agent READ the HUD numbers?

The focus overlay (focus.py) lights up the RINGS and TIME counters on most
frames, but it blurs pixels, and a blurred HUD is something the network never
saw in training, so a reaction there doesn't prove use. This test stays
in-distribution: it changes the numbers in RAM, the game redraws the HUD as
it would, and nothing else differs.

For start points spread over every act of a run (Tyler's recording):
  copy A: the agent plays STEPS agent steps (deterministic);
  copy B: the same start, one RAM edit, then exactly A's button presses.
Compared on the last COMPARE steps (after every HUD redraw, incl. the timer,
which only redraws when its second ticks): the decision (KL from A's action
probabilities to B's; does the chosen action change) and the value (B - A).
A sample counts only if Sonic ends in the same place in both copies and the
two pictures differ inside the HUD and nowhere else.

Edits: rings -> 0, rings + 50, timer + 20 s, timer + 1:00 (the questions);
score + 5000 and lives + 2 (controls: the same kind of digit change, on
numbers that don't matter to the reward); none (must be exactly 0).

    python3 attic/one_off_tools/hud_reading_test.py mg2 [--per-act 12]
(CPU only, one thread: fine beside training.)
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('run')
ap.add_argument('--per-act', type=int, default=12)
ap.add_argument('--model', help='a checkpoint zip (default: the run\'s latest save)')
args = ap.parse_args()

import train_main                                             # noqa: E402
train_main.configure_env(args.run)
os.environ['MG_LOG'] = os.devnull
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
torch.set_num_threads(1)
import coordinator as C                                       # noqa: E402
from stable_baselines3 import PPO                             # noqa: E402

STEPS, COMPARE = 24, 4                  # 96 frames: the timer's redraw comes within 60
HUD_X, HUD_Y = 34, 24                   # the 84 x 84 picture's top-left box holding SCORE/TIME/RINGS
LIVES_BOX = (slice(70, 84), slice(0, 22))   # the lives counter, bottom left

EXTRA_VARS = {                          # skdisasm: the HUD redraws a number when its flag is set
    'hud_upd_lives': (0xFFFE1C, '|u1'),
    'hud_upd_rings': (0xFFFE1D, '|u1'),
    'hud_upd_score': (0xFFFE1F, '|u1'),
}


def edit(name, d, info):
    """Apply one RAM edit; returns False when it doesn't apply here."""
    rings, score, lives = int(info['rings']), int(info['score']), int(info['lives'])
    mn, sec = int(info['mg_timer_min']), int(info['mg_timer_sec'])
    if name == 'none':
        return True
    if name == 'rings0':
        if rings == 0:
            return False
        d.set_value('rings', 0)
        d.set_value('hud_upd_rings', 0x80)      # $80: redraw from zero (ring loss uses it)
    elif name == 'rings+50':
        d.set_value('rings', rings + 50)
        d.set_value('hud_upd_rings', 1)
    elif name == 'timer+20s':
        if mn >= 8:
            return False
        sec += 20
        d.set_value('mg_timer_min', mn + sec // 60)
        d.set_value('mg_timer_sec', sec % 60)
    elif name == 'timer+1m':
        if mn >= 8:
            return False
        d.set_value('mg_timer_min', mn + 1)
    elif name == 'score+5000':
        d.set_value('score', score + 5000)
        d.set_value('hud_upd_score', 1)
    elif name == 'lives+2':
        d.set_value('lives', min(lives + 2, 99))
        d.set_value('hud_upd_lives', 1)
    return True


EDITS = ['none', 'rings0', 'rings+50', 'timer+20s', 'timer+1m', 'score+5000', 'lives+2']


@torch.no_grad()
def judge(policy, obs):
    t, _ = policy.obs_to_tensor(obs)
    dist = policy.get_distribution(t)
    return dist.distribution.probs[0].numpy(), float(policy.predict_values(t)[0, 0])


def play(env, key, kk, ed, actions=None):
    """One copy from start point kk with edit ed, the agent choosing (A) or
    replaying `actions` (B): (per-step (probs, value), observations, final
    info, actions), None if the episode ended, 'n/a' if the edit can't apply."""
    obs, _ = env.reset(options={'key': key, 'ckpt': kk})
    _, info = C.emu_frame(env.base_env, C.ACTION_BUTTONS[0])   # one no-op frame in every copy:
    if not edit(ed, env.base_env.data, info):                  # read the numbers, edit them
        return 'n/a'
    out, obss, acts_done = [], [], []
    for i in range(STEPS):
        a = actions[i] if actions is not None else int(model.predict(obs, deterministic=True)[0])
        obs, _, term, trunc, info = env.step(a)
        acts_done.append(a)
        out.append(judge(model.policy, obs))
        obss.append(obs)
        if term or trunc:
            return None
    return out, obss, info, acts_done


model = PPO.load(args.model or train_main.paths(args.run)['zip'] + '.zip', device='cpu')
model.policy.set_training_mode(False)
cfg = train_main.RUNS[args.run]
env = C.SonicCoordinator(acts=cfg['acts'], demo_prob=0.0)
for n, (addr, typ) in EXTRA_VARS.items():
    env.base_env.data.set_variable(n, {'address': addr, 'type': typ})
segs = env.demo_info()
print(f'{args.run} at {model.num_timesteps:,} steps; {len(segs)} acts, {args.per_act} start points each', flush=True)

res = collections.defaultdict(list)              # edit -> rows
natural = []                                     # A's own step-to-step change
skipped = collections.Counter()
for seg in segs:
    key, n = seg['key'], len(seg['fracs'])
    for kk in np.linspace(0, n - 2, args.per_act).astype(int):
        a = play(env, key, int(kk), 'none')
        if a is None:
            skipped['A ended early'] += 1
            continue
        outA, obsA, infoA, acts = a
        for (pa, va), (pb, vb) in zip(outA[-COMPARE - 1:-1], outA[-COMPARE:]):
            natural.append((float((pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum()), abs(vb - va)))
        for ed in EDITS:
            b = play(env, key, int(kk), ed, actions=acts)
            if b == 'n/a':
                continue
            if b is None:
                skipped[f'{ed}: B ended early'] += 1
                continue
            outB, obsB, infoB, _ = b
            if (int(infoA['x']), int(infoA['y'])) != (int(infoB['x']), int(infoB['y'])):
                skipped[f'{ed}: Sonic moved differently'] += 1
                continue
            d = np.abs(obsA[-1].astype(int) - obsB[-1].astype(int)).max(-1)
            outside = d.copy()
            outside[:HUD_Y, :HUD_X] = 0
            outside[LIVES_BOX] = 0
            if outside.any():
                skipped[f'{ed}: picture differs outside the HUD'] += 1
                continue
            if ed != 'none' and not d.any():
                skipped[f'{ed}: HUD did not change'] += 1
                continue
            for (pa, va), (pb, vb) in zip(outA[-COMPARE:], outB[-COMPARE:]):
                kl = float((pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum())
                res[ed].append(dict(act=seg['name'], kl=kl, flip=int(pa.argmax() != pb.argmax()),
                                    dv=vb - va, dp=pb - pa))
    print(f"  {seg['name']} done", flush=True)

nat = np.array(natural)
print(f"\nscale: the agent's own step-to-step change (copy A, {len(nat)} pairs): "
      f"KL median {np.median(nat[:, 0]):.4f}, |dV| median {np.median(nat[:, 1]):.3f}")
print(f"\n{'edit':12s} {'n':>4s} {'KL median':>10s} {'KL mean':>8s} {'flips':>6s} {'dV mean':>8s} {'|dV| med':>9s}   biggest action shifts (mean dprob)")
for ed in EDITS:
    rows = res[ed]
    if not rows:
        print(f'{ed:12s}    0')
        continue
    kl = np.array([r['kl'] for r in rows])
    dv = np.array([r['dv'] for r in rows])
    dp = np.mean([r['dp'] for r in rows], 0)
    top = ', '.join(f'{C.ACTION_NAMES[i]} {dp[i]:+.3f}' for i in np.argsort(-np.abs(dp))[:3])
    print(f"{ed:12s} {len(rows):4d} {np.median(kl):10.4f} {kl.mean():8.4f} {np.mean([r['flip'] for r in rows]):6.1%}"
          f" {dv.mean():+8.3f} {np.median(np.abs(dv)):9.3f}   {top}")
print('\nper act (KL median / mean dV):')
acts = [s['name'] for s in segs]
print(f"{'edit':12s} " + ' '.join(f'{a:>14s}' for a in acts))
for ed in EDITS:
    cells = []
    for a in acts:
        rows = [r for r in res[ed] if r['act'] == a]
        cells.append(f"{np.median([r['kl'] for r in rows]):.3f}/{np.mean([r['dv'] for r in rows]):+.2f}"
                     if rows else '-')
    print(f'{ed:12s} ' + ' '.join(f'{c:>14s}' for c in cells))
if skipped:
    print('\nskipped:', dict(skipped))
