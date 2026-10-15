"""MGZ1's end boss: fights per act; his fight vs the agent's (hits, route progress, stuck). CPU only."""
import os, sys, collections, json
REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate')
import numpy as np
import torch
torch.set_num_threads(1)
import coordinator as C
import demos
import pygame
from stable_baselines3 import PPO
OUT, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
env.demo_info()
print("fights per act (his frames) and how each act's clock stops:")
for s in env._segs:
    print(f"  {s['name']}: T {s['T']}, fights {s['route']['fights']}, boss-flag start points "
          f"{sum(c['boss'] for c in s['ckpts'])} of {len(s['ckpts'])}")
s = next(x for x in env._segs if x['name'] == 'MGZ1')
key = s['key']
em = env.base_env.em


def boss_slot():
    bt = env._bt
    if bt is None or bt.prev is None:
        return None
    cand = np.flatnonzero((bt.maxseen >= C.BOSS_MIN) & (bt.maxseen <= C.BOSS_MAX) & bt.prev[:, :4].any(1))
    if not len(cand):
        return None
    row = bt.prev[cand[0]]
    w = lambda o: int(row[o]) << 8 | int(row[o + 1])
    return w(0x10), w(0x14), int(row[0x29])


def shot(name):
    img = em.get_screen()
    pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), os.path.join(OUT, name))


def episode(choose, after_reset=None, tag=None):
    obs, _ = env.reset(options={'key': key, 'ckpt': K})
    if after_reset:
        after_reset()
    t0 = s['ckpts'][K]['t']
    done, prev, hits, trace, last_move = False, 0, [], [], 0
    best0 = env._route.best
    hist = collections.deque(maxlen=200)
    while not done:
        obs, _, term, trunc, info = env.step(choose(obs))
        done = term or trunc
        if env._boss_hits > prev:
            hits += [env.frames] * (env._boss_hits - prev)
            prev = env._boss_hits
        b = boss_slot()
        trace.append((env.frames, int(info['x']), int(info['y']), env._route.best - t0, b))
        hist.append(em.get_screen().copy())
    res = info['mg_result']
    if tag is not None and res['outcome'] != 'clear':
        for back in (150, 75, 1):
            img = hist[-back] if len(hist) >= back else hist[0]
            pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), os.path.join(OUT, f'{tag}_minus{back * 4}.png'))
    return res, hits, trace


def summary(name, res, hits, trace):
    prog = [(f, p) for f, _, _, p, _ in trace]
    at_hit = [next(p for f, p in prog if f >= h) for h in hits]
    bpos = [b for *_, b in trace if b]
    last = trace[-1]
    print(f"  {name}: {res['outcome']} at {res['frames']} frames (target {res['target']}); hits at {hits}; "
          f"route progress at each hit {at_hit}, at the end {last[3]}; Sonic ends ({last[1]},{last[2]}), boss there {last[4]}; "
          f"boss x range {min(b[0] for b in bpos) if bpos else '-'}..{max(b[0] for b in bpos) if bpos else '-'}, y "
          f"{min(b[1] for b in bpos) if bpos else '-'}..{max(b[1] for b in bpos) if bpos else '-'}", flush=True)
    return at_hit


print(f"\nMGZ1 start point {K} (t {s['ckpts'][K]['t']}), his remaining {s['T'] - s['ckpts'][K]['t']} frames")
path = next(p for p in env.demo_files if demos.demo_id(p) == s['demo'])
actions, meta = demos.load_play(path)
table = meta['action_table']
orig, at = C.emu_frame, {'f': 0}


def his_input(e, _mask):
    at['f'] += 1
    return orig(e, table[actions[at['f']]])


C.emu_frame = his_input
try:
    res, hits, trace = episode(lambda o: 0, after_reset=lambda: at.update(f=s['ckpts'][K]['frame']))
finally:
    C.emu_frame = orig
summary('Tyler', res, hits, trace)
gaps = [b - a for a, b in zip([0] + hits, hits)]
print(f"    his gaps between hits {gaps}")
model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
print(f"\n{N} sampled tries of mg2 ({model.num_timesteps:,} steps):")
out = collections.Counter()
for i in range(N):
    res, hits, trace = episode(lambda o: int(model.predict(o, deterministic=False)[0]), tag=f'try{i}' if i < 6 else None)
    summary(f'try {i}', res, hits, trace)
    out[res['outcome']] += 1
    gaps = [b - a for a, b in zip([0] + hits, hits)]
    if res['outcome'] != 'clear':
        print(f"    gaps {gaps}; frames since the last hit at the end {res['frames'] - (hits[-1] if hits else 0)}")
print('outcomes', dict(out))
