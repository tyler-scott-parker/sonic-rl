# mgz2_flight_actions.py (2026-10-17 MGZ2 dig): Tyler's vs the 83M agent's action mix from start point 270, hover (frames 0-640) vs Tails-carry flight (640+); jump presses; flight height. CPU.
import os, sys, random, collections, json
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ['MG_LOG'] = os.devnull
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
torch.set_num_threads(1)
import coordinator as C, demos
from stable_baselines3 import PPO
K = 270
env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
env.demo_info()
seg = next(s for s in env._segs if s['name'] == 'MGZ2')
key = seg['key']
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
f0 = seg['ckpts'][K]['frame']
# Tyler: his per-frame action indices from the start point; phases by frame
his = [int(actions[f0 + 1 + i]) for i in range(2940)]
def hist(v):
    c = collections.Counter(v); n = sum(c.values())
    return {C.ACTION_NAMES[a]: round(100 * c[a] / n) for a in sorted(c)}
print('Tyler hover (0-640):', hist(his[:640]))
print('Tyler flight (640-2940):', hist(his[640:]))
# his jump presses (rising edges) in flight
jb = [i for i in range(641, 2940) if C.ACTION_BUTTONS[his[i]][0] and not C.ACTION_BUTTONS[his[i-1]][0]]
print('Tyler fresh jump presses in flight:', len(jb), jb[:40])
model = PPO.load('checkpoints_main_mg2_keep/main_mg2_83045392_steps.zip', device='cpu')
random.seed(0); np.random.seed(0); torch.manual_seed(0)
hov, fl, ys, presses, eps = [], [], [], [], 0
for ep in range(16):
    obs, _ = env.reset(options={'key': key, 'ckpt': K})
    done, prev = False, 0
    n_press = 0
    while not done:
        a = int(model.predict(obs, deterministic=False)[0])
        (hov if env.frames < 640 else fl).append(a)
        if env.frames >= 640 and C.ACTION_BUTTONS[a][0] and not C.ACTION_BUTTONS[prev][0]:
            n_press += 1
        prev = a
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        if env.frames >= 640:
            ys.append(int(info['y']))
    presses.append((n_press, env.frames))
print('agent hover (0-640):', hist(hov))
print('agent flight (640+):', hist(fl))
print('agent fresh jump presses in flight per 1000 frames:', [round(1000 * p / max(1, f - 640), 1) for p, f in presses])
print('Tyler fresh jump presses per 1000 flight frames:', round(1000 * len(jb) / 2300, 1))
c = collections.Counter(y // 32 * 32 for y in ys); n = len(ys)
print('agent flight y (32px bands, %):', {k: round(100 * v / n) for k, v in sorted(c.items())})
