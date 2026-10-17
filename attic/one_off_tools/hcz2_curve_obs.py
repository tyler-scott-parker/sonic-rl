# hcz2_curve_obs.py CKPT (2026-10-17 corridor dig): at the curve's top the agent presses jump+right after its own spin
# dash (start point 6) and right after a start mid-roll (7), in near-identical states. Its observation is only 4 stacked
# 84x84 frames (+ overlays), so: P(any jump) at the decision step in each run, where the two observations differ, and
# P(jump) with only the HUD strip (top rows: TIME/SCORE/RINGS) swapped between them. CPU.
import os, sys, random
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np, torch
torch.set_num_threads(1)
import coordinator as C
from stable_baselines3 import PPO
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
model = PPO.load(sys.argv[1], device='cpu')
model.policy.set_training_mode(False)
JUMPS = [C.ACTION_NAMES.index(n) for n in ('jump', 'jump+left', 'jump+right', 'jump+down')]


def probs(obs):
    o, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return model.policy.get_distribution(o).distribution.probs[0].numpy()


def decision_obs(k, seed=0):
    """Run from start point k (sampled actions) and return the observation at the last step before x >= 1583."""
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    obs, _ = env.reset(options={'key': seg['key'], 'ckpt': k})
    trail = []
    for _ in range(400):
        p = probs(obs)
        a = int(model.predict(obs, deterministic=False)[0])
        trail.append((obs.copy(), p, a))
        obs, _, term, trunc, info = env.step(a)
        if int(info['x']) >= 1582 or term or trunc:
            break
    return obs, info, trail


o6, i6, t6 = decision_obs(6)
o7, i7, t7 = decision_obs(7)
pj = lambda p: float(p[JUMPS].sum())
print(f"obs shape {o6.shape} dtype {o6.dtype}")
print(f"k6 at ({int(i6['x'])},{int(i6['y'])}): P(any jump) {pj(probs(o6)):.2f}, P(jump+right) {probs(o6)[JUMPS[2]]:.2f}")
print(f"k7 at ({int(i7['x'])},{int(i7['y'])}): P(any jump) {pj(probs(o7)):.2f}, P(jump+right) {probs(o7)[JUMPS[2]]:.2f}")
print('P(any jump) over the last 8 steps before the curve top: k6', [round(pj(p), 2) for _, p, _ in t6[-8:]],
      '| k7', [round(pj(p), 2) for _, p, _ in t7[-8:]])
d = np.abs(o6.astype(int) - o7.astype(int))
latest = d[..., -1] if d.shape[-1] in (3, 4) else d[-1]
rows = latest.sum(axis=1)
print('newest frame: rows that differ (sum |diff| per row, rows 0-83):', [int(r) for r in rows])
for H in (8, 12, 16, 20):
    a, b = o6.copy(), o7.copy()
    if o6.shape[-1] in (3, 4):
        a[:H, :, :], b[:H, :, :] = o7[:H, :, :], o6[:H, :, :]
    else:
        a[:, :H, :], b[:, :H, :] = o7[:, :H, :], o6[:, :H, :]
    print(f'top {H} rows swapped: k6 with k7\'s HUD P(jump) {pj(probs(a)):.2f}; k7 with k6\'s HUD P(jump) {pj(probs(b)):.2f}')
from PIL import Image
f6 = o6[..., -1] if o6.shape[-1] in (3, 4) else o6[-1]
f7 = o7[..., -1] if o7.shape[-1] in (3, 4) else o7[-1]
Image.fromarray(np.concatenate([f6, np.full((84, 4), 255, np.uint8), f7], axis=1)).resize((172 * 4, 84 * 4), Image.NEAREST).save('logs/hcz2_curve_obs_k6_k7.png')
print('saved logs/hcz2_curve_obs_k6_k7.png (k6 left, k7 right, newest frame)')


# the critic along each run: does the value fall once Sonic is over the curve (the chase), i.e. does the agent
# expect climbing to go worse than stalling?
def values(k, seed, steps=120):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    obs, _ = env.reset(options={'key': seg['key'], 'ckpt': k})
    out = []
    for _ in range(steps):
        o, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            v = float(model.policy.predict_values(o)[0][0])
        a = int(model.predict(obs, deterministic=False)[0])
        obs, _, term, trunc, info = env.step(a)
        out.append((int(info['x']), int(info['y']), v))
        if term or trunc:
            out.append(('END', info['mg_result']['outcome'], None))
            break
    return out


for k, seed in ((7, 0), (6, 0)):
    tr = values(k, seed)
    pts = [p for p in tr if p[0] == 'END' or 1400 <= p[0] <= 2000][::3]
    print(f'k {k} value along x: ' + ' '.join(f'{p[0]},{p[1]}:{p[2]:+.2f}' if p[0] != 'END' else f'END {p[1]}' for p in pts))
