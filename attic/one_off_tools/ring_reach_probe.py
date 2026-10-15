"""For each giant ring mg2 hasn't been taking: how close do sampled tries from its ring-share start get,
and what's around it? CPU only."""
import os, sys, math
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
import pygame
from stable_baselines3 import PPO
OUT, N = sys.argv[1], int(sys.argv[2])
TAKEN = {(14008, 1488), (12608, 1208)}
env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
env.demo_info()
em = env.base_env.em
model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
print(f"mg2 save {model.num_timesteps:,} steps; {N} tries per ring from its ring-share start point", flush=True)


def save(img, name):
    pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), os.path.join(OUT, name))


for seg, k, ring in env.ring_starts():
    z, a, rx, ry = ring
    if (rx, ry) in TAKEN:
        continue
    r = seg['route']
    m = (r['zone'] == z) & (r['act'] == a)
    d = np.hypot(r['x'][m] - rx, r['y'][m] - ry)
    i = int(d.argmin())
    his = (int(r['x'][m][i]), int(r['y'][m][i]), int(r['t'][m][i]))
    print(f"\n{seg['name']} ring ({rx},{ry}): his route passes nearest at ({his[0]},{his[1]}), {d[i]:.0f} px, "
          f"dx {rx - his[0]:+d} dy {ry - his[1]:+d} (his frame {his[2]}); start point {k} (t {seg['ckpts'][k]['t']})", flush=True)
    for j in range(N):
        obs, _ = env.reset(options={'key': seg['key'], 'ckpt': k})
        if j == 0:
            save(em.get_screen(), f"{seg['name']}_{rx}_{ry}_start.png")
        done, best, best_img, best_at = False, 1e9, None, None
        while not done:
            obs, _, term, trunc, info = env.step(int(model.predict(obs, deterministic=False)[0]))
            done = term or trunc
            if int(info['zone']) == z and int(info['act']) == a:
                dist = math.hypot(int(info['x']) - rx, int(info['y']) - ry)
                if dist < best:
                    best, best_at = dist, (env.frames, int(info['x']), int(info['y']))
                    if j == 0:
                        best_img = em.get_screen().copy()
        res = info['mg_result']
        took = [list(x) for x in env._used_rings]
        print(f"  try {j}: {res['outcome']:6s} after {env.frames} frames; closest {best:.0f} px at frame {best_at[0] if best_at else '-'} "
              f"Sonic ({best_at[1] if best_at else '-'},{best_at[2] if best_at else '-'}){'; TOOK IT' if list(ring) in took else ''}", flush=True)
        if j == 0 and best_img is not None:
            save(best_img, f"{seg['name']}_{rx}_{ry}_closest.png")
