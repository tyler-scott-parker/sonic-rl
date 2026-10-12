"""One-off: where does run aiz1 (paused at start point 69 = 82%) get stuck? CPU only.
    python3 probe_aiz1_wall.py [K] [N] [ENV_VERSION] [RUN]     # RUN default aiz1; e.g. 69 30 2 mg1
Runs its saved policy (stochastic, as in training) from start point K on its own env (v1),
logs outcome, frames, the agent's final x/y, its x path and route progress, and saves a
screenshot of the last frame of each stuck episode."""
import os, sys, json, collections
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
import train_main
K = int(sys.argv[1]) if len(sys.argv) > 1 else 69
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
VER = sys.argv[3] if len(sys.argv) > 3 else '1'
RUN = sys.argv[4] if len(sys.argv) > 4 else 'aiz1'
TAG = f'v{VER}' if RUN == 'aiz1' else f'{RUN}_v{VER}'
train_main.configure_env(RUN)
os.environ['MG_LOG'] = os.devnull; os.environ['MG_ENV_VERSION'] = VER; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
torch.set_num_threads(1)
from PIL import Image
import coordinator as C
from stable_baselines3 import PPO
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aiz1_wall')
os.makedirs(OUT, exist_ok=True)
env = C.SonicCoordinator(demo_prob=0.0)
seg = env.demo_info()[0]; key = seg['key']; s = env._segs[0]
model = PPO.load(train_main.paths(RUN)['zip'] + '.zip', device='cpu')
print(f'{RUN} save at {model.num_timesteps:,} steps, env v{VER}')
route = s['route']
print(f"start point {K}: demo t {s['ckpts'][K]['t']} of {s['T']}; demo x there {route['x'][np.searchsorted(route['t'], s['ckpts'][K]['t'])]}")
rows = []
for ep in range(N):
    obs, _ = env.reset(options={'key': key, 'ckpt': K})
    xs = []
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=False)
        obs, r, term, trunc, info = env.step(int(a))
        xs.append((int(info['x']), int(info['y'])))
        done = term or trunc
    res = info['mg_result']
    rows.append(dict(ep=ep, outcome=res['outcome'], frames=res['frames'], target=res['target'],
                     progress=res['progress'], best=env._route.best, x=xs[-1][0], y=xs[-1][1],
                     xmax=max(x for x, _ in xs), ymin=min(y for _, y in xs),
                     # Tyler's climb: back up-left onto the ledge (3,730, 821), then over the top at y ~760
                     ledge=any(x <= 3780 and y <= 860 for x, y in xs),
                     over_top=any(x >= 3750 and y <= 790 for x, y in xs)))
    print(rows[-1], flush=True)
    if res['outcome'] in ('stuck', 'timeout') and ep < 6:
        Image.fromarray(env.base_env.em.get_screen()).save(os.path.join(OUT, f'{TAG}_k{K}_ep{ep}_{res["outcome"]}.png'))
print(collections.Counter(r['outcome'] for r in rows))
print(f"reached the ledge {sum(r['ledge'] for r in rows)}/{len(rows)}, over the top {sum(r['over_top'] for r in rows)}/{len(rows)}")
print('route best (demo t) reached:', sorted(r['best'] for r in rows))
# where does the demo go from the best point most episodes reach?
b = int(np.median([r['best'] for r in rows]))
i = np.searchsorted(route['t'], b)
print('demo route around the median best:', [(int(route['t'][j]), int(route['x'][j]), int(route['y'][j])) for j in range(max(0, i - 60), min(len(route['t']), i + 300), 30)])
json.dump(rows, open(os.path.join(OUT, f'rows_{TAG}_k{K}.json'), 'w'))
