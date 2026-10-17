# hcz2_curve_trace.py CKPT [K...] (2026-10-17 corridor dig): the agent's presses from x 1,300 on (the curve's
# approach and the curve), per sampled try from start point K: why its own dash (k 6) fails the curve when a start
# mid-roll (k 7) climbs it. CPU.
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
short = {'noop': '.', 'left': 'L', 'right': 'R', 'down': 'd', 'up': 'u', 'jump': 'J', 'jump+left': 'JL', 'jump+right': 'JR', 'jump+down': 'Jd'}
st = lambda v: ('roll' if v & 4 else 'run') + ('/air' if v & 2 else '')
for k in [int(v) for v in sys.argv[2:]] or [6, 7]:
    print(f"k {k} (his t {seg['ckpts'][k]['t']})")
    for i in range(6):
        random.seed(i), np.random.seed(i), torch.manual_seed(i)
        obs, _ = env.reset(options={'key': seg['key'], 'ckpt': k})
        done, trail, xmax, first = False, [], 0, []
        while not done and len(trail) < 70:
            a = int(model.predict(obs, deterministic=False)[0])
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            x, y = int(info['x']), int(info['y'])
            if len(first) < 14:
                first.append(short[C.ACTION_NAMES[a]])
            if x >= 1300:
                trail.append(f"{short[C.ACTION_NAMES[a]]}@{x},{y}{'r' if int(info['mg_status']) & 4 else ''}{'a' if int(info['mg_status']) & 2 else ''}")
            xmax = max(xmax, x)
            if x >= 1700:
                break
        print(f"  try {i}: {'UP ' if xmax >= 1650 else 'fail'} first presses {' '.join(first)} | from x 1300: {' '.join(trail[:26])}")
