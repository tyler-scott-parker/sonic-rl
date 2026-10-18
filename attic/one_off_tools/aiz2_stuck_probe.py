# aiz2_stuck_probe.py CKPT [N] (2026-10-17 night, Claude in charge; AIZ2's pointer stuck at 130 since ~15:00, 75% of its
# at-pointer tries 'stuck' at x 18,176-18,687, y ~384-511: the burning jungle's end and the rope bridge before the
# boss arena): N sampled tries from start point 130; per try the outcome, x every 100 frames, and for stuck ones the
# screen at the end (logs/boss_frames/AIZ2_130/stuck_<i>.png). CPU.
import os, sys, random
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np, torch
torch.set_num_threads(1)
from PIL import Image
import coordinator as C
from stable_baselines3 import PPO
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':AIZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
model = PPO.load(sys.argv[1], device='cpu')
model.policy.set_training_mode(False)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
short = {'noop': '.', 'left': 'L', 'right': 'R', 'down': 'd', 'up': 'u', 'jump': 'J', 'jump+left': 'JL', 'jump+right': 'JR', 'jump+down': 'Jd'}
os.makedirs('logs/boss_frames/AIZ2_130', exist_ok=True)
for i in range(N):
    random.seed(i), np.random.seed(i), torch.manual_seed(i)
    obs, _ = env.reset(options={'key': seg['key'], 'ckpt': 130})
    done, xs, acts = False, [], []
    while not done:
        a = int(model.predict(obs, deterministic=False)[0])
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        acts.append(short[C.ACTION_NAMES[a]])
        if env.frames % 100 < 4:
            xs.append(f"{int(info['x'])},{int(info['y'])}")
    oc = info['mg_result']['outcome']
    print(f"try {i:2d}: {oc:8s} {env.frames:5d} frames | " + ' '.join(xs[:30]), flush=True)
    if oc == 'stuck':
        Image.fromarray(np.array(env.render())).save(f'logs/boss_frames/AIZ2_130/stuck_{i}.png')
        print(f"         last 60 presses: {' '.join(acts[-60:])}", flush=True)
