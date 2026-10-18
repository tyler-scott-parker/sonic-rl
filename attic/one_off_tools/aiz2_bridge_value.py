# aiz2_bridge_value.py CKPT (2026-10-17 night): the critic along AIZ2 tries from 130: does the value fall once Sonic
# crosses the rope bridge into the boss arena (x > ~18,700), i.e. does the agent rate the fight below stalling before it?
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
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':AIZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
model = PPO.load(sys.argv[1], device='cpu')
model.policy.set_training_mode(False)
for seed in [int(s) for s in (sys.argv[2:] or ['7', '0'])]:
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    obs, _ = env.reset(options={'key': seg['key'], 'ckpt': 130})
    done, pts = False, []
    while not done:
        o, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            v = float(model.policy.predict_values(o)[0][0])
        a = int(model.predict(obs, deterministic=False)[0])
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        if env.frames % 80 < 4 and env.frames >= 640:
            pts.append(f"{int(info['x'])}:{v:+.2f}")
    print(f"seed {seed} ({info['mg_result']['outcome']}): x:value every 80 frames from 640: " + ' '.join(pts[:34]), flush=True)
