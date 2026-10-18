# act_fresh_eval.py ACT N CKPT... (2026-10-18 morning: AIZ1/AIZ2 fresh-start collapse overnight): N sampled fresh tries
# (start point 0) of ACT per checkpoint: outcomes, clears, median furthest x. CPU.
import os, sys, random, statistics, collections
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np, torch
torch.set_num_threads(1)
import coordinator as C
from stable_baselines3 import PPO
ACT, N = sys.argv[1], int(sys.argv[2])
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':' + ACT))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
for path in sys.argv[3:]:
    model = PPO.load(path, device='cpu')
    model.policy.set_training_mode(False)
    oc, xs = collections.Counter(), []
    for i in range(N):
        random.seed(i), np.random.seed(i), torch.manual_seed(i)
        obs, _ = env.reset(options={'key': seg['key'], 'ckpt': 0})
        done, xmax = False, 0
        while not done:
            obs, _, term, trunc, info = env.step(int(model.predict(obs, deterministic=False)[0]))
            done = term or trunc
            xmax = max(xmax, int(info['x']))
        oc[info['mg_result']['outcome']] += 1
        xs.append(xmax)
    print(f"{ACT} {model.num_timesteps:,}: {dict(oc)}; furthest x median {int(statistics.median(xs))}", flush=True)
