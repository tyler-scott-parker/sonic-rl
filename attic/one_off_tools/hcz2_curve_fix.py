"""HCZ2 curve, counterfactual (2026-10-16, CPU job Tyler picked): the same sampled tries of mg2 with and without jump blocked near the curve's
top (jump -> same direction without jump while x 1500-1640, y >= 1990). Paired: each try uses the same seeds
in both conditions, so they match until the first blocked jump. CPU only."""
import os, sys, random, collections
REPO = os.path.expanduser('~/Projects/sonic-rl'); sys.path.insert(0, REPO); os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np
import torch; torch.set_num_threads(2)
import coordinator as C
from stable_baselines3 import PPO
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0); env.demo_info()
key = env._segs[0]['key']
model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
UNJUMP = {5: 0, 6: 1, 7: 2, 8: 3}


def zone(x, y):
    return 1500 <= x <= 1640 and y >= 1990


def one(i, block):
    random.seed(i); np.random.seed(i); torch.manual_seed(i)
    obs, _ = env.reset(options={'key': key, 'ckpt': 0})
    done, top, xmax, blocked, info = False, 99999, 0, 0, None
    while not done:
        a = int(model.predict(obs, deterministic=False)[0])
        if block and a in UNJUMP and info is not None and zone(int(info['x']), int(info['y'])):
            a, blocked = UNJUMP[a], blocked + 1
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        x, y = int(info['x']), int(info['y'])
        if x < 2200:
            xmax = max(xmax, x)
            if 1400 <= x:
                top = min(top, y)
    r = info['mg_result']
    return dict(up=top <= 1950, over=xmax >= 1700, top=top, blocked=blocked, outcome=r['outcome'],
                progress=r['progress'], frames=env.frames)


res = {False: [], True: []}
for i in range(N):
    for block in (False, True):
        res[block].append(one(i, block))
    a, b = res[False][-1], res[True][-1]
    print(f"try {i:2d}: as is {'UP ' if a['up'] else 'not'} top {a['top']:5d} {a['outcome']:7s} {a['progress']:.1%} | "
          f"jump blocked ({b['blocked']} presses) {'UP ' if b['up'] else 'not'} top {b['top']:5d} {b['outcome']:7s} {b['progress']:.1%}", flush=True)
for block in (False, True):
    rs = res[block]
    print(f"{'jump blocked' if block else 'as is       '}: up the curve {sum(r['up'] for r in rs)}/{N}, over it {sum(r['over'] for r in rs)}/{N}, "
          f"median progress {np.median([r['progress'] for r in rs]):.1%}, outcomes {dict(collections.Counter(r['outcome'] for r in rs))}")
