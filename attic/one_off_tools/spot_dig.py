"""Stuck-spot dig (2026-10-16, Tyler picked it as a CPU job while gaming): sampled mg2 tries from a start point vs his own play through a box on the map.
    spot_dig.py ACT START_POINT N X0 X1 Y0 Y1 [T_LO T_HI]
Per try: outcome, where it ends, how long it spends in the box, the actions it uses there, and a sampled
trace through the box; then his trace (buttons) through the same box. T_LO/T_HI: only box frames whose
route progress (his frames) lies there (AIZ1's coordinates repeat after the fire). CPU only."""
import os, sys, random, collections
REPO = os.path.expanduser('~/Projects/sonic-rl'); sys.path.insert(0, REPO); os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np
import torch; torch.set_num_threads(2)
import coordinator as C, demos
from stable_baselines3 import PPO
ACT, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
X0, X1, Y0, Y1 = map(int, sys.argv[4:8])
TLO, THI = (int(sys.argv[8]), int(sys.argv[9])) if len(sys.argv) > 9 else (0, 10 ** 9)
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':' + ACT))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0); env.demo_info()
seg = env._segs[0]; key = seg['key']
model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
st = lambda v: ('roll' if v & 4 else 'run') + ('/air' if v & 2 else '')
inbox = lambda x, y: X0 <= x <= X1 and Y0 <= y <= Y1
print(f"{ACT} start point {K} (his t {seg['ckpts'][K]['t']} of {seg['T']}); box x {X0}-{X1} y {Y0}-{Y1}; model {model.num_timesteps:,}")
outs, ends = collections.Counter(), collections.Counter()
for i in range(N):
    random.seed(i); np.random.seed(i); torch.manual_seed(i)
    obs, _ = env.reset(options={'key': key, 'ckpt': K})
    done, trace, acts, info = False, [], collections.Counter(), None
    while not done:
        a = int(model.predict(obs, deterministic=False)[0])
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        x, y = int(info['x']), int(info['y'])
        t = env._route.best if getattr(env, '_route', None) is not None else 0
        if inbox(x, y) and TLO <= t <= THI:
            trace.append((env.frames, C.ACTION_NAMES[a], st(int(info['mg_status'])), x, y))
            acts[C.ACTION_NAMES[a]] += 1
    r = info['mg_result']
    outs[r['outcome']] += 1
    ends[(int(info['x']) // 64 * 64, int(info['y']) // 64 * 64, r['outcome'])] += 1
    print(f"try {i:2d}: {r['outcome']:7s} progress {r['progress']:.1%}, ends ({int(info['x'])},{int(info['y'])}); "
          f"{len(trace)} steps in the box; actions there {dict(acts.most_common(5))}", flush=True)
    if i < 4 and trace:
        print('      ' + ' '.join(f"{f}:{a}/{s}/({x},{y})" for f, a, s, x, y in trace[::max(1, len(trace) // 24)][:24]))
print('outcomes', dict(outs)); print('ends (64 px cells)', ends.most_common(8))
path = next(p for p in env.demo_files if demos.demo_id(p) == seg['demo'])
actions, meta = demos.load_play(path); table = meta['action_table']
env.reset(options={'key': key, 'ckpt': K}); ck = env._ck
BN = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
his, n0 = [], None
for n in range(min(6000, len(actions) - ck['frame'] - 1)):
    b = table[int(actions[ck['frame'] + 1 + n])]
    _, inf = C.emu_frame(env.base_env, b)
    x, y = int(inf['x']), int(inf['y'])
    if inbox(x, y) and TLO <= ck['t'] + n <= THI:
        n0 = n if n0 is None else n0
        if n % 8 == 0:
            his.append(f"{n}:{'+'.join(k for k, v in zip(BN, b) if v) or '-'}/{st(int(inf['mg_status']))}/({x},{y})")
    elif n0 is not None and n - n0 > 600:
        break
print(f"his ({len(his) * 8} frames in the box): " + ' '.join(his[:60]))
