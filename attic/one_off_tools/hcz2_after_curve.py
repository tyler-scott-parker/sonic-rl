"""HCZ2 after the curve (jump blocked near its top, as in hcz2_fix.py): what kills every try at ~4.8%,
vs what he did over the same stretch. CPU only."""
import os, sys, random, collections
REPO = os.path.expanduser('~/Projects/sonic-rl'); sys.path.insert(0, REPO); os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import numpy as np
import torch; torch.set_num_threads(2)
import coordinator as C, demos
from stable_baselines3 import PPO
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0); env.demo_info()
seg = env._segs[0]; key = seg['key']
model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
UNJUMP = {5: 0, 6: 1, 7: 2, 8: 3}
st = lambda v: ('roll' if v & 4 else 'run') + ('/air' if v & 2 else '')
deaths = collections.Counter()
for i in range(8):
    random.seed(i); np.random.seed(i); torch.manual_seed(i)
    obs, _ = env.reset(options={'key': key, 'ckpt': 0})
    done, info, trace = False, None, []
    while not done:
        a = int(model.predict(obs, deterministic=False)[0])
        if a in UNJUMP and info is not None and 1500 <= int(info['x']) <= 1640 and int(info['y']) >= 1990:
            a = UNJUMP[a]
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        if int(info['x']) >= 1650:
            trace.append((env.frames, C.ACTION_NAMES[a], st(int(info['mg_status'])), int(info['x']), int(info['y'])))
    deaths[(trace[-1][3] // 32 * 32, trace[-1][4] // 32 * 32)] += 1
    if i < 3:
        print(f"try {i} ({info['mg_result']['outcome']}): " + ' '.join(f"{f}:{a}/{s}/({x},{y})" for f, a, s, x, y in trace[::3][-30:]), flush=True)
print('ends (32 px cells):', dict(deaths))
# his frames over the same stretch
path = next(p for p in env.demo_files if demos.demo_id(p) == seg['demo'])
actions, meta = demos.load_play(path); table = meta['action_table']
env.reset(options={'key': key, 'ckpt': 0}); ck = env._ck
his = []
for n in range(1400):
    _, inf = C.emu_frame(env.base_env, table[int(actions[ck['frame'] + 1 + n])])
    if int(inf['x']) >= 1650 and n % 12 == 0:
        names = [k for k, v in zip(['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z'], table[int(actions[ck['frame'] + 1 + n])]) if v]
        his.append(f"{n}:{'+'.join(names) or '-'}/{st(int(inf['mg_status']))}/({int(inf['x'])},{int(inf['y'])})")
    if int(inf['x']) > 2600:
        break
print('his: ' + ' '.join(his[:45]))
