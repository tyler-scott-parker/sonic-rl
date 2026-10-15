"""What kills mg2 just past HCZ2's first curve? CPU only."""
import os, sys, collections
REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate')
import torch
torch.set_num_threads(2)
import coordinator as C
import demos
import pygame
from stable_baselines3 import PPO
OUT = sys.argv[1]
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
s = env._segs[0]
em = env.base_env.em


def save(img, name):
    pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), os.path.join(OUT, name))


path = next(p for p in env.demo_files if demos.demo_id(p) == s['demo'])
actions, meta = demos.load_play(path)
env.reset(options={'key': s['key'], 'ckpt': 0})
ck = env._ck
for n in range(1000):
    _, info = C.emu_frame(env.base_env, meta['action_table'][int(actions[ck['frame'] + 1 + n])])
    if n in (700, 740, 790, 880, 915):
        save(em.get_screen(), f'his_wait_{n}.png')
        print(f"his frame {n}: ({int(info['x'])},{int(info['y'])})")
model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
deaths = 0
for i in range(30):
    obs, _ = env.reset(options={'key': s['key'], 'ckpt': 0})
    done, hist = False, collections.deque(maxlen=40)
    while not done:
        a, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
        hist.append((em.get_screen().copy(), int(info['x']), int(info['y']), int(info['mg_routine'])))
    res = info['mg_result']
    if res['outcome'] == 'death' and int(info['x']) > 1600:
        for back in (40, 20, 8, 1):
            img, x, y, rt = hist[-back]
            save(img, f'death{deaths}_minus{back * 4:03d}.png')
        print(f"try {i}: death at ({info['x']},{info['y']}) cause {res.get('cause')} hurt {res.get('hurt')}; "
              f"last positions {[h[1:] for h in list(hist)[-8:]]}", flush=True)
        deaths += 1
        if deaths == 2:
            break
