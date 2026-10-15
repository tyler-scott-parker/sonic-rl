"""Does HCZ2's first curve need Sonic ROLLING? CPU only."""
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
OUT, N = sys.argv[1], int(sys.argv[2])
B = C.ACTION_BUTTONS
NOOP, LEFT, RIGHT, JUMP, DOWN = B[0], B[1], B[2], B[5], B[3]
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
s = env._segs[0]
em = env.base_env.em


def st(info):
    v = int(info['mg_status'])
    return ('roll' if v & 4 else 'run') + ('/air' if v & 2 else '')


def shot(name):
    img = em.get_screen()
    pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), os.path.join(OUT, name))


def run(seq, label):
    xm, ym, states, px = 0, 99999, [], None
    for btn, n in seq:
        for _ in range(n):
            _, info = C.emu_frame(env.base_env, btn)
            x, y = int(info['x']), int(info['y'])
            xm, ym = max(xm, x), min(ym, y)
            if 1450 <= x <= 1600 and y >= 2000:
                states.append(st(info))
            px = x
    print(f"  {label}: {'UP AND OVER' if xm >= 1700 else 'not up'} (x max {xm}, highest y {ym}); "
          f"Sonic at the curve: {dict(collections.Counter(states))}", flush=True)


path = next(p for p in env.demo_files if demos.demo_id(p) == s['demo'])
actions, meta = demos.load_play(path)
table = meta['action_table']
env.reset(options={'key': s['key'], 'ckpt': 0})
ck = env._ck
his = lambda n: table[int(actions[ck['frame'] + 1 + n])]
saved = {}
for n in range(600):
    _, info = C.emu_frame(env.base_env, his(n))
    if n in (360, 500):
        saved[n] = em.get_state()
        print(f"his frame {n}: ({int(info['x'])},{int(info['y'])}) {st(info)}")
print("from his frame 360 (standing at 306, before his spin dash):")
em.set_state(saved[360]); run([(RIGHT, 300)], 'RIGHT held 300 frames'); shot('A_right_from_306.png')
print("from his frame 500 (x ~1127, rolling at ~7 px/frame):")
em.set_state(saved[500]); run([(his(n), 1) for n in range(501, 640)], 'his own inputs (control)')
em.set_state(saved[500]); run([(NOOP, 140)], 'nothing pressed (stays rolling)')
em.set_state(saved[500]); run([(JUMP, 4), (RIGHT, 140)], 'one jump, then RIGHT (lands running)')
em.set_state(saved[500]); run([(LEFT, 30), (RIGHT, 140)], 'LEFT 30 frames (brakes), then RIGHT')

model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
print(f"\n{N} sampled fresh tries of mg2 ({model.num_timesteps:,} steps): how Sonic meets the curve")
for i in range(N):
    obs, _ = env.reset(options={'key': s['key'], 'ckpt': 0})
    done, first, jumps_after_dash, dashed = False, None, 0, False
    while not done:
        a, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
        x, y = int(info['x']), int(info['y'])
        if int(info['mg_spindash']):
            dashed = True
        if dashed and first is None and int(a) in (5, 6, 7, 8) and x > 420:
            jumps_after_dash += 1
        if first is None and 1450 <= x and y >= 2000:
            first = (env.frames, x, st(info))
            if i < 3:
                shot(f'agent{i}_at_curve.png')
    print(f"  try {i}: {info['mg_result']['outcome']:6s} progress {info['mg_result']['progress']:.1%}; first at the curve: "
          f"{first}; jump presses between its dash and the curve: {jumps_after_dash}", flush=True)
