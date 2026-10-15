"""Where and why mg2's MGZ1 tries get stuck near its pointer, vs his run. CPU only."""
import os, sys, collections, json
REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate')
import torch
torch.set_num_threads(1)
import coordinator as C
import demos
import pygame
from stable_baselines3 import PPO
OUT, N = sys.argv[1], int(sys.argv[2])
NAMES = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
SPOTS = {'A': (10850, 11150, 1650, 1950), 'B': (11650, 12150, 3550, 3950)}
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':MGZ1'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
s = env._segs[0]
em = env.base_env.em
P = json.load(open('main_mg2_curriculum.json'))[s['key']]['ptr']
ts = [c['t'] for c in s['ckpts']]
print(f"{s['key']}: T {s['T']}, pointer {P} at t {ts[P]} ({ts[P] / s['T']:.1%}); start points {P - 2}..{P + 3} at t {ts[P - 2:P + 4]}")


def shot(name):
    img = em.get_screen()
    pygame.image.save(pygame.surfarray.make_surface(img.swapaxes(0, 1)), os.path.join(OUT, name))


def spot(x, y):
    return next((k for k, (x0, x1, y0, y1) in SPOTS.items() if x0 <= x < x1 and y0 <= y < y1), None)


r = s['route']
print("his route from the pointer (t: x,y) every 30 frames:")
print('  ' + '  '.join(f"{int(t)}:{int(x)},{int(y)}" for t, x, y in zip(r['t'], r['x'], r['y'])
                       if ts[P] <= t <= s['T'] and t % 30 == 0))

path = next(p for p in env.demo_files if demos.demo_id(p) == s['demo'])
actions, meta = demos.load_play(path)
table = meta['action_table']
env.reset(options={'key': s['key'], 'ckpt': P})
ck = env._ck
prev_spot, prev_btn, seen = None, None, collections.Counter()
print("\nhis inputs from the pointer, in and around the stuck spots:")
for n in range(s['T'] - ck['t']):
    a = int(actions[ck['frame'] + 1 + n])
    _, info = C.emu_frame(env.base_env, table[a])
    x, y = int(info['x']), int(info['y'])
    sp = spot(x, y)
    btn = '+'.join(NAMES[i] for i, b in enumerate(table[a]) if b) or '-'
    if sp:
        seen[sp] += 1
        if sp != prev_spot or seen[sp] % 45 == 0:
            shot(f'his_{sp}_{n:05d}.png')
        if btn != prev_btn or sp != prev_spot:
            print(f"  {sp} t+{n} ({x},{y}) {btn} status {int(info['mg_status'])} routine {int(info['mg_routine'])}")
    elif prev_spot:
        print(f"  left {prev_spot} at t+{n} to ({x},{y})")
        shot(f'his_{prev_spot}_left_{n:05d}.png')
    prev_spot, prev_btn = sp, btn
    if C.frame_flags(info)[2]:
        print('  (died?)'); break
print(f"his frames in each spot: {dict(seen)}")

model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
print(f"\n{N} sampled tries of mg2 ({model.num_timesteps:,} steps) from the pointer {P}:")
for i in range(N):
    obs, _ = env.reset(options={'key': s['key'], 'ckpt': P})
    done, xs, acts, sp_frames = False, [], collections.deque(maxlen=150), collections.Counter()
    while not done:
        a, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
        x, y = int(info['x']), int(info['y'])
        acts.append(int(a))
        sp = spot(x, y)
        if sp:
            sp_frames[sp] += 1
            if sp_frames[sp] == 100 and i < 4:
                shot(f'agent{i}_{sp}_inside.png')
    res = info['mg_result']
    if i < 4 or res['outcome'] != 'clear':
        shot(f"agent{i}_end_{res['outcome']}.png")
    print(f"  try {i}: {res['outcome']:6s} after {env.frames} frames at ({x},{y}), x max {env._xmax}; steps in spots {dict(sp_frames)}; "
          f"last 150 actions {dict(collections.Counter(acts).most_common(5))}", flush=True)
print("agent actions: 0 none, 1 L, 2 R, 3 Down, 4 Up, 5 B, 6 B+L, 7 B+R, 8 B+Down")
