#!/usr/bin/env python3
"""mg_wall_objects.py — do the collapsing platforms at AIZ1's wall break under the agent, and how far
must Sonic go for them to come back?

    python3 attic/one_off_tools/mg_wall_objects.py [RUN] [N]

Tyler 2026-10-12: "I think there is normally a ledge there that collapses. If the ledge breaks before you
clear the area, then you may need to move far enough away so that the ledge despawns and respawns."
The ROM layout (giant_rings.py's table, AIZ internal act 2) has object $04 at (3696, 872) and, flipped,
at (3857, 807), on his climb (start point 72, the rock pile at ~(3822, 940); he goes up-left to
(3754, 838), up to (3747, 766), then right along y ~760). Watched in object RAM: a slot whose x/y words
are the platform's (an object slot is $4A bytes from $FFB000; x at +$10, y at +$14).

1. his own inputs from start point 72 (the control: he clears)
2. N sampled tries of RUN's save from start point 72: when each platform disappears, where Sonic was
   then, and how the try ended
3. after a platform is gone: Sonic held Left for 0.5 s .. 10 s, then Right back; does it come back?
CPU only.
"""
import os
import sys

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
RUN = sys.argv[1] if len(sys.argv) > 1 else 'mg1'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
import train_main                       # noqa: E402
train_main.configure_env(RUN)
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate')
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
torch.set_num_threads(1)
import coordinator as C                 # noqa: E402
import demos                            # noqa: E402
from stable_baselines3 import PPO       # noqa: E402

PLATFORMS = {'ledge (3696,872)': (3696, 872), 'top (3857,807)': (3857, 807)}
K = 72
env = C.SonicCoordinator(acts=['AIZ1'], demo_prob=0.0)
seg = env.demo_info()[0]
key = seg['key']
s = env._segs[0]


def ram():
    raw = np.frombuffer(bytes(env.base_env.get_ram()), dtype=np.uint8)
    return raw.reshape(-1, 2)[:, ::-1].reshape(-1) if SWAP else raw


def word(r, a):
    return int(r[a]) << 8 | int(r[a + 1])


def present(r):
    """Which platforms have an object slot at their position (x, y within 2 px)."""
    out = {}
    for name, (px, py) in PLATFORMS.items():
        out[name] = any(abs(word(r, b + 0x10) - px) <= 2 and abs(word(r, b + 0x14) - py) <= 2
                        for b in range(0xB000, 0xCFB6, 0x4A))
    return out


env.reset(options={'key': key, 'ckpt': K})
_raw = np.frombuffer(bytes(env.base_env.get_ram()), dtype=np.uint8)
_x = int(env.base_env.data.lookup_value('x'))
SWAP = (int(_raw[0xB010]) << 8 | int(_raw[0xB011])) != _x
print(f"start point {K} (demo t {s['ckpts'][K]['t']}): Sonic at ({_x}, {int(env.base_env.data.lookup_value('y'))}); "
      f"platforms present: {present(ram())}", flush=True)

# 1. his inputs (frame for frame, like check_play_acts)
actions, meta = demos.load_play(s['demo_path'] if 'demo_path' in s else env.demo_files[0])
table = meta['action_table']
orig, at = C.emu_frame, {'f': 0}
gone_his = {}


def his_input(e, _mask):
    at['f'] += 1
    out = orig(e, table[actions[at['f']]])
    for name, p in present(ram()).items():
        if not p and name not in gone_his:
            gone_his[name] = (env.frames, int(out[1]['x']), int(out[1]['y']))
    return out


C.emu_frame = his_input
env.reset(options={'key': key, 'ckpt': K})
at['f'] = s['ckpts'][K]['frame']
done = False
while not done:
    _, _, term, trunc, info = env.step(0)
    done = term or trunc
C.emu_frame = orig
print(f"1. his inputs: {info['mg_result']['outcome']} in {info['mg_result']['frames']} frames; "
      f"platforms gone (frame, Sonic x, y): {gone_his or 'none'}", flush=True)

# 2. sampled tries of the run's save
model = PPO.load(train_main.paths(RUN)['zip'] + '.zip', device='cpu')
print(f'2. {N} sampled tries of {RUN} ({model.num_timesteps:,} steps) from start point {K}:', flush=True)
broke_state = None
for i in range(N):
    obs, _ = env.reset(options={'key': key, 'ckpt': K})
    gone, ymin, done = {}, 9999, False
    while not done:
        a, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
        ymin = min(ymin, int(info['y']))
        for name, p in present(ram()).items():
            if not p and name not in gone:
                gone[name] = (env.frames, int(info['x']), int(info['y']))
                if broke_state is None and not done:
                    broke_state = (name, env.base_env.em.get_state())
    print(f"   try {i + 1:2d}: {info['mg_result']['outcome']:7s} after {env.frames:5d} frames, highest y {ymin:4d}; "
          f"gone: {gone or 'none'}", flush=True)

# 3. respawn: from the first try that broke a platform, hold Left T frames, then Right as long
if broke_state is None:
    print('3. no platform broke in the tries: nothing to respawn')
else:
    name, st = broke_state
    print(f'3. {name} broke; hold Left, then Right back as long, then check:', flush=True)
    L, R = C.ACTION_BUTTONS[1], C.ACTION_BUTTONS[2]
    for T in (30, 60, 120, 180, 240, 360, 480, 600):
        env.base_env.em.set_state(st)
        xs = []
        for _ in range(T):
            _, info = C.emu_frame(env.base_env, L)
            xs.append(int(info['x']))
        for _ in range(T):
            _, info = C.emu_frame(env.base_env, R)
        print(f"   Left {T / 60:4.1f} s (Sonic to x {min(xs)}), back: present {present(ram())[name]}", flush=True)
