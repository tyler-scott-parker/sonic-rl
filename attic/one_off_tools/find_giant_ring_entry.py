"""From Tyler's MHZ1 start points near the giant ring at (8768,1088), with all 7 chaos
emeralds granted (count + collected array, the capture recipe): random inputs until the
ring is entered. Saves states 60 and 1 frames before (states/handoff/mhz1_ring7_*)."""
import os, sys, zlib, random, collections
sys.path.insert(0, '/home/tyler/Projects/sonic-rl'); os.chdir('/home/tyler/Projects/sonic-rl')
os.environ['MG_LOG'] = os.devnull
import numpy as np, coordinator as C
env = C.SonicCoordinator(demo_files=['demos/play_20261010-202923.npz'], acts=['MHZ1'])
env.demo_info(); seg = env._segs[0]; emu = env.base_env
for i in range(7):
    emu.data.set_variable(f'em_arr{i}', {'address': 0xFFFFB2 + i, 'type': '|u1'})
r = seg['route']; i = int(np.argmin(np.hypot(r['x'] - 8768, r['y'] - 1088))); t_near = int(r['t'][i])
ks = [k for k, c in enumerate(seg['ckpts']) if t_near - 300 <= c['t'] <= t_near]
print('route nearest at t', t_near, 'start points', ks)
def grant():
    emu.data.set_value('chaos_emeralds', 7)
    for i in range(7):
        emu.data.set_value(f'em_arr{i}', 1)
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
for attempt in range(6000):
    k = random.choice(ks)
    emu.reset(); emu.em.set_state(zlib.decompress(seg['ckpts'][k]['state'])); grant()
    hist = collections.deque(maxlen=61); f, a, left = 0, 0, 0
    while f < 600:
        if left == 0:
            a, left = random.randrange(len(C.ACTION_BUTTONS)), random.randint(4, 30)
        hist.append(emu.em.get_state())
        raw, info = C.emu_frame(emu, C.ACTION_BUTTONS[a]); f += 1; left -= 1
        if C.frame_flags(info)[2]:
            break
        if int(info['mg_entry']) or int(info['mg_game_mode']) != C.GM_LEVEL:
            print(f'ENTRY attempt {attempt} start point {k} after {f} frames at x {info["x"]} y {info["y"]}: '
                  f'entry {int(info["mg_entry"])} gm {int(info["mg_game_mode"]):#x} zone {int(info["zone"]):#x}')
            open('states/handoff/mhz1_ring7_minus60.state', 'wb').write(hist[0])
            open('states/handoff/mhz1_ring7_minus1.state', 'wb').write(hist[-1])
            sys.exit(0)
    if attempt % 300 == 0:
        print('attempt', attempt, flush=True)
print('no entry')
