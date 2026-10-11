"""Replay Tyler's full playthrough; every 16 frames list giant-ring objects (code 0x061682)
in the object table: act, ring x/y, Sonic's nearest approach and the frame of it."""
import os, sys, json
sys.path.insert(0, '/home/tyler/Projects/sonic-rl'); os.chdir('/home/tyler/Projects/sonic-rl')
os.environ['MG_LOG'] = os.devnull
import numpy as np, coordinator as C, demos
GIANT = 0x061682
actions, meta = demos.load_play('demos/play_20261010-202923.npz')
table = meta['action_table']
emu = C.make_retro()
emu.reset(); emu.em.set_state(demos.play_start_state(meta))
split = C.ActSplitter()
rings = {}
def word(a, i): return (int(a[i]) << 8) | int(a[i + 1])
for i, a in enumerate(actions):
    raw, info = C.emu_frame(emu, table[a])
    split.feed(i, info)
    if i % 16 or not C.frame_flags(info)[0]:
        continue
    r = np.frombuffer(bytes(emu.get_ram()), dtype=np.uint8).reshape(-1, 2)[:, ::-1].reshape(-1)
    sx, sy = int(info['x']), int(info['y'])
    for slot in range(0xB000, 0xCFCC, 0x4A):
        if ((word(r, slot) << 16) | word(r, slot + 2)) == GIANT:
            x, y = word(r, slot + 0x10), word(r, slot + 0x14)
            act = split.cur['name'] if split.cur else '-'
            k = (act, int(info['zone']), int(info['act']), x, y)
            d = float(np.hypot(sx - x, sy - y))
            if k not in rings or d < rings[k]['dist']:
                rings[k] = dict(dist=round(d), frame=i, sonic=(sx, sy), t=split.cur['t'] if split.cur else None)
out = sorted(rings.items(), key=lambda kv: kv[1]['frame'])
for (act, z, ia, x, y), v in out:
    print(f'{act:6s} zone {z:#04x}/{ia} ring ({x},{y})  nearest {v["dist"]:4d} px at frame {v["frame"]} (act t {v["t"]}) Sonic {v["sonic"]}')
json.dump([dict(act=k[0], zone=k[1], iact=k[2], x=k[3], y=k[4], **v) for k, v in out],
          open('attic/one_off_tools/giant_rings.json', 'w'))
