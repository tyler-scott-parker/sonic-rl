#!/usr/bin/env python3
"""mg_air.py — how close did Tyler come to drowning? Per act of a recording: frames underwater, the
longest stretch, the lowest air left, how often the countdown would show.

    python3 attic/one_off_tools/mg_air.py [demos/play_<stamp>.npz]

Tyler 2026-10-12: "should we measure drowning as it's own type of death?" The env already counts
drowning as death (Player 1's routine $C). The agent sees 4 frames and no sound, so it can't tell how
long it has been underwater until the countdown digits show. Player 1 ($FFB000): status +$2A bit 6 =
underwater; air left +$2C (30 when out of water, one less per second under; the countdown digits from
12; drowned at 0) -- checked here against the underwater bit. CPU, one core, one replay (~6 min).
"""
import collections
import os
import sys

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MG_ENV_VERSION', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MG_LOG'] = os.devnull
import numpy as np                      # noqa: E402
import coordinator as C                 # noqa: E402
import demos                            # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else 'demos/play_20261010-202923.npz'
actions, meta = demos.load_play(path)
table = meta['action_table']
base = C.make_retro()
base.reset()
base.em.set_state(demos.play_start_state(meta))
split, swap = C.ActSplitter(), None
stats = collections.defaultdict(lambda: dict(under=0, longest=0, run=0, min_air=99, countdown=0, drowned=0,
                                             air_seen=collections.Counter()))
cur = None
for i, a in enumerate(actions):
    raw, info = C.emu_frame(base, table[a])
    event = split.feed(i, info)
    if event == 'start':
        cur = split.cur['name']
    elif event == 'clear':
        cur = None
    if cur is None:
        continue
    r = np.frombuffer(bytes(base.get_ram()), dtype=np.uint8)
    if swap is None:
        x = int(info['x'])
        if x >> 8 != x & 0xFF:
            swap = (int(r[0xB010]) << 8 | int(r[0xB011])) != x
        else:
            continue
    ram = r.reshape(-1, 2)[:, ::-1].reshape(-1) if swap else r
    s = stats[cur]
    under = bool(ram[0xB02A] & 0x40)
    air = int(ram[0xB02C])
    if under:
        s['under'] += 1
        s['run'] += 1
        s['longest'] = max(s['longest'], s['run'])
        s['min_air'] = min(s['min_air'], air)
        s['air_seen'][air] += 1
    else:
        s['run'] = 0
    if under and air <= 12 and (i == 0 or s.get('last_air', 99) > 12):
        s['countdown'] += 1
    s['last_air'] = air if under else 99
    if int(info['mg_routine']) == 0xC:
        s['drowned'] += 1
print(f'{os.path.basename(path)}: acts with time underwater (frames; air left counts seconds, drowned at 0)')
for name, s in stats.items():
    if s['under']:
        common = ', '.join(f'{k}' for k, _ in sorted(s['air_seen'].items())[:3])
        print(f"  {name:6s} underwater {s['under']:6d} frames ({s['under'] / 60:5.0f} s), longest stretch "
              f"{s['longest'] / 60:5.1f} s, lowest air left {s['min_air']:2d}, countdown shown {s['countdown']}x, "
              f"drowned frames {s['drowned']}; lowest air values seen: {common}")
print('no time underwater in:', ', '.join(n for n, s in stats.items() if not s['under']))
