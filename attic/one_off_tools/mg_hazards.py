#!/usr/bin/env python3
"""mg_hazards.py — where does a recording of the main game lean on items, loop, or ride a barrel?

    python3 attic/one_off_tools/mg_hazards.py [demos/play_<stamp>.npz] [--acts CNZ1,CNZ2]

Tyler 2026-10-12: "i used items to rush my way across dangerous areas ... a fire power up during the
eggman fight in the lava ... star power ups ... areas in levels that loop until a condition is met.
plus, the barrel in carnival night". The backward curriculum starts the agent from HIS save states,
so inside such a stretch it inherits his item; once the start point moves back past the monitor it
must break the monitor itself. This maps those stretches.

Replays every cleared act from its first start point with his per-frame inputs (as check_play_acts.py
does) and reads, every frame:
  - Player 1's status_secondary ($FFB02B): bit 1 invincible, 2 speed shoes, 4 fire / 5 lightning /
    6 bubble shield (bit 0 = any shield). Spans of each, where they began and ended, and how many
    rings he lost inside (a hit while invincible costs nothing; a hit with a shield costs the shield).
  - revisits: 128-px blocks he enters again and again (a new visit = back after >= 2 s away). Blocks
    with many visits are where the act loops (or he went round in circles).
  - Up/Down alternation (the Carnival Night barrels): windows where he switches between Up and Down
    at least 6 times within 5 s, with the shortest hold (the agent holds each action 4 frames).
CPU only; one core; ~1M frames per full game. JSON next to this file (mg_hazards_<stamp>.json).
"""
import collections
import json
import os
import sys
import time

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MG_ENV_VERSION', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MG_LOG'] = os.devnull
import numpy as np                      # noqa: E402

import coordinator as C                 # noqa: E402
import demos                            # noqa: E402

STATUS2 = 0xB02B                        # Player_1 ($FFB000) + status_secondary ($2B)
X, Y, RINGS = 0xB010, 0xB014, 0xFE20
BITS = {1: 'invincible', 2: 'speed shoes', 4: 'fire shield', 5: 'lightning shield', 6: 'bubble shield'}
UP, DOWN = 4, 5                         # button index in the 12-button Genesis mask
BLOCK, AWAY = 128, 120


def word(ram, a):
    return int(ram[a]) << 8 | int(ram[a + 1])


def spans(flags):
    """[(first, last)] runs of True."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else 'demos/play_20261010-202923.npz'
    only = None
    if '--acts' in sys.argv:
        only = set(sys.argv[sys.argv.index('--acts') + 1].split(','))
    actions, meta = demos.load_play(path)
    table = meta['action_table']
    t0 = time.time()
    env = C.SonicCoordinator(demo_files=[path], acts=sorted(only) if only else [])
    env.demo_info()
    print(f'{os.path.basename(path)}: {len(env._segs)} acts, segments built in {time.time() - t0:.0f} s', flush=True)
    orig = C.emu_frame
    at = {'f': 0}
    rec = {'s2': [], 'x': [], 'y': [], 'rings': [], 'up': [], 'down': []}

    def his_input(e, _mask):
        at['f'] += 1
        mask = table[actions[at['f']]]
        out = orig(e, mask)
        raw = np.frombuffer(bytes(e.get_ram()), dtype=np.uint8)
        if at.get('swap') is None:                  # stable-retro may keep Genesis RAM byte-swapped:
            x = int(out[1]['x'])                    # decide once, on a frame whose x bytes differ
            if x >> 8 != x & 0xFF:
                at['swap'] = (int(raw[X]) << 8 | int(raw[X + 1])) != x
        ram = raw.reshape(-1, 2)[:, ::-1].reshape(-1) if at.get('swap') else raw
        rec['s2'].append(int(ram[STATUS2]))
        rec['x'].append(word(ram, X))
        rec['y'].append(word(ram, Y))
        rec['rings'].append(word(ram, RINGS))
        rec['up'].append(bool(mask[UP]))
        rec['down'].append(bool(mask[DOWN]))
        return out

    C.emu_frame = his_input
    report = {}
    try:
        for seg in env._segs:
            name = seg['name']
            if only and name not in only:
                continue
            for v in rec.values():
                v.clear()
            env.reset(options={'key': seg['key'], 'ckpt': 0})
            at['f'] = seg['ckpts'][0]['frame']
            done = False
            while not done:
                _, _, term, trunc, info = env.step(0)
                done = term or trunc
            res = info['mg_result']
            s2 = np.array(rec['s2'])
            xs, ys, rings = np.array(rec['x']), np.array(rec['y']), np.array(rec['rings'])
            items = []
            for bit, label in BITS.items():
                for a, b in spans((s2 >> bit) & 1):
                    lost = int(sum(max(0, rings[i - 1] - rings[i]) for i in range(max(a, 1), b + 1)
                                   if rings[i] == 0 and rings[i - 1] > 0))
                    items.append(dict(item=label, t0=a, t1=b, frames=b - a + 1,
                                      at=(int(xs[a]), int(ys[a])), until=(int(xs[b]), int(ys[b])),
                                      ended='act end' if b == len(s2) - 1 else 'lost/expired',
                                      rings_lost_in_hits=lost))
            # revisits per 128-px block
            visits, last = collections.Counter(), {}
            for i in range(0, len(xs), 4):
                blk = (int(xs[i]) // BLOCK, int(ys[i]) // BLOCK)
                if blk not in last or i - last[blk] > AWAY:
                    visits[blk] += 1
                last[blk] = i
            loops = [dict(block=b, px=(b[0] * BLOCK, b[1] * BLOCK), visits=n) for b, n in visits.most_common(8) if n >= 4]
            # Up/Down alternation windows
            ud = [(i, 'U' if u else 'D') for i, (u, d) in enumerate(zip(rec['up'], rec['down'])) if u != d]
            switches = [ud[j][0] for j in range(1, len(ud)) if ud[j][1] != ud[j - 1][1]]
            barrels, j = [], 0
            while j < len(switches):
                k = j
                while k + 1 < len(switches) and switches[k + 1] - switches[j] <= 300:
                    k += 1
                if k - j + 1 >= 6:
                    holds = np.diff(switches[j:k + 1])
                    barrels.append(dict(t0=switches[j], t1=switches[k], switches=k - j + 1,
                                        shortest_hold=int(holds.min()), median_hold=float(np.median(holds)),
                                        at=(int(xs[switches[j]]), int(ys[switches[j]]))))
                    j = k + 1
                else:
                    j += 1
            report[name] = dict(outcome=res['outcome'], frames=res['frames'], target=res['target'],
                                items=items, loops=loops, barrels=barrels)
            print(f"\n{name}: {res['outcome']} in {res['frames']} (his {res['target']})", flush=True)
            for it in items:
                print(f"  {it['item']:16s} frames {it['t0']}-{it['t1']} ({it['frames']}), from {it['at']} to {it['until']},"
                      f" {it['ended']}, rings lost to hits inside: {it['rings_lost_in_hits']}", flush=True)
            for lp in loops:
                print(f"  revisited block at {lp['px']}: {lp['visits']} visits", flush=True)
            for br in barrels[:6]:
                print(f"  up/down rhythm frames {br['t0']}-{br['t1']}: {br['switches']} switches, hold shortest "
                      f"{br['shortest_hold']} median {br['median_hold']:.0f} frames, at {br['at']}", flush=True)
    finally:
        C.emu_frame = orig
    stamp = os.path.basename(path).replace('play_', '').replace('.npz', '')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mg_hazards_{stamp}.json')
    json.dump(report, open(out, 'w'), indent=1, default=str)
    print(f'\n{time.time() - t0:.0f} s; wrote {out}')


if __name__ == '__main__':
    main()
