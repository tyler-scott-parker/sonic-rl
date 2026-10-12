#!/usr/bin/env python3
"""giant_rings.py — every giant ring in the game, from the ROM, and how far each is from Tyler's route.

    python3 attic/one_off_tools/giant_rings.py [demos/play_<stamp>.npz]

Tyler 2026-10-12: "In a perfect world, I'd like the agent to get all 14 emeralds and beat the secret
final stage. I know that there are a number of giant rings that are almost unavoidable, however I don't
know how many there are."

The object layouts: a table of 4-byte pointers at $1E3D98 in the S&K half of the ROM, one list per
(zone, internal act), zones in the game's order (AIZ, HCZ, MGZ, CNZ, FBZ, ICZ, LBZ, MHZ, SOZ, LRZ, SSZ,
DEZ, DDZ); each list is 6-byte entries x.w, flags|y.w (y = low 12 bits), id.b, subtype.b, sorted by x,
ended by x = $FFFF. Object id $85 is the giant ring (found from AIZ1's ring at 7112,1216, the one the
special-stage handoff fixtures use). An act can span two internal acts (AIZ1's second half runs in
internal act 2's layout), so the route is matched by internal zone/act, not by name.
CPU only; builds the recording's segments (~6 min, ~1 GB).
"""
import collections
import json
import os
import struct
import sys

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
ROM = 'venv/lib/python3.12/site-packages/stable_retro/data/stable/SonicAndKnuckles3-Genesis-v0/rom.md'
TABLE, GIANT_RING = 0x1E3D98, 0x85
ZONES = ['AIZ', 'HCZ', 'MGZ', 'CNZ', 'FBZ', 'ICZ', 'LBZ', 'MHZ', 'SOZ', 'LRZ', 'SSZ', 'DEZ', 'DDZ']


def rings_in_rom():
    rom = open(ROM, 'rb').read()
    out = []
    for z, name in enumerate(ZONES):
        for a in (0, 1):
            p = struct.unpack('>I', rom[TABLE + 4 * (2 * z + a):TABLE + 4 * (2 * z + a) + 4])[0]
            while True:
                x, yf, oid, sub = struct.unpack('>HHBB', rom[p:p + 6])
                if x == 0xFFFF:
                    break
                if oid == GIANT_RING:
                    out.append(dict(zone=z, act=a, layout=f'{name}{a + 1}', x=x, y=yf & 0x0FFF, subtype=sub))
                p += 6
    return out


def main():
    rings = rings_in_rom()
    print(f'{len(rings)} giant rings in the object layouts:',
          ', '.join(f'{k} {v}' for k, v in collections.Counter(r['layout'] for r in rings).items()), flush=True)
    path = sys.argv[1] if len(sys.argv) > 1 else 'demos/play_20261010-202923.npz'
    os.environ.setdefault('MG_ENV_VERSION', '2')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MG_LOG'] = os.devnull
    import numpy as np
    import coordinator as C
    env = C.SonicCoordinator(demo_files=[path], acts=[])
    env.demo_info()
    for r in rings:
        best = None
        for seg in env._segs:
            rt = seg['route']
            m = (rt['zone'].astype(int) == r['zone']) & (rt['act'].astype(int) == r['act'])
            if not m.any():
                continue
            d = np.hypot(rt['x'][m].astype(float) - r['x'], rt['y'][m].astype(float) - r['y'])
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), seg['name'], int(rt['t'][m][i]), int(seg['T']))
        if best:
            r.update(dist=round(best[0]), act_name=best[1], his_t=best[2], his_T=best[3])
    rows = sorted(rings, key=lambda r: (r['zone'], r['act'], r['x']))
    print(f"\n{'layout':7s} {'x':>6s} {'y':>5s}  act   nearest his route   when")
    for r in rows:
        if 'dist' in r:
            print(f"{r['layout']:7s} {r['x']:6d} {r['y']:5d}  {r['act_name']:5s} {r['dist']:6d} px          "
                  f"{r['his_t'] / 60:5.1f} s of {r['his_T'] / 60:.0f} s")
        else:
            print(f"{r['layout']:7s} {r['x']:6d} {r['y']:5d}  (his route never in this layout)")
    near = [r for r in rows if r.get('dist', 1e9) <= C.ROUTE_RADIUS]
    close = [r for r in rows if C.ROUTE_RADIUS < r.get('dist', 1e9) <= 400]
    print(f"\nwithin {C.ROUTE_RADIUS} px of his route (on it, as the env sees it): {len(near)}; "
          f"within 400 px: {len(close)}; farther: {len(rows) - len(near) - len(close)}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'giant_rings.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
