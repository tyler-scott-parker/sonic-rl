"""Every boss-like fight in Tyler's recordings vs the Boss_flag the env builds fights from. CPU only.
A 'hit' here is BossTracker's rule without its 5..16 count window: an occupied slot (same object code)
whose +$29 dropped by exactly 1, touchable within 64 frames, >= 16 frames since that slot's last one.
Hits are grouped per act and object code into fights (gaps <= 900 frames)."""
import os, sys, glob, collections
REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
import numpy as np
import coordinator as C
import demos

files = [p for p in os.environ['MG_DEMOS'].split(',') if p.strip()]
files += [p for p in glob.glob('demos/play_20261011-095800*.npz') if p not in files]
env = C.make_retro()
for path in files:
    actions, meta = demos.load_play(path)
    table = meta['action_table']
    env.reset()
    env.em.set_state(demos.play_start_state(meta))
    split = C.ActSplitter()
    n = C.OBJ_SLOTS - C.BOSS_SLOT0
    prev = None
    peak, touch, last = np.zeros(n, int), np.full(n, -10 ** 9), np.full(n, -10 ** 9)
    flag = collections.defaultdict(list)      # act -> t's with Boss_flag
    hits = collections.defaultdict(list)      # act -> (t, code, peak, left, x, y, slot)
    clears = {}
    print(f"\n=== {demos.demo_id(path)}: {len(actions)} frames", flush=True)
    for i, a in enumerate(actions):
        raw, info = C.emu_frame(env, table[a])
        ev = split.feed(i, info)
        if ev == 'clear':
            clears[split.acts[-1]['name']] = split.acts[-1]
        cur = C.object_table(env, info)
        if cur is None:
            continue
        name = split.cur['name'] if split.cur else None
        t = split.cur['t'] if split.cur else None
        if name and int(info['mg_boss']):
            flag[name].append(t)
        occ = cur[:, :4].any(1)
        if prev is not None:
            same = occ & (cur[:, :4] == prev[:, :4]).all(1)
            peak[~same] = 0
            touch[~same] = -10 ** 9
            last[~same] = -10 ** 9
            v = cur[:, 0x29].astype(int)
            peak = np.maximum(peak, np.where(occ, v, 0))
            touch[(prev[:, 0x28] != 0) | (cur[:, 0x28] != 0)] = i
            hit = same & (v == prev[:, 0x29].astype(int) - 1) & (peak >= 2) & (i - touch <= 64) & (i - last >= 16)
            last[hit] = i
            if name:
                for sl in np.flatnonzero(hit):
                    row = cur[sl]
                    w = lambda o: int(row[o]) << 8 | int(row[o + 1])
                    code = int.from_bytes(bytes(row[:4]), 'big')
                    hits[name].append((t, code, int(peak[sl]), int(v[sl]), w(0x10), w(0x14), int(sl)))
        else:
            peak = np.where(occ, cur[:, 0x29].astype(int), 0)
        prev = cur
    for act in [a['name'] for a in split.acts] + ([split.cur['name']] if split.cur else []):
        T = clears[act]['t'] if act in clears else None
        fl = sorted(set(flag.get(act, [])))
        iv = []
        for x in fl:
            if iv and x - iv[-1][1] <= C.FIGHT_GAP:
                iv[-1][1] = x
            else:
                iv.append([x, x])
        groups = []
        for h in sorted(hits.get(act, [])):
            g = next((g for g in groups if g['code'] == h[1] and h[0] - g['end'] <= 900), None)
            if g is None:
                groups.append(dict(code=h[1], start=h[0], end=h[0], n=0, peak=h[2], x=h[4], y=h[5], flagged=0))
                g = groups[-1]
            g['end'], g['n'], g['peak'] = h[0], g['n'] + 1, max(g['peak'], h[2])
            g['flagged'] += any(a <= h[0] <= b + 1 for a, b in iv)
        big = [g for g in groups if g['n'] >= 3]
        print(f"{act}: T {T}; Boss_flag stretches {[tuple(x) for x in iv]}")
        for g in big:
            endnote = f", the act's clock stops {T - g['end']} frames after its last hit" if T is not None and 0 <= T - g['end'] <= 600 else ''
            status = 'FLAGGED' if g['flagged'] == g['n'] else 'NOT FLAGGED' if g['flagged'] == 0 else f"partly flagged {g['flagged']}/{g['n']}"
            print(f"   fight: object ${g['code']:08X} (count {g['peak']}), {g['n']} hits, t {g['start']}..{g['end']} at ({g['x']},{g['y']}): {status}{endnote}")
        small = [g for g in groups if g['n'] < 3]
        if small:
            print(f"   (+{len(small)} objects hit once or twice: badniks etc.)")
