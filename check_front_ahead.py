#!/usr/bin/env python3
"""check_front_ahead.py — does MG_FRONT_AHEAD (RUNS 'front_ahead') run the front share's window (and the imitation
pull's front window) that many frames past each act's fresh-start frontier, and change nothing at 0?

    python3 check_front_ahead.py [RUN] [AHEAD]          # default mg2 1500

1. ahead 0 = the old front_ranges, for every act with its frontier set to every 60th frame of the act (and the
   current pointers), and the old imitation front windows at today's frontiers
2. ahead AHEAD: lo unchanged, hi = the last start point at or before frontier + AHEAD (below the pointer)
3. today's frontiers: each act's range before -> after; HCZ2's share of front-share starts inside the wall
   chase (his frames 560-2,000; doc 2026-10-17 ~09:00)
4. the imitation pull's front windows: + his frames up to frontier + AHEAD, nothing else; what HCZ2's new
   frames teach (his spin dashes: down / jump+down)
CPU only, no model. Ends with ALL CHECKS PASS or the failures.
"""
import collections
import json
import os
import sys
import time

HOME = os.path.dirname(os.path.abspath(__file__))
bad = []
CHASE = (560, 2000)


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    ahead = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    sys.path.insert(0, HOME)
    import numpy as np
    import train_main as T
    T.configure_env(run)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import coordinator as C
    print(f"RUNS {run} front_ahead={T.RUNS[run].get('front_ahead', 0)} (checking {ahead}); "
          f"FRONT_BACK {C.FRONT_BACK}, MG_FRONT_SHARE {C.MG_FRONT_SHARE}")
    env = C.SonicCoordinator(acts=T.RUNS[run]['acts'], demo_prob=0.0)
    env.demo_info()
    cur = json.load(open(T.paths(run)['curriculum']))
    env._ptr = {k: v['ptr'] for k, v in cur.items()}
    segs = env._segs

    def old_range(s, f):                                # front_ranges' arithmetic before the change (git 68dfde2)
        ts = [c['t'] for c in s['ckpts']]
        p = min(env._ptr[s['key']], len(ts) - 1)
        hi = min(p - 1, sum(t <= f for t in ts) - 1)
        lo = max(1, sum(t < f - C.FRONT_BACK for t in ts))
        return (lo, hi) if lo <= hi else None

    def new_range(s, f):
        env._front, env._front_read = {s['key']: f}, time.time() + 1e9     # pinned: no log re-read
        got = [(lo, hi) for g, lo, hi in env.front_ranges() if g['key'] == s['key']]
        return got[0] if got else None

    print('\n1. ahead 0 = the old ranges')
    C.MG_FRONT_AHEAD = 0
    n = same = 0
    for s in segs:
        for f in range(0, int(s['T']), 60):
            n += 1
            same += new_range(s, f) == old_range(s, f)
    check(same == n, f'{n:,} (act, frontier) pairs: front_ranges unchanged')

    print(f'\n2. ahead {ahead}: lo the same, hi out to frontier + {ahead}')
    C.MG_FRONT_AHEAD = ahead
    n = ok = 0
    for s in segs:
        ts = [c['t'] for c in s['ckpts']]
        p = min(env._ptr[s['key']], len(ts) - 1)
        for f in range(0, int(s['T']), 60):
            lo = max(1, sum(t < f - C.FRONT_BACK for t in ts))
            hi = min(p - 1, sum(t <= f + ahead for t in ts) - 1)
            n += 1
            ok += new_range(s, f) == ((lo, hi) if lo <= hi else None)
    check(ok == n, f'{n:,} (act, frontier) pairs as specified')

    print("\n3. today's frontiers")
    by = collections.defaultdict(list)
    for r in C.log_tail(C.FRONT_TAIL_BYTES):
        by[f"{r.get('demo')}:{r.get('act')}"].append(r)
    fronts = {s['key']: C.fresh_frontier(by[s['key']]) for s in segs}
    for s in segs:
        f = fronts[s['key']]
        if f is None:
            continue
        ts = [c['t'] for c in s['ckpts']]
        C.MG_FRONT_AHEAD = 0
        a = new_range(s, f)
        C.MG_FRONT_AHEAD = ahead
        b = new_range(s, f)
        fmt = lambda r: f'{r[0]}..{r[1]} (t {ts[r[0]]}..{ts[r[1]]})' if r else 'none'
        line = f"    {s['name']}: frontier t {f:.0f}, pointer {env._ptr[s['key']]}: {fmt(a)} -> {fmt(b)}"
        if s['name'] == 'HCZ2' and b:
            ks = range(b[0], b[1] + 1)
            inside = sum(CHASE[0] <= ts[k] <= CHASE[1] for k in ks)
            line += f'; {inside} of {len(ks)} start points inside the chase (t {CHASE[0]}..{CHASE[1]})'
            check(inside / len(ks) >= 0.5, 'HCZ2: most front-share starts inside the chase')
        print(line)

    print('\n4. the imitation pull\'s front windows')
    cfg = {**T.IMITATION_DEFAULTS, **T.RUNS[run]['imitation'], 'front': True}
    cb = T.ImitationCallback(cfg, run, T.paths(run)['curriculum'])
    cb.load_data()
    S = cb.S
    cb._front, cb._front_at = {k: fronts.get(k) for k in cb.keys}, time.time() + 1e9
    old = set()
    for key, idx in cb.keys.items():                    # front_windows as before the change
        f = fronts.get(key)
        if f is None or key not in cur or f >= cur[key]['frac'] * int(S['T'][idx[0]]):
            continue
        t = S['t'][idx]
        old |= set(idx[(t >= f - C.FRONT_BACK) & (t <= f)].tolist())
    C.MG_FRONT_AHEAD = 0
    w0 = set(np.concatenate(cb.front_windows(cur)).tolist())
    check(w0 == old, f'ahead 0: {len(w0):,} frames, the old windows')
    C.MG_FRONT_AHEAD = ahead
    wA = set(np.concatenate(cb.front_windows(cur)).tolist())
    extra = set()
    for key, idx in cb.keys.items():
        f = fronts.get(key)
        if f is None or key not in cur or f >= cur[key]['frac'] * int(S['T'][idx[0]]):
            continue
        t = S['t'][idx]
        extra |= set(idx[(t > f) & (t <= f + ahead)].tolist())
    check(wA == old | extra, f'ahead {ahead}: {len(w0):,} -> {len(wA):,} frames (+ his next {ahead} past each frontier)')
    names = C.ACTION_NAMES
    hk = next(k for k in cb.keys if k.endswith(':HCZ2'))
    idx = cb.keys[hk]
    t = S['t'][idx]
    ch = idx[(t >= CHASE[0]) & (t <= CHASE[1])]
    n = collections.Counter(int(a) for a in S['action'][ch])
    print(f'    HCZ2 chase frames now in the window: {len(set(ch.tolist()) & wA):,} of {len(ch):,}; his actions there: '
          + ', '.join(f'{names[k]} {v / len(ch):.0%}' for k, v in n.most_common(7)))
    print('\n' + ('ALL CHECKS PASS' if not bad else f'{len(bad)} FAILED: ' + '; '.join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
