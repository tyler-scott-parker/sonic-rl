#!/usr/bin/env python3
"""check_front_share.py — does a run's front share (MG_FRONT_SHARE, RUNS 'front_share') start episodes in
the FRONT_BACK frames up to each act's fresh-start frontier, below its pointer?

    python3 check_front_share.py [RUN]          # default mg2; reads a copy of mg_log_<run>.jsonl's tail

1. fresh_frontier on made-up rows: the median of the last FRONT_N fresh starts' reach (progress x
   target); share starts (start point 0 drawn by a share included) and later start points left out;
   None under FRONT_MIN_N
2. front_ranges from the real log's tail, each act's range recomputed here independently: start
   points from 1 and below the pointer whose time is in [frontier - FRONT_BACK, frontier]; an act whose
   frontier is at or past its pointer, or whose pointer is 0, has none
3. 60,000 drawn starts (episode draws): MG_FRONT_SHARE of them (within 1 point) beside the boss and
   ring shares, acts evenly, every start point in its act's range (both ends reached), none at or
   past a pointer
4. the frontier follows the log: fresh starts appended that get further move that act's range at the
   next read (FRONT_EVERY_S after the last; aged here), and not before
5. a front-share reset runs and logs front_share; draw_key files it as ('front', act); it is not a
   fresh start; mg_stones.since_arrival doesn't take it for a rollback; it began below the pointer, so
   the curriculum's at-pointer rule never counts it
6. MG_FRONT_SHARE=0: no front starts
CPU only; writes no run file (the log tail is copied, episodes go to the copy). Ends with ALL CHECKS
PASS or the failures.
"""
import collections
import json
import os
import random
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
N = 60_000
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    real_log = os.environ['MG_LOG']
    tail = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    with open(real_log, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - (4 << 20)))
        data = f.read()
    with open(tail, 'wb') as f:
        f.write(data[data.index(b'\n') + 1:])
    os.environ.update(MG_LOG=tail, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    import mg_stones
    C.MG_BALANCE_TIME = False                       # the episode draws; check_balance_time.py does the time
    cfg = train_main.RUNS[run]
    check(C.MG_FRONT_SHARE == cfg.get('front_share', 0) > 0, f'MG_FRONT_SHARE {C.MG_FRONT_SHARE}')

    print('\n1. fresh_frontier on made-up rows')
    rows = []
    for i in range(40):                             # reach 0, 10, ..., 390 frames
        rows.append(dict(start=0, progress=i / 100, target=1000))
        rows.append(dict(start=0, progress=0.9, target=1000, ring_share=[0, 0, 1, 1]))
        rows.append(dict(start=0, progress=0.9, target=1000, boss_share=True))
        rows.append(dict(start=0, progress=0.9, target=1000, front_share=True))
        rows.append(dict(start=3, progress=0.9, target=1000))
    # the last 32 fresh: reach 80 .. 390, sorted; the median (index 16) is 240
    check(abs(C.fresh_frontier(rows) - 240) < 1e-9, f'median of the last {C.FRONT_N}: {C.fresh_frontier(rows)} (240)')
    few = [r for r in rows if C.is_fresh(r)][:C.FRONT_MIN_N - 1]
    check(C.fresh_frontier(few) is None, f'None with {len(few)} fresh starts')

    env = C.SonicCoordinator()
    env.demo_info()
    cur = json.load(open(os.path.join(HOME, f'main_{run}_curriculum.json')))
    ptrs = {k: v['ptr'] for k, v in cur.items()}
    env.set_demo_pointers(ptrs)
    segs = {s['key']: s for s in env._segs}

    print('\n2. the ranges from the real log')
    by = collections.defaultdict(list)
    for line in open(tail):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r['start'] == 0 and not (r.get('boss_share') or r.get('ring_share') or r.get('front_share')):
            by[f"{r['demo']}:{r['act']}"].append(r['progress'] * r['target'])
    fronts = env.front_ranges()
    got = {it[0]['key']: (it[1], it[2]) for it in fronts}
    for key, s in segs.items():
        reach = sorted(by[key][-C.FRONT_N:])
        f = reach[len(reach) // 2] if len(reach) >= C.FRONT_MIN_N else None
        ts = [c['t'] for c in s['ckpts']]
        p = min(ptrs.get(key, len(ts) - 1), len(ts) - 1)
        ks = [k for k in range(1, p) if f is not None and f - C.FRONT_BACK <= ts[k] <= f]
        want = (ks[0], ks[-1]) if ks else None
        desc = (f"frontier {f:.0f} of {s['T']} ({f / s['T']:.1%}), pointer {p} (t {ts[p]})" if f is not None
                else 'no frontier')
        check(got.get(key) == want, f"{s['name']}: {desc}: "
              + (f"start points {want[0]}..{want[1]} (t {ts[want[0]]}..{ts[want[1]]})" if want else 'none'))
    check(len(fronts) > 0, f'{len(fronts)} acts practise their fronts')

    print(f'\n3. {N:,} drawn starts')
    random.seed(11)
    draws = [env._pick({}) for _ in range(N)]
    fr = [(s['key'], k) for s, k, b, ring, front in draws if front]
    check(abs(len(fr) / N - C.MG_FRONT_SHARE) < 0.01, f'front share {len(fr) / N:.1%} (set {C.MG_FRONT_SHARE:.0%})')
    check(not any(front and (b or ring) for _, _, b, ring, front in draws), 'no start drawn by two shares')
    n_act = collections.Counter(key for key, _ in fr)
    even = len(fr) / max(1, len(got))
    worst = max(abs(n_act[k] - even) / even for k in got) if got else 0
    check(worst < 0.06, f'acts evenly: ' + ', '.join(f"{segs[k]['name']} {n_act[k]}" for k in got)
          + f' (worst {worst:.1%} off)')
    ks = collections.defaultdict(set)
    for key, k in fr:
        ks[key].add(k)
    check(all(got[key][0] <= k <= got[key][1] for key, k in fr), 'every start point inside its act\'s range')
    check(all(min(ks[k]) == got[k][0] and max(ks[k]) == got[k][1] for k in got), 'both ends of every range drawn')
    check(all(k < ptrs[key] for key, k in fr), 'none at or past a pointer')

    print('\n4. the frontier follows the log')
    key = max(got, key=lambda k: got[k][1] - got[k][0])         # the act with the widest range
    s = segs[key]
    ts = [c['t'] for c in s['ckpts']]
    p = min(ptrs[key], len(ts) - 1)
    new_f = min(ts[p] - 1, ts[got[key][1]] + 907)                # 15 s further, off a start point (float)
    with open(tail, 'a') as f:
        for _ in range(C.FRONT_N):
            f.write(json.dumps(dict(t=0, act=s['name'], demo=s['demo'], start=0, start_frac=0.0,
                                    outcome='stuck', frames=600, target=s['T'], progress=new_f / s['T'])) + '\n')
    same = {it[0]['key']: (it[1], it[2]) for it in env.front_ranges()}
    check(same.get(key) == got[key], f"{s['name']}: unchanged until the next read ({same.get(key)})")
    env._front_read -= C.FRONT_EVERY_S                         # a minute later
    moved = {it[0]['key']: (it[1], it[2]) for it in env.front_ranges()}
    ks = [k for k in range(1, p) if new_f - C.FRONT_BACK <= ts[k] <= new_f]
    check(moved.get(key) == (ks[0], ks[-1]), f"{s['name']}: {got[key]} -> {moved.get(key)} "
                                             f"(frontier {new_f:.0f}, expected {(ks[0], ks[-1])})")

    print('\n5. a front-share reset')
    C.MG_BOSS_SHARE, C.MG_RING_SHARE, share = 0.0, 0.0, C.MG_FRONT_SHARE
    C.MG_FRONT_SHARE = 1.0
    env.reset()
    k0, key0 = env._k, env._seg['key']
    check(env._front_start and env._src == ('front', key0), f"{env._seg['name']} start point {k0}: src {env._src}")
    done, n = False, 0
    while not done and n < 5000:
        _, _, term, trunc, info = env.step(random.randrange(len(C.ACTION_BUTTONS)))
        done, n = term or trunc, n + 1
    with open(tail) as f:
        r = json.loads(f.read().splitlines()[-1])
    check(done and r.get('front_share') is True and r['start'] == k0,
          f"logged: {r['outcome']} after {r['frames']} frames, front_share {r.get('front_share')}, start {r['start']}")
    check(C.draw_key(r, key0) == ('front', key0), f'draw_key {C.draw_key(r, key0)}')
    check(not C.is_fresh(r), 'not a fresh start')
    p0 = ptrs[key0]
    at = dict(r, start=p0, t=r['t'] - 10, front_share=False)
    later = dict(at, t=r['t'] + 10)
    check(len(mg_stones.since_arrival([at, r, later], p0)) == 2, 'since_arrival: no rollback marked')
    check(not (p0 <= r['start'] <= p0), 'below the pointer: the at-pointer rule skips it')
    C.MG_FRONT_SHARE = share

    print('\n6. MG_FRONT_SHARE=0')
    C.MG_FRONT_SHARE = 0.0
    check(not any(d[4] for d in (env._pick({}) for _ in range(5000))), 'no front starts')
    os.unlink(tail)
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
