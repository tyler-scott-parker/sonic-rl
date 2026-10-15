#!/usr/bin/env python3
"""check_ring_share.py — does a run's ring share (MG_RING_SHARE, RUNS 'ring_share') start episodes
where his route passes nearest each giant ring, and does MG_BAR_FORGIVE leave out of the curriculum's
time bar exactly what the reward forgives?

    python3 check_ring_share.py [RUN]          # default mg2

1. the ring starts: every giant ring within RING_SHARE_PX of his route, each start point the one at
   or before where he passes nearest (computed here independently of the env)
2. 60,000 drawn starts with the run's boss share on too: ring share MG_RING_SHARE and boss share
   MG_BOSS_SHARE of them (within 1 point), rings evenly, each start point k or k - 1
3. MG_RING_SHARE=0: no ring starts
4. a ring-share reset runs and logs ring_share with its ring; mg_stones.since_arrival ignores it
5. the AIZ1 try that takes the giant ring and gets back (check_ss_detour's ring_back): the log's
   'forgiven' = trip_there + the stage's cost + trip_back, and its giant_rings has the ring
6. the bar: a clear that is slower than the bar only by forgiven frames matches him with
   MG_BAR_FORGIVE and not without
CPU only; the episode log goes to a temp file. Ends with ALL CHECKS PASS or the failures.
"""
import collections
import json
import os
import random
import sys
import tempfile

import numpy as np

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
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    C.MG_BALANCE_TIME = False                           # the episode draws; check_balance_time.py does the time
    import mg_stones
    cfg = train_main.RUNS[run]
    check(C.MG_RING_SHARE == cfg.get('ring_share', 0) > 0, f'MG_RING_SHARE {C.MG_RING_SHARE}')
    check(C.MG_BAR_FORGIVE == bool(cfg.get('bar_forgive')), f'MG_BAR_FORGIVE {C.MG_BAR_FORGIVE}')
    check(C.MG_EMERALD_BONUS == cfg.get('emerald_bonus', 0), f'MG_EMERALD_BONUS {C.MG_EMERALD_BONUS}')
    log = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    C.LOG_FILE = log
    env = C.SonicCoordinator()
    env.demo_info()
    cur = json.load(open(os.path.join(HOME, f'main_{run}_curriculum.json')))
    ptrs = {k: v['ptr'] for k, v in cur.items()}
    env.set_demo_pointers(ptrs)

    print('\n1. the ring starts')
    rings = C.giant_rings()
    want = {}
    for s in env._segs:
        r = s['route']
        for i in range(len(r['t'])):
            za = (int(r['zone'][i]), int(r['act'][i]))
            for rx, ry in rings.get(za, ()):
                d = float(np.hypot(r['x'][i] - rx, r['y'][i] - ry))
                key = (s['key'],) + za + (rx, ry)
                if d <= C.RING_SHARE_PX and (key not in want or d < want[key][0]):
                    want[key] = (d, int(r['t'][i]))
    got = {(s['key'],) + tuple(ring): (s, k) for s, k, ring in env.ring_starts()}
    check(set(got) == set(want), f'{len(got)} rings within {C.RING_SHARE_PX} px of his route (expected {len(want)})')
    for key in sorted(want, key=lambda x: (x[0], want[x][1])):
        if key not in got:
            continue
        s, k = got[key]
        ts = [c['t'] for c in s['ckpts']]
        d, t = want[key]
        check(ts[k] <= t and (k + 1 == len(ts) or ts[k + 1] > t),
              f"{s['name']} giant ring at ({key[3]},{key[4]}): {d:.0f} px off, nearest at t {t}, start point {k} (t {ts[k]})")

    print(f'\n2. {N:,} drawn starts (boss share {C.MG_BOSS_SHARE:.0%} on too)')
    random.seed(2)
    draws = [env._pick({}) for _ in range(N)]
    ring_d = [(s, k, ring) for s, k, b, ring in draws if ring]
    boss_d = [1 for _, _, b, _ in draws if b]
    expect_boss = C.MG_BOSS_SHARE if env.boss_ranges() else 0.0
    check(abs(len(ring_d) / N - C.MG_RING_SHARE) < 0.01, f'ring share {len(ring_d) / N:.1%} (set {C.MG_RING_SHARE:.0%})')
    check(abs(len(boss_d) / N - expect_boss) < 0.01, f'boss share {len(boss_d) / N:.1%} (set {expect_boss:.0%})')
    check(not any(b and ring for _, _, b, ring in draws), 'no start drawn by both')
    per = collections.Counter((s['key'],) + tuple(ring) for s, _, ring in ring_d)
    even = len(ring_d) / max(1, len(got))
    check(all(abs(n - even) < 0.2 * even for n in per.values()) and len(per) == len(got),
          f'rings drawn evenly ({min(per.values())}-{max(per.values())} each, even {even:.0f})')
    check(all(k in (max(0, got[(s['key'],) + tuple(ring)][1] - 1), got[(s['key'],) + tuple(ring)][1])
              for s, k, ring in ring_d), 'each at its start point or the one before')

    print('\n3. MG_RING_SHARE=0')
    share = C.MG_RING_SHARE
    C.MG_RING_SHARE = 0.0
    check(not any(ring for *_, ring in (env._pick({}) for _ in range(5000))), 'no ring starts')

    print('\n4. a ring-share reset')
    C.MG_RING_SHARE, boss = 1.0, C.MG_BOSS_SHARE
    C.MG_BOSS_SHARE = 0.0
    env.reset()
    target = env._ring_target
    n, done = 0, False
    while not done and n < 100:
        _, _, term, trunc, info = env.step(random.randrange(len(C.ACTION_BUTTONS)))
        done, n = term or trunc, n + 1
    if not done:
        env._result('stuck', info)
    line = json.loads(open(log).read().strip().splitlines()[-1])
    check(line.get('ring_share') == list(target) and line['start'] == env._k and not line.get('boss_share'),
          f"{line['act']} start {line['start']}: log ring_share {line.get('ring_share')}")
    rows = [dict(t=1, start=50), dict(t=2, start=7, ring_share=[0, 0, 1, 1]), dict(t=3, start=50)]
    check(len(mg_stones.since_arrival(rows, 50)) == 2, 'mg_stones.since_arrival: a ring-share start below the pointer is no rollback')
    C.MG_RING_SHARE, C.MG_BOSS_SHARE = share, boss

    print('\n5. the AIZ1 ring try, forgiven frames')
    tries = json.load(open(os.path.join(HOME, 'logs', 'check_ss_detour_mg1.json')))
    t = tries['ring_back']
    key = next(s['key'] for s in env._segs if s['name'] == 'AIZ1')
    env.reset(options={'key': key, 'ckpt': t['kk']})
    i = j = 0
    done = False
    while not done:
        if t.get('then') and env._stages:
            a = t['then'][j % len(t['then'])]
            j += 1
        elif i < len(t['actions']):
            a = t['actions'][i]
            i += 1
        else:
            a = 0
        _, _, term, trunc, info = env.step(a)
        done = term or trunc
    line = json.loads(open(log).read().strip().splitlines()[-1])
    st = line.get('stages', [{}])[0]
    parts = st.get('trip_there', 0) + st.get('cost', 0) + st.get('trip_back', 0)
    check(st.get('emeralds', 0) > 0 and st.get('trip_back', 0) > 0, f"won the stage and got back ({st.get('outcome')}, "
          f"trip there {st.get('trip_there')}, stage {st.get('cost')}, back {st.get('trip_back')})")
    check(line.get('forgiven') == parts > 0, f"forgiven {line.get('forgiven')} = {parts}")
    check(len(line.get('giant_rings', [])) == 1, f"giant_rings {line.get('giant_rings')}")

    print('\n6. the bar')
    for forgive in (True, False):
        C.MG_BAR_FORGIVE = forgive
        env.reset(options={'key': key, 'ckpt': 0})
        _, _, _, _, info = env.step(0)
        bar = env._target * (1 + C.MATCH_SLACK) + C.MATCH_ADD
        env.frames, env._forgiven = int(bar) + 100, 150
        res = env._result('clear', info)
        check(res['success'] == forgive, f"MG_BAR_FORGIVE={forgive}: {env.frames} frames, bar {bar:.0f}, "
                                          f"150 forgiven -> matched him: {res['success']}")
    C.MG_BAR_FORGIVE = bool(cfg.get('bar_forgive'))
    os.unlink(log)
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
