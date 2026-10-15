#!/usr/bin/env python3
"""check_boss_share.py — does a run's boss share (MG_BOSS_SHARE, RUNS 'boss_share') start episodes
inside the boss fights its acts are stuck in, and do the act weights (MG_ACT_WEIGHTS, RUNS
'act_weights') draw the other starts as set?

    python3 check_boss_share.py [RUN]          # default mg2; pointers from main_<run>_curriculum.json

1. his boss fights per act as start points, and which acts' pointers sit inside one: each range
   runs from the fight's first start point to the pointer's band
2. 60,000 drawn starts: the boss share's are all inside a stuck act's range, its share is
   MG_BOSS_SHARE (within 1 point), split evenly between the stuck acts; the other starts' acts
   follow the weights and their start points follow the old rule (pointer band, or 0)
3. an act whose pointer leaves its fight drops out; with no act in a fight nothing is drawn by it
4. with MG_BOSS_SHARE=0 and no weights the draws are the old ones (acts uniform)
5. mg_stones.since_arrival: a boss-share start below the pointer doesn't count as a rollback
6. real resets: a boss-share start runs and its log line carries boss_share; an ordinary one doesn't
CPU only; writes no run file (the episode log goes to a temp file). Ends with ALL CHECKS PASS or
the failures.
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
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    C.MG_BALANCE_TIME = False                           # the episode draws; check_balance_time.py does the time
    import mg_stones
    C.MG_RING_SHARE = 0.0                               # the ring share has its own check (check_ring_share.py)
    cfg = train_main.RUNS[run]
    check(C.MG_BOSS_SHARE == cfg.get('boss_share', 0) > 0,
          f"MG_BOSS_SHARE {C.MG_BOSS_SHARE} (RUNS {cfg.get('boss_share')})")
    check(C.MG_ACT_WEIGHTS == cfg.get('act_weights', {}), f'MG_ACT_WEIGHTS {C.MG_ACT_WEIGHTS}')

    env = C.SonicCoordinator()
    env.demo_info()
    cur = json.load(open(os.path.join(HOME, f'main_{run}_curriculum.json')))
    ptrs = {k: v['ptr'] for k, v in cur.items()}
    env.set_demo_pointers(ptrs)
    segs = {s['key']: s for s in env._segs}

    print('\n1. his fights (start points) and the pointers')
    for s in env._segs:
        ts = [c['t'] for c in s['ckpts']]
        fs = []
        for a, b in s['route']['fights']:
            ks = [k for k, t in enumerate(ts) if a <= t < b]
            fs.append(f'{ks[0]}-{ks[-1]}' if ks else f'(t {a}-{b}: none)')
        print(f"  {s['name']}: pointer {ptrs.get(s['key'])} of {len(ts) - 1}; fights at {', '.join(fs) or 'none'}")
    ranges = env.boss_ranges()
    allowed = {}
    for s, lo, hi in ranges:
        ts, p, last = [c['t'] for c in s['ckpts']], min(ptrs[s['key']], len(s['ckpts']) - 1), len(s['ckpts']) - 1
        a, b = next((a, b) for a, b in s['route']['fights'] if a <= ts[p] < b)
        allowed[s['key']] = (lo, hi)
        check(ts[lo] >= a and (lo == 0 or ts[lo - 1] < a) and lo <= p and hi == min(last, p + C.DEMO_BAND),
              f"{s['name']}: boss share draws start points {lo}..{hi} (fight t {a}-{b}, pointer {p})")
    for s in env._segs:
        if s['key'] not in allowed:
            p = min(ptrs.get(s['key'], 0), len(s['ckpts']) - 1)
            check(not any(a <= s['ckpts'][p]['t'] < b for a, b in s['route']['fights']),
                  f"{s['name']}: pointer {p} outside his fights: not in the boss share")

    print(f'\n2. {N:,} drawn starts')
    random.seed(1)
    draws = [env._pick({})[:3] for _ in range(N)]
    boss = [(s['key'], k) for s, k, b in draws if b]
    check(abs(len(boss) / N - C.MG_BOSS_SHARE) < 0.01 if allowed else not boss,
          f'boss share {len(boss) / N:.1%} of starts (set {C.MG_BOSS_SHARE:.0%})')
    check(all(key in allowed and allowed[key][0] <= k <= allowed[key][1] for key, k in boss),
          'every boss-share start inside its act\'s range')
    per = collections.Counter(key for key, _ in boss)
    for key in allowed:
        got, exp = per[key] / max(1, len(boss)), 1 / len(allowed)
        check(abs(got - exp) < 0.02, f"{segs[key]['name']}: {got:.1%} of the boss share (even split {exp:.1%})")
    los = {key: collections.Counter(k for kk, k in boss if kk == key) for key in allowed}
    for key, (lo, hi) in allowed.items():
        check(min(los[key]) == lo and max(los[key]) == hi, f"{segs[key]['name']}: draws reach both ends {lo} and {hi}")
    other = [(s, k) for s, k, b in draws if not b]
    w = {key: env.act_weight(s) for key, s in segs.items()}
    tot = sum(w.values())
    n_act = collections.Counter(s['key'] for s, _ in other)
    for key, s in segs.items():
        got, exp = n_act[key] / len(other), w[key] / tot
        check(abs(got - exp) < 0.01, f"{s['name']}: {got:.1%} of the other starts (weight {w[key]}: {exp:.1%})")
    rule = all(k == 0 or min(ptrs[s['key']], len(s['ckpts']) - 1) <= k
               <= min(len(s['ckpts']) - 1, ptrs[s['key']] + C.DEMO_BAND) for s, k in other)
    check(rule, 'the other starts: pointer band or the act start, as before')
    print('  all starts by act: ' + ', '.join(
        f"{s['name']} {sum(1 for x, _, _ in draws if x['key'] == key) / N:.1%}" for key, s in segs.items()))

    print('\n3. pointers leaving the fights')
    if allowed:
        key = next(iter(allowed))
        lo = allowed[key][0]
        env.set_demo_pointers({key: max(0, lo - 1)})
        left = {s['key'] for s, _, _ in env.boss_ranges()}
        check(key not in left and left == set(allowed) - {key},
              f"{segs[key]['name']} at {max(0, lo - 1)} (just before its fight): drops out, the others stay")
        env.set_demo_pointers({k: 0 for k in ptrs})
        none = [b for _, _, b, _ in (env._pick({}) for _ in range(5000))]
        check(not env.boss_ranges() and not any(none), 'every pointer at 0: no boss-share starts')
        env.set_demo_pointers(ptrs)

    print('\n4. MG_BOSS_SHARE=0, no weights (the old draws)')
    share, weights = C.MG_BOSS_SHARE, C.MG_ACT_WEIGHTS
    C.MG_BOSS_SHARE, C.MG_ACT_WEIGHTS = 0.0, {}
    old = [env._pick({})[:3] for _ in range(N)]
    check(not any(b for _, _, b in old), 'no boss-share starts')
    n_old = collections.Counter(s['key'] for s, _, _ in old)
    check(all(abs(n / N - 1 / len(segs)) < 0.01 for n in n_old.values()), 'acts uniform')
    C.MG_BOSS_SHARE, C.MG_ACT_WEIGHTS = share, weights

    print('\n5. mg_stones.since_arrival')
    rows = [dict(t=1, start=10), dict(t=2, start=8, boss_share=True), dict(t=3, start=10)]
    check(len(mg_stones.since_arrival(rows, 10)) == 2, 'a boss-share start below the pointer: both kept')
    rows[1].pop('boss_share')
    check(len(mg_stones.since_arrival(rows, 10)) == 1, 'an ordinary start below it (a rollback): only the later one')

    print('\n6. real resets and their log lines')
    log = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    C.LOG_FILE = log
    for want in ([True, False] if allowed else [False]):
        C.MG_BOSS_SHARE = 1.0 if want else 0.0
        env.reset()
        info, done, n = {}, False, 0
        while not done and n < 150:
            _, _, term, trunc, info = env.step(random.randrange(len(C.ACTION_BUTTONS)))
            done, n = term or trunc, n + 1
        if not done:
            env._result('stuck', info)                  # log the line as an episode end would
        line = json.loads(open(log).read().strip().splitlines()[-1])
        k, key = env._k, env._seg['key']
        inside = key in allowed and allowed[key][0] <= k <= allowed[key][1]
        check(line['start'] == k and line['act'] == env._seg['name'] and line.get('boss_share', False) == want
              and (inside or not want),
              f"{'boss-share' if want else 'ordinary'} reset: {env._seg['name']} start {k}, {n} steps, "
              f"log boss_share={line.get('boss_share')}")
    C.MG_BOSS_SHARE = share
    os.unlink(log)

    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
