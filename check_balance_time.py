#!/usr/bin/env python3
"""check_balance_time.py — with MG_BALANCE_TIME, are the boss share, the ring share and the act
weights shares of training TIME (env steps) instead of episodes?

    python3 check_balance_time.py [RUN]          # default mg2; reads a copy of mg_log_<run>.jsonl's tail

1. the lengths a worker starts from: each draw's mean steps over its last LEN_N episodes in the log's
   tail, recomputed here independently
2. 600,000 draws against made-up lengths (3-10x apart): each source's share of the steps is its share
   (within 1 point), split evenly in steps between its options (the rest by act weight, within 4%)
3. the same with the lengths from 1: the time each act is expected to get now
4. a real episode moves its draw's length by LEN_ALPHA toward its own steps
5. MG_BALANCE_TIME off: the episode draws, whatever the lengths
CPU only; writes no run file (the log tail is copied, episodes go to a temp file). Ends with ALL
CHECKS PASS or the failures.
"""
import collections
import json
import os
import random
import sys
import tempfile

import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
N = 600_000
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def shares(env, C, draws, lens):
    """Steps per source and per (source, option) for draws [(seg, k, boss, ring)]."""
    src, opt = collections.Counter(), collections.Counter()
    for seg, _, boss, ring in draws:
        key = ('boss', seg['key']) if boss else ('ring',) + tuple(ring) if ring else ('main', seg['key'])
        L = lens(key)
        src[key[0]] += L
        opt[key] += L
    return src, opt


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    real_log = os.environ['MG_LOG']
    tail = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    with open(real_log, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - (16 << 20)))
        data = f.read()
    with open(tail, 'wb') as f:
        f.write(data[data.index(b'\n') + 1:])
    os.environ.update(MG_LOG=tail, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    cfg = train_main.RUNS[run]
    check(C.MG_BALANCE_TIME == bool(cfg.get('balance_time')) is True, f'MG_BALANCE_TIME {C.MG_BALANCE_TIME}')
    env = C.SonicCoordinator()
    env.demo_info()
    cur = json.load(open(os.path.join(HOME, f'main_{run}_curriculum.json')))
    env.set_demo_pointers({k: v['ptr'] for k, v in cur.items()})
    segs = {s['key']: s for s in env._segs}
    ranges, rings = env.boss_ranges(), env.ring_starts()

    print('\n1. the lengths a worker starts from')
    env._load_lens()
    with open(tail, 'rb') as f:                      # what _load_lens reads: the last LEN_TAIL_BYTES
        f.seek(0, 2)
        f.seek(max(0, f.tell() - C.LEN_TAIL_BYTES))
        lines = f.read().decode(errors='replace').splitlines()[1:]
    by = collections.defaultdict(list)
    for line in lines:
        r = json.loads(line)
        key = f"{r['demo']}:{r['act']}"
        if key not in segs:
            continue
        k = (('boss', key) if r.get('boss_share') else ('ring',) + tuple(r['ring_share'])
             if r.get('ring_share') else ('main', key))
        st = sum(x.get('cost', 0) for x in r.get('stages', []))
        by[k].append(max(1, -(-(r['frames'] - st) // C.FRAME_SKIP)))
    want = {k: float(np.mean(v[-C.LEN_N:])) for k, v in by.items()}
    check(set(env._lens) == set(want) and all(abs(env._lens[k] - want[k]) < 1e-6 for k in want),
          f'{len(want)} draws, lengths as recomputed')
    for k in sorted(want, key=str):
        name = segs[k[1]]['name'] if k[0] != 'ring' else f'ring ({k[3]},{k[4]})'
        print(f'    {k[0]:4s} {name}: {want[k]:.0f} steps over {min(len(by[k]), C.LEN_N)} episodes')

    print(f'\n2. {N:,} draws against made-up lengths')
    rnd = random.Random(5)
    fake = {}
    for it in ranges:
        fake[('boss', it[0]['key'])] = rnd.choice([80, 250, 800])
    for it in rings:
        fake[('ring',) + tuple(it[2])] = rnd.choice([100, 300, 1000])
    for s in env._segs:
        fake[('main', s['key'])] = rnd.choice([150, 450, 1500])
    env._lens = dict(fake)
    random.seed(7)
    draws = [env._pick({}) for _ in range(N)]
    src, opt = shares(env, C, draws, lambda k: fake[k])
    tot = sum(src.values())
    exp_src = {'boss': C.MG_BOSS_SHARE if ranges else 0, 'ring': C.MG_RING_SHARE if rings else 0}
    exp_src['main'] = 1 - sum(exp_src.values())
    for kind, e in exp_src.items():
        check(abs(src[kind] / tot - e) < 0.01, f'{kind}: {src[kind] / tot:.1%} of the steps (set {e:.0%})')
    for kind in exp_src:
        keys = [k for k in fake if k[0] == kind]
        if not keys:
            continue
        w = {k: (env.act_weight(segs[k[1]]) if kind == 'main' else 1.0) for k in keys}
        sw = sum(w.values())
        # sampling noise: an option drawn n times is off by ~1/sqrt(n) of its share; allow 4 of those
        z = max(abs(opt[k] / src[kind] - w[k] / sw) / (w[k] / sw) * (opt[k] / fake[k]) ** 0.5 for k in keys)
        worst = max(abs(opt[k] / src[kind] - w[k] / sw) / (w[k] / sw) for k in keys)
        check(z < 4, f'{kind}: steps split as the weights across {len(keys)} options (worst {worst:.1%} off, '
                     f'{z:.1f} sigma; lengths {sorted(set(fake[k] for k in keys))})')

    print('\n3. with the lengths from the log: expected training time by act')
    env._load_lens()
    lens = dict(env._lens)
    random.seed(8)
    draws = [env._pick({}) for _ in range(N)]
    src, opt = shares(env, C, draws, env.ep_len)
    tot = sum(src.values())
    acts = collections.Counter()
    for k, v in opt.items():
        acts[segs[k[1]]['name'] if k[0] != 'ring' else
             next(it[0]['name'] for it in rings if ('ring',) + tuple(it[2]) == k)] += v
    print('    ' + ', '.join(f'{a} {100 * v / tot:.0f}%' for a, v in sorted(acts.items()))
          + f" (boss {100 * src['boss'] / tot:.0f}%, ring {100 * src['ring'] / tot:.0f}%)")
    stuck = [it[0]['name'] for it in ranges]
    bs = {it[0]['name']: opt[('boss', it[0]['key'])] for it in ranges}
    if bs:
        check(max(bs.values()) / min(bs.values()) < 1.08, 'boss share even in steps: '
              + ', '.join(f'{a} {100 * v / tot:.1f}%' for a, v in bs.items()))

    print('\n4. a real episode updates its length')
    C.LOG_FILE = os.devnull
    env._lens = dict(lens)
    obs, _ = env.reset()
    key, before = env._src, env.ep_len(env._src)
    raw = env._lens.get(key)
    done, n = False, 0
    while not done and n < 4000:
        _, _, term, trunc, _ = env.step(random.randrange(len(C.ACTION_BUTTONS)))
        done, n = term or trunc, n + 1
    expect = n if raw is None else raw + C.LEN_ALPHA * (n - raw)
    check(done and abs(env._lens[key] - expect) < 1e-6,
          f'{key[0]} {key[1] if key[0] != "ring" else key[3:]}: {raw} -> {env._lens[key]:.1f} after a {n}-step episode')
    env.reset(options={'key': env._segs[0]['key'], 'ckpt': 0})
    check(env._src is None, 'a forced start (a tool picking its start) updates nothing')

    print('\n5. MG_BALANCE_TIME off')
    C.MG_BALANCE_TIME = False
    env._lens = {k: (5.0 if k[0] == 'main' and segs[k[1]]['name'] == 'MGZ1' else 5000.0) for k in fake}
    random.seed(9)
    draws = [env._pick({}) for _ in range(60_000)]
    main = [s for s, _, b, r in draws if not b and not r]
    w = {k: env.act_weight(s) for k, s in segs.items()}
    worst = max(abs(sum(1 for s in main if s['key'] == k) / len(main) - w[k] / sum(w.values())) for k in segs)
    check(worst < 0.01, f'act shares of episodes as the weights whatever the lengths (worst {worst:.1%} off)')
    os.unlink(tail)
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
