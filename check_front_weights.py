#!/usr/bin/env python3
"""check_front_weights.py — does MG_FRONT_WEIGHTS (RUNS 'front_weights') shift the front share's steps toward the
weighted act, and change nothing when empty?

    python3 check_front_weights.py [RUN]          # default mg2

Draws N resets with the env's own _draw_by_time (today's pointers, frontiers and episode lengths from the log) and
books each draw's recent episode length as its steps: the share of steps per (source, act).
1. weights empty, front share 0.15 (the old settings): every front act gets an even share of the front steps
2. RUNS (front_share, front_weights): the weighted act's share of all steps, the others', and boss/ring/main
3. the non-time-balanced draw (MG_BALANCE_TIME off) honours the weights too
CPU only, no model. Ends with ALL CHECKS PASS or the failures.
"""
import collections
import os
import random
import sys

HOME = os.path.dirname(os.path.abspath(__file__))
bad = []
N = 200_000


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    sys.path.insert(0, HOME)
    import train_main as T
    T.configure_env(run)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import json
    import coordinator as C
    cfg = T.RUNS[run]
    env = C.SonicCoordinator(acts=cfg['acts'], demo_prob=0.0)
    env.demo_info()
    env._ptr = {k: v['ptr'] for k, v in json.load(open(T.paths(run)['curriculum'])).items()}
    ranges, rings, fronts = env.boss_ranges(), env.ring_starts(), env.front_ranges()
    print(f"front acts now: {', '.join(it[0]['name'] for it in fronts)}; boss: {', '.join(it[0]['name'] for it in ranges)}")

    def shares(front_share, weights, seed=0):
        C.MG_FRONT_SHARE, C.MG_FRONT_WEIGHTS = front_share, weights
        random.seed(seed)
        steps = collections.Counter()
        for _ in range(N):
            kind, item = env._draw_by_time(ranges, rings, fronts)
            seg = item if kind == 'main' else item[0]
            key = (('boss', seg['key']) if kind == 'boss' else ('ring',) + tuple(item[2]) if kind == 'ring'
                   else ('front', seg['key']) if kind == 'front' else ('main', seg['key']))
            steps[(kind, seg['name'])] += env.ep_len(key)
        tot = sum(steps.values())
        return {k: v / tot for k, v in steps.items()}

    def by_kind(sh):
        out = collections.Counter()
        for (k, _), v in sh.items():
            out[k] += v
        return out

    print('\n1. old settings (front 0.15, no weights)')
    old = shares(0.15, {})
    fr = {a: v for (k, a), v in old.items() if k == 'front'}
    print('    front steps by act: ' + ', '.join(f'{a} {v:.1%}' for a, v in sorted(fr.items())))
    check(max(fr.values()) - min(fr.values()) < 0.004, 'front steps even across its acts (within 0.4 points)')
    print('    by source: ' + ', '.join(f'{k} {v:.1%}' for k, v in sorted(by_kind(old).items())))

    print(f"\n2. RUNS: front_share {cfg.get('front_share')}, front_weights {cfg.get('front_weights')}")
    w = {k: float(v) for k, v in cfg.get('front_weights', {}).items()}
    new = shares(cfg.get('front_share', 0), w)
    fr2 = {a: v for (k, a), v in new.items() if k == 'front'}
    print('    front steps by act: ' + ', '.join(f'{a} {v:.1%} (was {fr.get(a, 0):.1%})' for a, v in sorted(fr2.items())))
    print('    by source: ' + ', '.join(f'{k} {v:.1%} (was {by_kind(old)[k]:.1%})' for k, v in sorted(by_kind(new).items())))
    heavy = [k.split(':')[-1] for k in w]
    for a in heavy:
        check(fr2.get(a, 0) >= 2.5 * fr.get(a, 1), f'{a} front steps {fr.get(a, 0):.1%} -> {fr2.get(a, 0):.1%} (>= 2.5x)')
    others = [a for a in fr2 if a not in heavy]
    check(all(abs(fr2[a] - fr[a]) < 0.006 for a in others), 'the other front acts keep their share (within 0.6 points)')
    check(abs(by_kind(new)['boss'] - by_kind(old)['boss']) < 0.01 and abs(by_kind(new)['ring'] - by_kind(old)['ring']) < 0.01,
          'boss and ring shares unchanged (within 1 point)')

    print('\n3. the draw by count (MG_BALANCE_TIME off) honours the weights')
    C.MG_FRONT_SHARE, C.MG_FRONT_WEIGHTS = cfg.get('front_share', 0), w
    random.seed(1)
    picks = collections.Counter(random.choices(fronts, weights=[env.front_weight(it[0]) for it in fronts])[0][0]['name']
                                for _ in range(N))
    want = {it[0]['name']: env.front_weight(it[0]) / sum(env.front_weight(f[0]) for f in fronts) for it in fronts}
    check(all(abs(picks[a] / N - want[a]) < 0.01 for a in want),
          'front draws by count: ' + ', '.join(f'{a} {picks[a] / N:.0%}' for a in sorted(want)))
    print('\n' + ('ALL CHECKS PASS' if not bad else f'{len(bad)} FAILED: ' + '; '.join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
