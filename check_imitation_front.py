#!/usr/bin/env python3
"""check_imitation_front.py — with RUNS imitation 'front', does the imitation pull's focused half also cover
his frames at each act's fresh-start frontier (the front share's window)?

    python3 check_imitation_front.py [RUN]          # default mg2

1. without 'front': the focused frames are the pointer windows only (his t within [-behind, +ahead] of
   each act's start point)
2. with it: also, for each act whose fresh-start frontier (coordinator.fresh_frontier on the run's log)
   is short of its pointer, his frames in [frontier - FRONT_BACK, frontier], and nothing else new
3. what those frames teach: his actions in each act's frontier window (HCZ2: the corridor, his spin
   dash, the roll up the curve), so a reader can see the pull points at the right thing
CPU only, no model. Ends with ALL CHECKS PASS or the failures.
"""
import collections
import json
import os
import sys

HOME = os.path.dirname(os.path.abspath(__file__))
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    sys.path.insert(0, HOME)
    import numpy as np
    import train_main as T
    T.configure_env(run)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import coordinator as C
    cfg = {**T.IMITATION_DEFAULTS, **T.RUNS[run]['imitation']}
    check(bool(cfg.get('front')), f"RUNS {run} imitation front={cfg.get('front')}")
    cur_path = T.paths(run)['curriculum']
    cur = json.load(open(cur_path))
    cb = T.ImitationCallback(dict(cfg, front=False), run, cur_path)
    cb.load_data()
    base = set(cb.frontier().tolist())
    S = cb.S

    print('1. without front: pointer windows only')
    want = set()
    for key, idx in cb.keys.items():
        t0 = cur[key]['frac'] * int(S['T'][idx[0]])
        t = S['t'][idx]
        want |= set(idx[(t >= t0 - cfg['behind']) & (t <= t0 + cfg['ahead'])].tolist())
    check(base == want, f'{len(base):,} frames, the pointer windows recomputed here')

    print('\n2. with front: plus each act\'s frontier window')
    cb.cfg['front'] = True
    with_front = set(cb.frontier().tolist())
    by = collections.defaultdict(list)
    for r in C.log_tail(C.FRONT_TAIL_BYTES):
        by[f"{r.get('demo')}:{r.get('act')}"].append(r)
    extra, acts = set(), []
    for key, idx in cb.keys.items():
        f = C.fresh_frontier(by[key])
        T_ = int(S['T'][idx[0]])
        if f is None or f >= cur[key]['frac'] * T_:
            continue
        t = S['t'][idx]
        w = idx[(t >= f - C.FRONT_BACK) & (t <= f)]
        extra |= set(w.tolist())
        acts.append((key.split(':')[-1], f, w))
    check(with_front == want | extra and extra - want,
          f"{len(with_front):,} frames: + {len(extra - want):,} at the frontiers of {', '.join(a for a, _, _ in acts)}")

    print('\n3. what the frontier frames teach (his actions there)')
    names = C.ACTION_NAMES if hasattr(C, 'ACTION_NAMES') else [str(i) for i in range(len(C.ACTION_BUTTONS))]
    for act, f, w in acts:
        n = collections.Counter(int(a) for a in S['action'][w])
        tot = sum(n.values()) or 1
        print(f"    {act}: frontier {f:.0f} of {int(S['T'][w[0]])} frames, his t {int(S['t'][w].min())}..{int(S['t'][w].max())}: "
              + ', '.join(f"{names[a]} {c / tot:.0%}" for a, c in n.most_common(5)))
    h = next((w for a, _, w in acts if a == 'HCZ2'), None)
    if h is not None:
        roll = h[(S['t'][h] >= 395) & (S['t'][h] <= 540)]            # after his spin dash, along the corridor
        jumps = sum(int(a) in (5, 6, 7, 8) for a in S['action'][roll])
        check(len(roll) > 0 and jumps == 0, f'HCZ2: his {len(roll)} corridor frames after the spin dash (t 395-540) '
                                            f'have {jumps} jump presses: the pull says keep rolling')
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
