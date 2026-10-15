#!/usr/bin/env python3
"""check_fight_bar.py — does a run's fight bar (MG_FIGHT_SLACK, RUNS 'fight_slack') loosen the match bar for
starts inside his boss fights, and only there?

    python3 check_fight_bar.py [RUN]          # default mg2

1. each act whose pointer sits inside his fight (Boss_flag at the start point): a clear just under
   (1 + MG_FIGHT_SLACK) x his frames + MATCH_ADD matches him, one just over doesn't
2. a start outside any fight: the usual bar, (1 + MATCH_SLACK) x + MATCH_ADD, both sides of it
3. MG_FIGHT_SLACK=0: starts inside fights get the usual bar again
CPU only; the episode log goes to a temp file. Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
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
    log = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    os.environ.update(MG_LOG=log, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    cfg = train_main.RUNS[run]
    check(C.MG_FIGHT_SLACK == cfg.get('fight_slack', 0) > 0, f'MG_FIGHT_SLACK {C.MG_FIGHT_SLACK}')
    env = C.SonicCoordinator()
    env.demo_info()
    ptrs = {k: v['ptr'] for k, v in json.load(open(os.path.join(HOME, f'main_{run}_curriculum.json'))).items()}

    def matched(key, k, mult, add):
        env.reset(options={'key': key, 'ckpt': k})
        _, _, _, _, info = env.step(0)
        env.frames = int(env._target * mult + add)
        env._forgiven = 0
        return env._result('clear', info)['success']

    def edges(key, k, slack, name):
        lo = matched(key, k, 1 + slack, C.MATCH_ADD - 2)
        hi = matched(key, k, 1 + slack, C.MATCH_ADD + 2)
        check(lo and not hi, f'{name}: matched just under {1 + slack:.2f}x + {C.MATCH_ADD}, not just over')

    print('1. pointers inside his fights')
    fights = 0
    for s in env._segs:
        p = min(ptrs.get(s['key'], 0), len(s['ckpts']) - 1)
        if s['ckpts'][p]['boss']:
            fights += 1
            edges(s['key'], p, C.MG_FIGHT_SLACK, f"{s['name']} start point {p} (in his fight)")
    check(fights >= 1, f'{fights} acts with the pointer in a fight')

    print('\n2. a start outside any fight')
    s = next(x for x in env._segs if x['name'] == 'AIZ2')
    k = next(i for i, c in enumerate(s['ckpts']) if not c['boss'] and i > 10)
    edges(s['key'], k, C.MATCH_SLACK, f'AIZ2 start point {k} (no fight)')

    print('\n3. MG_FIGHT_SLACK=0')
    C.MG_FIGHT_SLACK = 0.0
    for s in env._segs:
        p = min(ptrs.get(s['key'], 0), len(s['ckpts']) - 1)
        if s['ckpts'][p]['boss']:
            edges(s['key'], p, C.MATCH_SLACK, f"{s['name']} start point {p} back on the usual bar")
            break
    os.unlink(log)
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
