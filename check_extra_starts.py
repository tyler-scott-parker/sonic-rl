#!/usr/bin/env python3
"""check_extra_starts.py — are a run's extra start points (MG_EXTRA_STARTS, RUNS 'extra_starts')
where they should be, without moving the regular ones, and does Tyler's own play clear from them?

    python3 check_extra_starts.py [RUN]          # default mg1

1. builds the run's acts without and with the extras (each in its own process: the env reads its
   settings at import) and requires the regular start points unchanged and each extra inserted at
   its demo frame (or the first frame the act clock runs after it)
2. from every extra start point and the regular ones on either side: HIS per-frame inputs through
   the env must clear on his exact frame with return CLEAR_BONUS (the env's promise; as
   check_play_acts.py)
CPU only. Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import subprocess
import sys

HOME = os.path.dirname(os.path.abspath(__file__))


def build(run, extras):
    """Child: print the start points' demo times per act as JSON."""
    import train_main
    train_main.configure_env(run)
    if not extras:
        os.environ['MG_EXTRA_STARTS'] = ''
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    env = C.SonicCoordinator(demo_prob=0.0)
    env.demo_info()
    out = {s['name']: [c['t'] for c in s['ckpts']] for s in env._segs}
    if extras:                                  # 2. his inputs from the extras and their neighbours
        import demos
        bad, rows = [], []
        for s in env._segs:
            ts = [c['t'] for c in s['ckpts']]
            want = C.MG_EXTRA_STARTS.get(s['name'], [])
            ks = set()
            for w in want:
                k0 = next(k for k in range(len(ts)) if ts[k] >= w)      # the extra itself
                ks |= {k0 - 1, k0, k0 + 1}
            ks = sorted(ks)
            actions, meta = demos.load_play(next(p for p in env.demo_files
                                                 if demos.demo_id(p) == s['demo']))
            table = meta['action_table']
            orig, at = C.emu_frame, {'f': 0}

            def his_input(e, _mask):
                at['f'] += 1
                return orig(e, table[actions[at['f']]])
            C.emu_frame = his_input
            for k in ks:
                if not 0 <= k < len(ts):
                    continue
                env.reset(options={'key': s['key'], 'ckpt': k})
                at['f'] = s['ckpts'][k]['frame']
                ret, done = 0.0, False
                while not done:
                    _, r, term, trunc, info = env.step(0)
                    ret += r / C.REWARD_SCALE
                    done = term or trunc
                res = info['mg_result']
                ok = (res['outcome'] == 'clear' and res['frames'] == res['target']
                      and abs(ret - C.CLEAR_BONUS) < 1e-6)
                rows.append(f"{s['name']} start point {k} (t {ts[k]}): {res['outcome']} in {res['frames']} "
                            f"(his {res['target']}), return {ret:.1f}{'' if ok else '  <-- FAIL'}")
                if not ok:
                    bad.append(rows[-1])
            C.emu_frame = orig
        out['_his'] = dict(rows=rows, bad=bad)
    print('RESULT ' + json.dumps(out), flush=True)


def main():
    if sys.argv[1:2] == ['--build']:
        return build(sys.argv[2], sys.argv[3] == '1')
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg1'
    res = {}
    for ex in ('0', '1'):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--build', run, ex],
                           capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
        if not line:
            sys.exit(f'FAILED: the build crashed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
        res[ex] = json.loads(line[0][7:])
    import train_main
    want = train_main.RUNS[run].get('extra_starts', {})
    bad = []
    for name, base in res['0'].items():
        new = res['1'][name]
        extra = [t for t in new if t not in base]
        regular = [t for t in new if t in base]
        print(f'1. {name}: {len(base)} start points; with extras {len(new)}; added at demo t {extra}')
        if regular != base:
            bad.append(f'{name}: the regular start points moved')
        for w in want.get(name, []):
            if not any(w <= t <= w + 2 for t in extra):
                bad.append(f'{name}: no extra start point at t {w}')
        for t in extra:
            print(f'   new start point {new.index(t)} at t {t}')
    print('2. his inputs from the extras and their neighbours:')
    for r in res['1']['_his']['rows']:
        print('   ' + r)
    bad += res['1']['_his']['bad']
    if bad:
        print('\nFAILED:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print('\nALL CHECKS PASS')


if __name__ == '__main__':
    main()
