#!/usr/bin/env python3
"""check_emerald_split.py — does MG_EMERALD_PAY=split pay MG_EMERALD_SPLIT of the emerald bonus on getting back (where
=return paid it all) and the rest only after MG_EMERALD_HOLD of his frames of route progress since the stage (or at a
clear), and change nothing else?

    python3 check_emerald_split.py [RUN]      # default mg2; CPU only, a few minutes

The AIZ1 tries of check_emerald_pay.py (logs/check_emerald_pay_<run>.json: control and a clear without a ring, the
ring as played, ring_back = taken over after the stage by jump+right, right x 3, ring_stuck = no input after it),
replayed exactly under =return and under =split with HOLD small (120: ring_back gets there) and huge (never), each
setting in its own process (the env reads them at import). Requires:
  - the same outcome and frames in every setting (the game itself is untouched);
  - no stage (control, clear): identical rewards, step by step;
  - ring_stuck (never back): identical to =return (nothing paid either way);
  - ring_back: on the payment step =split pays SPLIT x bonus (=return all of it); with HOLD 120 the rest comes on the
    first later step where route progress since the stage reached 120 (return equal to =return), with HOLD huge never
    (return lower by (1 - SPLIT) x bonus);
  - clear_owed: a clear pays both what is owed on getting back and the rest.
Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import subprocess
import sys

HOME = os.path.dirname(os.path.abspath(__file__))
BONUS, SPLIT = 1200.0, 0.5


def replay(run, mode, hold, path):
    """Child: replay the tries under MG_EMERALD_PAY=mode (HOLD hold); print one JSON line."""
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    os.environ.update(MG_LOG=os.devnull, MG_SS='pilot', MG_EMERALD_PAY=mode, MG_EMERALD_BONUS=str(BONUS),
                      MG_EMERALD_SPLIT=str(SPLIT), MG_EMERALD_HOLD=str(hold), CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    tries = json.load(open(path))
    env = C.SonicCoordinator(acts=train_main.RUNS[run]['acts'][:1], demo_prob=0.0)
    key = env.demo_info()[0]['key']
    out = {}
    for name, t in tries.items():
        env.reset(options={'key': key, 'ckpt': t['kk']})
        rews, best, stage_at, i, j, done = [], [], None, 0, 0, False
        while not done:
            if t.get('then') and env._stages:
                a = t['then'][j % len(t['then'])]
                j += 1
            elif i < len(t['actions']):
                a = t['actions'][i]
                i += 1
            else:
                a = 0
            if name == 'clear_owed' and i == len(t['actions']):
                if mode == 'split':
                    env._em_owed, env._em_rest = SPLIT * BONUS, (1 - SPLIT) * BONUS
                else:
                    env._em_owed = BONUS
            n0 = len(env._stages)
            _, r, term, trunc, info = env.step(a)
            if len(env._stages) > n0 and stage_at is None:
                stage_at = len(rews)
            rews.append(r / C.REWARD_SCALE)
            best.append(float(env._route.best))
            done = term or trunc
        res = info['mg_result']
        out[name] = dict(outcome=res['outcome'], frames=res['frames'], ret=sum(rews), rews=rews, best=best,
                         stage_at=stage_at, stages=[{k: v for k, v in s.items() if k != 'info'} for s in env._stages])
    print('RESULT ' + json.dumps(out, default=float), flush=True)


def main():
    if sys.argv[1:2] == ['--replay']:
        return replay(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    path = os.path.join(HOME, 'logs', f'check_emerald_pay_{run}.json')
    if not os.path.exists(path):
        sys.exit(f'FAILED: {path} missing (run check_emerald_pay.py {run} first)')
    res = {}
    for tag, mode, hold in (('return', 'return', 1800), ('split120', 'split', 120), ('splitinf', 'split', 1e12)):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--replay', run, mode, str(hold), path],
                           capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
        if not line:
            sys.exit(f'FAILED: the {tag} replay crashed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
        res[tag] = json.loads(line[0][7:])
    R = res['return']
    bad = []
    for name in ('control', 'clear', 'ring', 'ring_back', 'ring_stuck'):
        r = R[name]
        for tag in ('split120', 'splitinf'):
            s = res[tag][name]
            if (s['outcome'], s['frames'], len(s['rews'])) != (r['outcome'], r['frames'], len(r['rews'])):
                bad.append(f'{name} {tag}: the game played differently')
                continue
            diff = [k for k, (x, y) in enumerate(zip(r['rews'], s['rews'])) if abs(x - y) > 1e-6]
            paid1 = [st.get('bonus_paid_at') for st in s['stages']]
            paid2 = [st.get('bonus2_paid_at') for st in s['stages']]
            print(f"  {name} {tag}: {s['outcome']} at {s['frames']}, {len(s['stages'])} stages; return {r['ret']:+.1f} "
                  f"vs split {s['ret']:+.1f}; steps differing {diff[:4]}; paid on return at {paid1}, rest at {paid2}")
            if not r['stages'] or r['stages'][0].get('emeralds', 0) < 1:
                if diff:
                    bad.append(f'{name} {tag}: rewards changed without a won stage')
                continue
            back = r['stages'][0].get('bonus_paid_at') is not None
            if not back:
                if diff:
                    bad.append(f'{name} {tag}: never back, yet rewards differ')
                continue
            p = next(k for k, x in enumerate(r['rews']) if k > r['stage_at'] and k in diff) if diff else None
            if p is None or abs(s['rews'][p] - r['rews'][p] + (1 - SPLIT) * BONUS) > 1e-6 and not (
                    tag == 'split120' and abs(s['rews'][p] - r['rews'][p]) < 1e-6):
                bad.append(f'{name} {tag}: the payment step did not pay SPLIT x bonus')
            got2 = paid2[0] is not None
            if tag == 'splitinf':
                if got2 or abs(s['ret'] - r['ret'] + (1 - SPLIT) * BONUS) > 1e-6:
                    bad.append(f'{name} {tag}: the rest was paid or the return is off ({s["ret"] - r["ret"]:+.1f})')
            else:
                if name == 'ring_back' and not got2:
                    bad.append(f'{name} {tag}: the rest never came with HOLD 120')
                if got2 and abs(s['ret'] - r['ret']) > 1e-6:
                    bad.append(f'{name} {tag}: rest paid but the return differs ({s["ret"] - r["ret"]:+.1f})')
                if got2:
                    q = [k for k in diff if k != p]
                    gain = s['best'][q[0]] - s['best'][r['stage_at']] if q else None
                    print(f'    rest paid {q} (payment step {p}); route progress since the stage {gain}')
                    if len(q) > 1 or (q and gain < 120):
                        bad.append(f'{name} {tag}: the rest came on {q}, progress {gain}')
    for tag in ('split120', 'splitinf'):
        c, co = res[tag]['clear'], res[tag]['clear_owed']
        d = co['ret'] - c['ret']
        print(f'  clear_owed {tag}: {co["outcome"]}, return {co["ret"]:+.1f} vs {c["ret"]:+.1f} ({d:+.1f})')
        if co['outcome'] != 'clear' or abs(d - BONUS) > 1e-6:
            bad.append(f'clear_owed {tag}: the clear paid {d:+.1f}, expected {BONUS:+.1f}')
    if bad:
        print('\nFAILED:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print('\nALL CHECKS PASS')


if __name__ == '__main__':
    main()
