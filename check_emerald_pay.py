#!/usr/bin/env python3
"""check_emerald_pay.py — does MG_EMERALD_PAY=return move the emerald bonus from the stage to the
first route progress after it (or a clear), drop it when Sonic never gets back, and change nothing
else?

    python3 check_emerald_pay.py [RUN]      # default mg2; CPU only, a few minutes

Tries, AIZ1 from its start, replayed exactly in RUN's env: check_ss_detour.py's three
(logs/check_ss_detour_mg1.json: control, ring as played, ring_back = taken over after the stage by
jump+right, right x 3, which gets back to his route), ring_stuck (taken over after the stage by no
input: stuck at the ring), and a clear without a ring sampled with RUN's save (clear_owed: the same
clear with an emerald's bonus owed before its last step). Each replayed under MG_EMERALD_PAY=stage
and =return with the same MG_EMERALD_BONUS, each mode in its own process (the env reads its
settings at import). Requires:
  - the same outcome and frames under both (the game itself is untouched);
  - no stage (control, clear): identical rewards, step by step;
  - ring_back: identical return; rewards differ on exactly two steps, the stage's (-bonus) and the
    payment's (+bonus), which comes after the stage, on a step where his route progress rose;
  - ring_stuck: never gets back: return lower by exactly the bonus, nothing paid;
  - ring (as played): equal if it got back (bonus_paid_at), lower by the bonus if not;
  - clear_owed: the clear pays what is owed (return + bonus against the same clear).
Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import subprocess
import sys

HOME = os.path.dirname(os.path.abspath(__file__))
BONUS = 1200.0
TRIES = os.path.join(HOME, 'logs', 'check_ss_detour_mg1.json')


def replay(run, mode, path):
    """Child: replay the tries in `path` under MG_EMERALD_PAY=mode; print one JSON line."""
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    os.environ.update(MG_LOG=os.devnull, MG_SS='pilot', MG_EMERALD_PAY=mode, MG_EMERALD_BONUS=str(BONUS),
                      CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    tries = json.load(open(path))
    env = C.SonicCoordinator(acts=train_main.RUNS[run]['acts'][:1], demo_prob=0.0)
    key = env.demo_info()[0]['key']
    out = {}
    for name, t in tries.items():
        if name == 'clear_owed' and mode != 'return':
            continue
        env.reset(options={'key': key, 'ckpt': t['kk']})
        rews, best, stage_at, i, j, done = [], [], None, 0, 0, False
        while not done:
            if t.get('then') and env._stages:           # taken over after the stage
                a = t['then'][j % len(t['then'])]
                j += 1
            elif i < len(t['actions']):
                a = t['actions'][i]
                i += 1
            else:
                a = 0
            if name == 'clear_owed' and i == len(t['actions']):
                env._em_owed = BONUS                   # as if an emerald were won and not yet paid
            n0 = len(env._stages)
            _, r, term, trunc, info = env.step(a)
            if len(env._stages) > n0 and stage_at is None:
                stage_at = len(rews)
            rews.append(r / C.REWARD_SCALE)
            best.append(float(env._route.best))
            done = term or trunc
        res = info['mg_result']
        out[name] = dict(outcome=res['outcome'], frames=res['frames'], ret=sum(rews), rews=rews, best=best,
                         stage_at=stage_at, stages=[{k: v for k, v in s.items() if k != 'info'}
                                                    for s in env._stages])
    print('RESULT ' + json.dumps(out, default=float), flush=True)


def find_clear(run):
    """A clear from the act's start without a ring, sampled with RUN's save."""
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    os.environ.update(MG_LOG=os.devnull, MG_SS='pilot', CUDA_VISIBLE_DEVICES='')
    import torch
    torch.set_num_threads(1)
    import coordinator as C
    from stable_baselines3 import PPO
    env = C.SonicCoordinator(acts=train_main.RUNS[run]['acts'][:1], demo_prob=0.0)
    key = env.demo_info()[0]['key']
    model = PPO.load(train_main.paths(run)['zip'] + '.zip', device='cpu')
    for i in range(20):
        obs, _ = env.reset(options={'key': key, 'ckpt': 0})
        acts, done = [], False
        while not done:
            a, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(int(a))
            acts.append(int(a))
            done = term or trunc
        print(f"  try {i + 1}: {info['mg_result']['outcome']}, {len(env._stages)} stages", flush=True)
        if info['mg_result']['outcome'] == 'clear' and not env._stages:
            return dict(kk=0, actions=acts)
    return None


def main():
    if sys.argv[1:2] == ['--replay']:
        return replay(sys.argv[2], sys.argv[3], sys.argv[4])
    if sys.argv[1:2] == ['--find']:
        print('FOUND ' + json.dumps(find_clear(sys.argv[2])), flush=True)
        return
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    path = os.path.join(HOME, 'logs', f'check_emerald_pay_{run}.json')
    tries = {k: v for k, v in json.load(open(TRIES)).items()}
    tries['ring_stuck'] = dict(tries['ring'], then=[0])
    if '--reuse' in sys.argv and os.path.exists(path):
        tries['clear'] = json.load(open(path))['clear']
    else:
        print(f'{run}: sampling a clear from the act start ...', flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--find', run],
                           capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('FOUND ')]
        print('\n'.join(ln for ln in p.stdout.splitlines() if ln.startswith('  try')))
        if not line or line[0] == 'FOUND null':
            sys.exit(f'FAILED: no clear found:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
        tries['clear'] = json.loads(line[0][6:])
    tries['clear_owed'] = tries['clear']
    json.dump(tries, open(path, 'w'))
    res = {}
    for mode in ('stage', 'return'):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--replay', run, mode, path],
                           capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
        if not line:
            sys.exit(f'FAILED: the {mode} replay crashed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
        res[mode] = json.loads(line[0][7:])
    S, R = res['stage'], res['return']
    bad = []
    for name in ('control', 'clear', 'ring', 'ring_back', 'ring_stuck'):
        s, r = S[name], R[name]
        diff = [k for k, (x, y) in enumerate(zip(s['rews'], r['rews'])) if abs(x - y) > 1e-6]
        paid = [st.get('bonus_paid_at') for st in r['stages']]
        print(f"  {name}: {s['outcome']} at {s['frames']} frames, {len(s['stages'])} stages "
              f"({sum(st['emeralds'] for st in s['stages'])} emeralds); return stage {s['ret']:+.1f}, "
              f"return {r['ret']:+.1f}; steps differing {diff[:4]}; bonus paid at {paid}")
        if (s['outcome'], s['frames'], len(s['rews'])) != (r['outcome'], r['frames'], len(r['rews'])):
            bad.append(f'{name}: the game played differently')
            continue
        if not s['stages']:
            if diff:
                bad.append(f'{name}: rewards changed without a stage')
            continue
        if s['stages'][0]['emeralds'] < 1:
            bad.append(f'{name}: the stage won no emerald (nothing tested)')
            continue
        back = paid[0] is not None
        want = {'ring_back': True, 'ring_stuck': False}.get(name, back)
        if back != want:
            bad.append(f"{name}: bonus {'paid' if back else 'not paid'}, expected the opposite")
        dret = r['ret'] - s['ret']
        if abs(dret - (0 if back else -BONUS)) > 1e-6:
            bad.append(f'{name}: return changed by {dret:+.1f}')
        st = s['stage_at']
        if st is None or st not in diff or abs(r['rews'][st] - s['rews'][st] + BONUS) > 1e-6:
            bad.append(f'{name}: the stage step did not drop the bonus')
        if back:
            pay = [k for k in diff if k != st]
            if (len(pay) != 1 or pay[0] < st or abs(r['rews'][pay[0]] - s['rews'][pay[0]] - BONUS) > 1e-6
                    or not r['best'][pay[0]] > r['best'][pay[0] - 1]):
                bad.append(f'{name}: payment steps {pay} (stage at {st}): not one step of new route progress')
            else:
                print(f'    paid {pay[0] - st} steps after the stage, as route progress rose '
                      f"{r['best'][pay[0] - 1]} -> {r['best'][pay[0]]}")
        elif len(diff) != 1:
            bad.append(f'{name}: {len(diff)} steps differ, expected only the stage step')
    c, co = R['clear'], R['clear_owed']
    d = co['ret'] - c['ret']
    print(f"  clear_owed: {co['outcome']} at {co['frames']}, return {co['ret']:+.1f} vs {c['ret']:+.1f} "
          f"({d:+.1f})")
    if co['outcome'] != 'clear' or abs(d - BONUS) > 1e-6:
        bad.append(f'clear_owed: the clear paid {d:+.1f}, expected {BONUS:+.1f}')
    if bad:
        print('\nFAILED:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print('\nALL CHECKS PASS')


if __name__ == '__main__':
    main()
