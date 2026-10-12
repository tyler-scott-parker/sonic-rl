#!/usr/bin/env python3
"""check_ss_detour.py — does MG_SS_DETOUR=refund make a won special stage's trip free, and change
nothing else?

    python3 check_ss_detour.py [RUN]      # default mg1; its latest save finds the tries

Samples tries from the act's start with the run's save until one goes into a giant ring and wins the
emerald (MG_SS=pilot), and keeps one that doesn't (the control). Replays both, exactly, under
MG_SS_DETOUR=charge and =refund (each in its own process: the env reads its settings at import).
Requires: identical outcome and frames in both modes (the game itself is untouched); the control's
return identical; the ring try's return higher under refund by exactly the frames forgiven
(trip_there + trip_back in its stage record, each <= DETOUR_CAP), and both legs > 0.
CPU only. Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import subprocess
import sys

HOME = os.path.dirname(os.path.abspath(__file__))


def replay(run, mode, path):
    """Child: replay the tries in `path` under MG_SS_DETOUR=mode; print one JSON line."""
    import train_main
    train_main.configure_env(run)
    os.environ.update(MG_LOG=os.devnull, MG_SS='pilot', MG_SS_DETOUR=mode, MG_EMERALD_BONUS='0',
                      CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    tries = json.load(open(path))
    env = C.SonicCoordinator(acts=train_main.RUNS[run]['acts'][:1], demo_prob=0.0)
    key = env.demo_info()[0]['key']
    out = {}
    for name, t in tries.items():
        env.reset(options={'key': key, 'ckpt': t['kk']})
        ret = 0.0
        for a in t['actions']:
            _, r, term, trunc, info = env.step(a)
            ret += r / C.REWARD_SCALE
            if term or trunc:
                break
        res = info['mg_result']
        out[name] = dict(outcome=res['outcome'], frames=res['frames'], ret=round(ret, 3),
                         stages=[{k: v for k, v in s.items() if k != 'info'} for s in env._stages])
    print('RESULT ' + json.dumps(out), flush=True)


def find(run, path):
    """Parent: a won-ring try and a no-ring try from the act's start."""
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
    found = {}
    for i in range(400):
        obs, _ = env.reset(options={'key': key, 'ckpt': 0})
        acts, done = [], False
        while not done:
            a, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, _ = env.step(int(a))
            acts.append(int(a))
            done = term or trunc
        won = any(s['emeralds'] > 0 for s in env._stages)
        name = 'ring' if won else 'control'
        if name not in found:
            found[name] = dict(kk=0, actions=acts)
            print(f'  try {i + 1}: {name} ({len(acts)} steps)', flush=True)
        if len(found) == 2:
            break
    json.dump(found, open(path, 'w'))
    return found


def main():
    if sys.argv[1:2] == ['--replay']:
        return replay(sys.argv[2], sys.argv[3], sys.argv[4])
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg1'
    path = os.path.join(HOME, 'logs', f'check_ss_detour_{run}.json')
    print(f'{run}: looking for a won giant-ring try and a control from the act start ...', flush=True)
    found = find(run, path)
    if len(found) < 2:
        sys.exit(f'FAILED: found only {list(found)} in 400 tries')
    res = {}
    for mode in ('charge', 'refund'):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--replay', run, mode, path],
                           capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
        if not line:
            sys.exit(f'FAILED: the {mode} replay crashed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
        res[mode] = json.loads(line[0][7:])
    bad = []
    for name in ('control', 'ring'):
        c, f = res['charge'][name], res['refund'][name]
        print(f"  {name}: charge -> {c['outcome']} at {c['frames']}, return {c['ret']:+.1f};  "
              f"refund -> {f['outcome']} at {f['frames']}, return {f['ret']:+.1f}")
        if (c['outcome'], c['frames']) != (f['outcome'], f['frames']):
            bad.append(f'{name}: the game played differently')
    if abs(res['charge']['control']['ret'] - res['refund']['control']['ret']) > 1e-6:
        bad.append('control: the return changed without a stage')
    st = res['refund']['ring']['stages'][0]
    there, back = st.get('trip_there', 0), st.get('trip_back', 0)
    print(f"  ring: stage {st['board']} {st['outcome']}, emeralds +{st['emeralds']}; "
          f"trip there {there} frames, back {back} frames forgiven")
    gain = res['refund']['ring']['ret'] - res['charge']['ring']['ret']
    if abs(gain - (there + back)) > 1e-6:
        bad.append(f'ring: refund gained {gain:+.1f}, forgiven {there + back}')
    if not (0 < there <= 1200 and 0 < back <= 1200):
        bad.append(f'ring: legs {there}, {back} not both in (0, 1200]')
    if bad:
        print('\nFAILED:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print('\nALL CHECKS PASS')


if __name__ == '__main__':
    main()
