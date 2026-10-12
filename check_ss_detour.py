#!/usr/bin/env python3
"""check_ss_detour.py — does MG_SS_DETOUR=refund make a won special stage's trip free, and change
nothing else?

    python3 check_ss_detour.py [RUN]      # default mg1; its latest save finds the tries

Samples tries from the act's start with the run's save until one goes into a giant ring and wins the
emerald (MG_SS=pilot), and keeps one that doesn't (the control); `--reuse` takes them from the last
run's logs/check_ss_detour_<run>.json. Replays, exactly, under MG_SS_DETOUR=charge and =refund (each
in its own process: the env reads its settings at import): the control; the ring try; and the ring
try taken over after the stage by jump+right, right, right, right (AIZ1's ring sits in a pocket you
jump out of: this one gets back to his route). Requires: identical outcome and frames in both modes
(the game itself is untouched); the control's return identical; each ring try's return higher under
refund by exactly the frames forgiven (trip_there + trip_back in its stage record, each <=
DETOUR_CAP); trip_there > 0; trip_back > 0 only for the try that got back to his route (time spent
not getting back is not refunded).
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
        ret, i, j, done = 0.0, 0, 0, False
        while not done:
            if t.get('then') and env._stages:           # taken over after the stage
                a = t['then'][j % len(t['then'])]
                j += 1
            elif i < len(t['actions']):
                a = t['actions'][i]
                i += 1
            else:
                a = 0
            _, r, term, trunc, info = env.step(a)
            ret += r / C.REWARD_SCALE
            done = term or trunc
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
    if '--reuse' in sys.argv and os.path.exists(path):
        found = json.load(open(path))
        print(f'{run}: the tries in {path}', flush=True)
    else:
        print(f'{run}: looking for a won giant-ring try and a control from the act start ...', flush=True)
        found = find(run, path)
    if len(found) < 2:
        sys.exit(f'FAILED: found only {list(found)} in 400 tries')
    found['ring_back'] = dict(found['ring'], then=[7, 2, 2, 2])     # jump+right, right x 3
    json.dump(found, open(path, 'w'))
    res = {}
    for mode in ('charge', 'refund'):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--replay', run, mode, path],
                           capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
        if not line:
            sys.exit(f'FAILED: the {mode} replay crashed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
        res[mode] = json.loads(line[0][7:])
    bad = []
    for name in ('control', 'ring', 'ring_back'):
        c, f = res['charge'][name], res['refund'][name]
        print(f"  {name}: charge -> {c['outcome']} at {c['frames']}, return {c['ret']:+.1f};  "
              f"refund -> {f['outcome']} at {f['frames']}, return {f['ret']:+.1f}")
        if (c['outcome'], c['frames']) != (f['outcome'], f['frames']):
            bad.append(f'{name}: the game played differently')
    if abs(res['charge']['control']['ret'] - res['refund']['control']['ret']) > 1e-6:
        bad.append('control: the return changed without a stage')
    for name, wants_back in (('ring', False), ('ring_back', True)):
        st = res['refund'][name]['stages'][0]
        there, back = st.get('trip_there', 0), st.get('trip_back', 0)
        gain = res['refund'][name]['ret'] - res['charge'][name]['ret']
        print(f"  {name}: stage {st['board']} {st['outcome']}, emeralds +{st['emeralds']}; "
              f"trip there {there} frames, back {back} frames forgiven; refund gained {gain:+.1f}")
        if abs(gain - (there + back)) > 1e-6:
            bad.append(f'{name}: refund gained {gain:+.1f}, forgiven {there + back}')
        if not 0 < there <= 1200:
            bad.append(f'{name}: trip there {there} not in (0, 1200]')
        if wants_back != (0 < back <= 1200):
            bad.append(f'{name}: trip back {back}, but it {"did" if wants_back else "did not"} get back')
    if bad:
        print('\nFAILED:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print('\nALL CHECKS PASS')


if __name__ == '__main__':
    main()
