#!/usr/bin/env python3
"""watch_tries.py — the runs watch_main.py --best shows, found in a process of
their own so the viewer always has one to play (Tyler 2026-10-12: "Can we
queue up footage so that there is always something playing?").

    python3 watch_tries.py RUN DIR [--act AIZ1] [--n 20] [--keep 2]
    (watch_main.py --best starts it and stops it; it also stops when the viewer is gone)

Builds its own env with every act of the run (stable-retro allows one
emulator per process), reloads the run's save when it changes, plays N
sampled tries from the start point the viewer asks for (DIR/flags.json: act =
which act, else --act or the run's first; fresh = the act's start, else where
training starts now; ring = a try that went into a giant ring first), and
writes the best one to DIR/run_<time>.json: the fastest clear, else the
furthest along his route (then the quickest). A switch flipped mid-batch
drops the batch after the try in progress. It keeps at most KEEP runs
waiting, so what plays is never far behind the latest save. The game is
deterministic, so the viewer replays a try exactly from its start point and
actions.
"""
import argparse
import collections
import json
import os
import time


def best_of(env, model, key, kk, n, ring, stale):
    """N sampled tries from start point kk; the best one as a dict, or None
    if stale() turns true (the viewer flipped a switch) between tries."""
    best, outcomes = None, collections.Counter()
    for _ in range(n):
        if stale():
            return None
        obs, _ = env.reset(options={'key': key, 'ckpt': kk})
        acts, done = [], False
        while not done:
            a, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(int(a))
            acts.append(int(a))
            done = term or trunc
        res = info['mg_result']
        outcomes[res['outcome']] += 1
        prog = env._route.best
        entered = res['outcome'] == 'stage' or bool(getattr(env, '_stages', None))
        if entered:
            outcomes['giant ring'] += 1
        score = (1, -res['frames']) if res['outcome'] == 'clear' else (0, prog, -res['frames'])
        if ring and entered:
            score = (2, -res['frames'])
        if best is None or score > best[0]:
            best = (score, acts, res, prog, entered)
    _, acts, res, prog, entered = best
    T = next(s['T'] for s in env.demo_info() if s['key'] == key)
    tally = ', '.join(f'{c} {o}' for o, c in outcomes.most_common() if o != 'stage')
    if res['outcome'] == 'clear':
        desc = f"best of {n}: cleared in {res['frames']} frames (you {res['target']})"
    elif ring and entered:
        desc = f"a giant-ring try: {prog / T:.0%} of the act, then {res['outcome']}"
    else:
        desc = f"best of {n}: {prog / T:.0%} of the act, then {res['outcome']}"
    return dict(kk=kk, actions=acts, outcome=res['outcome'], frames=res['frames'],
                desc=f'{desc}  ({tally})', steps=int(model.num_timesteps), key=key)


def flags(d):
    """(fresh, ring, act key or None) as the viewer last set them."""
    try:
        with open(os.path.join(d, 'flags.json')) as f:
            fl = json.load(f)
        return bool(fl.get('fresh')), bool(fl.get('ring')), fl.get('act')
    except (OSError, ValueError):
        return True, False, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('dir')
    ap.add_argument('--act')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--keep', type=int, default=2)
    a = ap.parse_args()
    parent = os.getppid()
    import train_main
    train_main.configure_env(a.run)
    os.environ['MG_LOG'] = os.devnull
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    import coordinator as C
    from stable_baselines3 import PPO
    cfg, P = train_main.RUNS[a.run], train_main.paths(a.run)
    env = C.SonicCoordinator(acts=cfg['acts'], demo_prob=0.0)
    keys = [s['key'] for s in env.demo_info()]          # builds every act (mg2: ~75 s)
    first = next((s['key'] for s in env.demo_info() if a.act in (s['key'], s['name'])), keys[0])
    os.makedirs(a.dir, exist_ok=True)
    path, model, mtime = P['zip'] + '.zip', None, None
    print(f'watch_tries: {a.run} ({len(keys)} acts, first {first}), best of {a.n}, '
          f'up to {a.keep} waiting in {a.dir}', flush=True)

    def want():                                         # what the viewer asks for, its act resolved
        fresh, ring, act = flags(a.dir)
        return fresh, ring, act if act in keys else first
    while os.getppid() == parent:                       # the viewer is gone: stop
        if len([f for f in os.listdir(a.dir) if f.startswith('run_')]) >= a.keep:
            time.sleep(0.5)
            continue
        try:
            m = os.path.getmtime(path)
            if m != mtime:
                model, mtime = PPO.load(path, device='cpu'), m
        except Exception:                               # no save yet, or caught mid-write
            if model is None:
                time.sleep(2)
                continue
        fl = want()
        fresh, ring, key = fl
        kk = 0
        if not fresh:
            try:
                with open(P['curriculum']) as f:
                    kk = int(json.load(f)[key]['ptr'])
            except (OSError, ValueError, KeyError):
                pass
        t0 = time.time()
        item = best_of(env, model, key, kk, a.n, ring, lambda: want() != fl)
        if item is None:
            print(f"{time.strftime('%H:%M:%S')} switched mid-batch: dropped", flush=True)
            continue
        item.update(fresh=fresh, ring=ring)
        out = os.path.join(a.dir, f'run_{time.time():.3f}.json')
        with open(out + '.tmp', 'w') as f:
            json.dump(item, f)
        os.replace(out + '.tmp', out)
        print(f"{time.strftime('%H:%M:%S')} {key.split(':')[-1]}: {item['desc']} [{time.time() - t0:.0f} s]",
              flush=True)


if __name__ == '__main__':
    main()
