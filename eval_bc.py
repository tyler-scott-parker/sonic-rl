#!/usr/bin/env python3
"""eval_bc.py — how far does a network get on its own, no training: from chosen start points of the
acts of a run's setup, N tries (argmax and sampled), outcome and how far along his route.

    python3 eval_bc.py MODEL_RUN [--acts-of mg2] [--points AIZ1:0,30,60,72,75 HCZ1:0 ...] [--n 5]
    python3 eval_bc.py bc1                       # every act of mg2 from its start, AIZ1 at the climb too

Tyler 2026-10-12 (imitation first): the first test of the pretrained network is whether it can, with
no reinforcement learning, clear AIZ1 from the rock pile (start point 72), and how far it gets from the
start of each act. Compare MODEL_RUN=mg1 (before) with the pretrained one. CPU, one thread.
"""
import argparse
import os
import sys

HOME = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('--acts-of', default='mg2')
    ap.add_argument('--points', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--air-bar', choices=['0', '1'], help="override the setup's air bar (MG_AIR_BAR)")
    ap.add_argument('--compass', choices=['0', '1'], help="override the setup's ring compass (MG_RING_COMPASS)")
    a = ap.parse_args()
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(a.acts_of)
    if a.air_bar is not None:
        os.environ['MG_AIR_BAR'] = a.air_bar
    if a.compass is not None:
        os.environ['MG_RING_COMPASS'] = a.compass
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate')
    import torch
    torch.set_num_threads(1)
    import coordinator as C
    from stable_baselines3 import PPO
    model = PPO.load(train_main.paths(a.model)['zip'] + '.zip', device='cpu')
    env = C.SonicCoordinator(demo_prob=0.0)
    segs = {s['name']: s for s in (env.demo_info(), env._segs)[1]}
    points = a.points or ['AIZ1:0,60,72,75,76'] + [f'{n}:0' for n in segs if n != 'AIZ1']
    print(f'{a.model} ({model.num_timesteps:,} steps of its own) on {a.acts_of}\'s acts, air bar '
          f'{C.MG_AIR_BAR}, compass {C.MG_RING_COMPASS}, {a.n} tries each:',
          flush=True)
    for spec in points:
        name, ks = spec.split(':')
        seg = segs[name]
        for k in map(int, ks.split(',')):
            row = []
            for det in (True, False):
                res = []
                for _ in range(a.n if not det else 1):
                    obs, _ = env.reset(options={'key': seg['key'], 'ckpt': k})
                    done = False
                    while not done:
                        act, _ = model.predict(obs, deterministic=det)
                        obs, _, term, trunc, info = env.step(int(act))
                        done = term or trunc
                    r = info['mg_result']
                    res.append(f"{r['outcome']}{':' + r['cause'] if r.get('cause') == 'drown' else ''} {env._route.best / seg['T']:.0%}"
                               + (f" in {r['frames']} (his {r['target']})" if r['outcome'] == 'clear' else ''))
                row.append(('argmax: ' if det else 'sampled: ') + ', '.join(res))
            print(f"  {name} start point {k} ({seg['ckpts'][k]['t'] / seg['T']:.0%}): " + ' | '.join(row),
                  flush=True)


if __name__ == '__main__':
    main()
