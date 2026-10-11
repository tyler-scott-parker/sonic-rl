#!/usr/bin/env python3
"""check_play_acts.py — can the main-game env train on every act of a recording?

    python3 check_play_acts.py demos/play_<stamp>.npz [points per act, default 6]
    MG_ENV_VERSION=1 python3 check_play_acts.py ...     # (default here: 2)

The env's promise (coordinator.py): from any start point, Tyler's own play
clears on his exact frame and returns exactly CLEAR_BONUS (time saved 0, the
route reward paying back every frame). This drives the real env from evenly
spaced start points of every cleared act with HIS per-frame inputs (the
emulator call is swapped for his recorded input at that frame, so the
4-frame action repeat doesn't get in the way) and requires, per episode:
outcome 'clear', frames == target, return == CLEAR_BONUS. A 'stuck' or
'timeout' here means the env would cut his own route — e.g. a boss fight the
game never raises Boss_flag for (MGZ1, SOZ1, SOZ2, SSZ1, DEZ3), where only
route progress keeps `stuck` away.
Ends with ALL ACTS PASS or the failures. CPU only; ~1M frames per full game.
"""
import os
import sys
import time

os.environ.setdefault('MG_ENV_VERSION', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MG_LOG', os.devnull)
import numpy as np

import coordinator as C
import demos


def main():
    path = sys.argv[1]
    per_act = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    actions, meta = demos.load_play(path)
    table = meta['action_table']
    t0 = time.time()
    env = C.SonicCoordinator(demo_files=[path], acts=[])
    env.demo_info()
    print(f'{os.path.basename(path)}: {len(env._segs)} acts, env v{C.MG_ENV_VERSION}, '
          f'segments built in {time.time() - t0:.0f} s', flush=True)
    orig = C.emu_frame
    at = {'f': 0}

    def his_input(e, _mask):
        at['f'] += 1
        return orig(e, table[actions[at['f']]])

    C.emu_frame = his_input
    bad, n_eps, n_frames = [], 0, 0
    try:
        for seg in env._segs:
            K = len(seg['ckpts'])
            ks = sorted(set(np.linspace(0, K - 1, per_act).round().astype(int).tolist()))
            rows = []
            for k in ks:
                env.reset(options={'key': seg['key'], 'ckpt': k})
                at['f'] = seg['ckpts'][k]['frame']
                ret, done = 0.0, False
                while not done:
                    _, r, term, trunc, info = env.step(0)
                    ret += r / C.REWARD_SCALE
                    done = term or trunc
                res = info['mg_result']
                ok = (res['outcome'] == 'clear' and res['frames'] == res['target']
                      and abs(ret - C.CLEAR_BONUS) < 1e-6)
                n_eps += 1
                n_frames += res['frames']
                rows.append(f"{k}{'' if ok else '!'}")
                if not ok:
                    bad.append((seg['name'], k, res['outcome'], res['frames'], res['target'], round(ret, 3)))
            print(f"  {seg['name']:6s} T={seg['T']:6d}  start points {' '.join(rows)}", flush=True)
    finally:
        C.emu_frame = orig
    print(f'{n_eps} episodes, {n_frames:,} frames, {time.time() - t0:.0f} s')
    if bad:
        print(f'\n{len(bad)} FAILED (act, start point, outcome, frames, target, return):')
        for b in bad:
            print('  ', b)
        sys.exit(1)
    print('\nALL ACTS PASS')


if __name__ == '__main__':
    main()
