#!/usr/bin/env python3
"""eval_finish_line.py — where a special-stage agent stands against Tyler's
finish line (2026-10-11): "fresh start clears on all 14 boards and beating my
times on every board".

    python3 eval_finish_line.py [run] [-n 40] [--model x.zip] [--boards 1,2] [--jobs 2]

A fixed snapshot of the run's save (default d9), fresh starts on each board
(sphere 0, the board's captured state: exactly what the main game hands over,
ss_pilot.py), actions sampled as in training, and the game's rules rather than
training's: no 600-frame stall rule (in the game Sonic is still running; the
pilot plays this way too), SS_MAX_STEPS (18,000 frames) as the only cap.
Frames count from the captured state, as Tyler's recordings do.

Per board: clear rate with a Wilson 95% interval, median clear frames with a
bootstrap 95% interval, his recording's frames, and whether the median beats
him (and whether even the interval's upper end does). CPU only; reads no
training files and writes only to stdout (+ --json).
"""
import argparse
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

HOME = os.path.dirname(os.path.abspath(__file__))


def his_frames(run):
    """{board: frames} of Tyler's recordings the run learned from (its RUNS
    'demos', read with ast: importing the trainer would start a run)."""
    import ast
    import demos
    tree = ast.parse(open(os.path.join(HOME, 'train_ss_multistage.py')).read())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RUNS' for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == run:
                    names = ast.literal_eval([x.value for x in v.keywords if x.arg == 'demos'][0])
    out = {}
    for did in names:
        meta = demos.load_demo(os.path.join(HOME, 'demos', f'{did}.npz'))[1]
        out[int(meta['stage'])] = int(meta['frames'])
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_median(xs, n=2000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return (meds[int(0.025 * n)], meds[int(0.975 * n) - 1])


def play_board(args):
    """One worker: n fresh starts on one board. Returns (board, [(cleared, frames, spheres, total)])."""
    run, model, board, n, seed = args
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, HOME)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import ss_pilot
    sse = ss_pilot.import_env(ss_pilot.run_preset(run))
    from sb3_contrib import RecurrentPPO
    env = sse.SpecialStageEnv()
    env._replaying = True                 # the game's rules: no stall rule, no training log
    pol = RecurrentPPO.load(model, device='cpu')
    np.random.seed(seed + board)
    torch.manual_seed(seed + board)
    out = []
    for _ in range(n):
        obs, _ = env.reset(options={'stage_file': sse.stage_file(board), 'stage': board})
        lstm, start = None, True
        while True:
            a, lstm = pol.predict(obs, state=lstm, episode_start=np.array([start]), deterministic=False)
            start = False
            obs, _, done, trunc, _ = env.step(int(np.asarray(a).reshape(-1)[0]))
            if done or trunc:
                break
        out.append((bool(env._board_cleared), int(env.steps), int(env._spheres_collected), int(env._stage_total)))
    env.close()
    return board, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?', default='d9')
    ap.add_argument('-n', type=int, default=40, help='fresh starts per board (default 40)')
    ap.add_argument('--model', help='a .zip (default: the run\'s latest save, snapshotted first)')
    ap.add_argument('--boards', default=','.join(str(b) for b in range(1, 15)))
    ap.add_argument('--jobs', type=int, default=2, help='worker processes (default 2)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', help='also write the per-board results here')
    a = ap.parse_args()
    boards = [int(b) for b in a.boards.split(',')]
    sys.path.insert(0, HOME)
    his = his_frames(a.run)
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, 'model.zip')          # a running trainer rewrites its save
        subprocess.run(['cp', a.model or os.path.join(HOME, f'sonic_ss_multistage_{a.run}.zip'), model], check=True)
        jobs = [(a.run, model, b, a.n, a.seed) for b in boards]
        res = {}
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for board, out in ex.map(play_board, jobs):
                res[board] = out
                clears = [f for c, f, _, _ in out if c]
                print(f'  board {board:2d} done: {len(clears)}/{len(out)} cleared', flush=True)
    print(f'\n{a.run} ({a.model or "latest save"}), {a.n} fresh starts per board, the game\'s rules (no stall rule)')
    print(f'{"board":>5s} {"clears":>8s} {"rate 95% CI":>14s} {"median":>7s} {"median 95% CI":>14s} {"his":>5s}  beats him?')
    summary = {}
    for b in boards:
        out = res[b]
        clears = sorted(f for c, f, _, _ in out if c)
        k, n = len(clears), len(out)
        lo, hi = wilson(k, n)
        med = st.median(clears) if clears else None
        ci = boot_median(clears)
        h = his.get(b)
        beats = med is not None and h is not None and med < h
        sure = ci is not None and h is not None and ci[1] < h
        print(f'{b:5d} {k:3d}/{n:<4d} {lo:6.0%}-{hi:<6.0%} {med if med is not None else "-":>7} '
              f'{(f"{ci[0]:.0f}-{ci[1]:.0f}" if ci else "-"):>14s} {h or "-":>5}  '
              f'{"YES" if sure else "yes (median)" if beats else "no" if clears else "-"}')
        summary[b] = dict(clears=k, n=n, rate_ci=[lo, hi], median=med, median_ci=ci, his=h,
                          beats=beats, beats_ci=sure, spheres=[s for _, _, s, _ in out])
    all_clear = all(summary[b]['clears'] > 0 for b in boards)
    all_beat = all(summary[b]['beats'] for b in boards)
    print(f'\nfresh clear on every board: {all_clear} ({sum(summary[b]["clears"] > 0 for b in boards)}/{len(boards)}); '
          f'median beats his time on every board: {all_beat} ({sum(summary[b]["beats"] for b in boards)}/{len(boards)})')
    if a.json:
        json.dump(summary, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
