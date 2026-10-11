#!/usr/bin/env python3
"""route_search.py — look for faster finishes along the route book's routes,
from EVERY start point (CPU, beside training).

    python3 route_search.py [boards] [--run d10] [--sweeps 5] [--model x.zip]
    python3 route_search.py 1,3,9,12,14 --sweeps 10

Training's curriculum only starts episodes at and just after each board's
current start point, and once that has walked back to the stage start
nobody tries the later stretches of the route any more. This plays the run's
agent (sampled, as in training; training's rules incl. the stall rule) from
each start point of each board's book route in turn, `sweeps` times over.
A clear from start point k is the route's first k frames + the agent's: when
that beats the book, special_stage_env saves it to routes/candidates/ exactly
as training does, and `route_book.py harvest` checks and promotes it.

The book is frozen at the start (copied to a temporary directory); restart
the search to pick up a newer book. Writes only candidates and stdout.
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

HOME = os.path.dirname(os.path.abspath(__file__))


def init_of(run):
    """RUNS[run]['init'] from train_ss_multistage.py, read with ast."""
    import ast
    tree = ast.parse(open(os.path.join(HOME, 'train_ss_multistage.py')).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'RUNS':
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == run:
                    return next(ast.literal_eval(x.value) for x in v.keywords if x.arg == 'init')
    raise KeyError(run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('boards', nargs='?', default=','.join(map(str, range(1, 15))))
    ap.add_argument('--run', default='d10', help='whose settings and save (default d10)')
    ap.add_argument('--model', help='a .zip (default: the run\'s latest save, snapshotted)')
    ap.add_argument('--sweeps', type=int, default=5, help='passes over every start point (default 5)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    boards = [int(b) for b in a.boards.split(',')]
    sys.path.insert(0, HOME)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import ss_pilot
    import route_book
    tmp = tempfile.mkdtemp()
    try:
        model = os.path.join(tmp, 'model.zip')
        src = a.model or os.path.join(HOME, f'sonic_ss_multistage_{a.run}.zip')
        if not a.model and not os.path.exists(src):           # a new run before its first save:
            init = init_of(a.run)                               # the save it started from
            src = os.path.join(HOME, f'sonic_ss_multistage_{init}.zip')
            print(f'{a.run} has no save yet: using {init}\'s, which it started from', flush=True)
        subprocess.run(['cp', src, model], check=True)
        demo_files = []
        for b in boards:
            demo_files.append(os.path.join(tmp, f'board{b:02d}.npz'))
            shutil.copy2(route_book.book_path(b), demo_files[-1])
        for k in ss_pilot.SS_KEYS:
            os.environ.pop(k, None)
        os.environ.update(ss_pilot.run_preset(a.run))
        os.environ.update(SS_STAGES=','.join(map(str, boards)), SS_DEMOS=','.join(demo_files),
                          SS_DEMO_PROB='0', SS_RESTART_PROB='0', SS_LOG=os.devnull)
        import numpy as np
        import torch
        torch.set_num_threads(1)
        import special_stage_env as sse
        from sb3_contrib import RecurrentPPO
        env = sse.SpecialStageEnv()
        pol = RecurrentPPO.load(model, device='cpu')
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        env.reset(options={'stage_file': sse.stage_file(boards[0])})     # builds the start points
        demos_all = list(env._demos)
        sse.DEMO_BAND = 0
        book0 = route_book.book_frames()
        best = dict(book0)
        print(f'route search: {a.run} ({a.model or "latest save"}), boards {boards}, '
              f'{a.sweeps} sweeps over every start point', flush=True)
        for sweep in range(a.sweeps):
            for d in demos_all:
                b, t0 = int(d['stage']), time.time()
                n_cl = 0
                found = []
                for i in range(len(d['ckpts'])):
                    env._demos, env._demo_ptr = [d], {d['id']: i}
                    obs, _ = env._reset_from_demo()
                    lstm, first = None, True
                    while True:
                        act, lstm = pol.predict(obs, state=lstm, episode_start=np.array([first]),
                                                deterministic=False)
                        first = False
                        obs, _, done, trunc, _ = env.step(int(np.asarray(act).reshape(-1)[0]))
                        if done or trunc:
                            break
                    if env._board_cleared:
                        n_cl += 1
                        if env.steps < best.get(b, 10 ** 9):
                            best[b] = env.steps
                            found.append((i, env.steps))
                print(f'  sweep {sweep + 1} board {b:2d}: {len(d["ckpts"])} start points, {n_cl} clears, '
                      f'best {best.get(b)} (book {book0.get(b)})'
                      + (f'; faster: {found}' if found else '') + f' [{time.time() - t0:.0f} s]', flush=True)
        env.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
