#!/usr/bin/env python3
"""check_route_book.py — does training save its clears as routes that replay?
(route_book.py, special_stage_env.ROUTE_DIR)

    python3 check_route_book.py [model.zip] [--board 10]   # default: d9's latest save

CPU only, a few minutes. In a training env (the run's settings, its demo for
the board, the stall rule on), into a temporary candidates directory and with
the book's threshold lifted so any clear is saved:
  1. the agent plays fresh starts until it clears: the clear is saved, and
     replayed frame by frame from the captured state it clears on exactly its
     last frame (route_book.check_captured)
  2. the same from a late demo start: the saved route is Tyler's frames up to
     the start point, then the agent's, and it clears from the captured state
     too, as one route
  3. a clear no faster than the book's route is not saved
Ends with ALL CHECKS PASS or the failures.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
TRIES = 60          # episodes per check before giving up on a clear
failures = []


def check(ok, what):
    print(f"  {'ok ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failures.append(what)


def run_child(model, board, out_dir):
    """Plays in a training env; prints one JSON line of what happened."""
    sys.path.insert(0, HOME)
    import ss_pilot
    preset = ss_pilot.run_preset('d9')
    for k in ss_pilot.SS_KEYS:
        os.environ.pop(k, None)
    demo = sorted(glob.glob(os.path.join(HOME, 'demos', f'stage{board}_*.npz')))[-1]
    os.environ.update(preset)
    os.environ.update(SS_STAGES=str(board), SS_DEMOS=demo, SS_DEMO_PROB='0', SS_RESTART_PROB='0',
                      SS_LOG=os.devnull, SS_ROUTES=out_dir)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import special_stage_env as sse
    from sb3_contrib import RecurrentPPO
    env = sse.SpecialStageEnv()
    pol = RecurrentPPO.load(model, device='cpu')
    np.random.seed(1)
    torch.manual_seed(1)
    env.demo_info()                                 # builds the demo's start points
    did = env._demos[0]['id']
    late = len(env._demos[0]['ckpts']) - 1 - 3

    def episode(start):
        obs, _ = start()
        lstm, first = None, True
        while True:
            a, lstm = pol.predict(obs, state=lstm, episode_start=np.array([first]), deterministic=False)
            first = False
            obs, _, done, trunc, _ = env.step(int(np.asarray(a).reshape(-1)[0]))
            if done or trunc:
                return bool(env._board_cleared), int(env.steps)

    out = {}
    for kind, start in (('fresh', lambda: env.reset(options={'stage_file': sse.stage_file(board)})),
                        ('demo', lambda: env._reset_from_demo())):
        env.set_demo_pointers({did: late})
        got = None
        for i in range(TRIES):
            before = set(glob.glob(os.path.join(out_dir, '*.npz')))
            env._route_best = {}                    # any clear is a candidate here
            cleared, steps = episode(start)
            new = sorted(set(glob.glob(os.path.join(out_dir, '*.npz'))) - before)
            if cleared:
                got = dict(episodes=i + 1, steps=steps, saved=new,
                           prefix=0 if env._route_prefix is None else len(env._route_prefix), from_restart=env._from_restart)
                break
            if new:
                got = dict(episodes=i + 1, error='saved a route without a clear', saved=new)
                break
        out[kind] = got
    # 3: no faster than the book's -> not saved
    env._route_best = {board: 1}
    before = set(glob.glob(os.path.join(out_dir, '*.npz')))
    for i in range(TRIES):
        cleared, steps = episode(lambda: env.reset(options={'stage_file': sse.stage_file(board)}))
        if cleared:
            break
    out['slow'] = dict(cleared=cleared, saved=sorted(set(glob.glob(os.path.join(out_dir, '*.npz'))) - before))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', nargs='?', default=os.path.join(HOME, 'sonic_ss_multistage_d9.zip'))
    ap.add_argument('--board', type=int, default=10, help='a board the agent often clears (d9: 10)')
    a = ap.parse_args()
    sys.path.insert(0, HOME)
    import demos
    import route_book
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, 'model.zip')
        subprocess.run(['cp', a.model, snap], check=True)      # a running trainer rewrites its save
        cand = os.path.join(tmp, 'cand')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        p = subprocess.run([sys.executable, __file__, '--child', snap, str(a.board), cand],
                           env=env, capture_output=True, text=True, cwd=HOME)
        lines = [l for l in p.stdout.splitlines() if l.startswith('{')]
        if p.returncode or not lines:
            print(p.stdout[-2000:], p.stderr[-3000:])
            sys.exit('the training env crashed')
        r = json.loads(lines[-1])
        for n, kind in ((1, 'fresh'), (2, 'demo')):
            print(f'{n}. a {kind} start\'s clear on board {a.board}')
            g = r[kind]
            if not g or 'error' in g:
                check(False, f'{kind}: {g or f"no clear in {TRIES} episodes"}')
                continue
            check(len(g['saved']) == 1, f"{kind}: cleared after {g['episodes']} episode(s) in "
                  f"{g['steps']} frames; saved {len(g['saved'])} route(s)")
            if len(g['saved']) != 1:
                continue
            path = g['saved'][0]
            actions, meta = demos.load_demo(path)
            c = route_book.check_captured([path])[path]
            check(len(actions) == g['steps'] == meta['frames'] and c['ok'],
                  f"{kind}: the route ({len(actions)} frames, {g['prefix']} of them Tyler's) replays "
                  f"from the captured state: cleared {c.get('cleared')} on frame "
                  f"{c.get('ended_at')} of {len(actions)} ({c.get('spheres')}/{c.get('total')})")
            if kind == 'demo':
                his, _ = demos.load_demo(sorted(glob.glob(os.path.join(HOME, 'demos', f'stage{a.board}_*.npz')))[-1])
                check(g['prefix'] > 0 and list(actions[:g['prefix']]) == list(his[:g['prefix']]),
                      f"demo: its first {g['prefix']} frames are Tyler's ({meta.get('source')})")
        print('3. a clear no faster than the book\'s route')
        check(r['slow']['cleared'] and not r['slow']['saved'],
              f"cleared {r['slow']['cleared']}, saved {len(r['slow']['saved'])}")
    print('\nALL CHECKS PASS' if not failures else f'\n{len(failures)} FAILED:\n  ' + '\n  '.join(failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        run_child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()
