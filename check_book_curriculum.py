#!/usr/bin/env python3
"""check_book_curriculum.py — pre-launch check for runs with demos='book' (d10):
the curriculum along the route book's routes, switched live as the book
improves (train_ss_multistage.py book_demos + DemoCurriculumCallback;
special_stage_env set_demo_files).

    python3 check_book_curriculum.py [run]      # default d10

CPU only, ~2 min, one env in this process; the trainer's callback is read out
of train_ss_multistage.py with ast (importing it would start a run) and driven
through a stand-in for the VecEnv. Nothing is trained and no run file is
touched (the book is copied to a temporary directory; the start points are
read from a copy of the init run's curriculum file). Checks:
  1. every book route builds as a demo; start points carried over from the
     init run's curriculum (d9: Tyler's recordings) by frame: a board whose
     book route IS his recording keeps its exact start point; a faster
     (spliced) route starts at the first start point at or after that frame
  2. the book changes (one board's route swapped): the workers' next reset
     rebuilds only that board, the other 13 demos are reused as they were;
     the callback waits for that, then drops the old route and carries its
     start point over; the curriculum file records stage and frame
  3. set_demo_files in the middle of an episode changes nothing in it (same
     observations and rewards as without the call)
Ends with ALL CHECKS PASS or the failures.
"""
import ast
import collections
import glob
import json
import os
import shutil
import sys
import tempfile
import time

HOME = os.path.dirname(os.path.abspath(__file__))
failures = []


def check(ok, what):
    print(f"  {'ok ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failures.append(what)


def trainer_parts(names):
    """Source of these top-level defs/assignments from train_ss_multistage.py."""
    src = open(os.path.join(HOME, 'train_ss_multistage.py')).read()
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        name = getattr(node, 'name', None)
        if name is None and isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', None)
        if name in names:
            out.append(ast.get_source_segment(src, node))
    return '\n\n'.join(out)


def run_cfg(run):
    tree = ast.parse(open(os.path.join(HOME, 'train_ss_multistage.py')).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'RUNS':
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == run:
                    kw = {x.arg: x.value for x in v.keywords}
                    return dict(obs={x.arg: ast.literal_eval(x.value) for x in kw['obs'].keywords},
                                demos=ast.literal_eval(kw['demos']), init=ast.literal_eval(kw['init']))
    raise KeyError(run)


class FakeVec:
    """VecEnv.env_method over one env."""
    def __init__(self, env):
        self.env = env

    def env_method(self, name, *a, indices=None, **kw):
        return [getattr(self.env, name)(*a, **kw)]


class FakeLogger:
    def record(self, *a, **k):
        pass


class FakeModel:
    """What BaseCallback reads through: model.get_env(), model.logger."""
    def __init__(self, env):
        self.vec, self.logger = FakeVec(env), FakeLogger()

    def get_env(self):
        return self.vec


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'd10'
    cfg = run_cfg(run)
    if cfg.get('demos') != 'book':
        sys.exit(f'run {run} does not use demos=\'book\'')
    sys.path.insert(0, HOME)
    import route_book
    tmp = tempfile.mkdtemp()
    try:
        book = os.path.join(tmp, 'routes')
        shutil.copytree(route_book.BOOK_DIR, book, ignore=shutil.ignore_patterns('candidates', 'curriculum'))
        route_book.BOOK_DIR = book                          # book_demos reads the copy
        ns = dict(os=os, json=json, time=time, shutil=shutil, collections=collections, HOME=HOME)
        exec(trainer_parts({'book_demos', 'DEMO_WINDOW', 'DEMO_MIN_N', 'DEMO_ADVANCE', 'BOOK_CHECK_S'}), ns)
        stages = list(range(1, 15))
        demo_files = ns['book_demos'](stages)
        for k in ('SS_ENV_VERSION', 'SS_USE_PIXELS', 'SS_PATCH_R', 'SS_DROP_POSITION', 'SS_POINTER', 'SS_SPEED_BONUS'):
            os.environ.pop(k, None)
        os.environ.update(cfg['obs'])
        os.environ.update(SS_STAGES=','.join(map(str, stages)), SS_DEMOS=','.join(demo_files),
                          SS_DEMO_PROB='0', SS_RESTART_PROB='0', SS_LOG=os.devnull, SS_ROUTES='')
        import special_stage_env as sse
        from stable_baselines3.common.callbacks import BaseCallback
        ns.update(special_stage_env=sse, BaseCallback=BaseCallback, book_demos=ns['book_demos'])
        exec(trainer_parts({'DemoCurriculumCallback'}), ns)
        env = sse.SpecialStageEnv()
        t0 = time.time()
        env.reset(options={'stage_file': sse.stage_file(1)})
        t_build = time.time() - t0

        print(f'1. the book as demos; start points carried over from {cfg.get("init")}')
        init_state = os.path.join(HOME, f'sonic_ss_multistage_{cfg["init"]}_demos.json')
        state = os.path.join(tmp, 'demos.json')
        shutil.copy(init_state, state)
        saved = json.load(open(state))
        cb = ns['DemoCurriculumCallback'](state, book_stages=stages)
        cb.model = FakeModel(env)
        cb._on_training_start()
        check(len(cb.ptr) == 14, f'{len(cb.ptr)} book routes built as demos in {t_build:.0f} s')
        import demos as demo_io
        old_by_board = {}
        for sid, e in saved.items():
            b = cb._board_of(sid, e)
            old_by_board[b] = (sid, e, cb._frame_of(sid, e))
        for did in sorted(cb.ptr, key=lambda d: cb.stage[d]):
            b = cb.stage[did]
            sid, e, f = old_by_board[b]
            steps = [fr * cb.frames[did] for fr in cb.fracs[did]]
            p = cb.ptr[did]
            same_route = list(demo_io.load_demo(route_book.book_path(b))[0]) == list(
                demo_io.load_demo(os.path.join(HOME, 'demos', f'{sid}.npz'))[0])
            if same_route:
                ok = p == e['ptr']
                what = f'same route as {sid}: start point {p} (was {e["ptr"]})'
            else:
                ok = steps[p] >= f - 0.5 and (p == 0 or steps[p - 1] < f - 0.5)
                what = (f'faster route ({cb.frames[did]} frames): starts at frame {steps[p]:.0f}, '
                        f'the first start point at or after {f:.0f} (his {sid})')
            check(ok, f'board {b:2d}: {what}')

        print('2. the book changes under a running curriculum')
        hist = sorted(glob.glob(os.path.join(book, 'history', 'board10_*.npz')))
        alt = hist[0] if hist else None
        if alt is None:
            check(False, 'no older board-10 route in history/ to swap in')
        else:
            old_id = next(d for d in cb.ptr if cb.stage[d] == 10)
            kept_before = {d['id']: d for d in env._demos}
            shutil.copy(alt, route_book.book_path(10))
            cb._check_book()
            check(cb._pending is not None and old_id not in cb._pending,
                  f'switch requested: {old_id} -> {sorted(cb._pending - set(cb.ptr))}')
            cb._adopt_pending()
            check(old_id in cb.ptr, 'nothing adopted before the worker has rebuilt (demo_info(build=False) '
                  f'-> {env.demo_info(build=False)})')
            t0 = time.time()
            env.reset(options={'stage_file': sse.stage_file(1)})
            t_re = time.time() - t0
            reused = sum(1 for d in env._demos if kept_before.get(d['id']) is d)
            check(reused == 13, f'the next reset rebuilt 1 demo and reused {reused} ({t_re:.1f} s)')
            cb._adopt_pending()
            new_id = next(d for d in cb.ptr if cb.stage[d] == 10)
            st = json.load(open(state))
            check(cb._pending is None and old_id not in cb.ptr and new_id != old_id
                  and all({'stage', 'frame', 'ptr', 'frac'} <= set(v) for v in st.values()) and len(st) == 14,
                  f'adopted {new_id}, start point {cb.ptr[new_id]} at frame {st[new_id]["frame"]} '
                  f'(the old route\'s was frame {cb._frame_of(old_id, saved.get(old_id, {})) if old_id in saved else "-"}); '
                  f'curriculum file: 14 entries with stage and frame')

        print('3. set_demo_files in the middle of an episode')
        import numpy as np

        def episode(call_at):
            obs, _ = env.reset(options={'stage_file': sse.stage_file(5)})
            out = [obs]
            rng = np.random.RandomState(3)
            for i in range(40):
                if i == call_at:
                    env.set_demo_files(demo_files)
                obs, r, done, trunc, _ = env.step(int(rng.randint(env.action_space.n)))
                out.append((obs, r))
                if done or trunc:
                    break
            return out
        a, b2 = episode(None), episode(2)
        same = len(a) == len(b2) and all(
            (np.array_equal(x['grid'], y['grid']) and np.array_equal(x['ram'], y['ram'])) if i == 0 else
            (x[1] == y[1] and all(np.array_equal(x[0][k], y[0][k]) for k in x[0]))
            for i, (x, y) in enumerate(zip(a, b2)))
        check(same and len(a) - 1 > 2, f'identical observations and rewards over {len(a) - 1} steps with the call at step 2')
        env.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nALL CHECKS PASS' if not failures else f'\n{len(failures)} FAILED:\n  ' + '\n  '.join(failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
