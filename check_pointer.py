"""Verify SS_POINTER (run d7's compass to the nearest blue) before any launch.

    python3 check_pointer.py          # all checks; must end with ALL CHECKS PASS

1. Flag off = the old env: the same seeded trajectory under git HEAD's
   special_stage_env.py and under the working copy with SS_POINTER=0 gives
   identical observations (ram + grid) and rewards.
2. Flag on changes nothing else: the first 10 features, the grid and the
   rewards equal the flag-off run; only 3 features are appended.
3. The pointer agrees with the patch: whenever the nearest blue is inside the
   15x15 patch, it sits exactly where the pointer says (forward = up the
   patch, right = right); no blue in the patch is ever nearer than the pointer.

Trajectory: d5's preset (env v4, stage 1, Tyler's demo), episodes started from
demo checkpoints 0/16/32/48 (full to nearly empty boards), seeded actions.
"""
import os, subprocess, sys, tempfile
import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HOME, 'demos', 'stage1_20260930-165817.npz')
STEPS_PER_START = 150
STARTS = (0, 16, 32, 48)


def record(out, pointer):
    """Run the seeded trajectory with whatever special_stage_env is first on sys.path."""
    os.environ.update(SS_ENV_VERSION='4', SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                      SS_POINTER=pointer, SS_STAGES='1', SS_RESTART_PROB='0', SS_DEMOS=DEMO,
                      SS_DEMO_PROB='1.0', CUDA_VISIBLE_DEVICES='',
                      SS_LOG=os.path.join(tempfile.gettempdir(), f'check_pointer_{os.getpid()}.jsonl'))
    import random
    import special_stage_env as sse
    print(f'  [{pointer=}] env module: {sse.__file__}', flush=True)
    env = sse.SpecialStageEnv()
    rng = np.random.default_rng(1234)
    rams, grids, rews = [], [], []
    env.reset()                                   # builds the demo checkpoints
    did = env._demos[0]['id']
    for k in STARTS:
        random.seed(k)                            # _reset_from_demo's band draw
        env.set_demo_pointers({did: k})
        obs, _ = env.reset()
        for _ in range(STEPS_PER_START):
            rams.append(obs['ram']); grids.append(obs['grid'])
            obs, r, done, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            rews.append(r)
            if done or trunc:
                random.seed(len(rews))
                obs, _ = env.reset()
    np.savez(out, ram=np.array(rams), grid=np.array(grids), rew=np.array(rews))


def main():
    tmp = tempfile.mkdtemp(prefix='check_pointer_')
    old = os.path.join(tmp, 'old_env')            # HEAD's env, everything else symlinked
    os.mkdir(old)
    for name in os.listdir(HOME):
        if name != 'special_stage_env.py':
            os.symlink(os.path.join(HOME, name), os.path.join(old, name))
    with open(os.path.join(old, 'special_stage_env.py'), 'wb') as f:
        f.write(subprocess.run(['git', 'show', 'HEAD:special_stage_env.py'], cwd=HOME,
                               check=True, capture_output=True).stdout)
    runs = {'old': (old, '0'), 'off': (HOME, '0'), 'on': (HOME, '1')}
    for tag, (where, ptr) in runs.items():
        print(f'recording {tag} ...', flush=True)
        subprocess.run([sys.executable, '-c',
                        f'import sys; sys.path.insert(0, {where!r}); sys.argv=["x"]; '
                        f'import importlib.util as u; s=u.spec_from_file_location("cp", {__file__!r}); '
                        f'm=u.module_from_spec(s); s.loader.exec_module(m); '
                        f'm.record({os.path.join(tmp, tag + ".npz")!r}, {ptr!r})'],
                       cwd=where, check=True)
    d = {t: np.load(os.path.join(tmp, t + '.npz')) for t in runs}
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f'  {"ok  " if cond else "FAIL"} {msg}')

    print('1. flag off == git HEAD')
    check(d['old']['ram'].shape == d['off']['ram'].shape, f"ram shape {d['off']['ram'].shape}")
    check(np.array_equal(d['old']['ram'], d['off']['ram']), 'ram features identical')
    check(np.array_equal(d['old']['grid'], d['off']['grid']), 'grid patches identical')
    check(np.array_equal(d['old']['rew'], d['off']['rew']), 'rewards identical')
    print('2. flag on changes only the appended 3')
    check(d['on']['ram'].shape[1] == d['off']['ram'].shape[1] + 3, f"ram width {d['on']['ram'].shape[1]}")
    check(np.array_equal(d['on']['ram'][:, :-3], d['off']['ram']), 'first features identical')
    check(np.array_equal(d['on']['grid'], d['off']['grid']), 'grid patches identical')
    check(np.array_equal(d['on']['rew'], d['off']['rew']), 'rewards identical')
    print('3. pointer vs patch')
    R = (d['on']['grid'].shape[1] - 1) // 2
    inside = agree = nearer = empty_ok = n_blue = 0
    for ram, grid in zip(d['on']['ram'], d['on']['grid']):
        f, rt, dist = ram[-3:].astype(float)
        dist *= 16
        pr, pc = np.nonzero(grid[..., 0])          # blues in the patch
        fwd_p, right_p = R - pr, pc - R
        if dist == 0:
            empty_ok += (f == 0 and rt == 0)
            continue
        n_blue += 1
        check_unit = abs(np.hypot(f, rt) - 1) < 1e-5
        if not check_unit:
            check(False, f'pointer not unit length: {f}, {rt}')
        if len(pr):
            dp = np.hypot(fwd_p, right_p)
            nearer += dp.min() < dist - 1e-4
        if dist <= R:
            inside += 1
            hit = (np.abs(fwd_p - f * dist) < 1e-3) & (np.abs(right_p - rt * dist) < 1e-3)
            agree += bool(hit.any())
    check(n_blue > 0, f'{n_blue} observations with a blue left')
    check(inside > 0 and agree == inside,
          f'nearest blue inside the patch on {inside}: patch has a blue exactly there on {agree}')
    check(nearer == 0, f'patch blue nearer than the pointer: {nearer}')
    print(f'  (observations with no blue / Sonic on it: {len(d["on"]["ram"]) - n_blue}, zeros {empty_ok})')
    print('\nALL CHECKS PASS' if ok else '\nSOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
