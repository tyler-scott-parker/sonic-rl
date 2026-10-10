"""Verify SS_SPEED_BONUS (run d8's clear-time bonus) before any launch.

    python3 check_speed_bonus.py      # all checks; must end with ALL CHECKS PASS

1. Flag off = the old env: under git HEAD's special_stage_env.py and under the
   working copy with SS_SPEED_BONUS unset, the same trajectories give
   identical observations and rewards.
2. Flag on (7) changes only the clear: every reward equals the flag-off run
   except on the step that clears the board, which pays exactly
   7 x (SS_MAX_STEPS - frames) more. Deaths and stalls are unchanged.

Trajectories, on d7's preset (env v4, stage 1, compass on):
  a. Tyler's stage-1 recording replayed frame by frame from a fresh start
     (ends in a clear at frame 2,925);
  b. his corner choices played through the env's own step() from a fresh
     start (a clear through the corner step, the path training uses);
  c. seeded random corner choices from demo checkpoints 0/16/32/48 (deaths
     and stalls).
"""
import os, subprocess, sys, tempfile
import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HOME, 'demos', 'stage1_20260930-165817.npz')
STEPS_PER_START = 150
STARTS = (0, 16, 32, 48)
BONUS = 7.0


def record(out, bonus):
    """Run the trajectories with whatever special_stage_env is first on sys.path."""
    os.environ.update(SS_ENV_VERSION='4', SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                      SS_POINTER='1', SS_STAGES='1', SS_RESTART_PROB='0', SS_DEMOS=DEMO,
                      SS_DEMO_PROB='1.0', CUDA_VISIBLE_DEVICES='',
                      SS_LOG=os.path.join(tempfile.gettempdir(), f'check_speed_{os.getpid()}.jsonl'))
    if bonus is not None:
        os.environ['SS_SPEED_BONUS'] = bonus
    import random
    import special_stage_env as sse
    import demos as demo_io
    print(f'  [{bonus=}] env module: {sse.__file__}', flush=True)
    env = sse.SpecialStageEnv()
    rams, rews, clears, frames = [], [], [], []

    def note(obs, r):
        rams.append(obs['ram']); rews.append(r)
        clears.append(env._board_cleared); frames.append(env.steps)

    # a. frame-by-frame replay of his recording from a fresh start
    actions, _ = demo_io.load_demo(DEMO)
    env.reset(options={'stage_file': sse.stage_file(1), 'replay': True})
    env._replaying = True
    for a in actions:
        obs, r, done, trunc, _ = env._step_frames(int(a), 1)
        note(obs, r)
        if done or trunc:
            break
    env._replaying = False
    # b. his corner choices through step(): the decision points and choices
    #    come from smoke_corners' reading, saved by main() beforehand
    choices = np.load(os.path.join(os.path.dirname(out), 'choices.npy'))
    obs, _ = env.reset(options={'stage_file': sse.stage_file(1)})
    for c in choices:
        obs, r, done, trunc, _ = env.step(int(c))
        note(obs, r)
        if done or trunc:
            break
    # c. seeded random corners from demo checkpoints
    rng = np.random.default_rng(1234)
    env.reset()                                   # builds the demo checkpoints
    did = env._demos[0]['id']
    for k in STARTS:
        random.seed(k)
        env.set_demo_pointers({did: k})
        obs, _ = env.reset()
        for _ in range(STEPS_PER_START):
            obs, r, done, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            note(obs, r)
            if done or trunc:
                random.seed(len(rews))
                obs, _ = env.reset()
    np.savez(out, ram=np.array(rams), rew=np.array(rews), clear=np.array(clears),
             frames=np.array(frames), n_replay=len(actions))


def his_choices(out):
    """His decision-by-decision choices on stage 1, as smoke_corners reads them
    (straight / left / right at every corner; stage 1 has no jumps)."""
    os.environ.update(SS_ENV_VERSION='4', SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                      SS_POINTER='1', SS_STAGES='1', SS_RESTART_PROB='0', SS_DEMOS='',
                      SS_DEMO_PROB='0', CUDA_VISIBLE_DEVICES='',
                      SS_LOG=os.path.join(tempfile.gettempdir(), f'check_speed_{os.getpid()}.jsonl'))
    import special_stage_env as sse
    import demos as demo_io
    env = sse.SpecialStageEnv()
    actions, _ = demo_io.load_demo(DEMO)
    env.reset(options={'stage_file': sse.stage_file(1), 'replay': True})
    env._replaying = True
    points, heads = [], []
    for j, a in enumerate(actions):
        _, _, done, trunc, info = env._step_frames(int(a), 1)
        heads.append(int(env._ram()[sse.ADDR_HEADING]))
        if not done and env._at_decision(info):
            points.append(j)
        if done or trunc:
            break
    choices = []
    for k, f in enumerate(points):
        g = points[k + 1] if k + 1 < len(points) else len(heads) - 1
        turn = (env._quadrant(heads[g]) - env._quadrant(heads[f])) % 4
        choices.append({0: 0, 1: 1, 3: 2}[turn])  # straight, left, right
    np.save(out, np.array(choices))


def main():
    tmp = tempfile.mkdtemp(prefix='check_speed_')
    old = os.path.join(tmp, 'old_env')            # HEAD's env, everything else symlinked
    os.mkdir(old)
    for name in os.listdir(HOME):
        if name != 'special_stage_env.py':
            os.symlink(os.path.join(HOME, name), os.path.join(old, name))
    with open(os.path.join(old, 'special_stage_env.py'), 'wb') as f:
        f.write(subprocess.run(['git', 'show', 'HEAD:special_stage_env.py'], cwd=HOME,
                               check=True, capture_output=True).stdout)
    load = (f'import importlib.util as u; s=u.spec_from_file_location("cs", {__file__!r}); '
            f'm=u.module_from_spec(s); s.loader.exec_module(m); ')
    subprocess.run([sys.executable, '-c', f'import sys; sys.path.insert(0, {HOME!r}); sys.argv=["x"]; '
                    + load + f'm.his_choices({os.path.join(tmp, "choices.npy")!r})'], cwd=HOME, check=True)
    runs = {'old': (old, None), 'off': (HOME, None), 'on': (HOME, f'{BONUS:g}')}
    for tag, (where, bonus) in runs.items():
        print(f'recording {tag} ...', flush=True)
        subprocess.run([sys.executable, '-c',
                        f'import sys; sys.path.insert(0, {where!r}); sys.argv=["x"]; ' + load
                        + f'm.record({os.path.join(tmp, tag + ".npz")!r}, {bonus!r})'],
                       cwd=where, check=True)
    d = {t: np.load(os.path.join(tmp, t + '.npz')) for t in runs}
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f'  {"ok  " if cond else "FAIL"} {msg}')

    o, f, n = d['old'], d['off'], d['on']
    check(o['rew'].shape == f['rew'].shape and np.array_equal(o['rew'], f['rew'])
          and np.array_equal(o['ram'], f['ram']),
          f'flag off = git HEAD: {len(f["rew"])} steps, rewards and observations identical')
    check(np.array_equal(f['ram'], n['ram']) and np.array_equal(f['clear'], n['clear'])
          and np.array_equal(f['frames'], n['frames']),
          'flag on: same observations, same trajectory (frames, clears)')
    # the step that clears = first step of each run of clear==True
    cl = f['clear']
    first = np.flatnonzero(cl & ~np.r_[False, cl[:-1]])
    diff = n['rew'] - f['rew']
    want = np.zeros_like(diff)
    want[first] = BONUS * np.maximum(0, 18000 - f['frames'][first])
    check(len(first) >= 2, f'clears in the trajectories: {len(first)} at frames '
          f'{f["frames"][first].tolist()} (need the replay and the corner-step clear)')
    check(np.array_equal(diff, want),
          f'flag on pays exactly {BONUS:g} x (18000 - frames) on each clearing step '
          f'({[round(float(x)) for x in diff[first]]}) and 0 on all {len(diff) - len(first)} others')
    print('\nALL CHECKS PASS' if ok else '\nCHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
