"""smoke_corners.py — check env v4 (one choice per corner) against your demos.

For each demo of a run (default d3):
  1. replay your frames, find every decision point v4 would offer and what you
     did at that corner (straight / left / right / jump)
  2. play those choices through v4's own step() from a fresh start: the same
     corners must come up in the same order and end the same way
  3. from EVERY demo start point the curriculum can use: the episode must
     begin on one of your decision points with nothing queued, and playing
     your remaining choices from there must collect exactly what you did

  python3 smoke_corners.py [run]      # must end "ALL CHECKS PASS"
"""
import os
import sys
import glob
import random

HOME = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1] if len(sys.argv) > 1 else 'd3'
# the trainer parses argv at import, so read RUNS out of its source instead
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read()
ns = {}
exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
if run not in ns['RUNS']:
    sys.exit(f'unknown run {run!r}. Known: {", ".join(ns["RUNS"])}')
cfg = ns['RUNS'][run]
if cfg['obs']['SS_ENV_VERSION'] != '4':
    sys.exit(f'{run} is not on env v4')
DEMOS = [p for pat in cfg.get('demos', [])
         for p in sorted(glob.glob(os.path.join(HOME, 'demos', f'{pat}.npz')))]
os.environ.update(cfg['obs'], SS_STAGES=','.join(map(str, cfg['stages'])),
                  SS_RESTART_PROB='0', SS_DEMOS=','.join(DEMOS), SS_DEMO_PROB='1.0',
                  SS_LOG=os.path.join(HOME, 'logs', f'smoke_corners_{run}.jsonl'))
import special_stage_env as sse      # noqa: E402
import demos as demo_io              # noqa: E402

env = sse.SpecialStageEnv()
env._log_episode = lambda *a, **k: None
env.reset()                          # builds the demo start points
fails = 0


def check(ok, msg):
    global fails
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {msg}")


def his_corners(actions, stage):
    """Your decision points (frame, cell, choice) along the recorded run."""
    env.reset(options={'stage_file': sse.stage_file(stage), 'replay': True})
    points, traj = [], []
    for j, a in enumerate(actions):
        _, _, done, trunc, info = env._step_frames(int(a), 1)
        ram = env._ram()
        traj.append((env._quadrant(int(ram[sse.ADDR_HEADING])), int(ram[sse.ADDR_JUMPING])))
        if not done and env._at_decision(info):
            points.append(dict(f=j, cell=env._decided))
        if done or trunc:
            break
    for k, p in enumerate(points):
        seg = traj[p['f']:(points[k + 1]['f'] if k + 1 < len(points) else len(traj) - 1) + 1]
        if any(j & 0x80 for _, j in seg):
            p['choice'] = 3
        else:
            p['choice'] = {0: 0, 1: 1, 3: 2}.get((seg[-1][0] - seg[0][0]) % 4)
    return points, env._spheres_collected, env._board_cleared


def follow(points, k0):
    """Play your choices from decision k0 on; None on the first divergence."""
    for p in points[k0:]:
        if env._decided != p['cell'] or p['choice'] is None:
            return None
        _, _, done, trunc, _ = env.step(p['choice'])
        if done or trunc:
            break
    return env._spheres_collected, env._board_cleared


for path, d in zip(DEMOS, env._demos):
    actions, meta = demo_io.load_demo(path)
    print(f"{os.path.basename(path)}: stage {meta['stage']}, {meta['spheres']} spheres, "
          f"{len(actions)} frames, outcome {meta['outcome']}")
    points, final, cleared = his_corners(actions, meta['stage'])
    print(f"  {len(points)} corners: " + ', '.join(
        f'{sse.CORNER_ACTIONS[c]} {sum(p["choice"] == c for p in points)}' for c in range(4)))
    check(all(p['choice'] is not None for p in points), 'every corner reads as straight/left/right/jump')

    env.reset(options={'stage_file': sse.stage_file(meta['stage'])})
    got = follow(points, 0)
    # v4 may finish a few frames SOONER, never later: it queues each turn before
    # the corner's window, and a human tap on the window's last frame fires one
    # frame late (stage 2, 2026-10-09: 3 frames over 213 corners)
    check(got == (final, cleared) and env.steps <= len(actions),
          f'your corners from a fresh start: {got} in {env.steps} frames '
          f'(recorded: {final} spheres, cleared={cleared}, {len(actions)} frames)')

    at = {p['f']: k for k, p in enumerate(points)}
    bad = []
    for i, ck in enumerate(d['ckpts']):
        env.set_demo_pointers({d['id']: i})
        random.seed(0)                     # the band draw: pin it to this point
        sse.DEMO_BAND, band = 0, sse.DEMO_BAND
        try:
            env.reset()
        finally:
            sse.DEMO_BAND = band
        k = at.get(ck['fields']['steps'] - 1)
        queued = int(env._ram()[0xE42A]) if env._ram_swapped is not None else None
        res = follow(points, k) if k is not None else None
        target = env._demo_episode['target'] if env._demo_episode else None
        if k is None or queued or res is None or res[0] != target or res[1] != cleared:
            bad.append((i, k, queued, res, target))
    check(not bad, f"all {len(d['ckpts'])} start points begin on your decision point and "
                   f"your choices from there reproduce your finish"
                   + (f' — failures (ptr, corner, queued, result, target): {bad[:5]}' if bad else ''))

print('\nALL CHECKS PASS' if not fails else f'\n{fails} CHECK(S) FAILED')
sys.exit(1 if fails else 0)
