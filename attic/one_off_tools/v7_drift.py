"""smoke_corners.py — check the corner envs (v4, v5) against your demos.

For each demo of a run (default d3):
  1. replay your frames, find every decision point the env would offer and
     what you did at that corner (straight / left / right / jump, and on v5
     left+jump / right+jump: a jump pressed during the turn, which fires as
     it ends). A move the env lacks fails here, by name.
  2. play those choices through the env's own step() from a fresh start; it
     must end the same way, in no more frames than you took
  3. from EVERY start point of that demo: the episode must begin on one of
     your decision points with nothing queued, and your remaining choices
     must collect exactly what you did

Corners are matched BY CELL, not by count: a jump taken a few frames
earlier or later lands at a different point in a cell, so the env can offer
a choice on a cell you passed without one (answered straight) or skip one
you had (fine only if you went straight there).

  python3 smoke_corners.py [run]      # must end "ALL CHECKS PASS"
"""
import os
import sys
import contextlib
import copy
import glob
import random

HOME = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, HOME)
STAGE, PAT = int(sys.argv[1]), sys.argv[2]
run = f'new{STAGE}'
# the trainer parses argv at import, so read RUNS out of its source instead
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read()
ns = {}
exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
if 'd4' not in ns['RUNS']:
    sys.exit(f'unknown run {run!r}. Known: {", ".join(ns["RUNS"])}')
cfg = dict(ns['RUNS']['d4'], stages=[STAGE], demos=[PAT])  # d4's setup, one recording
cfg['obs'] = dict(cfg['obs'], SS_ENV_VERSION=os.environ.get('CHK_ENV', cfg['obs']['SS_ENV_VERSION']))
if cfg['obs']['SS_ENV_VERSION'] not in ('4', '5', '6', '7'):
    sys.exit(f'{run} is not on a corner env (v4-v7)')
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


@contextlib.contextmanager
def replaying():
    """Your run, not an episode: no logging, and v6 skips the stall rule."""
    env._replaying = True
    try:
        yield
    finally:
        env._replaying = False


def his_corners(actions, stage):
    """Your decision points (frame, cell, choice) along the recorded run.
    choice = an index into CORNER_ACTIONS, or the name of a move it lacks."""
    env.reset(options={'stage_file': sse.stage_file(stage), 'replay': True})
    points, traj, extra = [], [], []
    with replaying():
        for j, a in enumerate(actions):
            _, _, done, trunc, info = env._step_frames(int(a), 1)
            ram = env._ram()
            x, y = int(info.get('ss_x', 0)), int(info.get('ss_y', 0))
            traj.append((int(ram[sse.ADDR_HEADING]), int(ram[sse.ADDR_JUMPING]),
                         x, y, int(info.get('ss_move_speed', 0)), env._cell(x, y)))
            if not done and env._at_decision(info):
                points.append(dict(f=j, cell=env._decided))
            if done or trunc:
                break
    # v7: frames where a backward run's speed starts climbing (Up) — the onset
    # only, so a turn-round that carries him into the next cell counts once
    onsets = {k for k in range(2, len(traj)) if traj[k - 1][4] < 0
              and traj[k][4] > traj[k - 1][4] and traj[k - 1][4] <= traj[k - 2][4]}
    for k, p in enumerate(points):
        seg = traj[p['f']:(points[k + 1]['f'] if k + 1 < len(points) else len(traj) - 1) + 1]
        turn = {0: '', 1: 'left', 3: 'right'}.get(
            (env._quadrant(seg[-1][0]) - env._quadrant(seg[0][0])) % 4, 'reverse')
        t_rot = next((i for i, (h, *_) in enumerate(seg) if h != seg[0][0]), None)
        # $80 = a jump (B); $81 = a spring launch, which no button chose (stage 8)
        t_air = next((i for i, (_, j, *_) in enumerate(seg) if j & 0x81 == 0x80), None)
        if t_air is None:
            name = turn or 'straight'
        elif not turn:
            name = 'jump'
        elif t_rot < t_air:
            name = turn + '+jump'               # jumped out of the turn
        else:
            name = 'jump, then ' + turn          # turned after landing
        if sse.V7:
            name = v7_name(name, p, seg, t_air, extra, onsets)
        p['choice'] = name
    points = sorted(points + extra, key=lambda q: q['f'])
    for p in points:
        p['choice'] = (sse.CORNER_ACTIONS.index(p['choice'])
                       if p['choice'] in sse.CORNER_ACTIONS else p['choice'])
    return points, env._spheres_collected, env._board_cleared


LATE_FROM = -48     # take-off offset from the centre: a jump fires ~-110, a late-jump ~+16


def v7_name(name, p, seg, t_air, extra, onsets):
    """v7's reading of a corner: Up turning round from a backward run, and
    each jump by where it took off. A take-off on the frame he entered the
    NEXT cell left that cell no decision point: read it as straight here and
    a jump there (stage 9, (13,27) -> (12,27))."""
    # a turn-round starting at this decision or after it (the last frame is the
    # next decision's own; stage 10 held Up through a lock: onset = decision)
    t_up = next((i for i in range(len(seg) - 1) if p['f'] + i in onsets), None)
    if t_up is not None:
        return 'up' if name == 'straight' else 'up, then ' + name
    if name != 'jump' or t_air == 0:
        return name
    (_, _, x0, y0, _, _), (_, _, x, y, _, cell) = seg[t_air - 1], seg[t_air]
    mx = (x - x0 + 4096) % 8192 - 4096
    my = (y - y0 + 4096) % 8192 - 4096
    ox = (x - (cell[1] << 8) + 4096) % 8192 - 4096
    oy = (y - (cell[0] << 8) + 4096) % 8192 - 4096
    off = (ox if mx > 0 else -ox) if abs(mx) >= abs(my) else (oy if my > 0 else -oy)
    jump = 'late-jump' if off >= LATE_FROM else 'jump'
    if cell == p['cell']:
        return jump
    extra.append(dict(f=p['f'] + t_air - 1, cell=cell, choice=jump))
    return 'straight'


SHARED = {'base_env', '_demos', 'action_space', 'observation_space', '_original_grids', '_actions'}
picked = []


def snapshot():
    return (env.base_env.em.get_state(),
            {k: v if k in SHARED else copy.deepcopy(v) for k, v in env.__dict__.items()})


def restore(snap):
    env.base_env.em.set_state(snap[0])
    env.__dict__.update({k: v if k in SHARED else copy.deepcopy(v) for k, v in snap[1].items()})


def best_jump(points, j, a):
    """v7: your take-off could be anywhere in the cell; the env has a jump
    and a late-jump, and which lands where you did depends on speed. Try
    both from here, and straight here + a jump at the next cell (a take-off
    just before the boundary, fast: stage 6 corner 376, +116 at 28/frame).
    Among those whose next decision is your next corner, keep the one that
    gets there in the frames you took (landing is a decision, so the cell
    alone rarely tells them apart). Returns the actions, one per decision."""
    names = ('jump', 'late-jump')
    other = sse.CORNER_ACTIONS.index(names[1 - names.index(sse.CORNER_ACTIONS[a])])
    want = points[j + 1]['cell'] if j + 1 < len(points) else None
    his_dt = points[j + 1]['f'] - points[j]['f'] if j + 1 < len(points) else None
    snap, steps0, best = snapshot(), env.steps, None
    for cands in ((a,), (other,), (0, sse.CORNER_ACTIONS.index('jump'))):
        restore(snap)
        for c in cands:
            _, _, done, trunc, _ = env.step(c)
            if done or trunc:
                break
        if not (done or trunc) and env._decided == want:
            miss = abs((env.steps - steps0) - his_dt)
            if best is None or miss < best[0]:
                best = (miss, cands, (env._decided, env.steps, env._spheres_collected))
    restore(snap)
    pick, seen = (best[1], best[2]) if best else ((a,), None)
    picked.append((pick != (a,), seen))
    return pick


def follow(points, k0):
    """Play your choices from decision k0 on, matching corners by cell.
    Returns (spheres, cleared), or None where the env can't follow you."""
    i, extra, queue, expect = k0, 0, [], None
    while True:
        here = env._decided
        ahead = [q['cell'] for q in points[i:i + 4]]
        if queue:
            a = queue.pop(0)            # the deferred jump, at the next cell
        elif here in ahead:
            j = i + ahead.index(here)
            if any(points[m]['choice'] != 0 for m in range(i, j)):
                return None             # skipped a corner where you turned or jumped
            a, i = points[j]['choice'], j + 1
            if not isinstance(a, int):
                return None             # a move this env lacks
            if sse.V7 and sse.CORNER_ACTIONS[a] in ('jump', 'late-jump'):
                a, *queue = best_jump(points, j, a)
                expect = picked[-1][1]
        else:
            a, extra = 0, extra + 1     # a cell you passed without a choice
            if extra > 40:
                return None
        _, _, done, trunc, _ = env.step(a)
        # yardstick: the picked timing replays to the decision it reached in the trial
        if expect and not queue:
            if (env._decided, env.steps, env._spheres_collected) != expect:
                print(f'  WARNING snapshot restore not faithful: {expect} vs '
                      f'{(env._decided, env.steps, env._spheres_collected)}')
            expect = None
        if done or trunc:
            return env._spheres_collected, env._board_cleared



log = []
_orig = env.step
def step(a):
    r = _orig(a)
    log.append((sse.CORNER_ACTIONS[a], env._decided, env.steps))
    return r
env.step = step
path = DEMOS[0]
actions, meta = demo_io.load_demo(path)
points, final, cleared = his_corners(actions, meta['stage'])
env.reset(options={'stage_file': sse.stage_file(meta['stage'])})
start = env.steps
with replaying():
    got = follow(points, 0)
print('result', got, 'frames', env.steps, 'his', len(actions))
# walk his points in order, matching each env decision to the next his point with that cell
i, prev_d = 0, None
hist = [(None, env._decided if False else None, start)] + log
for k in range(1, len(hist)):
    act, cell, steps = hist[k]
    for j in range(i, min(i + 6, len(points))):
        if points[j]['cell'] == cell:
            d = steps - (points[j]['f'] + 1)
            if d != prev_d:
                name = points[j-1]['choice'] if j else None
                print(f'  his corner {j} cell {cell} f={points[j]["f"]}: env steps {steps}, drift {d:+d}  (env action before: {hist[k][0]}, his prev choice {sse.CORNER_ACTIONS[name] if isinstance(name,int) else name})')
                prev_d = d
            i = j + 1
            break
