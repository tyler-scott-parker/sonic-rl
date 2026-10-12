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

HOME = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1] if len(sys.argv) > 1 else 'd3'
# the trainer parses argv at import, so read RUNS out of its source instead
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read()
ns = {}
exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
if run not in ns['RUNS']:
    sys.exit(f'unknown run {run!r}. Known: {", ".join(ns["RUNS"])}')
cfg = ns['RUNS'][run]
if cfg['obs']['SS_ENV_VERSION'] not in ('4', '5', '6', '7'):
    sys.exit(f'{run} is not on a corner env (v4-v7)')
if cfg.get('demos') == 'book':        # d10: the route book's current routes,
    sys.path.insert(0, HOME)             # frozen at the start (the harvest may promote
    import shutil                        # a route mid-run: the 2026-10-11 run read new
    import tempfile                      # routes against start points built from old ones)
    import route_book
    _frozen = tempfile.mkdtemp()
    DEMOS = []
    for b in cfg['stages']:
        DEMOS.append(os.path.join(_frozen, f'board{b:02d}.npz'))
        shutil.copy2(route_book.book_path(b), DEMOS[-1])
else:
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
    # only, so a turn-round that carries him into the next cell counts once.
    # Only where Up was pressed (on the onset frame in all 10 of Tyler's; up to 2
    # frames allowed): a jump out of a backward run also turns him round, mid-air
    # (d9's board-7 route, 2026-10-11: jump at 3,137, forwards at 3,141, no Up)
    onsets = {k for k in range(2, len(traj)) if traj[k - 1][4] < 0
              and traj[k][4] > traj[k - 1][4] and traj[k - 1][4] <= traj[k - 2][4]
              and 4 in actions[max(0, k - 2):k + 1]}
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
up_picked = []      # v7: up vs up-now, as for the jump timings


def snapshot():
    return (env.base_env.em.get_state(),
            {k: v if k in SHARED else copy.deepcopy(v) for k, v in env.__dict__.items()})


def restore(snap):
    env.base_env.em.set_state(snap[0])
    env.__dict__.update({k: v if k in SHARED else copy.deepcopy(v) for k, v in snap[1].items()})


def best_jump(points, j, a, cands=None, log=None):
    """v7: your take-off could be anywhere in the cell; the env has a jump
    and a late-jump, and which lands where you did depends on speed. Try
    both from here, and straight here + a jump at the next cell (a take-off
    just before the boundary, fast: stage 6 corner 376, +116 at 28/frame).
    Among those whose next decision is your next corner, prefer one from
    which your choice there leads on to the corner after it (a landing past
    the turn window queues that turn to the next cell: stage 4, (19,23)),
    then the one that gets there in the frames you took (landing is a
    decision, so the cell alone rarely tells them apart). Returns the
    actions, one per decision. cands/log: other alternatives (the two Up
    timings) and where to count the pick."""
    if cands is None:
        names = ('jump', 'late-jump')
        other = sse.CORNER_ACTIONS.index(names[1 - names.index(sse.CORNER_ACTIONS[a])])
        cands = ((a,), (other,), (0, sse.CORNER_ACTIONS.index('jump')))
    want = points[j + 1]['cell'] if j + 1 < len(points) else None
    his_dt = points[j + 1]['f'] - points[j]['f'] if j + 1 < len(points) else None
    then = points[j + 1]['choice'] if j + 2 < len(points) else None
    snap, steps0, best = snapshot(), env.steps, None
    for tried in cands:
        restore(snap)
        for c in tried:
            _, _, done, trunc, _ = env.step(c)
            if done or trunc:
                break
        if not (done or trunc) and env._decided == want:
            seen = (env._decided, env.steps, env._spheres_collected)
            miss = abs((env.steps - steps0) - his_dt)
            on = True                   # one corner of lookahead — not onto a
            # jump: its own timing is picked when we get there, and playing it
            # at the nominal one misjudged stage 6's chained hops (died 127/128)
            if isinstance(then, int) and sse.CORNER_ACTIONS[then] not in ('jump', 'late-jump'):
                _, _, done, trunc, _ = env.step(then)
                on = done or trunc or env._decided == points[j + 2]['cell']
            if best is None or (not on, miss) < best[0]:
                best = ((not on, miss), tried, seen)
    restore(snap)
    pick, seen = (best[1], best[2]) if best else ((a,), None)
    (picked if log is None else log).append((pick != (a,), seen))
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
            elif sse.V7 and sse.CORNER_ACTIONS[a] == 'up':
                # his Up may be held through the lock (up-now) or pressed past
                # the centre (up): measured, neither timing dominates
                a, *queue = best_jump(points, j, a, log=up_picked,
                                      cands=((a,), (sse.CORNER_ACTIONS.index('up-now'),)))
                expect = up_picked[-1][1]
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


for path, d in zip(DEMOS, env._demos):
    actions, meta = demo_io.load_demo(path)
    print(f"{os.path.basename(path)}: stage {meta['stage']}, {meta['spheres']} spheres, "
          f"{len(actions)} frames, outcome {meta['outcome']}")
    points, final, cleared = his_corners(actions, meta['stage'])
    names = [sse.CORNER_ACTIONS[c] if isinstance(c, int) else c for c in (p['choice'] for p in points)]
    print(f"  {len(points)} corners: " + ', '.join(f'{n} {names.count(n)}' for n in dict.fromkeys(names)))
    lacking = sorted({n for n in names if n not in sse.CORNER_ACTIONS})
    check(not lacking, f'every corner is a move env v{sse.ENV_VERSION} has'
                       + (f' — it lacks: {lacking}' if lacking else ''))

    env.reset(options={'stage_file': sse.stage_file(meta['stage'])})
    with replaying():
        got = follow(points, 0)
    # v4 may finish a few frames SOONER, never later: it queues each turn before
    # the corner's window, and a human tap on the window's last frame fires one
    # frame late (stage 2, 2026-10-09: 3 frames over 213 corners)
    # v7's Up goes in on the first frame he moves after a lock or a rotation;
    # you can hold it through one and turn round sooner (stage 10: 2 frames
    # for its first Up), so each Up may cost up to 2 frames
    slack = 2 * sum(p['choice'] == sse.CORNER_ACTIONS.index('up') for p in points) if sse.V7 else 0
    check(got == (final, cleared) and env.steps <= len(actions) + slack,
          f'your corners from a fresh start: {got} in {env.steps} frames '
          f'(recorded: {final} spheres, cleared={cleared}, {len(actions)} frames)')

    at = {p['f']: k for k, p in enumerate(points)}
    bad = []
    for i, ck in enumerate(d['ckpts']):
        env.set_demo_pointers({d['id']: i})
        random.seed(0)                     # the band draw: pin it to this point
        sse.DEMO_BAND, band = 0, sse.DEMO_BAND
        sse.DEMO_MIX, mix = 0, sse.DEMO_MIX
        all_demos, env._demos = env._demos, [d]   # this demo, not a random one
        try:
            env.reset()
        finally:
            sse.DEMO_BAND, sse.DEMO_MIX = band, mix
            env._demos = all_demos
        k = at.get(ck['fields']['steps'] - 1)
        queued = int(env._ram()[0xE42A]) if env._ram_swapped is not None else None
        with replaying():
            res = follow(points, k) if k is not None else None
        target = env._demo_episode['target'] if env._demo_episode else None
        if k is None or queued or res is None or res[0] != target or res[1] != cleared:
            bad.append((i, k, queued, res, target))
    check(not bad, f"all {len(d['ckpts'])} start points begin on your decision point and "
                   f"your choices from there reproduce your finish"
                   + (f' — failures (ptr, corner, queued, result, target): {bad[:5]}' if bad else ''))

if sse.V7:
    print(f'  jump timings tried: {len(picked)}, swapped to the other timing: '
          f'{sum(sw for sw, _ in picked)}, neither reached your next corner: '
          f'{sum(seen is None for _, seen in picked)}')
    print(f'  up timings tried: {len(up_picked)}, swapped to up-now: '
          f'{sum(sw for sw, _ in up_picked)}, neither reached your next corner: '
          f'{sum(seen is None for _, seen in up_picked)}')
print('\nALL CHECKS PASS' if not fails else f'\n{fails} CHECK(S) FAILED')
sys.exit(1 if fails else 0)
