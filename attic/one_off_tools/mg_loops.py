#!/usr/bin/env python3
"""mg_loops.py — what does the main-game env pay an agent that misses a loop's escape?

    python3 attic/one_off_tools/mg_loops.py ACT START_POINT POLICY [FRAMES]
    python3 attic/one_off_tools/mg_loops.py SOZ2 80 down          # Sandopolis 2: hold Down, never jump across

Tyler 2026-10-12: Ice Cap 1 (a pit you fall through again and again until you jump onto a ledge that
falls) and Sandopolis 2 (two crossing sand slides: swap the right way or the loop goes on). Both acts
wrap vertically at y 2048; his route loops 4 times (ICZ1, x ~25,450, demo t 10,231-12,449) and 3 times
(SOZ2, x ~4,130, t 2,997-4,745) before he escapes. RouteProgress matches the agent to his route within
[-300, +600] demo frames of its last match, so lap n of the agent is matched to lap n of his.

From his save state at START_POINT, plays a scripted POLICY (agent actions, 4-frame repeat, through the
real env) and prints, every 60 frames and at each wrap: position, the matched demo time, best progress,
the step reward and the running return; then how the episode ended. A policy that stays in the loop
should earn nothing once it is past his last lap, and be cut as `stuck`.
POLICY: a name below, a comma list of ACTION:STEPS (e.g. right:20,jump+left:2), or his:T0-T1 = HIS
recorded inputs from demo time T0 to T1, over and over (start at a start point at T0: the first pass is
his own lap, later passes are laps he never ran).
"""
import os
import sys

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MG_ENV_VERSION', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MG_LOG'] = os.devnull
import coordinator as C                 # noqa: E402

PLAY = 'demos/play_20261010-202923.npz'      # his whole-game recording

POLICIES = {
    'down': 'down:1',
    'right': 'right:1',
    'left': 'left:1',
    'noop': 'noop:1',
}


def parse(policy):
    seq = []
    for part in POLICIES.get(policy, policy).split(','):
        name, n = part.split(':')
        seq += [C.ACTION_NAMES.index(name)] * int(n)
    return seq


def main():
    act, k, policy = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 3600
    env = C.SonicCoordinator(demo_files=[PLAY], acts=[act])
    env.demo_info()
    seg = next(s for s in env._segs if s['name'] == act)
    seq = [0]
    if policy.startswith('his:'):                   # his inputs, frame for frame, in a cycle
        import demos
        t0, t1 = map(int, policy[4:].split('-'))
        actions, meta = demos.load_play(PLAY)
        table = meta['action_table']
        f0 = seg['ckpts'][0]['frame'] - seg['ckpts'][0]['t']     # recording frame of demo time 0
        orig, n = C.emu_frame, {'i': 0}

        def his_lap(e, _mask):
            f = f0 + t0 + 1 + n['i'] % (t1 - t0)
            n['i'] += 1
            return orig(e, table[actions[f]])
        C.emu_frame = his_lap
    else:
        seq = parse(policy)
    env.reset(options={'key': seg['key'], 'ckpt': k})
    rt = env._route
    print(f"{act} start point {k} (his demo t {seg['ckpts'][k]['t']}), policy {policy}; his act time {seg['T']}", flush=True)
    ret, y_prev, i, done, last_print = 0.0, None, 0, False, -999
    while not done and env.frames < limit:
        _, r, term, trunc, info = env.step(seq[i % len(seq)])
        i += 1
        r /= C.REWARD_SCALE
        ret += r
        done = term or trunc
        x, y = int(info['x']), int(info['y'])
        wrap = y_prev is not None and abs(y - y_prev) > 1000
        y_prev = y
        if wrap or env.frames - last_print >= 60 or done:
            last_print = env.frames
            print(f"  frame {env.frames:5d}  ({x:6d},{y:5d}){'  WRAP' if wrap else '      '}  matched t {rt.p:6d}"
                  f"{'' if rt.on else ' (off)'}  best {rt.best:6d}  step r {r:+7.1f}  return {ret:+9.1f}", flush=True)
    res = info.get('mg_result', {})
    print(f"ended: {res.get('outcome', 'limit')} after {env.frames} frames; best progress {rt.best} "
          f"(start {seg['ckpts'][k]['t']}, +{rt.best - seg['ckpts'][k]['t']} demo frames); return {ret:+.1f}")


if __name__ == '__main__':
    main()
