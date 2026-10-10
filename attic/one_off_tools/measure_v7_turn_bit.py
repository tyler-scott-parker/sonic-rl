# usage: MV=7 SP=<scratch dir> python3 attic/one_off_tools/measure_v7_turn_outcome.py   (MV=6 for the v6 baseline)
"""Env v7: when the agent picks a turn, does it happen inside that step, or
is it still queued when the step ends (it fires a cell late)? Split by the
kind of decision point. Random agent, stages 1-14, 25 episodes each."""
import os, sys, random, inspect, textwrap, collections
HOME = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, HOME)
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read()
ns = {}
exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
os.environ.update(ns['RUNS']['d4']['obs'], SS_ENV_VERSION=os.environ.get('MV','7'),
                  SS_STAGES=','.join(map(str, range(1, 15))), SS_RESTART_PROB='0',
                  SS_DEMOS='', SS_DEMO_PROB='0', SS_LOG=os.path.join(os.environ['SP'], 'measure2.jsonl'))
import special_stage_env as sse
s = inspect.getsource(sse.SpecialStageEnv._at_decision)
for o, n in [("self._v7_pending = True\n            elif speed < 0", "self._v7_pending = True; self._sub = 'fwd-after-up'\n            elif speed < 0"),
             ("self._v7_pending = True\n            elif self._v7_air", "self._v7_pending = True; self._sub = 'back-restart'\n            elif self._v7_air"),
             ("self._v7_pending = True     # landed", "self._v7_pending = True; self._sub = 'landing'  # landed"),
             ("return True                     # Up needs", "self._kind = self._sub; return True  # Up needs"),
             ("return off < 0 if up else off >= v", "self._kind = 'normal'; return off < 0 if up else off >= v")]:
    assert s.count(o) == 1, o
    s = s.replace(o, n)
g = dict(vars(sse)); exec(textwrap.dedent(s), g)
sse.SpecialStageEnv._at_decision = g['_at_decision']
env = sse.SpecialStageEnv()
env._kind = env._sub = None
import collections as _c
ring = _c.deque(maxlen=40)
_of = env._frame
def _fr(b):
    r = _of(b); i = r[4]; x, y = int(i.get('ss_x', 0)), int(i.get('ss_y', 0))
    ring.append((env.steps, x, y, env._cell(x, y), int(env._ram()[sse.ADDR_HEADING]), int(i.get('ss_next_corner', 0)), int(env._ram()[sse.ADDR_JUMPING]), int(i.get('ss_move_speed', 0)), env._decided, 'B' if b[0] else ''.join('LRU'[k] for k, v in enumerate((b[6], b[7], b[4])) if v)))
    return r
env._frame = _fr
traces = []

def turn_here_bit(env):
    """candidate obs bit: moving forwards, grounded, and this cell's turn
    window still ahead along the actual motion (the v4 decision rule)"""
    if env._last_pose is None or env._prev_xy is None:
        return None
    r = ring[-1] if ring else None
    if r is None or len(ring) < 2:
        return None
    (_, x0, y0, *_), (_, x, y, cell, _, _, jmp, speed, _, _) = ring[-2], ring[-1]
    if jmp:
        return 0
    mx = (x - x0 + 4096) % 8192 - 4096
    my = (y - y0 + 4096) % 8192 - 4096
    if mx == my == 0:
        return 0
    ox = (x - (cell[1] << 8) + 4096) % 8192 - 4096
    oy = (y - (cell[0] << 8) + 4096) % 8192 - 4096
    off, up, v = ((ox, mx > 0, abs(mx)) if abs(mx) >= abs(my) else (oy, my > 0, abs(my)))
    return int(off < 0 if up else off >= v)

TURNS = {i for i, n in enumerate(sse.CORNER_ACTIONS) if n.split('+')[0] in ('left', 'right')}
out = collections.Counter(); kinds = collections.Counter()
random.seed(2); ex = []; xt = collections.Counter()
for stage in range(1, 15):
    for ep in range(25):
        env.reset(options={'stage_file': sse.stage_file(stage)})
        env._kind = 'start'; prev = ('reset', 0, False)
        while True:
            kind = env._kind
            kinds[kind] += 1
            a = random.randrange(len(sse.CORNER_ACTIONS))
            b0 = env.bumper_hit; c0 = env._decided; f0 = env.steps; nc0 = int(env._ram()[0xE42A])
            hb = turn_here_bit(env); onb = int(env._original_grids[env._cur_stage][env._decided] == 3) if env._decided else 0
            q0 = env._quadrant(int(env._ram()[sse.ADDR_HEADING]))
            _, _, done, trunc, info = env.step(a)
            if a in TURNS and not (done or trunc):
                turned = env._quadrant(int(env._ram()[sse.ADDR_HEADING])) != q0
                queued = int(info.get('ss_next_corner', 0)) != 0
                if queued and not turned and len(traces) < 2: traces.append(list(ring))
                if queued and not turned and len(ex) < 14: ex.append((stage, 'prev step:', prev, '| now', sse.CORNER_ACTIONS[a], c0, env._decided, env.steps - f0, info.get('frames'), nc0, int(info.get('ss_next_corner', 0)), done))
                xt[(kind, 'bit=%s' % hb, 'bumper' if onb else '', 'turned' if turned else 'queued' if queued else 'none')] += 1
                out[(kind, 'turned' if turned else ('QUEUED+bumper' if env.bumper_hit != b0 else 'QUEUED no bumper') if queued else 'no turn')] += 1
            prev = (sse.CORNER_ACTIONS[a], info.get('frames'), int(env._ram()[sse.ADDR_JUMPING]))
            if done or trunc:
                break
print('decision points by kind:', dict(kinds))
for kind in sorted({k for k, _ in out}):
    n = sum(v for (k, _), v in out.items() if k == kind)
    print(f'{kind:13s} turns chosen {n:4d}: ' + ', '.join(f'{r} {v} ({100*v/n:.0f}%)' for (k, r), v in sorted(out.items()) if k == kind))

print('examples (stage, kind, action, cell, next decision cell, frames, step frames, queued before, queued after):')
for e in ex: print('  ', e)

for t in traces:
    print('--- trace (steps, x, y, cell, heading, next_corner, jumping, speed, decided, buttons)')
    for r in t[-38:]: print('  ', r)

print('--- candidate bit vs outcome (kind, bit, on a bumper cell, outcome): count')
for k, v in sorted(xt.items(), key=str): print('  ', k, v)
