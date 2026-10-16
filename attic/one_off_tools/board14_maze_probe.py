"""Board 14 at d10's stuck start points (old route board14_2921): does the straight-line compass point where
the maze lets Sonic go? CPU only."""
import os, sys, json, collections
REPO = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, REPO); os.chdir(REPO)
DID = 'board14_2921_96f1dc25'
_src = open('train_ss_multistage.py').read(); _ns = {}
exec(_src[_src.index('ALL_14 ='):_src.index('# ── Shared config')], _ns)
os.environ.update(_ns['RUNS']['d10']['obs'], SS_STAGES='14', SS_RESTART_PROB='0', SS_DEMO_MIX='0',
                  SS_LOG=os.devnull, SS_DEMOS=f'routes/curriculum/{DID}.npz', SS_DEMO_PROB='1.0', CUDA_VISIBLE_DEVICES='')
import numpy as np
import special_stage_env as sse
import demos as demo_io
env = sse.SpecialStageEnv()
env._log_episode = lambda *a, **k: None
env._build_demos() if env._demos is None else None
dm = env._demos[0]
actions, meta = demo_io.load_demo(f'routes/curriculum/{DID}.npz')
PTR = json.load(open('sonic_ss_multistage_d10_demos.json'))[DID]['ptr']
PASS = {0, 2, 4, 5}                       # empty, blue, ring, spring; red (1) and bumpers (3) block
SYM = {0: '.', 1: 'x', 2: 'o', 3: '*', 4: 'r', 5: 's'}


def bfs(grid, start):
    dist = {start: 0}
    q = collections.deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = ((r + dr) % 32, (c + dc) % 32)
            if n not in dist and grid[n] in PASS:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


seen = {}
for _ in range(200):
    env.set_demo_pointers({DID: PTR})
    obs, info = env.reset()
    k = env._demo_episode['start']
    if k in seen:
        continue
    grid = env._read_grid().copy()
    ram = env._ram()
    w = lambda a: int(ram[a]) << 8 | int(ram[a + 1])
    x_raw, y_raw = w(sse.ADDR_X), w(sse.ADDR_X + 2)
    cell = env._cell(x_raw, y_raw)
    if not seen:                                   # the y address: check against one step's info
        st = env.env.em.get_state() if hasattr(env, 'env') else None
        _, _, _, _, i1 = env._step_frames(0, 1)
        print(f'RAM x,y {x_raw},{y_raw} vs info after one frame {i1.get("ss_x")},{i1.get("ss_y")}', flush=True)
        if st is not None:
            env.env.em.set_state(st)
    heading = int(env._ram()[sse.ADDR_HEADING])
    blues = [tuple(b) for b in np.argwhere(grid == 2)]
    d2 = lambda b: (((b[0] - cell[0] + 16) % 32 - 16) ** 2 + ((b[1] - cell[1] + 16) % 32 - 16) ** 2)
    straight = min(blues, key=d2)
    dist = bfs(grid, cell)
    reach = [b for b in blues if b in dist]
    by_path = min(reach, key=lambda b: dist[b]) if reach else None
    # the blue the route collects next from here
    f0 = dm['ckpts'][k]['fields']['steps']
    g0, nxt = env._spheres_collected, None
    for a in actions[f0 + 1:f0 + 600]:
        _, _, done, trunc, inf2 = env._step_frames(int(a), 1)
        if env._spheres_collected > g0:
            nxt = env._cell(int(inf2.get('ss_x', 0)), int(inf2.get('ss_y', 0)))
            break
        if done or trunc:
            break
    seen[k] = dict(cell=cell, heading=heading, straight=straight, straight_d=round(d2(straight) ** 0.5, 1),
                   straight_path=dist.get(straight), by_path=by_path, by_path_len=dist.get(by_path) if by_path else None,
                   route_next=nxt, route_next_path=dist.get(nxt) if nxt else None, grid=grid)
    if len(seen) >= 6:
        break
for k in sorted(seen):
    s = seen[k]
    print(f"start point {k}: Sonic at {s['cell']} heading {s['heading']}; compass -> {s['straight']} "
          f"({s['straight_d']} cells straight, {s['straight_path']} by path); nearest by path -> {s['by_path']} "
          f"({s['by_path_len']} steps); your route's next blue -> {s['route_next']} ({s['route_next_path']} steps)")
k = min(seen)
s = seen[k]
r0, c0 = s['cell']
print(f"\nthe maze around start point {k} (S = Sonic, C = compass target, R = route's next blue; o blue, x red, * bumper, r ring, s spring):")
for dr in range(-8, 9):
    row = ''
    for dc in range(-12, 13):
        p = ((r0 + dr) % 32, (c0 + dc) % 32)
        ch = SYM.get(int(s['grid'][p]), '?')
        if p == s['cell']: ch = 'S'
        elif p == tuple(s['straight']): ch = 'C'
        elif s['route_next'] and p == tuple(s['route_next']): ch = 'R'
        row += ch + ' '
    print('   ' + row)
