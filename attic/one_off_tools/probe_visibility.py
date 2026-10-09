"""Can d2 see the last spheres? Plays the current d2 model from the 79% start
point (and replays Tyler's demo as the yardstick) and records, every step,
whether any blue sphere is inside the 15x15 patch the agent observes.

Read-only for the training run: own copies of the model, own episode log,
no curriculum-pool captures, CPU only.
"""
import os, sys, json, shutil, time
REPO = '/home/tyler/Projects/sonic-rl'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
DEMO = os.path.join(REPO, 'demos', 'stage1_20260930-165817.npz')
os.environ.update(SS_ENV_VERSION='3', SS_USE_PIXELS='0', SS_PATCH_R='7',
                  SS_DROP_POSITION='1', SS_STAGES='1', SS_RESTART_PROB='0',
                  SS_LOG=os.path.join(OUT, 'probe_ss_log.jsonl'),
                  SS_DEMOS=DEMO, SS_DEMO_PROB='1.0', CUDA_VISIBLE_DEVICES='')

import numpy as np
import torch
torch.set_num_threads(1)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import special_stage_env as sse
import demos as demo_io

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 120
PTR = 77                                     # d2's current start point (79%)
R = sse.PATCH_R

# a consistent model + normalizer pair (autosave rewrites both every 100K steps)
for _ in range(10):
    zp, pp = (os.path.join(REPO, f'sonic_ss_multistage_d2{s}') for s in ('.zip', '_vecnormalize.pkl'))
    m0 = (os.path.getmtime(zp), os.path.getmtime(pp))
    shutil.copy(zp, os.path.join(OUT, 'd2.zip')); shutil.copy(pp, os.path.join(OUT, 'd2.pkl'))
    if m0 == (os.path.getmtime(zp), os.path.getmtime(pp)) and abs(m0[0] - m0[1]) < 60:
        break
    time.sleep(20)
print(f'model saved at {time.strftime("%H:%M:%S", time.localtime(m0[0]))}', flush=True)

env = sse.SpecialStageEnv()
vn = VecNormalize.load(os.path.join(OUT, 'd2.pkl'), DummyVecEnv([lambda: env]))
vn.training = False
model = RecurrentPPO.load(os.path.join(OUT, 'd2.zip'), device='cpu')


def walk_dist(grid, r0, c0):
    """Cells to the nearest blue along grid lines without touching red (1, or 10 = turning red) or
    a bumper (3), on the wrapping 32x32 board; None if every blue is walled
    off (reachable only by jumping)."""
    from collections import deque
    seen = {(r0, c0)}; q = deque([(r0, c0, 0)])
    while q:
        r, c, d = q.popleft()
        if d and grid[r, c] == 2:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = (r + dr) % 32, (c + dc) % 32
            if (nr, nc) not in seen and grid[nr, nc] not in (1, 3, 10):
                seen.add((nr, nc)); q.append((nr, nc, d + 1))
    return None


def measure(e, obs, info):
    """(blue visible in the agent's patch per its OWN obs, same per RAM,
    toroidal Chebyshev distance to the nearest blue, blues on the board)."""
    grid = e._read_grid()
    blues = np.argwhere(grid == 2)
    r0, c0 = e._cell(int(info.get('ss_x', 0)), int(info.get('ss_y', 0)))
    if len(blues):
        dr = np.abs(blues[:, 0] - r0); dr = np.minimum(dr, 32 - dr)
        dc = np.abs(blues[:, 1] - c0); dc = np.minimum(dc, 32 - dc)
        nearest = int(np.max([dr, dc], axis=0).min())
    else:
        nearest = 99
    vis_obs = bool(obs['grid'][..., 0].sum() > 0)
    measure.walk = walk_dist(grid, r0, c0) if len(blues) else None
    measure.board = (grid, r0, c0)
    return vis_obs, nearest <= R, nearest, len(blues)


# ── 1. Yardstick: Tyler's own run, frame by frame, from 79% on ──────────────
actions, meta = demo_io.load_demo(DEMO)
obs, _ = env.reset(options={'stage_file': sse.stage_file(meta['stage'])})
start_frame = int(0.79 * len(actions))
demo_rows = []
for j, a in enumerate(actions):
    obs, _, done, trunc, info = env._step_frames(int(a), 1)
    if j >= start_frame and not (done or trunc):
        v, vr, near, nb = measure(env, obs, info)
        demo_rows.append(dict(f=j, vis=v, vis_ram=vr, near=near, nblue=nb,
                              got=env._spheres_collected, walk=measure.walk))
json.dump(demo_rows, open(os.path.join(OUT, 'probe_demo.json'), 'w'))
print(f'demo from frame {start_frame}: {len(demo_rows)} frames, '
      f'{env._spheres_collected} spheres total', flush=True)

# ── 2. The agent from the 79% start point ───────────────────────────────────
env._build_demos() if env._demos is None else None
did = env._demos[0]['id']
episodes = []
for ep in range(N_EPISODES):
    env.set_demo_pointers({did: PTR})
    obs, _ = env.reset()
    target = env._demo_episode['target']; k = env._demo_episode['start']
    state, start = None, np.array([True])
    rows = []
    while True:
        o = vn.normalize_obs({key: v[None] for key, v in obs.items()})
        a, state = model.predict(o, state=state, episode_start=start, deterministic=False)
        start = np.array([False])
        obs, _, done, trunc, info = env.step(int(a[0]))
        v, vr, near, nb = measure(env, obs, info)
        rows.append(dict(f=env.steps, vis=v, vis_ram=vr, near=near, nblue=nb,
                         got=env._spheres_collected, last=env._last_pickup,
                         rem=int(info.get('blue_spheres_remaining', -1)),
                         walk=measure.walk))
        if done or trunc:
            break
    episodes.append(dict(start=k, target=target, got=env._spheres_collected,
                         cleared=bool(env._board_cleared), stalled=bool(env._stalled),
                         timeout=bool(trunc), rows=rows,
                         board=measure.board[0].tolist(), pos=[int(x) for x in measure.board[1:]]))
    if (ep + 1) % 20 == 0:
        print(f'{ep + 1} episodes', flush=True)
json.dump(episodes, open(os.path.join(OUT, 'probe_episodes.json'), 'w'))
print('done', flush=True)
