"""What does d2 DO in the 600 frames it spends not reaching a visible,
reachable straggler? Follow-up to probe_visibility.py (which showed it sees
them). Plays the current d2 model from the 79% start point and replays Tyler's
whole demo as the yardstick, recording raw data only — every frame's position,
heading, speed, queued corner and held action; every decision's action
probabilities; and the board each time it changes. steering_analyze.py turns
that into route decisions at intersections.

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
import torch as th
th.set_num_threads(1)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import special_stage_env as sse
import demos as demo_io

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PTR = 77                                     # d2's current start point (79%)

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
policy = model.policy
policy.set_training_mode(False)


class Recorder:
    """Per-frame pose + the board whenever it changes."""
    def __init__(self):
        self.frames, self.boards, self._last = [], [], None

    def frame(self, action, info):
        ram = env._ram()
        grid = env._read_grid()
        key = grid.tobytes()
        if key != self._last:
            self._last = key
            self.boards.append(dict(i=len(self.frames), grid=grid.tolist()))
        self.frames.append(dict(
            f=env.steps, x=int(info.get('ss_x', 0)), y=int(info.get('ss_y', 0)),
            h=int(ram[sse.ADDR_HEADING]), spd=int(info.get('ss_move_speed', 0)),
            corner=int(info.get('ss_next_corner', 0)),
            jump=int(ram[sse.ADDR_JUMPING]), a=int(action),
            got=env._spheres_collected,
            rem=int(info.get('blue_spheres_remaining', -1))))

    def dump(self):
        return dict(frames=self.frames, boards=self.boards)


# ── 1. Yardstick: Tyler's whole run, frame by frame ─────────────────────────
actions, meta = demo_io.load_demo(DEMO)
obs, _ = env.reset(options={'stage_file': sse.stage_file(meta['stage'])})
rec = Recorder()
for a in actions:
    obs, _, done, trunc, info = env._step_frames(int(a), 1)
    rec.frame(a, info)
    if done or trunc:
        break
demo = rec.dump()
demo['start_frame'] = int(0.79 * len(actions))
json.dump(demo, open(os.path.join(OUT, 'steer_demo.json'), 'w'))
print(f'demo: {len(rec.frames)} frames, {env._spheres_collected} spheres', flush=True)

# ── 2. The agent from the 79% start point ───────────────────────────────────
if env._demos is None:
    env._build_demos()
did = env._demos[0]['id']
episodes = []
t0 = time.time()
for ep in range(N_EPISODES):
    env.set_demo_pointers({did: PTR})
    obs, info = env.reset()
    target = env._demo_episode['target']; k = env._demo_episode['start']
    rec = Recorder()
    state = None
    start = np.array([True])
    decisions = []
    done = trunc = False
    while not (done or trunc):
        o = vn.normalize_obs({key: v[None] for key, v in obs.items()})
        obs_t, _ = policy.obs_to_tensor(o)
        if state is None:
            state = (np.zeros(policy.lstm_hidden_state_shape),) * 2
        with th.no_grad():
            st = (th.tensor(state[0], dtype=th.float32), th.tensor(state[1], dtype=th.float32))
            dist, new_st = policy.get_distribution(obs_t, st, th.tensor(start, dtype=th.float32))
            probs = dist.distribution.probs[0].numpy()
            a = int(dist.sample()[0])
        # exactly predict(): get_distribution, sample, carry the actor LSTM state
        state = (new_st[0].numpy(), new_st[1].numpy())
        start = np.array([False])
        decisions.append(dict(i=len(rec.frames), a=a, p=[round(float(x), 4) for x in probs]))
        for _ in range(sse.FRAME_SKIP):          # == env.step(a), frame by frame
            obs, _, done, trunc, info = env._step_frames(a, 1)
            rec.frame(a, info)
            if done or trunc:
                break
    ep_rec = rec.dump()
    ep_rec.update(start=k, target=target, got=env._spheres_collected,
                  cleared=bool(env._board_cleared), stalled=bool(env._stalled),
                  timeout=bool(trunc), last_pickup=env._last_pickup,
                  decisions=decisions)
    episodes.append(ep_rec)
    if (ep + 1) % 10 == 0:
        print(f'{ep + 1} episodes, {time.time() - t0:.0f}s', flush=True)
json.dump(episodes, open(os.path.join(OUT, 'steer_episodes.json'), 'w'))
print('done', flush=True)
