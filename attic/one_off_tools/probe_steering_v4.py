"""probe_steering.py for env v4 (run d3): does the agent go straight on the
ground where only straight is right? (d2 on v3: 6-9%.) d3's gate check.

Same raw recording as probe_steering.py — every frame's position, heading,
speed, queued corner and held buttons; every decision's action probabilities;
the board each time it changes — so steering_analyze_v4.py can score the same
intersections the same way. Two v4 differences: a decision is one corner (the
env plays on alone between them, so frames are recorded inside the env's own
frame loop), and the probabilities are over straight / left / right / jump.

Read-only for the training run: own copies of the model, own episode log,
no curriculum-pool captures (restarts off), CPU only.

usage: [PROBE_RUN=d5] probe_steering_v4.py [episodes] [start fraction, e.g. 0.79]
       (default run d3; default start: the run's current curriculum pointer)
"""
import os, sys, json, shutil, time
REPO = '/home/tyler/Projects/sonic-rl'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
RUN = os.environ.get('PROBE_RUN', 'd3')   # any stage-1 run on env v4
DEMO = os.path.join(REPO, 'demos', 'stage1_20260930-165817.npz')
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else None
TAG = (f'{RUN}_' if RUN != 'd3' else '') + (f'v4_{round(100 * FRAC)}' if FRAC is not None else 'v4_ptr')
os.environ.update(SS_ENV_VERSION='4', SS_USE_PIXELS='0', SS_PATCH_R='7',
                  SS_DROP_POSITION='1', SS_STAGES='1', SS_RESTART_PROB='0',
                  SS_LOG=os.path.join(OUT, f'probe_ss_log_{TAG}.jsonl'),
                  SS_DEMOS=DEMO, SS_DEMO_PROB='1.0', CUDA_VISIBLE_DEVICES='')

import numpy as np
import torch as th
th.set_num_threads(1)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import special_stage_env as sse
import demos as demo_io

assert sse.V4 and not sse.V5 and sse.CORNER_ACTIONS == ('straight', 'left', 'right', 'jump')

# a consistent model + normalizer pair (autosave rewrites both every 100K steps)
for _ in range(10):
    zp, pp = (os.path.join(REPO, f'sonic_ss_multistage_{RUN}{s}') for s in ('.zip', '_vecnormalize.pkl'))
    m0 = (os.path.getmtime(zp), os.path.getmtime(pp))
    shutil.copy(zp, os.path.join(OUT, f'{RUN}.zip')); shutil.copy(pp, os.path.join(OUT, f'{RUN}.pkl'))
    if m0 == (os.path.getmtime(zp), os.path.getmtime(pp)) and abs(m0[0] - m0[1]) < 60:
        break
    time.sleep(20)
print(f'model saved at {time.strftime("%H:%M:%S", time.localtime(m0[0]))}', flush=True)

env = sse.SpecialStageEnv()
vn = VecNormalize.load(os.path.join(OUT, f'{RUN}.pkl'), DummyVecEnv([lambda: env]))
vn.training = False
model = RecurrentPPO.load(os.path.join(OUT, f'{RUN}.zip'), device='cpu')
policy = model.policy
policy.set_training_mode(False)
print(f'model at {model.num_timesteps:,} steps', flush=True)


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


# record every frame the env plays, including the ones inside a v4 corner step;
# 'a' is the ACTION_BUTTONS index held that frame (0 none, 1 B, 2 left, 3 right)
rec = None
_frame = env._frame


def recorded_frame(buttons):
    out = _frame(buttons)
    if rec is not None:
        rec.frame(next(i for i, b in enumerate(sse.ACTION_BUTTONS) if b is buttons), out[4])
    return out


env._frame = recorded_frame

# ── 1. Yardstick: Tyler's whole run, frame by frame ─────────────────────────
actions, meta = demo_io.load_demo(DEMO)
if env._demos is None:
    env._build_demos()
dm = env._demos[0]
did = dm['id']
if FRAC is None:
    PTR = json.load(open(os.path.join(REPO, f'sonic_ss_multistage_{RUN}_demos.json')))[did]['ptr']
else:   # the last start point at or before FRAC of the run
    PTR = max(k for k, ck in enumerate(dm['ckpts']) if ck['fields']['steps'] <= FRAC * dm['frames'])
start_frame = dm['ckpts'][PTR]['fields']['steps']
print(f'start point {PTR} of {len(dm["ckpts"]) - 1}: frame {start_frame} = '
      f'{start_frame / dm["frames"]:.0%} of the run (episodes start in [{PTR}, {PTR + sse.DEMO_BAND}])', flush=True)

obs, _ = env.reset(options={'stage_file': sse.stage_file(meta['stage']), 'replay': True})
rec = Recorder()
for a in actions:
    obs, _, done, trunc, info = env._step_frames(int(a), 1)
    if done or trunc:
        break
demo = rec.dump()
demo['start_frame'] = start_frame
json.dump(demo, open(os.path.join(OUT, f'steer_demo_{TAG}.json'), 'w'))
print(f'demo: {len(rec.frames)} frames, {env._spheres_collected} spheres', flush=True)

# ── 2. The agent from the start point ───────────────────────────────────────
episodes = []
t0 = time.time()
for ep in range(N_EPISODES):
    env.set_demo_pointers({did: PTR})
    rec = None                       # the reset's own frames aren't the agent's
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
        # i = index of the first frame this choice plays; the decision frame
        # itself is i - 1 (or the reset frame, for the first choice)
        decisions.append(dict(i=len(rec.frames), a=a, p=[round(float(x), 4) for x in probs],
                              cell=list(env._decided) if env._decided else None,
                              xy=list(env._prev_xy) if env._prev_xy else None))
        obs, _, done, trunc, info = env.step(a)
    ep_rec = rec.dump()
    ep_rec.update(start=k, target=target, got=env._spheres_collected,
                  cleared=bool(env._board_cleared), stalled=bool(env._stalled),
                  timeout=bool(trunc), last_pickup=env._last_pickup,
                  decisions=decisions)
    episodes.append(ep_rec)
    if (ep + 1) % 10 == 0:
        print(f'{ep + 1} episodes, {time.time() - t0:.0f}s', flush=True)
json.dump(episodes, open(os.path.join(OUT, f'steer_episodes_{TAG}.json'), 'w'))
print('done', flush=True)
