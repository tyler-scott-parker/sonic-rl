"""Board 14 (route board14_2921), counterfactual: d10's latest save from its stuck start-point band, paired
tries with and without the junction turn forced: at (13,3) on the way back up from (14,3), 'right' (the
route's turn east) whatever the policy picks. Then the policy plays on. Where does each try end, and does it
reach and clear the jump at (13,8) onto the spring at (13,10)? CPU only."""
import os, sys, json, shutil, random, collections
REPO = '/home/tyler/Projects/sonic-rl'; sys.path.insert(0, REPO); os.chdir(REPO)
import tempfile
T = tempfile.mkdtemp(prefix='b14_fix_')   # copies of the live save, so a running trainer's autosave can't race the load
DID = 'board14_2921_96f1dc25'
_src = open('train_ss_multistage.py').read(); _ns = {}
exec(_src[_src.index('ALL_14 ='):_src.index('# ── Shared config')], _ns)
obs_cfg = {k: v for k, v in _ns['RUNS']['d10']['obs'].items() if k != 'SS_STAGE_WEIGHTS'}
os.environ.update(obs_cfg, SS_STAGES='14', SS_RESTART_PROB='0', SS_DEMO_MIX='0', SS_LOG=os.devnull,
                  SS_DEMOS=f'routes/curriculum/{DID}.npz', SS_DEMO_PROB='1.0', CUDA_VISIBLE_DEVICES='')
import numpy as np, torch as th
th.set_num_threads(2)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import special_stage_env as sse
sse.ROUTE_DIR = ''
shutil.copy('sonic_ss_multistage_d10_vecnormalize.pkl', f'{T}/b14.pkl')
shutil.copy('sonic_ss_multistage_d10.zip', f'{T}/b14.zip')
env = sse.SpecialStageEnv(); env._log_episode = lambda *a, **k: None
vn = VecNormalize.load(f'{T}/b14.pkl', DummyVecEnv([lambda: env])); vn.training = False
model = RecurrentPPO.load(f'{T}/b14.zip', device='cpu'); policy = model.policy; policy.set_training_mode(False)
env.demo_info()
PTR = json.load(open('sonic_ss_multistage_d10_demos.json'))[DID]['ptr']
RIGHT = sse.CORNER_ACTIONS.index('right')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def cell():
    r = env._ram(); w = lambda a: int(r[a]) << 8 | int(r[a + 1])
    return env._cell(w(sse.ADDR_X), w(sse.ADDR_X + 2))


def one(i, force):
    random.seed(i); np.random.seed(i); th.manual_seed(i)
    env.set_demo_pointers({DID: PTR})
    obs, _ = env.reset(); k = env._demo_episode['start']; tgt = env._demo_episode['target']; g0 = env._spheres_collected
    state, start, done, trunc = None, np.array([True]), False, False
    seen14, forced, path, picked = False, False, [], None
    while not (done or trunc):
        o = vn.normalize_obs({key: v[None] for key, v in obs.items()})
        obs_t, _ = policy.obs_to_tensor(o)
        if state is None:
            state = (np.zeros(policy.lstm_hidden_state_shape),) * 2
        with th.no_grad():
            st_ = (th.tensor(state[0], dtype=th.float32), th.tensor(state[1], dtype=th.float32))
            dist, new_st = policy.get_distribution(obs_t, st_, th.tensor(start, dtype=th.float32))
            a = int(dist.sample()[0])
        state, start = (new_st[0].numpy(), new_st[1].numpy()), np.array([False])
        c = cell()
        seen14 |= c == (14, 3)
        if seen14 and c == (13, 3) and picked is None:
            picked = sse.CORNER_ACTIONS[a]
            if force and a != RIGHT:
                a, forced = RIGHT, True
        path.append((c, sse.CORNER_ACTIONS[a]))
        obs, _, done, trunc, info = env.step(a)
    cells = [p[0] for p in path] + [cell()]
    east = max((c[1] for c in cells if c[0] == 13 and 3 <= c[1] <= 16), default=0)
    return dict(start=k, got=env._spheres_collected - g0, end=cells[-1], picked=picked, forced=forced,
                east=east, spring=any(c[0] == 13 and c[1] >= 10 for c in cells), success=env._board_cleared or env._spheres_collected - g0 >= tgt,
                after=[f"{c[0]},{c[1]}:{m}" for c, m in path if seen14][:14])


print(f'model {model.num_timesteps:,} steps; start band {PTR}..{PTR + sse.DEMO_BAND}', flush=True)
res = {False: [], True: []}
for i in range(N):
    for force in (False, True):
        res[force].append(one(i, force))
    a, b = res[False][-1], res[True][-1]
    print(f"try {i:2d} from {a['start']}: as is: picked {a['picked']}, {a['got']} spheres, ends {a['end']}, past col {a['east']} "
          f"| forced right: {b['got']} spheres, ends {b['end']}, past col {b['east']}{', ON THE SPRING' if b['spring'] else ''}", flush=True)
    if i < 4 and b['picked'] is not None:
        print('      forced, after (14,3): ' + ' '.join(b['after']))
for force in (False, True):
    rs = res[force]
    reached = [r for r in rs if r['picked'] is not None]
    print(f"{'forced right' if force else 'as is       '}: reached the junction on the way back {len(reached)}/{N}; picks there "
          f"{dict(collections.Counter(r['picked'] for r in reached))}; on the spring (13,10+) {sum(r['spring'] for r in rs)}; "
          f"cleared {sum(r['success'] for r in rs)}; median spheres {np.median([r['got'] for r in rs]):.0f}; "
          f"ends {collections.Counter(r['end'] for r in rs).most_common(5)}")
