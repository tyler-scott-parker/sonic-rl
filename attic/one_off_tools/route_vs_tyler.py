"""Where does a stage-1 agent lose time against Tyler's recording? CPU only.

  CUDA_VISIBLE_DEVICES= taskset -c 6,7 nice -n 19 python3 attic/one_off_tools/route_vs_tyler.py [run] [episodes]

Runs the run's saved policy (default d7) from fresh starts of stage 1,
stochastic like training (plus a few deterministic), and replays Tyler's
recording through the same env. Per run: frames, frames to each quarter of
the spheres, cells travelled, jumps, bumper hits, multi-sphere frames (the
outline trick: enclosed blues turn to rings at once) and spheres they took.
"""
import os, sys, json, statistics as st
HOME = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, HOME)
RUN = sys.argv[1] if len(sys.argv) > 1 else 'd7'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read(); ns = {}
exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
cfg = ns['RUNS'][RUN]
os.environ.update(cfg['obs'], SS_STAGES='1', SS_RESTART_PROB='0', SS_DEMOS='', SS_DEMO_PROB='0',
                  CUDA_VISIBLE_DEVICES='', SS_LOG=f'/tmp/claude-1000/route_{os.getpid()}.jsonl')
import numpy as np                   # noqa: E402
import torch                         # noqa: E402
torch.set_num_threads(1)
import special_stage_env as sse      # noqa: E402
import demos as demo_io              # noqa: E402
from sb3_contrib import RecurrentPPO  # noqa: E402

env = sse.SpecialStageEnv()
trace = []
_frame = env._frame


def frame(buttons):
    r = _frame(buttons)
    info = r[4]
    x, y = int(info.get('ss_x', 0)), int(info.get('ss_y', 0))
    trace.append((info.get('blue_spheres_remaining'), env._cell(x, y),
                  int(env._ram()[sse.ADDR_JUMPING]) & 0x81 == 0x80, int(info.get('bumper_contact', 0)) > 0,
                  x, y, env._quadrant(int(env._ram()[sse.ADDR_HEADING]))))
    return r


env._frame = frame


def summarise(tr, total):
    blues = [t[0] for t in tr]
    got = [total - b if b is not None else 0 for b in blues]
    quarters = []
    for q in (0.25, 0.5, 0.75, 1.0):
        k = next((i for i, g in enumerate(got) if g >= round(q * total)), None)
        quarters.append(k + 1 if k is not None else None)
    cells = sum(1 for a, b in zip(tr, tr[1:]) if a[1] != b[1])
    jumps = sum(1 for a, b in zip(tr, tr[1:]) if b[2] and not a[2])
    bumps = sum(1 for a, b in zip(tr, tr[1:]) if b[3] and not a[3])
    turns = sum(1 for a, b in zip(tr, tr[1:]) if a[6] != b[6])
    still = sum(1 for a, b in zip(tr, tr[1:]) if (a[4], a[5]) == (b[4], b[5]))
    air = sum(1 for t in tr if t[2])
    drops = [p - b for p, b in zip(blues, blues[1:]) if p is not None and b is not None and p - b > 1]
    return dict(frames=len(tr), quarters=quarters, cells=cells, jumps=jumps, bumps=bumps,
                turns=turns, still=still, air=air,
                multi=len(drops), converted=sum(d - 1 for d in drops), final=got[-1])


# Tyler's recording, frame by frame through the same env
actions, meta = demo_io.load_demo(os.path.join(HOME, 'demos', 'stage1_20260930-165817.npz'))
env.reset(options={'stage_file': sse.stage_file(1), 'replay': True})
env._replaying = True
trace.clear()
for a in actions:
    _, _, done, trunc, _ = env._step_frames(int(a), 1)
    if done or trunc:
        break
env._replaying = False
total = meta['spheres']
his = summarise(trace, total)
print(f'Tyler: {his}', flush=True)

model = RecurrentPPO.load(os.path.join(HOME, f'sonic_ss_multistage_{RUN}.zip'), device='cpu')
rows = []
for ep in range(N + 5):
    det = ep >= N                    # the last 5 deterministic
    obs, _ = env.reset(options={'stage_file': sse.stage_file(1)})
    trace.clear()
    lstm, start = None, np.array([True])
    while True:
        a, lstm = model.predict({k: v[None] for k, v in obs.items()}, state=lstm,
                                episode_start=start, deterministic=det)
        start = np.array([False])
        obs, _, done, trunc, _ = env.step(int(a[0]))
        if done or trunc:
            break
    r = summarise(trace, total)
    r.update(clear=bool(env._board_cleared), det=det)
    rows.append(r)
    print(f'  ep {ep:3d} {"det " if det else ""}clear={r["clear"]} frames={r["frames"]} spheres={r["final"]} '
          f'cells={r["cells"]} turns={r["turns"]} still={r["still"]} jumps={r["jumps"]} air={r["air"]} '
          f'bumps={r["bumps"]} converted={r["converted"]}', flush=True)

clears = [r for r in rows if r['clear'] and not r['det']]
print(f'\n{RUN}: {len(clears)}/{N} stochastic fresh starts cleared')


def med(key, rs, i=None):
    v = [r[key][i] if i is not None else r[key] for r in rs]
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


if clears:
    print(f'{"":24s} {"Tyler":>8s} {RUN + " median":>12s}')
    print(f'{"frames to clear":24s} {his["frames"]:8d} {med("frames", clears):12.0f}')
    for i, q in enumerate(('25%', '50%', '75%', '100%')):
        print(f'{"frames to " + q + " spheres":24s} {his["quarters"][i]:8d} {med("quarters", clears, i):12.0f}')
    for k, label in (('cells', 'cells travelled'), ('turns', 'turns'), ('still', 'frames standing still'),
                     ('jumps', 'jumps'), ('air', 'frames in the air'), ('bumps', 'bumper hits'),
                     ('multi', 'multi-sphere frames'), ('converted', 'spheres by outline')):
        print(f'{label:24s} {his[k]:8d} {med(k, clears):12.1f}')
json.dump(dict(tyler=his, rows=rows), open(os.path.join(HOME, 'attic', 'one_off_tools', f'route_vs_tyler_{RUN}.json'), 'w'))
