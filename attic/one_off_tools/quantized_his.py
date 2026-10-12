#!/usr/bin/env python3
"""quantized_his.py — can the agent's controls do what Tyler did? His per-frame inputs, rounded to the
agent's 9 actions held 4 frames each, replayed through the env from start points.

    python3 attic/one_off_tools/quantized_his.py RUN ACT K [K ...]
    python3 attic/one_off_tools/quantized_his.py mg1 AIZ1 70 72 75 76 80

Tyler 2026-10-12: "This game is riddled with precise jumps across all many of its stages." The agent
picks one of 9 button sets (none, left, right, down, up, jump, jump+left, jump+right, jump+down) every
4 frames; he pressed per frame. Each 4-frame window of his inputs becomes the agent action that agrees
with him on the most frames (on jump, left, right, down, up), with each of the 4 possible alignments of
the windows (the agent's decisions fall every 4th frame from the start point; his presses don't). If his
own play, rounded this way, can't clear from a start point while his exact inputs do, the agent's
controls can't express what he did there, and no training will find it that way.
CPU only.
"""
import os
import sys

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main                       # noqa: E402
RUN, ACT, KS = sys.argv[1], sys.argv[2], [int(k) for k in sys.argv[3:]]
train_main.configure_env(RUN)
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate')
import numpy as np                      # noqa: E402
import coordinator as C                 # noqa: E402
import demos                            # noqa: E402

BUTTONS = [0, 6, 7, 5, 4]               # B (jump), LEFT, RIGHT, DOWN, UP in the 12-button mask
acts = np.array([[a[b] for b in BUTTONS] for a in C.ACTION_BUTTONS])      # 9 x 5

cfg_act = next(a for a in train_main.RUNS[RUN]['acts'] if a.split(':')[-1] == ACT)
env = C.SonicCoordinator(acts=[cfg_act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
actions, meta = demos.load_play(next(p for p in env.demo_files if demos.demo_id(p) == seg['demo']))
table = np.array(meta['action_table'])
his = table[np.asarray(actions)][:, BUTTONS]                              # frames x 5


def rounded(f0, n, shift):
    """Agent actions for his frames f0+1 .. f0+n, windows of 4 starting `shift` frames early."""
    out = []
    for w in range(0, n, C.FRAME_SKIP):
        lo = max(f0 + 1, f0 + 1 + w - shift)
        win = his[lo:lo + C.FRAME_SKIP]
        out.append(int(np.argmax((acts[:, None, :] == win[None, :, :]).all(-1).sum(-1))))
    return out


def play(k, acts_seq=None):
    env.reset(options={'key': seg['key'], 'ckpt': k})
    done, i = False, 0
    if acts_seq is None:                # his exact inputs, frame for frame
        orig, at = C.emu_frame, {'f': seg['ckpts'][k]['frame']}

        def his_input(e, _m):
            at['f'] += 1
            return orig(e, table[actions[at['f']]])
        C.emu_frame = his_input
        try:
            while not done:
                _, _, term, trunc, info = env.step(0)
                done = term or trunc
        finally:
            C.emu_frame = orig
    else:
        while not done:
            a = acts_seq[i] if i < len(acts_seq) else 0
            _, _, term, trunc, info = env.step(a)
            i += 1
            done = term or trunc
    r = info['mg_result']
    return r['outcome'], r['frames'], r['target'], env._route.best


for k in KS:
    t, f0 = seg['ckpts'][k]['t'], seg['ckpts'][k]['frame']
    exact = play(k)
    row = [f"{ACT} start point {k} (t {t}): his inputs -> {exact[0]} in {exact[1]} (his {exact[2]})"]
    for shift in range(C.FRAME_SKIP):
        o, fr, tg, best = play(k, rounded(f0, exact[2] + 600, shift))
        row.append(f"rounded, offset {shift}: {o}" + (f" in {fr}" if o == 'clear' else f", got to t {best} of {seg['T']}"))
    print('\n   '.join(row), flush=True)
