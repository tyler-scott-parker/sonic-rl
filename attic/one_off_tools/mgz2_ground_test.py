# mgz2_ground_test.py (2026-10-18; Tyler: "I don't think that MGZ2 ground phase really exists. It's more like a cutscene
# ... I'm not sure if Eggman can take damage during that time, even though it shows the damage animation"): his fight
# from start point 270 replayed exactly vs with his inputs during the ground part (frames 0..640) replaced (no input /
# hold left = no attacks), then his exact inputs from the flight on. If the ground hits count, skipping them leaves the
# boss alive after his flight hits; if they're cosmetic it still dies at ~2,940. Prints hits, hits-left and outcome. CPU.
import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
import numpy as np
import coordinator as C, demos
K, GROUND = 270, 640
env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
env.demo_info()
seg = next(s for s in env._segs if s['name'] == 'MGZ2')
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
table = meta['action_table']
orig = C.emu_frame
f0 = seg['ckpts'][K]['frame']


def boss_slot(env):
    bt = env._bt
    if bt is None or bt.prev is None:
        return None
    cand = np.flatnonzero((bt.maxseen >= C.BOSS_MIN) & (bt.maxseen <= C.BOSS_MAX) & bt.prev[:, :4].any(1))
    if not len(cand):
        return None
    row = bt.prev[cand[0]]
    return int(row[0x29])


def play(label, ground_action=None):
    at = {'f': f0}

    def inp(e, _m):
        at['f'] += 1
        a = int(actions[at['f']])
        if ground_action is not None and at['f'] - f0 <= GROUND:
            a = ground_action
        return orig(e, table[a])
    C.emu_frame = inp
    env.reset(options={'key': seg['key'], 'ckpt': K})
    hits, prev, left, done, info = [], 0, [], False, None
    while not done and env.frames < 4200:
        _, _, term, trunc, info = env.step(0)
        done = term or trunc
        if env._boss_hits > prev:
            hits += [env.frames] * (env._boss_hits - prev)
            prev = env._boss_hits
        if env.frames % 160 < 4:
            left.append(f'{env.frames}:{boss_slot(env)}')
    C.emu_frame = orig
    res = info['mg_result']['outcome'] if done else 'alive at 4200'
    print(f"{label:40s} {res:12s} hits at {hits}; hurt at {[h[0] for h in env._hurt]}\n    hits left over time: {' '.join(left)}", flush=True)


play('his inputs, exactly')
play('ground part: no input, then his inputs', ground_action=0)
play('ground part: hold left, then his inputs', ground_action=C.ACTION_NAMES.index('left'))
