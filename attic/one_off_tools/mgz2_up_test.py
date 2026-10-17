# mgz2_up_test.py (2026-10-17; Tyler: "I held up during my playthrough because I thought it helped me gain air.
# However, I don't think it matters"): his MGZ2 boss fight from start point 270, replayed exactly vs with every
# 'up' press replaced by noop from frame FROM on (his other presses and timing kept). Same emulator, deterministic:
# do the hits still land? Prints boss hits, outcome and Sonic's y at each of his jump presses. CPU.
import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
import coordinator as C, demos
K = 270
env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
env.demo_info()
seg = next(s for s in env._segs if s['name'] == 'MGZ2')
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
table = meta['action_table']
orig = C.emu_frame
UP, NOOP, JUMPS = C.ACTION_NAMES.index('up'), 0, {5, 6, 7, 8}
f0 = seg['ckpts'][K]['frame']


def play(label, swap_from=None, swap=None):
    at = {'f': f0}

    def inp(e, _m):
        at['f'] += 1
        a = int(actions[at['f']])
        if swap_from is not None and at['f'] - f0 >= swap_from and a in swap:
            a = swap[a]
        return orig(e, table[a])
    C.emu_frame = inp
    env.reset(options={'key': seg['key'], 'ckpt': K})
    hits, prev, ys, done, info = [], 0, [], False, None
    while not done and env.frames < 7200:
        _, _, term, trunc, info = env.step(0)
        done = term or trunc
        if env._boss_hits > prev:
            hits += [env.frames] * (env._boss_hits - prev)
            prev = env._boss_hits
        rel = at['f'] - f0
        if int(actions[at['f']]) in JUMPS and int(actions[at['f'] - 4]) not in JUMPS and rel > 640:
            ys.append((rel, int(info['y'])))
    C.emu_frame = orig
    res = info['mg_result']['outcome'] if done else 'running'
    print(f'{label:38s} {res:8s} at {env.frames}; boss hits {hits}; y at his flight jump presses {ys}', flush=True)


play('his inputs, exactly')
play("'up' -> noop for the whole flight (640+)", 640, {UP: NOOP})
play("'up' -> noop from 2,700 (last 2 hits)", 2700, {UP: NOOP})
play("noop -> 'up' for the whole flight", 640, {NOOP: UP})
