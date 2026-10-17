# mgz2_carry_branch.py [FRAME] (2026-10-17 MGZ2 dig): replay Tyler from start point 270 to FRAME (default 1760, carried by Tails at y ~1822), then branch: noop / up / down / one jump / jump spam; prints Sonic's y. CPU.
import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ['MG_LOG'] = os.devnull
import coordinator as C, demos
K, AT = 270, int(sys.argv[1]) if len(sys.argv) > 1 else 1760
env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
env.demo_info()
seg = next(s for s in env._segs if s['name'] == 'MGZ2')
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
table = meta['action_table']
orig = C.emu_frame
N = C.ACTION_NAMES.index
def run(plan, label):
    at = {'f': seg['ckpts'][K]['frame']}
    def his_input(e, _m):
        at['f'] += 1
        return orig(e, table[actions[at['f']]])
    C.emu_frame = his_input
    env.reset(options={'key': seg['key'], 'ckpt': K})
    while env.frames < AT:
        _, _, term, trunc, info = env.step(0)
    C.emu_frame = orig
    ys, t = [], 0
    for a in plan:
        _, _, term, trunc, info = env.step(a)
        ys.append((env.frames, int(info['x']), int(info['y']), int(info['mg_routine'])))
        if term or trunc:
            break
    print(f'{label:24s}', ' '.join(f'{y}' for _, _, y, _ in ys[::8]), '| end', ys[-1])
run([0] * 80, 'noop x80 steps')
run([N('up')] * 80, 'up held')
run([N('down')] * 80, 'down held')
run([N('jump')] + [0] * 79, 'one jump, then noop')
run([N('jump')] + [N('up')] * 79, 'one jump, then up')
run(([N('jump')] + [0] * 3) * 20, 'jump every 4 steps')
