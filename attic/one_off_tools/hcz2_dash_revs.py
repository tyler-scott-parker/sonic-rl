# hcz2_dash_revs.py (2026-10-17 corridor dig): from his standing state at HCZ2 start point 6 (t 360, before his own spin
# dash), a scripted spin dash with R revs (crouch, R fresh jump+down presses, release) then hold right, in the agent's
# 4-frame actions: speed at the curve's foot and up the curve or not. Also: hold right from a stand, and roll from a run.
import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import coordinator as C
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
N = C.ACTION_NAMES.index
st = lambda v: ('roll' if v & 4 else 'run') + ('/air' if v & 2 else '')


def run(plan, label, k=6):
    env.reset(options={'key': seg['key'], 'ckpt': k})
    prev, foot, xmax, start_speed, info = None, None, 0, None, None
    for i, a in enumerate(plan + [N('right')] * 200):
        _, _, term, trunc, info = env.step(a)
        x = int(info['x'])
        if prev is not None and i == len(plan) + 2:
            start_speed = (x - prev) / 4
        if foot is None and prev is not None and 1440 <= x <= 1500:
            foot = ((x - prev) / 4, st(int(info['mg_status'])))
        prev, xmax = x, max(xmax, x)
        if term or trunc or x >= 1700:
            break
    print(f'{label:34s} speed after release {start_speed if start_speed is None else f"{start_speed:+.1f}"}, at the foot '
          f'{f"{foot[0]:+.1f} {foot[1]}" if foot else "never reached"} -> {"UP the curve" if xmax >= 1650 else f"not up (x max {xmax})"}', flush=True)


crouch = [N('down')] * 2
for r in range(0, 5):
    run(crouch + [N('jump+down'), N('down')] * r + ([N('jump+down')] if r == 0 else []), f'spin dash, {max(r, 1) if r else 1} press{"es" if r > 1 else ""}' if r else 'spin dash, 1 press (no extra rev)')
run([], 'hold right from a stand')
run([N('right')] * 30 + [N('down')], 'run 30 steps then roll (down)')
