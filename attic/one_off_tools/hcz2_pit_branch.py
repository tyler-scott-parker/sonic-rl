# hcz2_pit_branch.py [FRAME] (2026-10-17 HCZ2 opening dig; Tyler: "let's work on the opening"): replay Tyler's
# HCZ2 from its start to FRAME (default 780: in the basin past the curve, before his spin dash at the quarter
# pipe), then branch on scripted 4-frame actions (the agent's action set): can a spin dash get out, and what kills
# Sonic if he stays? Prints x/y/status per branch. CPU.
import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/sonic-rl'))
os.chdir(os.path.expanduser('~/Projects/sonic-rl'))
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK='')
import coordinator as C, demos
AT = int(sys.argv[1]) if len(sys.argv) > 1 else 780
act = next(a for a in train_main.RUNS['mg2']['acts'] if a.endswith(':HCZ2'))
env = C.SonicCoordinator(acts=[act], demo_prob=0.0)
env.demo_info()
seg = env._segs[0]
path = next(p for p in os.environ['MG_DEMOS'].split(',') if os.path.basename(p).startswith(seg['demo']))
actions, meta = demos.load_play(path)
table = meta['action_table']
orig = C.emu_frame
N = C.ACTION_NAMES.index
st = lambda v: ('roll' if v & 4 else 'run') + ('/air' if v & 2 else '')


def run(plan, label, show=12):
    at = {'f': seg['ckpts'][0]['frame']}

    def his_input(e, _m):
        at['f'] += 1
        return orig(e, table[actions[at['f']]])
    C.emu_frame = his_input
    env.reset(options={'key': seg['key'], 'ckpt': 0})
    info = None
    while env.frames < AT:
        _, _, term, trunc, info = env.step(0)
    C.emu_frame = orig
    trace, done = [], False
    for a in plan:
        _, _, term, trunc, info = env.step(a)
        trace.append((env.frames, int(info['x']), int(info['y']), st(int(info['mg_status'])), int(info['mg_routine']), int(info['mg_air'])))
        if term or trunc:
            done = True
            break
    res = info.get('mg_result', {}).get('outcome') if done else 'running'
    step = max(1, len(trace) // show)
    print(f'{label:34s} -> {res:8s} at frame {AT + trace[-1][0] - trace[0][0] + 4} ({trace[-1][1]},{trace[-1][2]}) air {trace[-1][5]}; '
          f'max x {max(t[1] for t in trace)}, min y {min(t[2] for t in trace)} | air '
          + ' '.join(str(t[5]) for t in trace[::step]), flush=True)
    return trace


if not os.environ.get('REACTIVE'):
  sd = lambda revs: [N('down')] * 2 + [N('jump+down'), N('down')] * revs   # crouch, then rev (fresh jump presses)
  run([0] * 200, 'noop x200 (wait in the basin)')
  run([N('right')] * 200, 'hold right')
  run([N('left')] * 3 + [N('right')] * 197, 'step left, then right')
  for revs in (1, 2, 3):
      run(sd(revs) + [N('right')] * 150, f'spin dash, {revs} rev(s), then right')
      run([N('left')] * 3 + sd(revs) + [N('right')] * 150, f'step left, spin dash {revs}, right')
  run([N('down')] * 3 + [N('right')] * 150, 'crouch then right (no rev)')
  run([N('jump+right')] * 200, 'jump+right spam')


def reactive(revs, label, steps=260):
    """Spin dash (crouch 2, `revs` revs, release) whenever grounded and slow in the basin (x < 1930) or on the
    ledge before the yellow pipe (x 1930-2030, y 1740-1800); otherwise hold right."""
    at = {'f': seg['ckpts'][0]['frame']}

    def his_input(e, _m):
        at['f'] += 1
        return orig(e, table[actions[at['f']]])
    C.emu_frame = his_input
    env.reset(options={'key': seg['key'], 'ckpt': 0})
    info = None
    while env.frames < AT:
        _, _, term, trunc, info = env.step(0)
    C.emu_frame = orig
    queue, trace, res, prev_x, cool = [], [], 'running', int(info['x']), 0
    for _ in range(steps):
        x, y, s = int(info['x']), int(info['y']), int(info['mg_status'])
        grounded, slow = not s & 2, abs(x - prev_x) < 4
        if not queue and cool <= 0 and grounded and slow and (x < 1930 or (x < 2030 and 1740 <= y <= 1800)):
            queue = [N('down')] * 2 + [N('jump+down'), N('down')] * revs
            cool = len(queue) + 12                     # let the dash play out before judging 'slow' again
        cool -= 1
        a = queue.pop(0) if queue else N('right')
        prev_x = x
        _, _, term, trunc, info = env.step(a)
        trace.append((env.frames, int(info['x']), int(info['y'])))
        if term or trunc:
            res = info['mg_result']['outcome']
            break
    print(f'{label:34s} -> {res:8s} at frame {AT + trace[-1][0] - trace[0][0] + 4} ({trace[-1][1]},{trace[-1][2]}); '
          f'max x {max(t[1] for t in trace)} | ' + ' '.join(f'{x},{y}' for _, x, y in trace[::20]), flush=True)


if os.environ.get('REACTIVE'):
    for r in (1, 2, 3):
        reactive(r, f'reactive spin dashes, {r} rev(s)')
    # his own inputs from AT on, as the yardstick
    at = {'f': seg['ckpts'][0]['frame']}

    def his_input(e, _m):
        at['f'] += 1
        return orig(e, table[actions[at['f']]])
    C.emu_frame = his_input
    env.reset(options={'key': seg['key'], 'ckpt': 0})
    xs = []
    while env.frames < AT + 1000:
        _, _, term, trunc, info = env.step(0)
        if env.frames >= AT:
            xs.append((env.frames, int(info['x']), int(info['y'])))
        if term or trunc:
            break
    C.emu_frame = orig
    print('his inputs from', AT, '->', ' '.join(f'{f}:{x},{y}' for f, x, y in xs[::20]))


if os.environ.get('CHASE'):
    # where does the chase end? from his state at frame F, stand still for 900 frames: crushed or not
    for F in range(560, 2401, 120):
        at = {'f': seg['ckpts'][0]['frame']}

        def his_input(e, _m):
            at['f'] += 1
            return orig(e, table[actions[at['f']]])
        C.emu_frame = his_input
        env.reset(options={'key': seg['key'], 'ckpt': 0})
        info, dead = None, False
        while env.frames < F:
            _, _, term, trunc, info = env.step(0)
            if term or trunc:
                dead = True
                break
        C.emu_frame = orig
        x0, y0, res = int(info['x']), int(info['y']), 'his run ended first' if dead else 'alive'
        for _ in range(225):
            if dead:
                break
            _, _, term, trunc, info = env.step(0)
            if term or trunc:
                res = f"{info['mg_result']['outcome']} at frame {env.frames}"
                break
        print(f'stand still from his frame {F} at ({x0},{y0}): {res}', flush=True)
