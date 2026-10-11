"""Do Tyler's recordings clear their boards from the main game, every board reachable?
Super boards 8-14: MHZ1's ring with 7 chaos emeralds, the shrine controller picks each
board's emerald, then his inputs. Also board 1 from AIZ1's ring. Frames must equal his."""
import os, sys, glob
sys.path.insert(0, '/home/tyler/Projects/sonic-rl'); os.chdir('/home/tyler/Projects/sonic-rl')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
import torch; torch.set_num_threads(1)
import coordinator as C, demos, ss_pilot
SCR = None
emu = C.make_retro()
pilot = ss_pilot.SSPilot(emu, run='d9', model=None)
for i in range(7):
    emu.data.set_variable(f'em_arr{i}', {'address': 0xFFFFB2 + i, 'type': '|u1'})
buttons = dict(zip(C.ACTION_NAMES, C.ACTION_BUTTONS))
def his(b):
    a, m = demos.load_demo(glob.glob(f'demos/stage{b}_*.npz')[0]); return a, m
rows = []
# board 1 from AIZ1
emu.reset(); emu.em.set_state(open('states/handoff/aiz1_ring_minus1.state', 'rb').read()); C.emu_frame(emu, buttons['noop'])
a, m = his(1); r = pilot.play(script=a)
rows.append((1, r['board'], r['outcome'], r['spheres'], pilot.env.steps, len(a)))
for b in range(8, 15):
    emu.reset(); emu.em.set_state(open('states/handoff/mhz1_ring7_minus1.state', 'rb').read())
    for i in range(7):
        emu.data.set_value(f'em_arr{i}', 2)
    C.emu_frame(emu, buttons['noop'])                    # the ring frame: Hidden Palace
    n, slot, exp = pilot.shrine([b], buttons)
    a, m = his(b); r = pilot.play(script=a)
    rows.append((b, r['board'], r['outcome'], r['spheres'], pilot.env.steps, len(a)))
for b, got, out, sph, steps, n in rows:
    print(f'board {b:2d}: loaded board {got}, {out} with {sph} spheres in {steps} frames (his recording {n}) '
          f'{"IDENTICAL" if out == "clear" and steps == n and got == b else "DIFFERENT"}')
