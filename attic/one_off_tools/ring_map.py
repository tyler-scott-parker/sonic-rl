"""A stitched map of the area around a giant ring, with Tyler's route drawn on: screens taken with Sonic
held at points on a grid around the ring (re-set every frame while the game's camera scrolls there),
pasted at the camera's position. CPU only.  python3 ring_map.py OUT N [--span 2]"""
import os, sys, json, math
REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(CUDA_VISIBLE_DEVICES='', MG_LOG=os.devnull)
import numpy as np
import coordinator as C
import ring_clips as RC
import pygame
OUT, N = sys.argv[1], int(sys.argv[2])
SPAN = int(sys.argv[3]) if len(sys.argv) > 3 else 2
g = next(x for x in RC.load_index() if x['n'] == N)
z, a, rx, ry = g['ring']
start = C.integration_state(g['starts']['0']['file'])
env = C.make_retro()
env.reset()
tiles = []
STEP_X, STEP_Y = 160, 112                          # half a screen: every HUD corner is covered by a neighbour
NX, NY = 4 * SPAN, 4 * SPAN
for j in range(-NY, NY + 1):
    for i in range(-NX, NX + 1):
        px, py = rx + i * STEP_X, ry + j * STEP_Y
        if math.hypot(px - rx, py - ry) < 80:
            py = ry + 100                          # not on the ring: it would start the stage
        env.em.set_state(start)
        info = None
        for f in range(150):
            env.data.set_value('x', max(0, px))
            env.data.set_value('y', max(0, py))
            _, info = C.emu_frame(env, C.ACTION_BUTTONS[0])
            if C.frame_flags(info)[1]:
                break
        if C.frame_flags(info)[1]:
            print(f'  ({px},{py}) started a special stage: skipped')
            continue
        tiles.append((int(info['screen_x']), int(info['screen_y']), env.em.get_screen().copy()))
x0 = min(t[0] for t in tiles); y0 = min(t[1] for t in tiles)
W = max(t[0] for t in tiles) - x0 + 320; H = max(t[1] for t in tiles) - y0 + 224
canvas = np.zeros((H, W, 3), np.uint8)
hud = np.zeros((224, 320), bool)
hud[:60, :120] = True                              # score / time / rings
hud[196:, :80] = True                              # lives
for sx, sy, img in tiles:                          # everything first, then the parts without a HUD over it
    canvas[sy - y0:sy - y0 + 224, sx - x0:sx - x0 + 320] = img
for sx, sy, img in tiles:
    view = canvas[sy - y0:sy - y0 + 224, sx - x0:sx - x0 + 320]
    view[~hud] = img[~hud]
pygame.init()
surf = pygame.surfarray.make_surface(canvas.swapaxes(0, 1))
# his route through this act's internal zone/act, within the map
env.close()                                     # one emulator per process
env2 = C.SonicCoordinator(acts=[g['act_key']], demo_prob=0.0)
env2.demo_info()
r = env2._segs[0]['route']
m = (r['zone'] == z) & (r['act'] == a)
pts = [(int(x) - x0, int(y) - y0) for x, y in zip(r['x'][m], r['y'][m]) if 0 <= x - x0 < W and 0 <= y - y0 < H]
if len(pts) > 1:
    pygame.draw.lines(surf, (255, 60, 200), False, pts, 2)
pygame.draw.circle(surf, (255, 0, 0), (rx - x0, ry - y0), 40, 3)
h = g['his_nearest']
pygame.draw.circle(surf, (255, 255, 255), (h['x'] - x0, h['y'] - y0), 8, 2)
st = g['starts']['0']
pygame.draw.circle(surf, (0, 255, 0), (st['x'] - x0, st['y'] - y0), 8, 2)
font = pygame.font.SysFont('monospace', 14, bold=True)
surf.blit(font.render(f"ring {N} {g['act']} ({rx},{ry}): red = giant ring, pink = your route, green = clip start, white = nearest", True, (255, 255, 0), (0, 0, 0)), (4, 4))
fn = os.path.join(OUT, f'ring{N:02d}_map.png')
pygame.image.save(surf, fn)
print(f'{len(tiles)} screens, map {W}x{H} px covering x {x0}..{x0 + W}, y {y0}..{y0 + H} -> {fn}')
