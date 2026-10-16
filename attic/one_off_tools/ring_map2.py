"""Readable map of a giant ring's area (2026-10-16; Tyler: "these maps are just a little hard to read").
v2 of ring_map.py: only the box around the ring and the nearest point of his route (padded half a screen),
no stamped Sonics/Tails (each screen's centre, where the held Sonic is, is filled from a neighbour), his route
as a thick line with direction arrows, the ring circled and labelled, and his clip's path (if any) in green.

    python3 attic/one_off_tools/ring_map2.py OUT_DIR N [N ...]       (CPU only)
"""
import glob
import json
import math
import os
import sys
import zlib

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main                                  # noqa: E402
train_main.configure_env('mg2')
os.environ.update(CUDA_VISIBLE_DEVICES='', MG_LOG=os.devnull)
import numpy as np                                 # noqa: E402
import coordinator as C                            # noqa: E402
import demos                                       # noqa: E402
import ring_clips as RC                            # noqa: E402
import pygame                                      # noqa: E402

OUT = sys.argv[1]
RINGS = [int(a) for a in sys.argv[2:]]
os.makedirs(OUT, exist_ok=True)
index = {g['n']: g for g in RC.load_index()}
acts = sorted({index[n]['act_key'] for n in RINGS})
env = C.SonicCoordinator(acts=acts, demo_prob=0.0)
env.demo_info()
segs = {s['key']: s for s in env._segs}
pygame.init()
font = pygame.font.SysFont('monospace', 16, bold=True)
small = pygame.font.SysFont('monospace', 13, bold=True)
HUD = np.zeros((224, 320), bool)
HUD[:60, :120] = True                              # score / time / rings
HUD[196:, :80] = True                              # lives
HUD[60:170, 110:210] = True                        # the held Sonic and Tails behind him


def screens(start_state, x0, x1, y0, y1):
    """[(camera x, camera y, screen)] with Sonic held on a half-screen grid over the box."""
    out = []
    for py in range(y0, y1 + 1, 112):
        for px in range(x0, x1 + 1, 160):
            env.base_env.reset()
            env.base_env.em.set_state(start_state)
            info = None
            for _ in range(150):
                env.base_env.data.set_value('x', max(0, px))
                env.base_env.data.set_value('y', max(0, py))
                _, info = C.emu_frame(env.base_env, C.ACTION_BUTTONS[0])
                if C.frame_flags(info)[1]:
                    break
            sx, sy = int(info['screen_x']), int(info['screen_y'])
            held = abs(int(info['x']) - max(0, px)) <= 64 and abs(int(info['y']) - max(0, py)) <= 64
            near = abs(sx + 160 - px) <= 320 and abs(sy + 112 - py) <= 240      # the camera got there
            if not C.frame_flags(info)[1] and held and near:
                out.append((sx, sy, env.base_env.em.get_screen().copy()))
    return out


def clip_path(g):
    """His clip's positions into ring g (the first clip that went in), or []."""
    for p in sorted(glob.glob(f"demos/clips/ring{g['n']:02d}_*.npz")):
        actions, meta = demos.load_play(p)
        c = meta.get('clip') or {}
        if c.get('entered_at') is None:
            continue
        env.base_env.reset()
        env.base_env.em.set_state(demos.play_start_state(meta))
        pts = []
        for a in actions[:c['entered_at']]:
            _, info = C.emu_frame(env.base_env, meta['action_table'][int(a)])
            pts.append((int(info['x']), int(info['y'])))
        return pts
    return []


for n in RINGS:
    g = index[n]
    z, a, rx, ry = g['ring']
    seg = segs[g['act_key']]
    h = g['his_nearest']
    cp = clip_path(g)
    xs = [rx, h['x']] + [p[0] for p in cp[-400:]]       # the clip's last approach, not the whole clip
    ys = [ry, h['y']] + [p[1] for p in cp[-400:]]
    x0, x1, y0, y1 = min(xs) - 320, max(xs) + 320, min(ys) - 224, max(ys) + 224
    start = zlib.decompress(seg['ckpts'][g['ring_start_point']]['state'])
    tiles = screens(start, x0, x1, y0, y1)
    cx0, cy0 = min(t[0] for t in tiles), min(t[1] for t in tiles)
    W, H = max(t[0] for t in tiles) - cx0 + 320, max(t[1] for t in tiles) - cy0 + 224
    canvas = np.zeros((H, W, 3), np.uint8)
    for sx, sy, img in tiles:
        canvas[sy - cy0:sy - cy0 + 224, sx - cx0:sx - cx0 + 320] = img
    for sx, sy, img in tiles:
        view = canvas[sy - cy0:sy - cy0 + 224, sx - cx0:sx - cx0 + 320]
        view[~HUD] = img[~HUD]
    CL, CT, CB = 120, 60, 28                            # crop the outer edge, where HUD scraps stay
    canvas = canvas[CT:H - CB, CL:]
    cx0, cy0, W, H = cx0 + CL, cy0 + CT, W - CL, H - CT - CB
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(canvas.swapaxes(0, 1)))
    r = seg['route']
    m = (r['zone'] == z) & (r['act'] == a)
    rp = [(int(x) - cx0, int(y) - cy0) for x, y in zip(r['x'][m], r['y'][m])
          if 0 <= x - cx0 < W and 0 <= y - cy0 < H]
    if len(rp) > 1:
        pygame.draw.lines(surf, (255, 60, 200), False, rp, 4)
        for i in range(60, len(rp) - 1, 90):        # direction arrows along his route
            (ax, ay), (bx, by) = rp[i - 8], rp[i]
            ang = math.atan2(by - ay, bx - ax)
            tip = (bx + 12 * math.cos(ang), by + 12 * math.sin(ang))
            l = (bx + 9 * math.cos(ang + 2.5), by + 9 * math.sin(ang + 2.5))
            rt = (bx + 9 * math.cos(ang - 2.5), by + 9 * math.sin(ang - 2.5))
            pygame.draw.polygon(surf, (255, 60, 200), [tip, l, rt])
    if cp:
        pts = [(x - cx0, y - cy0) for x, y in cp if 0 <= x - cx0 < W and 0 <= y - cy0 < H]
        if len(pts) > 1:
            pygame.draw.lines(surf, (40, 255, 80), False, pts, 4)
    pygame.draw.circle(surf, (255, 0, 0), (rx - cx0, ry - cy0), 44, 5)
    surf.blit(font.render('RING', True, (255, 255, 255), (200, 0, 0)), (rx - cx0 - 20, ry - cy0 - 70))
    pygame.draw.circle(surf, (255, 255, 255), (h['x'] - cx0, h['y'] - cy0), 10, 3)
    legend = (f"ring {n}  {g['act']} ({rx},{ry}): your run passes {h['px']} px away "
              f"at {h['t'] // 3600}:{h['t'] // 60 % 60:02d}")
    key = 'pink = your route (arrows: your direction)   white = nearest point   ' + ('green = your clip in' if cp else '(no clip yet)')
    bar = pygame.Surface((W, 44))
    bar.fill((0, 0, 0))
    bar.blit(font.render(legend, True, (255, 255, 0)), (6, 4))
    bar.blit(small.render(key, True, (220, 220, 220)), (6, 24))
    full = pygame.Surface((W, H + 44))
    full.blit(bar, (0, 0))
    full.blit(surf, (0, 44))
    fn = os.path.join(OUT, f'ring{n:02d}.png')
    pygame.image.save(full, fn)
    print(f'ring {n}: {len(tiles)} screens, {W}x{H} px, clip path {len(cp)} frames -> {fn}', flush=True)
