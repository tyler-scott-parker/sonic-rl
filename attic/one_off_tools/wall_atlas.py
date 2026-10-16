"""Wall atlas (2026-10-16; a CPU job Tyler picked): for each mg2 act, maps of the spots where training's
failed tries end, from the episode log (each line's end x, y). The ends are grouped into the act's top
clusters; each map shows his route (pink, arrows for direction), every failed end in the box as a dot
(red = death, orange = stuck, cyan = out of time) and a caption with the counts.

The background is the ring_map2 mosaic (Sonic held on a half-screen grid from his nearest start point);
where that can't work (boss arenas lock the camera, scripted scenes) it falls back to screens from his own
play through the spot.

    python3 attic/one_off_tools/wall_atlas.py OUT_DIR [--last N] [--top K]      (CPU only)
"""
import argparse
import collections
import datetime
import math
import os
import sys
import zlib

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--last', type=int, default=1500, help='failed tries per act to read at most (newest)')
ap.add_argument('--hours', type=float, default=8, help="only the log's last HOURS of training (at least 150 tries an act)")
ap.add_argument('--top', type=int, default=3, help='clusters per act')
ap.add_argument('--acts', nargs='*', help='only these acts (default all of mg2)')
ap.add_argument('--min-share', type=float, default=0.04, help="smallest cluster drawn, as a share of the act's failures")
A = ap.parse_args()
import train_main                                  # noqa: E402
train_main.configure_env('mg2')
import coordinator as C                            # noqa: E402
RECS = C.log_tail(32 << 20)                        # read before the log is pointed away
os.environ.update(CUDA_VISIBLE_DEVICES='', MG_LOG=os.devnull)
C.LOG_FILE = os.devnull                            # this tool's own episodes are never logged
import numpy as np                                 # noqa: E402
import demos                                       # noqa: E402
import pygame                                      # noqa: E402

os.makedirs(A.out, exist_ok=True)
env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
env.demo_info()
pygame.init()
font = pygame.font.SysFont('monospace', 16, bold=True)
small = pygame.font.SysFont('monospace', 13, bold=True)
HUD = np.zeros((224, 320), bool)
HUD[:60, :120] = True                              # score / time / rings
HUD[196:, :80] = True                              # lives
HUD_ONLY = HUD.copy()
HUD[60:170, 110:210] = True                        # the held Sonic and Tails behind him
COLOURS = dict(death=(255, 40, 40), stuck=(255, 160, 0), timeout=(0, 220, 255))
BOX_W, BOX_H = 400, 280                            # half-size of the box around a cluster
RADIUS = 192                                       # ends this close to a cluster's centre belong to it


def screens(start_state, x0, x1, y0, y1):
    """[(camera x, camera y, screen)] with Sonic held on a half-screen grid over the box (ring_map2)."""
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
            near = abs(sx + 160 - px) <= 320 and abs(sy + 112 - py) <= 240
            if not C.frame_flags(info)[1] and held and near:
                out.append((sx, sy, env.base_env.em.get_screen().copy()))
    return out


DEMO = {}


def his_screens(seg, k, t_end, x0, x1, y0, y1):
    """[(camera x, camera y, screen)] from his own play: his frames from start point k up to his time
    t_end, a screen kept whenever the camera is over the box and 80+ px from every kept one."""
    if seg['demo'] not in DEMO:
        DEMO[seg['demo']] = demos.load_play(os.path.join(demos.DEMO_DIR, f"{seg['demo']}.npz"))
    actions, meta = DEMO[seg['demo']]
    ck = seg['ckpts'][k]
    env.base_env.reset()
    env.base_env.em.set_state(zlib.decompress(ck['state']))
    out = []
    for f in range(ck['frame'] + 1, min(len(actions), ck['frame'] + 1 + int(t_end - ck['t']) + 240)):
        _, info = C.emu_frame(env.base_env, meta['action_table'][int(actions[f])])
        sx, sy = int(info['screen_x']), int(info['screen_y'])
        over = sx + 320 > x0 and sx < x1 and sy + 224 > y0 and sy < y1
        if over and all(abs(sx - a) >= 80 or abs(sy - b) >= 80 for a, b, _ in out):
            out.append((sx, sy, env.base_env.em.get_screen().copy()))
    return out


def his_t(seg, e):
    """How far into his act a try got, in his frames: its start point's time plus progress x target
    (target = his frames from that start point to the act's end; progress stays 0 in a boss fight)."""
    return seg['ckpts'][min(e['start'], len(seg['ckpts']) - 1)]['t'] + e['progress'] * e['target']


def clusters(seg, rs):
    """The act's top clusters of failed ends: [(cx, cy, [rows])], biggest first. A cluster near a bigger
    one AND at about the same point of his route (progress x target) is the same spot (a wide boss
    arena) and joins it."""
    left = list(rs)
    out = []
    his = lambda rows: np.median([his_t(seg, e) for e in rows])
    while left and len(out) < A.top:
        cells = collections.Counter((r['x'] // 128, r['y'] // 128) for r in left)
        (gx, gy), _ = cells.most_common(1)[0]
        cell = [r for r in left if (r['x'] // 128, r['y'] // 128) == (gx, gy)]
        cx, cy = np.median([r['x'] for r in cell]), np.median([r['y'] for r in cell])
        mine = [r for r in left if math.hypot(r['x'] - cx, r['y'] - cy) <= RADIUS]
        if len(mine) < A.min_share * len(rs):
            break
        cx, cy = int(np.median([r['x'] for r in mine])), int(np.median([r['y'] for r in mine]))
        left = [r for r in left if math.hypot(r['x'] - cx, r['y'] - cy) > RADIUS]
        same = [c for c in out if math.hypot(c[0] - cx, c[1] - cy) <= 2.5 * RADIUS and abs(his(c[2]) - his(mine)) <= 900]
        if same:
            same[0][2].extend(mine)
        else:
            out.append((cx, cy, mine))
    return out


def draw(seg, cx, cy, mine, rs, n, fn):
    r = seg['route']
    m = np.arange(len(r['t']))                     # the whole act's route (AIZ1's burnt half reads as act 2)
    # his route point by the spot AND by how far the tries got (progress x target = his frame): AIZ1's
    # coordinates restart after the fire, so position alone can land in the wrong half of the act
    t_hint = np.median([his_t(seg, e) for e in mine])
    d = np.hypot(r['x'][m] - cx, r['y'][m] - cy)
    close = m[d <= 256] if (d <= 256).any() else m[d <= d.min() + 64]
    near = close[int(np.argmin(np.abs(r['t'][close] - t_hint)))]
    t_near = int(r['t'][near])
    ts = [c['t'] for c in seg['ckpts']]
    k = max(0, int(np.searchsorted(ts, t_near - 60, 'right')) - 1)   # a start point a second before
    x0, x1, y0, y1 = cx - BOX_W, cx + BOX_W, cy - BOX_H, cy + BOX_H
    src = 'mosaic'
    in_fight = any(f0 - 120 <= t_near <= f1 for f0, f1 in r['fights'])   # arenas lock the camera
    tiles = [] if seg['ckpts'][k]['boss'] or in_fight else screens(zlib.decompress(seg['ckpts'][k]['state']), x0, x1, y0, y1)
    if len(tiles) < 6:
        tiles, src = his_screens(seg, k, t_near, x0, x1, y0, y1), 'his play'
    if not tiles:
        print(f'{fn}: no screens (start point {k}, his frame {t_near}, box x {x0}-{x1} y {y0}-{y1})', flush=True)
        return None
    gx0, gy0 = min(t[0] for t in tiles), min(t[1] for t in tiles)
    W, H = max(t[0] for t in tiles) - gx0 + 320, max(t[1] for t in tiles) - gy0 + 224
    canvas = np.zeros((H, W, 3), np.uint8)
    for sx, sy, img in tiles:
        canvas[sy - gy0:sy - gy0 + 224, sx - gx0:sx - gx0 + 320] = img
    mask = HUD if src == 'mosaic' else HUD_ONLY  # his play: Sonic is wherever he was, keep him
    for sx, sy, img in tiles:                      # each screen's HUD (and held Sonic) from a neighbour
        view = canvas[sy - gy0:sy - gy0 + 224, sx - gx0:sx - gx0 + 320]
        view[~mask] = img[~mask]
    if W > 320 + 120 and H > 224 + 88:
        CL, CT, CB = 120, 60, 28                   # crop the outer edge, where HUD scraps stay
        canvas = canvas[CT:H - CB, CL:]
        gx0, gy0, W, H = gx0 + CL, gy0 + CT, W - CL, H - CT - CB
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(canvas.swapaxes(0, 1)))
    win = m[np.abs(r['t'][m] - t_near) <= 2400]      # his route near that time only (AIZ1's two halves)
    inside = (r['x'][win] - gx0 >= 0) & (r['x'][win] - gx0 < W) & (r['y'][win] - gy0 >= 0) & (r['y'][win] - gy0 < H)
    runs, cur = [], []
    for i, ok in zip(win, inside):                  # one line per stretch inside the box, no jumps across
        if ok:
            cur.append((int(r['x'][i]) - gx0, int(r['y'][i]) - gy0))
        elif cur:
            runs.append(cur)
            cur = []
    runs.append(cur)
    for rp in runs:
        if len(rp) < 2:
            continue
        pygame.draw.lines(surf, (255, 60, 200), False, rp, 4)
        for i in range(60, len(rp) - 1, 90):
            (ax, ay), (bx, by) = rp[i - 8], rp[i]
            ang = math.atan2(by - ay, bx - ax)
            pygame.draw.polygon(surf, (255, 60, 200), [
                (bx + 12 * math.cos(ang), by + 12 * math.sin(ang)),
                (bx + 9 * math.cos(ang + 2.5), by + 9 * math.sin(ang + 2.5)),
                (bx + 9 * math.cos(ang - 2.5), by + 9 * math.sin(ang - 2.5))])
    jit = np.random.default_rng(0)
    for e in rs:                                   # every failed end in the box, jittered so piles show
        if abs(his_t(seg, e) - t_near) > 2400:
            continue                               # the other half of AIZ1, same coordinates
        px, py = e['x'] - gx0 + jit.integers(-5, 6), e['y'] - gy0 + jit.integers(-5, 6)
        if 0 <= px < W and 0 <= py < H:
            pygame.draw.circle(surf, COLOURS[e['outcome']], (int(px), int(py)), 4)
            pygame.draw.circle(surf, (0, 0, 0), (int(px), int(py)), 4, 1)
    oc = collections.Counter(e['outcome'] for e in mine)
    starts = sorted(e['start'] for e in mine)
    fresh = sum(C.is_fresh(e) for e in mine)
    line1 = (f"{seg['name']} #{n}: {len(mine)} of the last {len(rs)} failed tries ended here "
             f"({len(mine) / len(rs):.0%}); your route passes at {t_near / seg['T']:.0%} into the act")
    line2 = (f"{oc['death']} deaths (red), {oc['stuck']} stuck (orange), {oc['timeout']} out of time (cyan); "
             f"start points {starts[0]}-{starts[-1]} (median {starts[len(starts) // 2]}), {fresh} fresh starts")
    line3 = f"pink = your route, arrows = your direction; background: {src}"
    bar_w = max(W, 8 + 8 * max(len(line1), len(line2)))
    bar = pygame.Surface((bar_w, 64))
    bar.fill((0, 0, 0))
    bar.blit(font.render(line1, True, (255, 255, 0)), (6, 2))
    bar.blit(small.render(line2, True, (255, 255, 255)), (6, 24))
    bar.blit(small.render(line3, True, (200, 200, 200)), (6, 44))
    full = pygame.Surface((bar_w, H + 64))
    full.blit(bar, (0, 0))
    full.blit(surf, (0, 64))
    pygame.image.save(full, os.path.join(A.out, fn))
    print(f'{fn}: {line1}\n    {line2} [{src}, {len(tiles)} screens]', flush=True)
    return f'{fn}\n  {line1}\n  {line2}\n'


index = []
for seg in env._segs:
    if A.acts and seg['name'] not in A.acts:
        continue
    rs = [r for r in RECS if r.get('demo') == seg['demo'] and r['act'] == seg['name']
          and not r['success'] and r['outcome'] in COLOURS]
    recent = [r for r in rs if r['t'] >= RECS[-1]['t'] - A.hours * 3600]
    rs = (recent if len(recent) >= 150 else rs[-150:])[-A.last:]
    if not rs:
        print(f"{seg['name']}: no failed tries in the log tail")
        continue
    span = (datetime.datetime.fromtimestamp(rs[0]['t']).strftime('%m-%d %H:%M'),
            datetime.datetime.fromtimestamp(rs[-1]['t']).strftime('%m-%d %H:%M'))
    print(f"{seg['name']}: {len(rs)} failed tries, {span[0]} to {span[1]}", flush=True)
    index.append(f"== {seg['name']}: {len(rs)} failed tries, {span[0]} to {span[1]}\n")
    for n, (cx, cy, mine) in enumerate(clusters(seg, rs), 1):
        got = draw(seg, cx, cy, mine, rs, n, f"{seg['name']}_{n}.png")
        if got:
            index.append(got)
open(os.path.join(A.out, 'index.txt'), 'w').write(''.join(index))
