#!/usr/bin/env python3
"""check_compass.py — does the ring compass (MG_RING_COMPASS, coordinator.draw_compass) point at the
right ring, and forget a ring once it's used?

    python3 check_compass.py [TRIES]          # default logs/check_ss_detour_mg1.json (a try into AIZ1's ring)

1. the ROM gives 77 giant rings, AIZ1's at (7112, 1216) among them
2. a real try into AIZ1's giant ring, replayed with the compass and the air bar on (MG_SS=pilot): at
   every step the dot's offset from the box centre has the signs of the true offset to the nearest
   unused ring (where it isn't rounded to 0), and its distance from the centre is
   COMPASS_R * sqrt(min(d, COMPASS_RANGE) / COMPASS_RANGE) within rounding (0.71 px); after the stage the ring is in the used set and the dot no longer points at it
3. pictures: the agent's view (x4) at the start, near the ring, and after the stage
   (scratchpad compass_*.png; the box is the top right, under the air bar)
CPU only. Ends with ALL CHECKS PASS or the failures.
"""
import json
import math
import os
import sys

HOME = os.path.dirname(os.path.abspath(__file__))
OUT = '/tmp/claude-1000/-home-tyler/f5282037-a414-481d-a718-80564fabc923/scratchpad'


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, 'logs', 'check_ss_detour_mg1.json')
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env('mg2')
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='pilot', MG_RING_COMPASS='1', MG_AIR_BAR='1')
    import numpy as np
    import cv2
    import coordinator as C
    bad = []
    n = sum(len(v) for v in C.GIANT_RINGS.values())
    print(f'1. {n} giant rings; AIZ1 (zone 0, act 0): {C.GIANT_RINGS.get((0, 0))}')
    if n != 77 or (7112, 1216) not in C.GIANT_RINGS.get((0, 0), []):
        bad.append('1: the ring list')
    acts = json.load(open(path))['ring']['actions']
    env = C.SonicCoordinator(acts=['play_20260930-153946:AIZ1'], demo_prob=0.0)
    seg = env.demo_info()[0]
    obs, _ = env.reset(options={'key': seg['key'], 'ckpt': 0})
    cy, cx = C.COMPASS_Y0 + C.COMPASS_R, C.COMPASS_X0 + C.COMPASS_R
    checked, wrong, edge_wrong, shots, after = 0, 0, 0, {}, None
    for i, a in enumerate(acts):
        _, _, term, trunc, info = env.step(a)
        f = obs_last = env._obs()[:, :, -1]
        box = f[C.COMPASS_Y0:C.COMPASS_Y0 + 2 * C.COMPASS_R + 1, C.COMPASS_X0:C.COMPASS_X0 + 2 * C.COMPASS_R + 1]
        near = C.nearest_ring(info, env._used_rings)
        dots = np.argwhere(box == 255)
        if near is not None and len(dots) == 1:
            (dy_px, dx_px), (_, dx, dy) = dots[0] - C.COMPASS_R, near
            checked += 1
            if (dx_px and math.copysign(1, dx_px) != math.copysign(1, dx)) or \
               (dy_px and math.copysign(1, dy_px) != math.copysign(1, dy)):
                wrong += 1
            want = C.COMPASS_R * math.sqrt(min(math.hypot(dx, dy), C.COMPASS_RANGE) / C.COMPASS_RANGE)
            if abs(math.hypot(dx_px, dy_px) - want) > math.sqrt(0.5) + 1e-9:     # rounding: <= 0.71 px
                edge_wrong += 1
        elif near is not None:
            bad.append(f'2: step {i}: {len(dots)} dots for one ring')
            break
        if i == 0:
            shots['start'] = f.copy()
        if near is not None and math.hypot(near[1], near[2]) < 200 and 'near' not in shots:
            shots['near'] = f.copy()
        if env._stages and after is None:
            after = (dict(info), set(env._used_rings))
            shots['after'] = f.copy()
        if term or trunc:
            break
    print(f'2. {checked} steps checked: dot direction wrong in {wrong}, distance wrong in {edge_wrong}')
    if checked < 100 or wrong or edge_wrong:
        bad.append(f'2: {checked} checked, {wrong} wrong direction, {edge_wrong} wrong distance')
    if after is None:
        bad.append('2: the try never played a stage')
    else:
        info, used = after
        print(f'   after the stage: used {used}; nearest unused now {C.nearest_ring(info, used)}')
        if (0, 0, 7112, 1216) not in used:
            bad.append('2: AIZ1\'s ring not marked used after its stage')
    for name, f in shots.items():
        cv2.imwrite(os.path.join(OUT, f'compass_{name}.png'), cv2.resize(f, (336, 336), interpolation=cv2.INTER_NEAREST))
    print(f'3. pictures: {sorted(shots)} in {OUT}/compass_*.png')
    if bad:
        print('\nFAILED:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print('\nALL CHECKS PASS')


if __name__ == '__main__':
    main()
