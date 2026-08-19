#!/usr/bin/env python3
"""
ram_log.py — Records RAM values during special stage steering and outputs
a ranked report of which addresses are most likely the rotation angle.

Usage:
  python3 ram_log.py

Controls:
  LEFT arrow  — steer left
  RIGHT arrow — steer right
  SPACE       — jump
  R           — reset to save state
  Q / ESC     — quit and print report

The script records all RAM values in the $E400-$E4FF range every frame.
After quitting it analyzes which addresses:
  1. Changed consistently when steering left (decreased)
  2. Changed consistently when steering right (increased)
  3. Reversed direction between left and right
  4. Did NOT change when no input was given

This is the signature of a rotation/heading angle.
Output is saved to ~/sonic-rl/ram_report.txt for easy sharing.
"""

import os
import sys
import time
import pygame
import stable_retro
import numpy as np
from collections import defaultdict

STATE_PATH  = os.path.expanduser('~/sonic-rl/states/curriculum_stage_1.state')
REPORT_PATH = os.path.expanduser('~/sonic-rl/ram_report.txt')

# Genesis RAM: address $FFxxxx → offset $xxxx into get_ram()
# Search $E400–$E4FF (256 bytes = 128 unsigned 16-bit values)
SEARCH_START = 0xE400
SEARCH_END   = 0xE4FF
KNOWN = {
    0xE438: 'blue_spheres_remaining',
    0xE43A: 'ss_rings_collected',
    0xE43E: 'ss_speed_timer',
}

SCALE = 2

def read_u16(ram, offset):
    if offset + 1 >= len(ram):
        return 0
    return (int(ram[offset]) << 8) | int(ram[offset + 1])

def read_s16(ram, offset):
    v = read_u16(ram, offset)
    return v if v < 32768 else v - 65536

def main():
    pygame.init()
    screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
    pygame.display.set_caption('RAM Logger — Steer around then press Q to get report')
    clock = pygame.time.Clock()

    env = stable_retro.make('SonicAndKnuckles3-Genesis-v0', render_mode='rgb_array')
    env.reset()
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'rb') as f:
            env.em.set_state(f.read())
        print(f'Loaded: {STATE_PATH}')
    else:
        print(f'ERROR: Save state not found at {STATE_PATH}')
        env.close()
        return

    # Recording structure
    # Each frame: {'input': 'left'|'right'|'none', 'values': {addr: value}}
    frames = []
    frame_count = 0
    font = pygame.font.SysFont('monospace', 13)

    print('\nControls: LEFT/RIGHT arrows to steer, SPACE to jump, R to reset, Q/ESC to quit and report')
    print('Try to steer LEFT for several seconds, then RIGHT for several seconds, then stop.')
    print('Recording started...\n')

    running = True
    while running:
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    with open(STATE_PATH, 'rb') as f:
                        env.em.set_state(f.read())
                    frames.clear()
                    frame_count = 0
                    print('Reset — recording cleared')

        # Determine input state
        left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        jump  = keys[pygame.K_SPACE] or keys[pygame.K_z]

        if left and not right:
            input_label = 'left'
        elif right and not left:
            input_label = 'right'
        else:
            input_label = 'none'

        # Build button array: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
        buttons = [int(jump), 0, 0, 0, 0, 0, int(left), int(right), 0, 0, 0, 0]
        obs, reward, term, trunc, info = env.step(buttons)
        frame_count += 1

        if term or trunc:
            env.reset()
            with open(STATE_PATH, 'rb') as f:
                env.em.set_state(f.read())

        # Read RAM
        ram = env.get_ram()
        values = {}
        for offset in range(SEARCH_START, SEARCH_END + 1, 2):
            values[offset] = read_s16(ram, offset)

        frames.append({'input': input_label, 'values': values})

        # Render
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            scaled = pygame.transform.scale(surf, (320 * SCALE, 224 * SCALE))
            screen.blit(scaled, (0, 0))

        # HUD
        left_frames  = sum(1 for f in frames if f['input'] == 'left')
        right_frames = sum(1 for f in frames if f['input'] == 'right')
        none_frames  = sum(1 for f in frames if f['input'] == 'none')
        hud = font.render(
            f'Frames: {frame_count}  LEFT:{left_frames}  RIGHT:{right_frames}  NONE:{none_frames}  |  Q=quit+report',
            True, (255, 255, 100)
        )
        screen.blit(hud, (5, 5))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    env.close()

    if len(frames) < 30:
        print('Not enough frames recorded. Run longer next time.')
        return

    print(f'\nAnalyzing {len(frames)} frames...')
    generate_report(frames)


def generate_report(frames):
    left_frames  = [f for f in frames if f['input'] == 'left']
    right_frames = [f for f in frames if f['input'] == 'right']
    none_frames  = [f for f in frames if f['input'] == 'none']

    addrs = list(frames[0]['values'].keys())
    results = []

    for addr in addrs:
        # Deltas during left steering
        left_deltas = []
        for i in range(1, len(left_frames)):
            d = left_frames[i]['values'][addr] - left_frames[i-1]['values'][addr]
            # Handle wraparound for 16-bit values
            if d > 16384: d -= 32768
            if d < -16384: d += 32768
            left_deltas.append(d)

        # Deltas during right steering
        right_deltas = []
        for i in range(1, len(right_frames)):
            d = right_frames[i]['values'][addr] - right_frames[i-1]['values'][addr]
            if d > 16384: d -= 32768
            if d < -16384: d += 32768
            right_deltas.append(d)

        # Deltas during no input
        none_deltas = []
        for i in range(1, len(none_frames)):
            d = none_frames[i]['values'][addr] - none_frames[i-1]['values'][addr]
            if d > 16384: d -= 32768
            if d < -16384: d += 32768
            none_deltas.append(d)

        if not left_deltas or not right_deltas:
            continue

        avg_left  = np.mean(left_deltas)  if left_deltas  else 0
        avg_right = np.mean(right_deltas) if right_deltas else 0
        avg_none  = np.mean(none_deltas)  if none_deltas  else 0
        std_left  = np.std(left_deltas)   if left_deltas  else 0
        std_right = np.std(right_deltas)  if right_deltas else 0

        # Rotation angle signature:
        # 1. Consistent direction when steering (low std relative to mean)
        # 2. Opposite direction between left and right
        # 3. Small or zero delta when not steering
        # 4. Non-trivial magnitude

        # Score based on these criteria
        direction_score = 0
        if abs(avg_left) > 0.5 and abs(avg_right) > 0.5:
            # Check if they go in opposite directions
            if (avg_left > 0) != (avg_right > 0):
                direction_score = 1
            # Check magnitude is consistent
            consistency_left  = abs(avg_left)  / (std_left  + 0.01)
            consistency_right = abs(avg_right) / (std_right + 0.01)
            consistency = min(consistency_left, consistency_right)
        else:
            consistency = 0

        stability_score = max(0, 1 - abs(avg_none) / (abs(avg_left) + 0.01))
        magnitude = (abs(avg_left) + abs(avg_right)) / 2

        score = direction_score * consistency * stability_score * min(magnitude, 10)

        results.append({
            'addr': addr,
            'avg_left': avg_left,
            'avg_right': avg_right,
            'avg_none': avg_none,
            'consistency': consistency,
            'magnitude': magnitude,
            'direction_score': direction_score,
            'score': score,
            'known': KNOWN.get(addr, ''),
        })

    # Sort by score descending
    results.sort(key=lambda x: -x['score'])

    # Build report
    lines = []
    lines.append('=' * 70)
    lines.append('RAM ROTATION ANGLE REPORT')
    lines.append(f'Total frames: {len(frames)}  |  Left: {sum(1 for f in frames if f["input"]=="left")}  Right: {sum(1 for f in frames if f["input"]=="right")}  None: {sum(1 for f in frames if f["input"]=="none")}')
    lines.append('=' * 70)
    lines.append('')
    lines.append('TOP 20 CANDIDATES (most likely rotation angle first):')
    lines.append(f'{"ADDR":8}  {"AVG_LEFT":10}  {"AVG_RIGHT":10}  {"AVG_NONE":10}  {"CONSIST":8}  {"SCORE":8}  NOTE')
    lines.append('-' * 70)

    for r in results[:20]:
        note = r['known'] or ('[OPPOSITE]' if r['direction_score'] else '')
        lines.append(
            f'${r["addr"]:04X}     {r["avg_left"]:+8.3f}    {r["avg_right"]:+8.3f}    {r["avg_none"]:+8.3f}    {r["consistency"]:6.2f}    {r["score"]:6.3f}  {note}'
        )

    lines.append('')
    lines.append('COLUMN GUIDE:')
    lines.append('  ADDR      - hex address (add $FF0000 for absolute, e.g. $E408 = $FFE408)')
    lines.append('  AVG_LEFT  - avg change per frame while steering LEFT')
    lines.append('  AVG_RIGHT - avg change per frame while steering RIGHT')
    lines.append('  AVG_NONE  - avg change per frame with no input (should be ~0 for angle)')
    lines.append('  CONSIST   - consistency score (higher = more reliable direction)')
    lines.append('  SCORE     - overall rotation angle likelihood score')
    lines.append('')
    lines.append('WHAT TO LOOK FOR:')
    lines.append('  - AVG_LEFT and AVG_RIGHT should be opposite signs (+/-)') 
    lines.append('  - AVG_NONE should be near 0')
    lines.append('  - CONSIST should be high (>2.0)')
    lines.append('  - Known addresses (blue_spheres etc) can be ignored')
    lines.append('=' * 70)

    report = '\n'.join(lines)
    print('\n' + report)

    with open(REPORT_PATH, 'w') as f:
        f.write(report + '\n')
    print(f'\nReport saved to {REPORT_PATH}')


if __name__ == '__main__':
    main()
