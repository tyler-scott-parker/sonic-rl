#!/usr/bin/env python3
"""
map_overlay.py — Overlays agent training data onto the Angel Island Act 1 map.

Modes:
  --calibrate   Interactive: click two points on the map, enter their x values.
                Saves calibration to map_calibration.json.
                Run this first.

  (default)     Generate annotated map PNG from curriculum_log.jsonl.

Usage:
  python3 map_overlay.py --calibrate    # set up coordinate mapping
  python3 map_overlay.py                # generate overlay image
  python3 map_overlay.py --out my.png  # custom output filename

Controls during calibration:
  Click on a known x position → enter its game x coordinate in the terminal.
  Do this for two well-separated points (e.g. spawn area and a far right obstacle).
  The map scrolls: left/right arrows, or click and drag.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.expanduser('~/sonic-rl')
MAP_PATH    = os.path.join(BASE_DIR, 'angel_island_map.png')
LOG_PATH    = os.path.join(BASE_DIR, 'curriculum_log.jsonl')
CALIB_PATH  = os.path.join(BASE_DIR, 'map_calibration.json')
LANDMARKS_PATH = os.path.join(BASE_DIR, 'landmarks.json')

# ── Visual constants ──────────────────────────────────────────────────────────
OVERLAY_H   = 120   # pixels of overlay panel height (top and bottom)
BAR_ALPHA   = 180   # 0-255 transparency of bars
STUCK_COLOR = (50,  140, 255, BAR_ALPHA)   # blue
DEATH_COLOR = (255,  50,  60, BAR_ALPHA)   # red
MARKER_W    = 8     # bar width in map pixels


# ══════════════════════════════════════════════════════════════════════════════
# Coordinate helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration():
    if not os.path.exists(CALIB_PATH):
        return None
    with open(CALIB_PATH) as f:
        return json.load(f)

def save_calibration(calib):
    with open(CALIB_PATH, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'Calibration saved to {CALIB_PATH}')

def game_x_to_pixel(x, calib):
    """Convert game x coordinate to image pixel x position."""
    px1, gx1 = calib['point1']['pixel_x'], calib['point1']['game_x']
    px2, gx2 = calib['point2']['pixel_x'], calib['point2']['game_x']
    if gx2 == gx1:
        return px1
    t = (x - gx1) / (gx2 - gx1)
    return int(px1 + t * (px2 - px1))

def pixel_to_game_x(px, calib):
    """Convert image pixel x to game x coordinate."""
    px1, gx1 = calib['point1']['pixel_x'], calib['point1']['game_x']
    px2, gx2 = calib['point2']['pixel_x'], calib['point2']['game_x']
    if px2 == px1:
        return gx1
    t = (px - px1) / (px2 - px1)
    return int(gx1 + t * (gx2 - gx1))


# ══════════════════════════════════════════════════════════════════════════════
# Log parsing
# ══════════════════════════════════════════════════════════════════════════════

def load_episodes():
    if not os.path.exists(LOG_PATH):
        print(f'No log found at {LOG_PATH}')
        return []
    episodes = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return episodes

def cluster(values, bucket=100):
    return Counter(round(v, -2) for v in values)

def load_cluster_data(episodes):
    stuck_all   = []
    death_all   = []
    best_x_all  = []

    for ep in episodes:
        sx = ep.get('stuck_xs', [])
        if isinstance(sx, list):
            stuck_all.extend(sx)
        dx = ep.get('death_x')
        if dx:
            death_all.append(dx)
        bx = ep.get('best_x') or ep.get('best_screen_x')
        if bx:
            best_x_all.append(bx)

    return cluster(stuck_all), cluster(death_all), best_x_all

def load_landmarks():
    if not os.path.exists(LANDMARKS_PATH):
        return {}
    try:
        with open(LANDMARKS_PATH) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get('landmarks', {}).items()}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Calibration mode (interactive pygame)
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration():
    if not os.path.exists(MAP_PATH):
        print(f'ERROR: Map image not found at {MAP_PATH}')
        print('Place angel_island_map.png in ~/sonic-rl/')
        return

    pygame.init()
    map_img = pygame.image.load(MAP_PATH)
    map_w, map_h = map_img.get_size()

    WIN_W, WIN_H = min(map_w, 1400), min(map_h + 60, 700)
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('Map Calibration — Click two known x positions')
    font  = pygame.font.SysFont('monospace', 14)
    clock = pygame.time.Clock()

    scroll_x  = 0
    points    = []
    dragging  = False
    drag_start = (0, 0)
    drag_scroll = 0

    print('\n=== MAP CALIBRATION ===')
    print('Click on the SPAWN POINT on the map (where Sonic starts).')
    print('This will be set to game x=7066.')
    print('Use +/- in the terminal to adjust scale if the overlay looks off.\n')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_LEFT:
                    scroll_x = max(0, scroll_x - 100)
                if event.key == pygame.K_RIGHT:
                    scroll_x = min(map_w - WIN_W, scroll_x + 100)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if my < WIN_H - 60:  # not in footer
                        actual_px = mx + scroll_x
                        try:
                            scale = float(input(f'  → Spawn pixel set to {actual_px} (x=7066). Scale [default 1.6]: ').strip() or '1.6')
                            calib = {
                                'point1': {'pixel_x': actual_px,                    'game_x': 7066},
                                'point2': {'pixel_x': actual_px + int(1000/scale),  'game_x': 7066 + 1000},
                            }
                            save_calibration(calib)
                            print(f'  ✓ Spawn at pixel {actual_px} = x=7066, scale={scale}')
                            print('\nCalibration done! Run without --calibrate to generate overlay.')
                            pygame.quit()
                            return
                        except (ValueError, EOFError):
                            print('  Invalid input, try again')

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                dragging = True
                drag_start = event.pos
                drag_scroll = scroll_x
            if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                dragging = False
            if event.type == pygame.MOUSEMOTION and dragging:
                dx = drag_start[0] - event.pos[0]
                scroll_x = max(0, min(map_w - WIN_W, drag_scroll + dx))

        # Draw
        screen.fill((20, 20, 30))
        screen.blit(map_img, (-scroll_x, 0))

        # Draw calibration points
        for i, pt in enumerate(points):
            px = pt['pixel_x'] - scroll_x
            if 0 <= px < WIN_W:
                pygame.draw.line(screen, (255, 255, 0), (px, 0), (px, WIN_H - 60), 2)
                label = font.render(f'P{i+1}: x={pt["game_x"]}', True, (255, 255, 0))
                screen.blit(label, (px + 3, 10))

        # Footer
        pygame.draw.rect(screen, (30, 30, 45), (0, WIN_H - 60, WIN_W, 60))
        msg = f'LEFT CLICK = set calibration point  |  ARROW KEYS = scroll  |  ESC = quit'
        calib_now = load_calibration()
        x_info = f'game_x≈{pixel_to_game_x(scroll_x, calib_now)}' if calib_now else 'not calibrated'
        msg2 = f'Scroll: {scroll_x}/{map_w - WIN_W}  |  Points set: {len(points)}/2  |  {x_info}'
        screen.blit(font.render(msg,  True, (200, 200, 200)), (5, WIN_H - 55))
        screen.blit(font.render(msg2, True, (150, 150, 150)), (5, WIN_H - 35))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ══════════════════════════════════════════════════════════════════════════════
# Overlay generation (PIL)
# ══════════════════════════════════════════════════════════════════════════════

def generate_overlay(output_path):
    calib = load_calibration()
    if not calib:
        print('ERROR: No calibration found. Run with --calibrate first.')
        return

    if not os.path.exists(MAP_PATH):
        print(f'ERROR: Map not found at {MAP_PATH}')
        return

    episodes = load_episodes()
    if not episodes:
        print('No episode data found.')
        return

    stuck_clusters, death_clusters, best_x_all = load_cluster_data(episodes)
    landmarks = load_landmarks()

    print(f'Loaded {len(episodes)} episodes')
    print(f'Stuck clusters: {len(stuck_clusters)}  Death clusters: {len(death_clusters)}')

    # Load map
    base = Image.open(MAP_PATH).convert('RGBA')
    map_w, map_h = base.size

    # Create output canvas with extra space top and bottom for bars
    canvas_h = map_h + OVERLAY_H * 2
    canvas = Image.new('RGBA', (map_w, canvas_h), (15, 15, 25, 255))
    canvas.paste(base, (0, OVERLAY_H))

    overlay = Image.new('RGBA', (map_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_sm = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 11)
        font_lg = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 14)
    except Exception:
        font_sm = ImageFont.load_default()
        font_lg = font_sm

    # ── Stuck clusters (top panel, blue) ─────────────────────────────────────
    max_stuck = max(stuck_clusters.values()) if stuck_clusters else 1
    for x_val, count in stuck_clusters.items():
        px = game_x_to_pixel(x_val, calib)
        if not (0 <= px < map_w):
            continue
        bar_h = int((count / max_stuck) * (OVERLAY_H - 20))
        bar_x1 = max(0, px - MARKER_W // 2)
        bar_x2 = min(map_w, px + MARKER_W // 2)
        bar_y1 = OVERLAY_H - bar_h
        bar_y2 = OVERLAY_H
        draw.rectangle([bar_x1, bar_y1, bar_x2, bar_y2], fill=STUCK_COLOR)
        # Count label
        draw.text((px - 8, bar_y1 - 14), str(count), font=font_sm, fill=(150, 200, 255, 220))

    # ── Death clusters (bottom panel, red) ───────────────────────────────────
    max_deaths = max(death_clusters.values()) if death_clusters else 1
    for x_val, count in death_clusters.items():
        px = game_x_to_pixel(x_val, calib)
        if not (0 <= px < map_w):
            continue
        bar_h = int((count / max_deaths) * (OVERLAY_H - 20))
        bar_x1 = max(0, px - MARKER_W // 2)
        bar_x2 = min(map_w, px + MARKER_W // 2)
        bar_y1 = OVERLAY_H + map_h
        bar_y2 = bar_y1 + bar_h
        draw.rectangle([bar_x1, bar_y1, bar_x2, bar_y2], fill=DEATH_COLOR)
        draw.text((px - 8, bar_y2 + 2), str(count), font=font_sm, fill=(255, 150, 150, 220))

    # ── Vertical guide lines on map ───────────────────────────────────────────
    all_x = sorted(set(stuck_clusters.keys()) | set(death_clusters.keys()))
    for x_val in all_x:
        px = game_x_to_pixel(x_val, calib)
        if not (0 <= px < map_w):
            continue
        draw.line([(px, OVERLAY_H), (px, OVERLAY_H + map_h)],
                  fill=(255, 255, 255, 30), width=1)

    # ── Landmark labels on the map ────────────────────────────────────────────
    for gx, label in landmarks.items():
        px = game_x_to_pixel(gx, calib)
        if not (0 <= px < map_w):
            continue
        draw.line([(px, OVERLAY_H), (px, OVERLAY_H + 30)], fill=(255, 220, 0, 160), width=2)
        draw.text((px + 3, OVERLAY_H + 4), label, font=font_sm, fill=(255, 220, 0, 220))

    # ── Best x reached line ───────────────────────────────────────────────────
    if best_x_all:
        best_ever = max(best_x_all)
        px = game_x_to_pixel(best_ever, calib)
        if 0 <= px < map_w:
            draw.line([(px, OVERLAY_H), (px, OVERLAY_H + map_h)],
                      fill=(0, 255, 120, 180), width=3)
            draw.text((px + 4, OVERLAY_H + 40),
                      f'Best: x={best_ever}', font=font_sm, fill=(0, 255, 120, 220))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_x, legend_y = 20, 10
    draw.rectangle([legend_x, legend_y, legend_x + 16, legend_y + 16],
                   fill=STUCK_COLOR)
    draw.text((legend_x + 22, legend_y + 1),
              f'Stuck events (n={sum(stuck_clusters.values())})', font=font_sm,
              fill=(200, 220, 255, 255))

    legend_y2 = canvas_h - OVERLAY_H + 10
    draw.rectangle([legend_x, legend_y2, legend_x + 16, legend_y2 + 16],
                   fill=DEATH_COLOR)
    draw.text((legend_x + 22, legend_y2 + 1),
              f'Deaths (n={sum(death_clusters.values())})', font=font_sm,
              fill=(255, 200, 200, 255))

    draw.rectangle([legend_x + 200, legend_y, legend_x + 216, legend_y + 16],
                   fill=(0, 255, 120, 180))
    draw.text((legend_x + 222, legend_y + 1), 'Best x reached', font=font_sm,
              fill=(200, 255, 220, 255))

    # ── Episode count ─────────────────────────────────────────────────────────
    draw.text((map_w - 300, 10),
              f'{len(episodes)} episodes  |  best x: {max(best_x_all) if best_x_all else "?"}',
              font=font_sm, fill=(180, 180, 180, 200))

    # Composite and save
    final = Image.alpha_composite(canvas, overlay)
    final = final.convert('RGB')
    final.save(output_path, quality=95)
    print(f'Saved: {output_path}  ({map_w}x{canvas_h}px)')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Angel Island map overlay tool')
    parser.add_argument('--calibrate', action='store_true',
                        help='Interactive calibration: click the spawn point (x=7066) on the map')
    parser.add_argument('--spawn-pixel', type=int, default=None,
                        help='Skip calibration: directly set spawn pixel x position (x=7066)')
    parser.add_argument('--scale', type=float, default=1.6,
                        help='Game units per image pixel (default: 1.6). Adjust if overlay looks off.')
    parser.add_argument('--out', default=os.path.join(BASE_DIR, 'map_overlay.png'),
                        help='Output PNG path (default: ~/sonic-rl/map_overlay.png)')
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    if args.spawn_pixel is not None:
        # Quick calibration: spawn pixel + scale → save calibration and generate
        spawn_px = args.spawn_pixel
        scale    = args.scale
        # Two points: spawn (x=7066) and one further right derived from scale
        calib = {
            'point1': {'pixel_x': spawn_px,                    'game_x': 7066},
            'point2': {'pixel_x': spawn_px + int(1000/scale),  'game_x': 7066 + 1000},
        }
        save_calibration(calib)
        print(f'Calibration set: pixel {spawn_px} = x=7066, scale={scale} game units/px')
        generate_overlay(args.out)
    elif args.calibrate:
        run_calibration()
    else:
        generate_overlay(args.out)
