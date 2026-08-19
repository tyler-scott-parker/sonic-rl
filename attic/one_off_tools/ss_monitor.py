#!/usr/bin/env python3
"""
ss_monitor.py v2 — Monitors unknown SS RAM addresses via info dict.
Prints any address that changes value each frame.
Controls: Arrow keys + Space. Q to quit.
"""

import os
import pygame
import stable_retro

STATE_PATH = os.path.expanduser('~/sonic-rl/states/curriculum_stage_1.state')

WATCH_KEYS = [
    # Position tracking — verifying absolute vs relative coordinates
    'ss_x',             # $E422 — player X in stage grid ($100 per field, 32 fields = 0-8192)
    'ss_y',             # $E424 — player Y in stage grid ($100 per field, 32 fields = 0-8192)
    # Keys we're actively verifying
    'ss_next_corner',   # $E42A — expect -4, 0, or 4 at corners
    'bumper_contact',   # $E42B — expect 1 while touching bumper, 0 otherwise
    # Unknown addresses from earlier sessions
    'ss_unk_42c', 'ss_unk_42f', 'ss_unk_430',
    'ss_forward',
    'ss_unk_432', 'ss_unk_433', 'ss_unk_434', 'ss_unk_436',
    'ss_unk_440', 'ss_unk_442', 'ss_speed_val',
    'ss_unk_44a', 'ss_unk_44c', 'ss_unk_44d', 'ss_unk_44e', 'ss_unk_450',
    # Known for reference
    'ss_angle', 'ss_move_speed', 'ss_direction',
]

ACTION_MAP = {
    pygame.K_SPACE: [1,0,0,0,0,0,0,0,0,0,0,0],
    pygame.K_LEFT:  [0,0,0,0,0,0,1,0,0,0,0,0],
    pygame.K_RIGHT: [0,0,0,0,0,0,0,1,0,0,0,0],
    pygame.K_DOWN:  [0,0,0,0,0,1,0,0,0,0,0,0],
    pygame.K_UP:    [0,0,0,0,1,0,0,0,0,0,0,0],
}

SCALE = 3

def main():
    pygame.init()
    screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
    pygame.display.set_caption("SS RAM Monitor v2 — Q to quit")
    clock = pygame.time.Clock()

    env = stable_retro.make(
        game='SonicAndKnuckles3-Genesis-v0',
        use_restricted_actions=stable_retro.Actions.FILTERED,
        render_mode='rgb_array'
    )
    env.reset()

    with open(STATE_PATH, 'rb') as f:
        env.em.set_state(f.read())
    print("State loaded. Play and hit bumpers.")
    print("Only changed values print. Q to quit.\n")

    # Print initial values
    _, _, _, _, info = env.step([0]*12)
    print("Initial values:")
    for k in WATCH_KEYS:
        v = info.get(k)
        if v is not None and v != 0:
            print(f"  {k}: {v}")
    print()

    prev_vals = {k: info.get(k, 0) for k in WATCH_KEYS}
    frame_num = 1
    running = True

    while running:
        buttons = [0] * 12
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        for key, btn in ACTION_MAP.items():
            if keys[key]:
                for i, b in enumerate(btn):
                    buttons[i] |= b

        _, _, done, truncated, info = env.step(buttons)
        frame_num += 1

        # Print position every 60 frames (1 second) regardless of change
        if frame_num % 60 == 0:
            x = info.get('ss_x', '?')
            y = info.get('ss_y', '?')
            print(f"[POS] f{frame_num:>5}: ss_x={x:>6}  ss_y={y:>6}  "
                  f"(norm: x={x/8192:.3f}  y={y/8192:.3f})" if isinstance(x, int) else
                  f"[POS] f{frame_num:>5}: ss_x={x}  ss_y={y}")

        changes = []
        for k in WATCH_KEYS:
            curr = info.get(k, 0)
            prev = prev_vals.get(k, 0)
            if curr != prev:
                changes.append(f"{k}:{prev}→{curr}")
            prev_vals[k] = curr

        if changes:
            print(f"f{frame_num:>5}: {' | '.join(changes)}")

        raw = env.render()
        if raw is not None and not isinstance(raw, bool):
            surface = pygame.surfarray.make_surface(raw.swapaxes(0, 1))
            scaled = pygame.transform.scale(surface, (320 * SCALE, 224 * SCALE))
            screen.blit(scaled, (0, 0))
        pygame.display.flip()

        if done or truncated:
            with open(STATE_PATH, 'rb') as f:
                env.em.set_state(f.read())
            prev_vals = {k: 0 for k in WATCH_KEYS}

        clock.tick(60)

    pygame.quit()
    env.close()

if __name__ == '__main__':
    main()
