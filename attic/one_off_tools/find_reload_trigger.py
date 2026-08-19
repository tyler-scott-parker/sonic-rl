#!/usr/bin/env python3
"""
find_reload_trigger.py -- find the RAM flag that forces a level (re)load.

Strategy: we know writing `zone` alone is inert. The game reads `zone` only
when a load routine fires. This tool captures full RAM continuously and lets
you mark two moments by keypress in a pygame window:
  [1] = "about to transition" (normal gameplay, pre-goalpost/pre-death)
  [2] = "transition happening" (the frame the level is reloading)
It diffs RAM between the two marks and reports bytes that changed, ranked —
those are candidates for the load/restart trigger. Then we test writing them.

Controls (pygame window must be focused):
  Arrows/Z/X/A/Enter = play (reach a goalpost or walk into a hazard)
  1 = snapshot A (mark "before transition")
  2 = snapshot B (mark "during/after transition"), then prints the diff
  R = reset
  Q = quit
"""
import os, pygame, stable_retro
import numpy as np

GAME = 'SonicAndKnuckles3-Genesis-v0'
SCALE = 3
HELD = {pygame.K_z:0, pygame.K_a:1, pygame.K_RETURN:3, pygame.K_UP:4,
        pygame.K_DOWN:5, pygame.K_LEFT:6, pygame.K_RIGHT:7, pygame.K_x:8}

def ram(env):
    return np.frombuffer(bytes(env.get_ram()), dtype=np.uint8).copy()

def main():
    pygame.init()
    screen = pygame.display.set_mode((320*SCALE, 224*SCALE))
    pygame.display.set_caption("find_reload_trigger: 1=markBefore 2=markAfter/diff R=reset Q=quit")
    font = pygame.font.SysFont('monospace', 18, bold=True)
    clock = pygame.time.Clock()
    env = stable_retro.make(GAME, use_restricted_actions=stable_retro.Actions.FILTERED,
                            render_mode='rgb_array')
    env.reset()
    snapA = None
    msg = "Play to a goalpost/hazard. Press 1 just BEFORE it loads, 2 DURING the load."
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    running = False
                elif e.key == pygame.K_r:
                    env.reset(); snapA = None; msg = "reset"
                elif e.key == pygame.K_1:
                    snapA = ram(env); msg = "snapshot A taken (before). Now trigger the load, then press 2."
                    print(msg)
                elif e.key == pygame.K_2:
                    if snapA is None:
                        msg = "press 1 first"; print(msg)
                    else:
                        snapB = ram(env)
                        diff = np.nonzero(snapA != snapB)[0]
                        print(f"\n{len(diff)} bytes changed between A and B.")
                        # Focus on the classic S3&K control-RAM region $FE00-$FFFF
                        # (offsets 0xFE00+) where mode/restart flags usually live.
                        interesting = [i for i in diff if 0xF000 <= i <= 0xFFFF]
                        print(f"{len(interesting)} of them in $F000-$FFFF (likely control region):")
                        for i in interesting[:60]:
                            print(f"  ${i:04X}: {snapA[i]:3d} -> {snapB[i]:3d}")
                        msg = f"diff printed to terminal: {len(diff)} changed, {len(interesting)} in control region"
        keys = pygame.key.get_pressed()
        buttons = [0]*12
        for k, idx in HELD.items():
            if keys[k]:
                buttons[idx] = 1
        env.step(buttons)
        raw = env.render()
        if raw is not None and not isinstance(raw, bool):
            surf = pygame.surfarray.make_surface(raw.swapaxes(0,1))
            screen.blit(pygame.transform.scale(surf, (320*SCALE, 224*SCALE)), (0,0))
            try:
                z = env.data.lookup_value('zone'); a = env.data.lookup_value('act')
                screen.blit(font.render(f"zone:{z} act:{a}", True, (255,255,0)), (8,8))
            except Exception:
                pass
            screen.blit(font.render(msg[:70], True, (0,255,128)), (8,28))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit(); env.close()

if __name__ == '__main__':
    main()
