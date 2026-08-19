#!/usr/bin/env python3
"""
capture_stages.py — Mint special-stage entry savestates by hand.

Navigate to each special stage (level-select cheat, or just play the giant
rings), and tap S to dump the current frame to the next numbered
~/sonic-rl/states/curriculum_stage_<N>.state. A full RAM snapshot
curriculum_stage_<N>.ram.bin is written alongside so find_stage_byte.py can
later locate the stage-number address by diffing across captures.

States are written with env.em.get_state() — the same raw (uncompressed)
format your curriculum pool and ss_monitor already use, so they load straight
into special_stage_env with no conversion. Captured against YOUR rom + YOUR
stable_retro build, so they're guaranteed loadable (unlike anything online).

Controls
  Arrows ............ D-pad
  Z ................. B   (jump / action)
  X ................. C
  A ................. A
  Enter ............. Start
  S ................. save state + RAM to current target index, then +1
  E ................. set chaos_emeralds = target-1  (aim next ring at stage N)
  [ / ] ............. target index -1 / +1   (no save)
  R ................. reload START_STATE      (redo / panic)
  Q ................. quit
"""

import os
import sys
import pygame
import stable_retro

STATES_DIR  = os.path.expanduser('~/sonic-rl/states')
GAME        = 'SonicAndKnuckles3-Genesis-v0'
SCALE       = 3

# Optional state to boot from. Leave None to use the integration default
# (e.g. to reach the title screen for the level-select cheat). Set to a path
# if starting from a known state helps you navigate. Overridable via argv[1].
START_STATE = None
START_INDEX = 2          # stage 1 already exists; first save lands on 2

# retro Genesis button order:
# 0=B 1=A 2=MODE 3=START 4=UP 5=DOWN 6=LEFT 7=RIGHT 8=C 9=Y 10=X 11=Z
HELD_KEYS = {
    pygame.K_z:      0,   # B
    pygame.K_a:      1,   # A
    pygame.K_RETURN: 3,   # START
    pygame.K_KP_ENTER: 3, # START (numpad Enter)
    pygame.K_TAB: 3,      # START (extra, in case Return is intercepted)
    pygame.K_UP:     4,
    pygame.K_DOWN:   5,
    pygame.K_LEFT:   6,
    pygame.K_RIGHT:  7,
    pygame.K_x:      8,   # C
}


def state_path(idx):
    return os.path.join(STATES_DIR, f'curriculum_stage_{idx}.state')


def ram_path(idx):
    return os.path.join(STATES_DIR, f'curriculum_stage_{idx}.ram.bin')


_confirm_overwrite = [None]


def main():
    global START_STATE
    if len(sys.argv) > 1:
        START_STATE = os.path.expanduser(sys.argv[1])

    os.makedirs(STATES_DIR, exist_ok=True)
    pygame.init()
    screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
    pygame.display.set_caption("capture_stages — S=save  E=emeralds  [ ]=index  R=reload  Q=quit")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 18, bold=True)

    env = stable_retro.make(
        game=GAME,
        use_restricted_actions=stable_retro.Actions.FILTERED,
        render_mode='rgb_array',
    )
    env.reset()

    if START_STATE:
        with open(START_STATE, 'rb') as f:
            env.em.set_state(f.read())
        print(f"Booted from {START_STATE}")
    else:
        print("Booted from integration default (no start state).")

    target = START_INDEX
    last_msg = ""
    print(f"Next save -> stage {target}. Reach a special stage, then press S.\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    _tgt = state_path(target)
                    if os.path.exists(_tgt) and _confirm_overwrite[0] != target:
                        _confirm_overwrite[0] = target
                        last_msg = f'!! stage {target} EXISTS — press S again to overwrite, or [ ] to change index'
                        print(last_msg)
                    else:
                        _confirm_overwrite[0] = None
                        # ---- original save body runs below, indented one more level ----

                        sp, rp = state_path(target), ram_path(target)
                        existed = os.path.exists(sp)
                        with open(sp, 'wb') as f:
                            f.write(env.em.get_state())
                        try:
                            with open(rp, 'wb') as f:
                                f.write(bytes(env.get_ram()))
                            ram_note = ""
                        except Exception as e:
                            ram_note = f"  (RAM dump failed: {e})"
                        verb = "OVERWROTE" if existed else "saved"
                        last_msg = f"{verb} stage {target}{ram_note}"
                        print(f"{last_msg}  -> {sp}")
                        target += 1
                        print(f"Next save -> stage {target}")
                elif event.key == pygame.K_LEFTBRACKET:
                    target = max(1, target - 1)
                    last_msg = f"target index -> {target}"
                    print(last_msg)
                elif event.key == pygame.K_RIGHTBRACKET:
                    target += 1
                    last_msg = f"target index -> {target}"
                    print(last_msg)
                elif event.key == pygame.K_e:
                    # Poke the emerald counter to (target-1): in S3&K the giant
                    # ring sends you to stage = emeralds+1, so this aims the next
                    # entry at the current target index. Set BEFORE entering a ring.
                    want = max(0, target - 1)
                    try:
                        env.data.set_value('chaos_emeralds', want)
                        last_msg = f"chaos_emeralds -> {want} (aiming stage {target})"
                    except Exception as e:
                        last_msg = f"emerald poke failed: {e}"
                    print(last_msg)
                elif event.key == pygame.K_w:
                    wp = os.path.join(STATES_DIR, 'waypoint_mushroom_hill.state')
                    with open(wp, 'wb') as fh:
                        fh.write(env.em.get_state())
                    last_msg = f"WAYPOINT saved -> {wp}"
                    print(last_msg)
                elif event.key == pygame.K_p:
                    wp = os.path.join(STATES_DIR, 'waypoint_mushroom_hill.state')
                    with open(wp, 'wb') as fh:
                        fh.write(env.em.get_state())
                    last_msg = f"WAYPOINT saved -> {wp}"
                    print(last_msg)
                elif event.key == pygame.K_g:
                    try:
                        env.data.set_value('chaos_emeralds', 7)
                        for _ in range(2):
                            env.step([0] * 12)
                        got = env.data.lookup_value('chaos_emeralds')
                        last_msg = f"GRANT 7 chaos -> reads back {got}"
                    except Exception as e:
                        last_msg = f"grant failed: {e}"
                    print(last_msg)
                elif event.key == pygame.K_v:
                    # WARP to zone = current target index: poke zone/act, then
                    # force a pit-death to trigger the reload (confirmed method).
                    z = target
                    try:
                        env.data.set_value('zone', z)
                        env.data.set_value('act', 0)
                        env.data.set_value('y', 60000)   # drop below level -> death reload
                        for _ in range(400):
                            env.step([0] * 12)
                        got = env.data.lookup_value('zone')
                        last_msg = f"WARP zone->{z}  now in zone {got}"
                    except Exception as ex:
                        last_msg = f"warp failed: {ex}"
                    print(last_msg)
                elif event.key == pygame.K_r and START_STATE:
                    with open(START_STATE, 'rb') as f:
                        env.em.set_state(f.read())
                    last_msg = "reloaded start state"
                    print(last_msg)

        buttons = [0] * 12
        keys = pygame.key.get_pressed()
        for key, idx in HELD_KEYS.items():
            if keys[key]:
                buttons[idx] = 1

        _, _, done, truncated, info = env.step(buttons)

        raw = env.render()
        if raw is not None and not isinstance(raw, bool):
            surface = pygame.surfarray.make_surface(raw.swapaxes(0, 1))
            scaled = pygame.transform.scale(surface, (320 * SCALE, 224 * SCALE))
            screen.blit(scaled, (0, 0))
            screen.blit(font.render(f"next save: stage {target}", True,
                                    (255, 255, 0)), (8, 8))
            try:
                em = env.data.lookup_value('chaos_emeralds')
                ss = env.data.lookup_value('special_stage')
                screen.blit(font.render(f"emeralds:{em}  special_stage:{ss}", True,
                                        (120, 200, 255)), (8, 28))
            except Exception:
                pass
            if last_msg:
                screen.blit(font.render(last_msg[:64], True,
                                        (0, 255, 128)), (8, 48))
        pygame.display.flip()

        if done or truncated:
            env.reset()
            if START_STATE:
                with open(START_STATE, 'rb') as f:
                    env.em.set_state(f.read())

        clock.tick(60)

    pygame.quit()
    env.close()
    print("done.")


if __name__ == '__main__':
    main()
