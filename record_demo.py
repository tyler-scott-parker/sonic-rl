#!/usr/bin/env python3
"""
record_demo.py — play a special stage yourself and save it as a demo the agent
learns from (see DemoCurriculumCallback in train_ss_multistage.py).

  python3 record_demo.py 1              # stage 1, at 50% speed
  python3 record_demo.py 4 --speed 25   # a hard board, at quarter speed

You play with exactly the agent's controls, so every recorded frame is a move
the agent could have made, and the agent later starts episodes from states
your run passed through. It never copies your button presses — the reward
decides what's good — so a sloppy run still helps: it's a map of reachable
positions, not a script. Slow motion and rewind don't change what gets
recorded: the demo is one clean full-speed input sequence.

Controls
  Left / Right ....... turn            Up ........ advance (after a bumper)
  Z / X / C / Space .. jump            Backspace . rewind ~2 s (hold to keep going)
  P .................. pause           1 2 3 4 ... speed 25 / 50 / 75 / 100 %
  Enter .............. save and quit   R ......... restart the stage (discards)
  Esc ................ quit without saving
  Gamepad: d-pad or stick to turn/advance, face buttons jump, LB rewinds,
           Start pauses.

When the run ends (you touch red, or clear the board) the game freezes: rewind
past the mistake and keep going, or Enter to save it as it is. Saving replays
your inputs through the training env to confirm the demo reproduces your run.
"""
import os
import sys
import time
import zlib
import argparse

import numpy as np

# The demo is replayed by env v2 (termination on contact, whole-board grid).
os.environ['SS_ENV_VERSION'] = '2'
os.environ['SS_DEMOS'] = ''                 # never build demos inside the recorder

import stable_retro                                           # noqa: E402
import special_stage_env as sse                               # noqa: E402
import demos                                                  # noqa: E402

GAME = 'SonicAndKnuckles3-Genesis-v0'
SCALE = 3
SPEEDS = {1: 25, 2: 50, 3: 75, 4: 100}


def keys_to_action(left, right, up, jump):
    """Held controls -> one of the agent's ACTION_BUTTONS indices."""
    if jump:
        return 5 if (left and not right) else 6 if (right and not left) else 1
    if left and not right:
        return 2
    if right and not left:
        return 3
    return 4 if up else 0


class Recorder:
    """The emulator side of a recording session — no pygame, so it can be
    driven by a script (and tested). The play loop mirrors SpecialStageEnv
    exactly: load the stage state, one no-op reset frame, then one action per
    frame; that is what makes the saved actions replay into the same run."""

    SNAP_EVERY = 20         # frames between rewind points (~75 KB each)
    REWIND_FRAMES = 120     # one rewind press: ~2 s of game time

    def __init__(self, stage):
        self.stage = stage
        self.state_path = sse.stage_file(stage)
        with open(self.state_path, 'rb') as f:
            self.start_state = f.read()
        self.env = stable_retro.make(
            game=GAME, use_restricted_actions=stable_retro.Actions.FILTERED,
            render_mode='rgb_array')
        self.restart()

    def restart(self):
        self.env.reset()
        self.env.em.set_state(self.start_state)
        _, _, _, _, info = self.env.step(sse.ACTION_BUTTONS[0])   # the reset frame
        self.total = int(info['blue_spheres_remaining'])
        self.blue = self.total
        self.actions = []
        self._screen = None
        self.ended = None       # None, or 'clear' / 'death' / 'timeout'
        self.rewinds = 0
        self._swapped = self._detect_swap()
        self.snaps = [(0, zlib.compress(self.env.em.get_state(), 1), self.blue)]

    def _detect_swap(self):
        raw = np.frombuffer(bytes(self.env.get_ram()), dtype=np.uint8)
        for swapped, ram in ((True, sse.SpecialStageEnv._deswap(raw)), (False, raw)):
            if (int(ram[sse.ADDR_SPHERES]) << 8 | int(ram[sse.ADDR_SPHERES + 1])) == self.blue:
                return swapped
        raise RuntimeError('cannot locate the sphere counter in RAM — is this a '
                           'special-stage state?')

    def _exit_timer(self):
        raw = np.frombuffer(bytes(self.env.get_ram()), dtype=np.uint8)
        ram = sse.SpecialStageEnv._deswap(raw) if self._swapped else raw
        return int(ram[sse.ADDR_EXIT_TIMER])

    @property
    def collected(self):
        return self.total - self.blue

    def step(self, action):
        if self.ended:
            return
        self._screen = None
        _, _, _, _, info = self.env.step(sse.ACTION_BUTTONS[action])
        self.actions.append(int(action))
        self.blue = int(info['blue_spheres_remaining'])
        if len(self.actions) % self.SNAP_EVERY == 0:
            self.snaps.append((len(self.actions),
                               zlib.compress(self.env.em.get_state(), 1), self.blue))
        # the same end conditions env v2 uses, checked in the same order
        if self.blue == 0 and self.collected > 0:
            self.ended = 'clear'
        elif self._exit_timer() != 0 or (int(info.get('special_stage', 1)) == 0
                                         and len(self.actions) > 10):
            self.ended = 'death'
        elif len(self.actions) >= sse.SS_MAX_STEPS:
            self.ended = 'timeout'

    def rewind(self, frames=REWIND_FRAMES):
        target = max(0, len(self.actions) - frames)
        while len(self.snaps) > 1 and self.snaps[-1][0] > target:
            self.snaps.pop()
        n, state, blue = self.snaps[-1]
        self.env.em.set_state(zlib.decompress(state))
        del self.actions[n:]
        self.blue = blue
        self.ended = None
        self.rewinds += 1
        # loading a state leaves the screen black until a frame runs: run one
        # for the display and put the state back (the run stays identical)
        state = self.env.em.get_state()
        self.env.step(sse.ACTION_BUTTONS[0])
        self._screen = self.env.em.get_screen().copy()
        self.env.em.set_state(state)

    def screen(self):
        return self._screen if self._screen is not None else self.env.em.get_screen()

    def save(self, path=None):
        path = path or os.path.join(
            demos.DEMO_DIR, f'stage{self.stage}_{time.strftime("%Y%m%d-%H%M%S")}.npz')
        demos.save_demo(path, self.actions, dict(
            stage=self.stage, stage_total=self.total,
            outcome=self.ended or 'quit', spheres=self.collected,
            frames=len(self.actions), rewinds=self.rewinds,
            recorded_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            state_sha1=demos.file_sha1(self.state_path),
            actions_sha1=demos.actions_sha1(sse.ACTION_BUTTONS),
            env_version=sse.ENV_VERSION))
        return path

    def close(self):
        self.env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', type=int, help='special stage 1-14')
    ap.add_argument('--speed', type=int, choices=sorted(SPEEDS.values()), default=50,
                    help='game speed in %% (default 50; 1-4 change it while playing)')
    args = ap.parse_args()

    import pygame
    rec = Recorder(args.stage)
    pygame.init()
    pygame.key.set_repeat(400, 150)           # held Backspace keeps rewinding
    screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
    pygame.display.set_caption(f'record_demo — stage {args.stage}')
    font = pygame.font.SysFont('monospace', 18, bold=True)
    clock = pygame.time.Clock()
    pad = None
    if pygame.joystick.get_count():
        pad = pygame.joystick.Joystick(0)
        pad.init()

    # open paused: in Blue Spheres Sonic is already running, and the window
    # needs a click for focus first
    speed, paused, saved, msg = (args.speed, True, None,
                                 'paused — any move key (or P) starts; Backspace rewinds')
    jump_keys = (pygame.K_z, pygame.K_x, pygame.K_c, pygame.K_SPACE)
    move_keys = jump_keys + (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if rec.actions:
                        saved = rec.save()
                    running = False
                elif ev.key == pygame.K_BACKSPACE:
                    rec.rewind()
                    paused, msg = True, 'rewound — any move key (or P) resumes'
                elif ev.key == pygame.K_p:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    rec.restart()
                    paused, msg = True, 'restarted — any move key (or P) resumes'
                elif pygame.K_1 <= ev.key <= pygame.K_4:
                    speed = SPEEDS[ev.key - pygame.K_0]
                elif ev.key in move_keys and paused and not rec.ended:
                    paused = False
            elif ev.type == pygame.JOYBUTTONDOWN:
                if ev.button == 4:
                    rec.rewind()
                    paused, msg = True, 'rewound — any move (or Start) resumes'
                elif ev.button == 7:
                    paused = not paused
                elif ev.button in (0, 1, 2, 3) and paused and not rec.ended:
                    paused = False

        keys = pygame.key.get_pressed()
        left, right = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        up = keys[pygame.K_UP]
        jump = any(keys[k] for k in jump_keys)
        if pad is not None:
            hx, hy = pad.get_hat(0) if pad.get_numhats() else (0, 0)
            ax, ay = pad.get_axis(0), pad.get_axis(1)
            left |= hx < 0 or ax < -0.5
            right |= hx > 0 or ax > 0.5
            up |= hy > 0 or ay < -0.5
            jump |= any(pad.get_button(b) for b in (0, 1, 2, 3)
                        if b < pad.get_numbuttons())

        if not paused and not rec.ended:
            rec.step(keys_to_action(left, right, up, jump))
            if rec.ended == 'clear':
                msg = 'BOARD CLEARED — Enter saves, Backspace rewinds to improve'
            elif rec.ended == 'death':
                msg = (f'hit red at sphere {rec.collected} — Backspace rewinds, '
                       f'Enter saves as-is')
            elif rec.ended == 'timeout':
                msg = 'time limit — Enter saves, Backspace rewinds'

        surf = pygame.surfarray.make_surface(rec.screen().swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (320 * SCALE, 224 * SCALE)), (0, 0))
        state = 'ENDED' if rec.ended else 'PAUSED' if paused else 'REC'
        hud = (f'{state}  stage {args.stage}  spheres {rec.collected}/{rec.total}  '
               f'frame {len(rec.actions)}  speed {speed}%  rewinds {rec.rewinds}')
        screen.blit(font.render(hud, True, (255, 255, 0)), (8, 8))
        screen.blit(font.render(msg[:80], True, (0, 255, 128)), (8, 30))
        screen.blit(font.render('Bksp rewind  P pause  1-4 speed  Enter save  '
                                'R restart  Esc quit', True, (200, 200, 200)),
                    (8, 224 * SCALE - 26))
        pygame.display.flip()
        clock.tick(60 * speed / 100)

    pygame.quit()
    rec.close()
    if saved:
        print(f'saved {saved}')
        ok, note = demos.replay_check(saved)
        print(('OK: ' if ok else '!! ') + note)
        if not ok:
            sys.exit(1)
    else:
        print('nothing saved')


if __name__ == '__main__':
    main()
