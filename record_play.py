#!/usr/bin/env python3
"""
record_play.py — play Sonic 3 & Knuckles yourself and save it as a playthrough
the base-game agent starts from and tries to beat (coordinator.py,
train_main.py).

  python3 record_play.py                                  # new, from Angel Island 1
  python3 record_play.py --speed 25                       # quarter speed
  python3 record_play.py --state HydrocityZone.Act1       # start at another act
  python3 record_play.py --resume demos/play_<stamp>.npz  # keep going where you left off

You play with exactly the agent's buttons: whatever you hold is mapped to one
of its 9 actions the way the game itself would read it, so every frame of the
recording is a move the agent has. Slow motion and rewind don't change what is
recorded — the file is one clean full-speed input sequence (inputs only, no
game data) — and saving replays it through the training env to prove it.

Play as well as you comfortably can. The agent starts from states your run
passes through and is paid for every frame it beats you by; it never copies
your inputs, so a slow or messy section just sets a lower bar. Rewind past
deaths if you like; a death left in is fine (the agent won't copy it).
Special stages and bonus stages are yours to take or skip — they're left out
of the act times on both sides.

Controls
  Arrows ............ move, look up, duck   Z X C Space ... jump
  Backspace ......... rewind ~2 s (hold)     P ............. pause
  1 2 3 4 ........... speed 25/50/75/100 %   F5 ............ save now, keep playing
  Enter ............. save and quit          Esc (twice) ... quit without saving
  Gamepad: d-pad or stick, face buttons jump, LB rewinds, Start pauses,
           Back saves now.

Every act clear saves automatically, and so does closing the window. Saving
writes demos/play_<stamp>.npz; --resume continues that same file.
"""
import os
import sys
import copy
import time
import zlib
import pickle
import hashlib
import argparse

import numpy as np

import coordinator as C
import demos

SCALE = 3
SPEEDS = {1: 25, 2: 50, 3: 75, 4: 100}
CACHE_DIR = os.path.join(demos.DEMO_DIR, '.cache')


def keys_to_action(left, right, up, down, jump, spindashing=False):
    """Held controls -> an ACTION_BUTTONS index, the way the game reads them.
    Holding left or right blocks ducking and rolling (SonicKnux_Roll), so down
    only counts without them — except while revving a spindash, where down
    keeps the charge and left/right do nothing."""
    horiz = 1 if (left and not right) else 2 if (right and not left) else 0
    if spindashing and down:
        return 8 if jump else 3
    if jump:
        return {1: 6, 2: 7}.get(horiz) or (8 if down else 5)
    if horiz:
        return horiz
    return 3 if down else 4 if up else 0


def _actions_sha1(actions):
    return hashlib.sha1(np.asarray(actions, dtype=np.uint8).tobytes()).hexdigest()


class PlayRecorder:
    """The emulator side of a recording session — no pygame, so a script can
    drive it (smoke_main.py does). It plays exactly like demos.replay_play:
    reset, load the start state, then one emu_frame per recorded action."""

    SNAP_EVERY = 20         # frames between rewind points
    REWIND_FRAMES = 120     # one rewind press: ~2 s
    FINE_WINDOW = 3600      # keep every rewind point of the last minute...
    COARSE_EVERY = 600      # ...and one per 10 s before that (~100 KB each)

    def __init__(self, start_state=C.DEFAULT_START, resume=None, path=None):
        self.env = C.make_retro()
        prior, meta = [], None
        if resume:
            prior, meta = demos.load_play(resume)
            if meta['action_table'] != C.ACTION_BUTTONS:
                raise ValueError(f'{resume} was recorded with a different action '
                                 f'table; it can be trained on but not extended')
            start_state = meta['start_state']
        self.path = resume or path or os.path.join(
            demos.DEMO_DIR, f'play_{time.strftime("%Y%m%d-%H%M%S")}.npz')
        self.start_name = start_state
        self.start = C.integration_state(start_state)
        self.meta0 = meta or {}
        self.rewinds = int(self.meta0.get('rewinds', 0))
        self.sessions = list(self.meta0.get('sessions', []))
        self.session_start = len(prior)
        self.env.reset()
        self.env.em.set_state(self.start)
        self.split = C.ActSplitter()
        self.actions = []
        self.info = None
        self._screen = None
        self.snaps = []
        self._snap()
        self.from_cache = False
        if len(prior) and not self._load_cache(prior):
            for a in prior:
                self._advance(int(a))
        if not len(prior):
            self._peek()
        self.saved_len = len(self.actions)

    # ── resume cache: skip replaying an hour of inputs when nothing changed
    def _cache_path(self):
        return os.path.join(CACHE_DIR, demos.demo_id(self.path) + '.resume')

    def _load_cache(self, prior):
        """Resume from the cache if it matches the file: the emulator state,
        the ActSplitter AND the rewind points, so Backspace still goes back
        past where the last session stopped. An older cache without rewind
        points doesn't count — replaying rebuilds them."""
        try:
            with open(self._cache_path(), 'rb') as f:
                c = pickle.load(f)
            if (c['n'] != len(prior) or c['sha1'] != _actions_sha1(prior)
                    or not c.get('snaps')):
                return False
        except (OSError, pickle.PickleError, KeyError, EOFError):
            return False
        self.env.em.set_state(zlib.decompress(c['state']))
        self.split = c['split']
        self.info = c['info']
        self.actions = [int(a) for a in prior]
        self.snaps = c['snaps']
        if self.snaps[-1][0] != len(self.actions):
            self._snap()
        self.from_cache = True
        self._peek()
        return True

    def _save_cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self._cache_path(), 'wb') as f:
            pickle.dump(dict(n=len(self.actions), sha1=_actions_sha1(self.actions),
                             state=zlib.compress(self.env.em.get_state(), 1),
                             split=self.split, info=self.info, snaps=self.snaps), f)

    # ── playing
    def _snap(self):
        self.snaps.append((len(self.actions), zlib.compress(self.env.em.get_state(), 1),
                           copy.deepcopy(self.split), self.info))
        n = len(self.actions)
        if n % self.COARSE_EVERY == 0:          # thin the old ones, now and then
            cut = n - self.FINE_WINDOW
            self.snaps = [s for s in self.snaps
                          if s[0] >= cut or s[0] % self.COARSE_EVERY == 0]

    def _peek(self):
        """Draw the current state. Loading a state leaves the emulator's
        screen black until a frame runs, so run one and put the state back
        (the game stays frame-for-frame the same)."""
        state = self.env.em.get_state()
        self.env.step(C.ACTION_BUTTONS[0])
        self._screen = self.env.em.get_screen().copy()
        self.env.em.set_state(state)

    def _advance(self, action):
        self._screen = None
        _, info = C.emu_frame(self.env, C.ACTION_BUTTONS[action])
        event = self.split.feed(len(self.actions), info)
        self.actions.append(int(action))
        self.info = info
        if len(self.actions) % self.SNAP_EVERY == 0:
            self._snap()
        return event

    def step(self, action):
        """Play one frame. Returns the ActSplitter event, if any."""
        return self._advance(action)

    @property
    def spindashing(self):
        return bool(self.info and int(self.info['mg_spindash']) & 1)

    def rewind(self, frames=REWIND_FRAMES):
        target = max(0, len(self.actions) - frames)
        while len(self.snaps) > 1 and self.snaps[-1][0] > target:
            self.snaps.pop()
        n, state, split, info = self.snaps[-1]
        self.env.em.set_state(zlib.decompress(state))
        del self.actions[n:]
        self.split = copy.deepcopy(split)
        self.info = info
        self.rewinds += 1
        self._peek()

    def screen(self):
        return self._screen if self._screen is not None else self.env.em.get_screen()

    @property
    def unsaved(self):
        return len(self.actions) != self.saved_len

    def save(self):
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        sessions = self.sessions + [dict(ended=now, frames=len(self.actions) - self.session_start)]
        demos.save_play(self.path, self.actions, dict(
            start_state=self.start_name, start_sha1=C.sha1(self.start),
            action_table=C.ACTION_BUTTONS, action_names=C.ACTION_NAMES,
            frames=len(self.actions), acts=self.split.summary(),
            rewinds=self.rewinds, sessions=sessions,
            recorded_at=self.meta0.get('recorded_at', now), updated_at=now))
        self._save_cache()
        self.saved_len = len(self.actions)
        return self.path

    def close(self):
        self.env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--state', default=C.DEFAULT_START,
                    help=f'integration start state (default {C.DEFAULT_START})')
    ap.add_argument('--resume', help='a playthrough to continue (demos/play_*.npz)')
    ap.add_argument('--speed', type=int, choices=sorted(SPEEDS.values()), default=50,
                    help='game speed in %% (default 50; 1-4 change it while playing)')
    args = ap.parse_args()

    if args.resume:
        print(f'loading {args.resume} ...', flush=True)
    rec = PlayRecorder(start_state=args.state, resume=args.resume)
    if args.resume:
        print(f'resumed at frame {len(rec.actions)}', flush=True)

    import pygame
    pygame.init()
    pygame.key.set_repeat(400, 150)                 # held Backspace keeps rewinding
    screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
    pygame.display.set_caption(f'record_play — {os.path.basename(rec.path)}')
    font = pygame.font.SysFont('monospace', 18, bold=True)
    clock = pygame.time.Clock()
    pad = None
    if pygame.joystick.get_count():
        pad = pygame.joystick.Joystick(0)
        pad.init()

    speed, paused = args.speed, True
    msg = 'paused — any move key (or P) starts. Backspace rewinds, F5 saves'
    esc_at = 0.0
    jump_keys = (pygame.K_z, pygame.K_x, pygame.K_c, pygame.K_SPACE)
    move_keys = jump_keys + (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)
    quit_save = None
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:          # closing the window saves
                quit_save = True
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if not rec.unsaved or time.time() - esc_at < 2:
                        running = False
                    else:
                        esc_at = time.time()
                        msg = 'unsaved play — Esc again quits WITHOUT saving; Enter saves'
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    quit_save = True
                    running = False
                elif ev.key == pygame.K_F5:
                    msg = f'saved {os.path.basename(rec.save())}'
                elif ev.key == pygame.K_BACKSPACE:
                    rec.rewind()
                    paused, msg = True, 'rewound — any move key (or P) resumes'
                elif ev.key == pygame.K_p:
                    paused = not paused
                elif pygame.K_1 <= ev.key <= pygame.K_4:
                    speed = SPEEDS[ev.key - pygame.K_0]
                elif ev.key in move_keys and paused:
                    paused = False
            elif ev.type == pygame.JOYBUTTONDOWN:
                if ev.button == 4:
                    rec.rewind()
                    paused, msg = True, 'rewound — any move (or Start) resumes'
                elif ev.button == 7:
                    paused = not paused
                elif ev.button == 6:
                    msg = f'saved {os.path.basename(rec.save())}'
                elif ev.button in (0, 1, 2, 3) and paused:
                    paused = False

        keys = pygame.key.get_pressed()
        left, right = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        up, down = keys[pygame.K_UP], keys[pygame.K_DOWN]
        jump = any(keys[k] for k in jump_keys)
        if pad is not None:
            hx, hy = pad.get_hat(0) if pad.get_numhats() else (0, 0)
            ax, ay = pad.get_axis(0), pad.get_axis(1)
            left |= hx < 0 or ax < -0.5
            right |= hx > 0 or ax > 0.5
            up |= hy > 0 or ay < -0.5
            down |= hy < 0 or ay > 0.5
            jump |= any(pad.get_button(b) for b in (0, 1, 2, 3)
                        if b < pad.get_numbuttons())

        if not paused:
            event = rec.step(keys_to_action(left, right, up, down, jump, rec.spindashing))
            if event == 'death':
                paused = True
                msg = 'died — Backspace rewinds (recommended), P plays on'
            elif event == 'clear':
                a = rec.split.acts[-1]
                rec.save()
                msg = (f"{a['name']} CLEAR in {a['time']} (saved) — keep playing, "
                       f"Enter saves and quits")
            elif event == 'stage':
                msg = 'special/bonus stage — take it or skip it; not part of act times'
            elif event == 'start':
                msg = f"{rec.split.cur['name']} — go!"

        surf = pygame.surfarray.make_surface(rec.screen().swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (320 * SCALE, 224 * SCALE)), (0, 0))
        info = rec.info or {}
        act = rec.split.cur['name'] if rec.split.cur else 'between acts'
        clock_s = C.fmt_time(C.game_time(info)) if info else '-'
        hud = (f"{'PAUSED' if paused else 'REC'}  {act}  {clock_s}  "
               f"rings {info.get('rings', 0)}  lives {info.get('lives', 0)}  "
               f"frame {len(rec.actions)}  {speed}%  rewinds {rec.rewinds}"
               + ('  *' if rec.unsaved else ''))
        screen.blit(font.render(hud, True, (255, 255, 0)), (8, 8))
        screen.blit(font.render(msg[:88], True, (0, 255, 128)), (8, 30))
        screen.blit(font.render('Bksp rewind  P pause  1-4 speed  F5 save  Enter save+quit  '
                                'Esc quit', True, (200, 200, 200)), (8, 224 * SCALE - 26))
        pygame.display.flip()
        clock.tick(60 * speed / 100)

    pygame.quit()
    if quit_save and rec.actions:
        path = rec.save()
        rec.close()                         # one emulator per process
        print(f'saved {path} — checking it replays ...', flush=True)
        ok, note = demos.replay_play_check(path)
        print(('OK: ' if ok else '!! ') + note)
        sys.exit(0 if ok else 1)
    rec.close()
    if os.path.exists(rec.path):
        print(f'quit without saving the last {abs(len(rec.actions) - rec.saved_len)} '
              f'frames; {rec.path} has everything up to the last save')
    else:
        print('nothing saved')


if __name__ == '__main__':
    main()
