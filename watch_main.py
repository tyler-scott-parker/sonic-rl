#!/usr/bin/env python3
"""
watch_main.py — watch a base-game run's agent play an act, or watch your own
recording of it from the same point, to compare.

  python3 watch_main.py aiz1                  # the agent, from the start of the act
  python3 watch_main.py aiz1 --frac 0.6       # ... from 60% into your recording
  python3 watch_main.py aiz1 --you            # your recording, same start
  python3 watch_main.py aiz1 --model checkpoints_main_aiz1/main_aiz1_1000000_steps.zip
  python3 watch_main.py aiz1 --stochastic     # sample actions instead of the argmax
  python3 watch_main.py mg1 --live            # follow a run while it trains
  python3 watch_main.py mg1 --live --fresh    # ... always from the start of the act
  python3 watch_main.py mg1 --live --fresh --best 20   # ... its best of 20 tries each time

--live reloads the run's save before each episode (the trainer rewrites it
every 100k steps) and starts where training currently starts (its curriculum
start point); F switches between that and the start of the act. --best N plays
N sampled tries off-screen first (as training samples its actions) and shows
the best one: the fastest clear, else the one that got furthest along your
route. The game is deterministic, so the replay is exactly that try. G (or
--ring) shows a try that went into a giant ring instead, when one did.

Special stages (a run with MG_SS=pilot, e.g. mg1) are shown too; S skips the
one on screen (it runs off-screen at full speed; S again watches the next),
--skip-ss starts with skipping on.

--stream PORT: no window; the picture and sound go out as a live video stream
for a phone instead (stream_out.py), on the tailnet address at that port, with
buttons for the switches. The on-screen text sits in a panel under the picture.
  python3 watch_main.py mg1 --live --fresh --best 20 --ring --stream 8790

Sound plays at 100% speed (game_audio.py, as in the recorders); M mutes,
--no-sound starts muted.

Keys: 1-4 speed 25/50/100/200 %   P pause   R restart   F fresh/training start   M sound
      G giant-ring tries (--best)   S skip special stages   Esc quit
"""
import os
import sys
import json
import collections
import textwrap
import time
import argparse

import train_main

SCALE = 3
SPEEDS = {1: 25, 2: 50, 3: 100, 4: 200}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('run', help='run name from train_main.RUNS')
ap.add_argument('--act', help='act to watch (default: the run\'s first)')
ap.add_argument('--frac', type=float, default=0.0,
                help='start this far into your recording of the act (0 = its start)')
ap.add_argument('--you', action='store_true', help='play back your recording instead')
ap.add_argument('--model', help='a checkpoint zip (default: the run\'s latest save)')
ap.add_argument('--stochastic', action='store_true')
ap.add_argument('--live', action='store_true',
                help='follow a training run: reload its save before each episode, start where it trains')
ap.add_argument('--fresh', action='store_true', help='with --live: start at the act\'s start (F toggles)')
ap.add_argument('--no-sound', action='store_true', help='start muted (M toggles)')
ap.add_argument('--best', type=int, default=0, metavar='N',
                help='play N sampled tries off-screen, then show the best one')
ap.add_argument('--ring', action='store_true', help='with --best: show a try that entered a giant ring (G toggles)')
ap.add_argument('--skip-ss', action='store_true', help="don't show special stages (S toggles)")
ap.add_argument('--stream', type=int, metavar='PORT',
                help='no window: stream picture + sound to a phone on the tailnet at PORT')
args = ap.parse_args()

if args.run not in train_main.RUNS:
    sys.exit(f'unknown run {args.run!r}; known: {", ".join(train_main.RUNS)}')
cfg = train_main.RUNS[args.run]
if not train_main.configure_env(args.run):
    sys.exit(f"no recordings match {cfg['demos']}")
os.environ['MG_LOG'] = os.devnull                   # watching isn't training data

import numpy as np                                  # noqa: E402
import pygame                                       # noqa: E402
import coordinator as C                             # noqa: E402
import demos                                        # noqa: E402

env = C.SonicCoordinator(acts=[args.act] if args.act else cfg['acts'][:1], demo_prob=0.0)
seg = env.demo_info()[0]
key = seg['key']
k = int(np.searchsorted(seg['fracs'], args.frac, 'right')) - 1 if args.frac > 0 else 0
k = max(0, k)
model, model_mtime, path = None, None, None
if not args.you:
    import torch
    torch.set_num_threads(1)                        # one step's decision is tiny; more threads only fight training
    from stable_baselines3 import PPO
    path = args.model or train_main.paths(args.run)['zip'] + '.zip'
    if not os.path.exists(path) and not args.live:
        sys.exit(f'no model at {path} (train first, or --you to watch your recording)')
    if not args.live:
        model = PPO.load(path, device='cpu')
fresh = args.fresh                                  # --live: start at the act's start (F toggles)
ring = args.ring                                    # --best: show a giant-ring try when there is one (G)


def refresh_model():
    """--live: load the run's save if it changed since the last episode."""
    global model, model_mtime
    try:
        m = os.path.getmtime(path)
    except OSError:
        return
    if m == model_mtime:
        return
    try:
        model = PPO.load(path, device='cpu')
        model_mtime = m
        print(f'loaded the save at {model.num_timesteps:,} steps', flush=True)
    except Exception as e:                          # caught mid-write: keep the old one, retry next episode
        print(f'save not readable yet ({e!r}); retrying next episode', flush=True)


def live_start():
    """--live: the start point training uses now (its curriculum file), or 0 with F."""
    if fresh:
        return 0
    try:
        with open(train_main.paths(args.run)['curriculum']) as f:
            return int(json.load(f)[key]['ptr'])
    except (OSError, ValueError, KeyError):
        return k
demo_actions = table = None
if args.you:                                        # replayed with its own button table
    s = next(s for s in env._segs if s['key'] == key)
    demo_actions, meta = demos.load_play(next(p for p in env.demo_files
                                              if demos.demo_id(p) == s['demo']))
    table = meta['action_table']

if args.stream:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'         # no window: the frames go to the stream
    PIC, PANEL = (640, 448), 100                    # the picture x2, then a text panel under it
else:
    PIC, PANEL = (320 * SCALE, 224 * SCALE), 0
pygame.init()
screen = pygame.display.set_mode((PIC[0], PIC[1] + PANEL))
who = 'your recording' if args.you else os.path.basename(args.model or train_main.paths(args.run)['zip'])
if args.live:
    who = f'{args.run} live'
pygame.display.set_caption(f'watch_main — {key} — {who}')
font = pygame.font.SysFont('monospace', 16 if args.stream else 18, bold=True)
clock = pygame.time.Clock()
speed, paused, running = 100, False, True

audio = None
try:
    import game_audio                               # noqa: E402
    if args.stream:
        import stream_out

        def switch(name):
            return (lambda: globals()[name]), (lambda v: globals().__setitem__(name, v))
        audio = stream_out.StreamOut(os.path.join(train_main.HOME, 'logs', f'stream_{args.run}'),
                                     (PIC[0], PIC[1] + PANEL), args.stream,
                                     {n: switch(n) for n in ('skip_ss', 'ring', 'fresh')},
                                     rate=env.base_env.em.get_audio_rate())
        print(f'streaming at {audio.url}', flush=True)
    else:
        audio = game_audio.GameAudio(env.base_env.em.get_audio_rate())
except Exception as e:                              # no device / no SDL audio: watch in silence
    print(f'no sound: {e!r}', flush=True)
sound = audio is not None and not args.no_sound
stepping = False                                    # shown and heard only while an episode steps (not resets)
hud_lines = []                                      # (text, colour), drawn over every frame
_base_step = env.base_env.step


fps_count = {'n': 0, 't': time.time()}


last_img = None


def draw(img, extra=None, force=False):
    """One frame: the picture, the text (over it, or in the stream's panel)
    and, streaming, every other frame (the stream runs at 30 fps) to it."""
    global last_img
    last_img = img
    fps_count['n'] += 1
    if time.time() - fps_count['t'] >= 10:          # the log says how smooth the picture is
        print(f"drawing {fps_count['n'] / (time.time() - fps_count['t']):.1f} frames/s", flush=True)
        fps_count.update(n=0, t=time.time())
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    screen.blit(pygame.transform.scale(surf, PIC), (0, 0))
    lines = hud_lines + ([(extra, (255, 255, 0))] if extra else [])
    if args.stream:
        screen.fill((0, 0, 0), (0, PIC[1], PIC[0], PANEL))
        y = PIC[1] + 4
        for text, colour in lines:
            for part in textwrap.wrap(text, 62) or ['']:
                if y + 18 <= PIC[1] + PANEL:
                    screen.blit(font.render(part, True, colour), (8, y))
                y += 19
    else:
        for i, (text, colour) in enumerate(lines):
            screen.blit(font.render(text, True, colour), (8, 8 + 22 * i))
    pygame.display.flip()
    if args.stream and (force or fps_count['n'] % 2 == 0):
        audio.frame(pygame.image.tostring(screen, 'RGB'))


def show_text(text):
    """A status line between runs: on a black window, or streaming, under the
    last picture (the phone keeps seeing the game)."""
    if args.stream and last_img is not None:
        return draw(last_img, extra=text, force=True)
    screen.fill((0, 0, 0))
    screen.blit(font.render(text, True, (255, 255, 0)), (8, 8))
    pygame.display.flip()
    if args.stream:
        audio.frame(pygame.image.tostring(screen, 'RGB'))


skip_ss = args.skip_ss                              # S: special stages run off-screen
stage = {'on': False, 'n': 0}


def stage_keys():
    """Keys during a special stage: the whole stage is one agent step, so the
    episode loop can't see them. S and M act now, the rest wait for it."""
    global skip_ss, sound
    later = []
    for ev in pygame.event.get():
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
            skip_ss = not skip_ss
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_m and audio is not None:
            sound = not sound
        else:
            later.append(ev)
    for ev in later:
        pygame.event.post(ev)


def step_shown(*a, **kw):
    """The emulator's step, plus that frame on screen and its sound. The env
    steps 4 frames per agent step, and a special stage (MG_SS=pilot) thousands
    in one agent step, all through here, so every frame is seen and heard
    (drawing once per agent step gave 15 fps). At 100% with sound the sound
    card keeps the time, as in the recorders. Only this viewer patches it."""
    out = _base_step(*a, **kw)
    if stepping and stage['on']:
        stage['n'] += 1
        if stage['n'] % 10 == 1:
            stage_keys()
        if skip_ss:                                 # off-screen at full speed
            if audio is not None:
                audio.clear()
            if stage['n'] % 120 == 1:
                show_text('special stage: skipping ...' + ('' if args.stream else '  (S: watch them)'))
            return out
    if stepping:
        draw(env.base_env.em.get_screen())
        if sound and speed == 100:
            audio.push(env.base_env.em.get_audio())
            audio.pace()
        else:
            clock.tick(60 * speed / 100)
    return out


env.base_env.step = step_shown
_special_stage = env._special_stage


def special_stage_shown(info):
    """coordinator's special stage, marked so step_shown can skip it (S)."""
    stage.update(on=True, n=0)
    hud_lines.append(('special stage' + ('' if args.stream else '  (S: skip)'), (255, 255, 0)))
    try:
        return _special_stage(info)
    finally:
        stage['on'] = False


env._special_stage = special_stage_shown


def wait_for_save():
    """--live before the run's first save: a waiting screen; returns False to quit."""
    while True:
        refresh_model()
        if model is not None:
            return True
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return False
        show_text(f'waiting for {args.run}\'s first save ...')
        time.sleep(1)


def set_hud(kk, msg, shown=None):
    hud_lines[:] = [(f"{who[:28]}  {key.split(':')[-1]}  start {seg['fracs'][kk]:.0%}  frames "
                     f"{env.frames}/{env._target}  {speed}%" + ('  PAUSED' if paused else '')
                     + ('' if audio is None else '  sound' if sound and speed == 100 else '  muted'),
                     (255, 255, 0))]
    if args.live:
        hud_lines.append((f"save at {model.num_timesteps:,} steps   "
                          + ('from the act start' if fresh else 'from where it trains now')
                          + ('' if args.stream else ' (F: switch)'), (255, 255, 0)))
    if shown:
        hud_lines.append((shown[3][:80], (255, 255, 0)))
    if msg:
        hud_lines.append((msg[:80], (0, 255, 128)))


def best_try(kk):
    """--best: N sampled tries from start point kk, off-screen; returns (actions, result, progress,
    description) of the best one, or None to quit. Best = the fastest clear, else the furthest
    along his route (then the quickest)."""
    best, outcomes = None, collections.Counter()
    for i in range(args.best):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return None
        show_text(f'try {i + 1} of {args.best} (save at {model.num_timesteps:,} steps) ...')
        obs, _ = env.reset(options={'key': key, 'ckpt': kk})
        acts, done = [], False
        while not done:
            a, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(int(a))
            acts.append(int(a))
            done = term or trunc
        res = info['mg_result']
        outcomes[res['outcome']] += 1
        prog = env._route.best
        score = (1, -res['frames']) if res['outcome'] == 'clear' else (0, prog, -res['frames'])
        entered = res['outcome'] == 'stage' or bool(getattr(env, '_stages', None))
        if entered:
            outcomes['giant ring'] += 1
        if ring and entered:
            score = (2, -res['frames'])
        if best is None or score > best[0]:
            best = (score, acts, res, prog, entered)
    _, acts, res, prog, entered = best
    tally = ', '.join(f'{n} {o}' for o, n in outcomes.most_common() if o != 'stage')
    if res['outcome'] == 'clear':
        desc = f"best of {args.best}: cleared in {res['frames']} frames (you {res['target']})"
    elif ring and entered:
        desc = f"a giant-ring try: {prog / seg['T']:.0%} of the act, then {res['outcome']}"
    else:
        desc = f"best of {args.best}: {prog / seg['T']:.0%} of the act, then {res['outcome']}"
    print(f'{args.best} tries: {tally}; showing {res["outcome"]}', flush=True)
    return acts, res, prog, f'{desc}  ({tally})'


def episode():
    """One episode; returns False to quit."""
    global speed, paused, fresh, sound, stepping, ring
    kk = k
    if args.live:
        if not wait_for_save():
            return False
        kk = live_start()
    replay, shown = None, None
    if args.best and not args.you:
        found = best_try(kk)
        if found is None:
            return False
        replay, shown = list(found[0]), found
    obs, _ = env.reset(options={'key': key, 'ckpt': kk})
    ck = env._ck
    step, outcome, msg = 0, None, ''
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_p:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    return True
                elif ev.key == pygame.K_f and args.live:
                    fresh = not fresh
                    return True
                elif ev.key == pygame.K_m and audio is not None:
                    sound = not sound
                elif ev.key == pygame.K_g and args.best:
                    ring = not ring
                    return True
                elif pygame.K_1 <= ev.key <= pygame.K_4:
                    speed = SPEEDS[ev.key - pygame.K_0]
        paced = sound and speed == 100 and not paused and outcome is None
        if audio is not None and not paced:
            audio.clear()                           # slowed, paused, muted or between episodes
        stepped = not paused and outcome is None
        if stepped:
            set_hud(kk, msg, shown)
            stepping = True
            if args.you:                            # your inputs, frame for frame
                f = ck['frame'] + 1 + env.frames
                a = int(demo_actions[f]) if f < len(demo_actions) else 0
                _, info = C.emu_frame(env.base_env, table[a])
                env.frames += 1
                in_level, in_stage, dead, timer_on = C.frame_flags(info)
                if dead:
                    outcome = 'death'
                elif in_level and not timer_on:
                    outcome = 'clear'
                elif in_stage:
                    outcome = 'stage'
            else:
                if replay:                          # --best: the try being shown
                    a = replay.pop(0)
                else:
                    a, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, r, term, trunc, info = env.step(int(a))
                if term or trunc:
                    outcome = info['mg_result']['outcome']
                    if shown and (outcome != shown[1]['outcome'] or env.frames != shown[1]['frames']):
                        print(f"replay differs from the try: {outcome} at {env.frames} vs "
                              f"{shown[1]['outcome']} at {shown[1]['frames']}", flush=True)
            stepping = False
            step += 1
            if outcome:
                margin = env._target - env.frames
                msg = (f'{outcome.upper()} after {env.frames} frames; you took {env._target} '
                       f'from here ({margin:+d})' if outcome == 'clear' else outcome.upper())
                print(msg, flush=True)
                done_at = time.time()
        set_hud(kk, msg, shown)
        if not stepped or outcome:                  # stepped frames were drawn as they ran
            draw(env.render())
            clock.tick(30)
        if outcome and time.time() - done_at > 2.5:
            return True


while episode():
    pass
if audio is not None:
    audio.close()
pygame.quit()
env.close()
