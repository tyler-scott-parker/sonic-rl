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

--live reloads the run's save before each episode (the trainer rewrites it
every 100k steps) and starts where training currently starts (its curriculum
start point); F switches between that and the start of the act.

Sound plays at 100% speed (game_audio.py, as in the recorders); M mutes,
--no-sound starts muted.

Keys: 1-4 speed 25/50/100/200 %   P pause   R restart   F fresh/training start   M sound   Esc quit
"""
import os
import sys
import json
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
    from stable_baselines3 import PPO
    path = args.model or train_main.paths(args.run)['zip'] + '.zip'
    if not os.path.exists(path) and not args.live:
        sys.exit(f'no model at {path} (train first, or --you to watch your recording)')
    if not args.live:
        model = PPO.load(path, device='cpu')
fresh = args.fresh                                  # --live: start at the act's start (F toggles)


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

pygame.init()
screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
who = 'your recording' if args.you else os.path.basename(args.model or train_main.paths(args.run)['zip'])
if args.live:
    who = f'{args.run} live'
pygame.display.set_caption(f'watch_main — {key} — {who}')
font = pygame.font.SysFont('monospace', 18, bold=True)
clock = pygame.time.Clock()
speed, paused, running = 100, False, True

audio = None
try:
    import game_audio                               # noqa: E402
    audio = game_audio.GameAudio(env.base_env.em.get_audio_rate())
except Exception as e:                              # no device / no SDL audio: watch in silence
    print(f'no sound: {e!r}', flush=True)
sound = audio is not None and not args.no_sound
stepping = False                                    # sound only while an episode steps (not resets)
_emu_frame = C.emu_frame


def emu_frame_with_sound(base_env, mask):
    """coordinator.emu_frame plus that frame's sound. The env steps 4 frames per
    agent step through it, so every frame is heard; at 100% the sound card
    keeps the time, as in the recorders. Only this viewer patches it."""
    out = _emu_frame(base_env, mask)
    if stepping and sound and speed == 100:
        audio.push(base_env.em.get_audio())
        audio.pace()
    return out


C.emu_frame = emu_frame_with_sound


def wait_for_save():
    """--live before the run's first save: a waiting screen; returns False to quit."""
    while True:
        refresh_model()
        if model is not None:
            return True
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return False
        screen.fill((0, 0, 0))
        screen.blit(font.render(f'waiting for {args.run}\'s first save ...', True, (255, 255, 0)), (8, 8))
        pygame.display.flip()
        time.sleep(1)


def episode():
    """One episode; returns False to quit."""
    global speed, paused, fresh, sound, stepping
    kk = k
    if args.live:
        if not wait_for_save():
            return False
        kk = live_start()
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
                elif pygame.K_1 <= ev.key <= pygame.K_4:
                    speed = SPEEDS[ev.key - pygame.K_0]
        paced = sound and speed == 100 and not paused and outcome is None
        if audio is not None and not paced:
            audio.clear()                           # slowed, paused, muted or between episodes
        if not paused and outcome is None:
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
                a, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, r, term, trunc, info = env.step(int(a))
                if term or trunc:
                    outcome = info['mg_result']['outcome']
            stepping = False
            step += 1
            if outcome:
                margin = env._target - env.frames
                msg = (f'{outcome.upper()} after {env.frames} frames; you took {env._target} '
                       f'from here ({margin:+d})' if outcome == 'clear' else outcome.upper())
                print(msg, flush=True)
                done_at = time.time()
        surf = pygame.surfarray.make_surface(env.render().swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (320 * SCALE, 224 * SCALE)), (0, 0))
        hud = (f"{who[:28]}  {key.split(':')[-1]}  start {seg['fracs'][kk]:.0%}  frames "
               f"{env.frames}/{env._target}  {speed}%" + ('  PAUSED' if paused else '')
               + ('' if audio is None else '  sound' if sound and speed == 100 else '  muted'))
        if args.live:
            hud2 = (f"save at {model.num_timesteps:,} steps   "
                    + ('from the act start (F: training start)' if fresh
                       else 'from where it trains now (F: act start)'))
            screen.blit(font.render(hud2, True, (255, 255, 0)), (8, 30))
        screen.blit(font.render(hud, True, (255, 255, 0)), (8, 8))
        if msg:
            screen.blit(font.render(msg[:80], True, (0, 255, 128)), (8, 52 if args.live else 30))
        pygame.display.flip()
        if paced:
            clock.tick()                            # the sound card kept the time
        else:
            clock.tick(60 * speed / 100 / (1 if args.you else C.FRAME_SKIP))
        if outcome and time.time() - done_at > 2.5:
            return True


while episode():
    pass
if audio is not None:
    audio.close()
pygame.quit()
env.close()
