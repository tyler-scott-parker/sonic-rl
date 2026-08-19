"""
watch_ss.py — Watch a checkpoint play one game at a time.

Run manually in a foreground terminal:
    python3 watch_ss.py v5           # a multistage run (see train_ss_multistage.py --list)
    python3 watch_ss.py specialist   # the paused single-board specialist

Ctrl+C or close the window to stop. Creates a single env, reloads the
model whenever the checkpoint file changes, plays one episode at a time.

The observation preset must match the checkpoint's, so it is resolved from the
run name — a mismatch is a shape error, not a silently wrong policy.
"""

import os
import sys
import time
import traceback

SCALE = 3

_which = sys.argv[1] if len(sys.argv) > 1 else 'v5'
if _which == 'specialist':
    os.environ.update(SS_USE_PIXELS='1', SS_PATCH_R='5', SS_DROP_POSITION='0')
    MODEL_PATH = './sonic_ss_model_v2.zip'
    PKL_PATH = './sonic_ss_vecnormalize_v2.pkl'
else:
    _src = open('train_ss_multistage.py').read()
    _ns = {}
    exec(_src[_src.index('ALL_14 ='):_src.index('# ── Shared config')], _ns)
    if _which not in _ns['RUNS']:
        sys.exit(f"unknown run {_which!r}. Known: specialist, {', '.join(_ns['RUNS'])}")
    os.environ.update(_ns['RUNS'][_which]['obs'])
    MODEL_PATH = f'./sonic_ss_multistage_{_which}.zip'
    PKL_PATH = f'./sonic_ss_multistage_{_which}_vecnormalize.pkl'

import numpy as np                                            # noqa: E402
import pygame                                                 # noqa: E402

from stable_baselines3.common.vec_env import (DummyVecEnv,    # noqa: E402
                                              VecNormalize)
from special_stage_env import SpecialStageEnv                 # noqa: E402

try:
    from sb3_contrib import RecurrentPPO                      # noqa: E402
except ImportError:
    raise ImportError("sb3-contrib not installed. Run: pip install sb3-contrib --break-system-packages")

print(f'watching {MODEL_PATH}')


def make_env():
    return SpecialStageEnv()


def get_model_mtime():
    try:
        return os.path.getmtime(MODEL_PATH)
    except FileNotFoundError:
        return None


def load_model(env):
    print(f"[watch] Loading {MODEL_PATH}...")
    try:
        model = RecurrentPPO.load(MODEL_PATH, device='auto')
        print("[watch] Model loaded.")
        return model
    except Exception:
        print("[watch] Failed to load model:")
        traceback.print_exc()
        return None


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Waiting for {MODEL_PATH} to appear...")
        while not os.path.exists(MODEL_PATH):
            time.sleep(10)

    # Single env — no SubprocVecEnv, no extra windows
    env = DummyVecEnv([make_env])
    if os.path.exists(PKL_PATH):
        env = VecNormalize.load(PKL_PATH, env)
        env.training = False
        env.norm_reward = False
        print(f"[watch] Loaded VecNormalize from {PKL_PATH}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)

    model = load_model(env)
    last_mtime = get_model_mtime()

    pygame.init()
    screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
    pygame.display.set_caption("SS Watcher")
    clock = pygame.time.Clock()

    obs = env.reset()
    lstm_states   = None
    episode_start = np.array([True])
    running = True

    print("[watch] Running. Close window or Ctrl+C to stop.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Reload model if checkpoint updated
        current_mtime = get_model_mtime()
        if current_mtime != last_mtime and model is not None:
            new_model = load_model(env)
            if new_model is not None:
                model = new_model
                last_mtime = current_mtime
                print("[watch] Model reloaded.")

        if model is None:
            time.sleep(5)
            model = load_model(env)
            continue

        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.array([False])

        obs, reward, done, info = env.step(action)

        # Render from the underlying base_env
        raw_frame = env.envs[0].base_env.render()
        if raw_frame is not None:
            surface = pygame.surfarray.make_surface(raw_frame.swapaxes(0, 1))
            scaled = pygame.transform.scale(surface, (320 * SCALE, 224 * SCALE))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()

        if done[0]:
            obs = env.reset()
            lstm_states   = None
            episode_start = np.array([True])

        clock.tick(60)

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
