import os
import time
import glob
import traceback
import numpy as np
import cv2
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from coordinator import SonicCoordinator

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
    print("[warn] sb3-contrib not installed — SS handoff disabled")

SCALE = 3
CHECKPOINT_DIR = "./checkpoints/"
INFINITE_LIVES = 5  # lives to restore when agent dies

SS_MODEL_PATH   = "./sonic_ss_model.zip"
SS_PKL_PATH     = "./sonic_ss_vecnormalize.pkl"
IMG_SIZE        = 84
NUM_RAM_FEATURES = 7


def make_env():
    return SonicCoordinator(render_mode="rgb_array")


def get_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_latest_pkl():
    files = glob.glob("./*.pkl")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def make_vec_env(pkl_path=None):
    env = DummyVecEnv([make_env])
    if pkl_path:
        env = VecNormalize.load(pkl_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)
    return env


def load_ss_model():
    """Load the SS RecurrentPPO model if available. Returns model or None."""
    if not HAS_RECURRENT_PPO:
        return None
    if not os.path.exists(SS_MODEL_PATH):
        print(f"[SS] No SS model found at {SS_MODEL_PATH} — handoff disabled")
        return None
    try:
        model = RecurrentPPO.load(SS_MODEL_PATH, device='auto')
        print(f"[SS] Loaded SS model from {SS_MODEL_PATH}")
        return model
    except Exception:
        print("[SS] Failed to load SS model:")
        traceback.print_exc()
        return None


def build_ss_obs(raw_frame, info, speed_level, ss_current_streak):
    """
    Build the Dict observation the SS model expects from a raw RGB frame + info dict.
    Mirrors SpecialStageEnv._get_obs() exactly.
    raw_frame: (H, W, 3) uint8 from base_env.render()
    """
    pixels = cv2.resize(raw_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    blue_remaining = float(info.get('blue_spheres_remaining', 102))
    rings          = float(info.get('ss_rings_collected', 0))
    speed_timer    = float(info.get('ss_speed_timer', 600))
    ss_angle       = float(info.get('ss_angle', 0))
    spheres_done   = 102.0 - blue_remaining

    ram = np.array([
        blue_remaining / 102.0,
        spheres_done   / 102.0,
        rings          / 20.0,
        min(speed_timer / 600.0, 1.0),
        float(speed_level)       / 3.0,
        float(ss_current_streak) / 20.0,
        ss_angle / 32768.0,
    ], dtype=np.float32)

    # RecurrentPPO.predict() expects a batch dimension on each entry
    return {
        'pixels': pixels[np.newaxis],   # (1, 84, 84, 3)
        'ram':    ram[np.newaxis],       # (1, 7)
    }


def watch(checkpoint_path):
    print(f"Loading: {checkpoint_path}")

    pkl_path = get_latest_pkl()
    if pkl_path:
        print(f"Using VecNormalize: {pkl_path}")
    env = make_vec_env(pkl_path)

    try:
        model = PPO.load(checkpoint_path, env=env)
    except ValueError as e:
        if "Action spaces do not match" in str(e):
            print("PKL action space mismatch — retrying without pkl...")
            env.close()
            env = make_vec_env(pkl_path=None)
            try:
                model = PPO.load(checkpoint_path, env=env)
            except Exception:
                print("Failed to load checkpoint:")
                traceback.print_exc()
                env.close()
                return
        else:
            print("Failed to load checkpoint:")
            traceback.print_exc()
            env.close()
            return
    except Exception:
        print("Failed to load checkpoint:")
        traceback.print_exc()
        env.close()
        return

    # Load SS model (optional — degrades gracefully if missing)
    ss_model = load_ss_model()

    pygame.init()
    screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
    pygame.display.set_caption(f"Sonic Agent — {os.path.basename(checkpoint_path)}")
    clock = pygame.time.Clock()

    obs = env.reset()
    running = True

    # ── SS handoff state ──────────────────────────────────────────────────
    in_ss           = False
    lstm_states     = None   # RecurrentPPO hidden state, carried across SS steps
    episode_start   = np.array([True])
    ss_speed_level  = 0
    ss_streak       = 0
    ss_prev_speed_timer = None
    ss_prev_blue    = None
    step_info       = {}     # populated each step; seeded empty for first iteration

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Write infinite lives every frame so terminated never fires from death
        try:
            env.envs[0].base_env.em.set_value('lives', INFINITE_LIVES)
        except Exception:
            pass

        # ── SS model drives action when inside special stage ──────────────
        if in_ss and ss_model is not None:
            raw_frame = env.envs[0].base_env.render()

            # Track speed level (mirrors SpecialStageEnv logic)
            speed_timer = step_info.get('ss_speed_timer', None)
            if speed_timer is not None and ss_prev_speed_timer is not None:
                if speed_timer > ss_prev_speed_timer + 100:
                    ss_speed_level += 1
            ss_prev_speed_timer = speed_timer

            # Track streak (mirrors coordinator.py SS logic)
            blue_remaining = step_info.get('blue_spheres_remaining', None)
            if blue_remaining is not None and ss_prev_blue is not None:
                if blue_remaining < ss_prev_blue:
                    collected = ss_prev_blue - blue_remaining
                    ss_streak += collected
            ss_prev_blue = blue_remaining

            ss_obs = build_ss_obs(raw_frame, step_info, ss_speed_level, ss_streak)

            ss_action, lstm_states = ss_model.predict(
                ss_obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.array([False])

            obs, reward, done, info = env.step(ss_action)
            step_info = info[0] if isinstance(info, (list, tuple)) else info
            special = step_info.get('special_stage', 0)

        else:
            # Main model drives action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step_info = info[0] if isinstance(info, (list, tuple)) else info
            special = step_info.get('special_stage', 0)

        # ── Detect SS entry ───────────────────────────────────────────────
        if special == 1 and not in_ss:
            in_ss           = True
            lstm_states     = None
            episode_start   = np.array([True])
            ss_speed_level  = 0
            ss_streak       = 0
            ss_prev_speed_timer = None
            ss_prev_blue    = None
            print("[SS] Entered special stage — handing off to SS model")

        # ── Detect SS exit ────────────────────────────────────────────────
        if in_ss and special == 0:
            in_ss = False
            lstm_states = None
            ss_streak = 0
            print("[SS] Exited special stage — returning to main model")

        raw_frame = env.envs[0].render()
        if raw_frame is not None:
            surface = pygame.surfarray.make_surface(raw_frame.swapaxes(0, 1))
            scaled = pygame.transform.scale(surface, (320 * SCALE, 224 * SCALE))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    env.close()


def main():
    print("Watching for checkpoints... Press ESC or close window to exit.")
    last_checkpoint = None

    while True:
        latest = get_latest_checkpoint()
        if latest and latest != last_checkpoint:
            last_checkpoint = latest
            watch(latest)
            print("Ready for next checkpoint...")
        else:
            print("Waiting...")
            time.sleep(10)


if __name__ == "__main__":
    main()
