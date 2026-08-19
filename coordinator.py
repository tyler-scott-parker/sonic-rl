import gymnasium as gym
import numpy as np
import cv2
import stable_retro
import json
import os
import time
from gymnasium import spaces
from collections import deque

# Correct button order: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z

CREDITS_ZONE = 23
DOOMSDAY_ZONE = 11

ACT_COMPLETION_BASE     = 5000.0
ACT_TIME_BONUS_MAX      = 2000.0
ACT_MAX_STEPS           = 30000
GAME_COMPLETION_BONUS   = 50000.0

CHAOS_EMERALD_REWARD    = 10000.0
SUPER_EMERALD_REWARD    = 15000.0
GIANT_RING_REWARD       = 2000.0

SS_COMPLETION_BASE      = 2000.0
SS_SPHERE_REWARD        = 2000.0   # reward per blue sphere collected
SS_RED_SPHERE_BASE      = 2000.0   # base penalty for touching a red sphere
SS_RED_SPHERE_SCALE     = 2000.0   # additional penalty per blue sphere already collected
SS_TIME_BONUS_MAX       = 300.0
SS_MAX_STEPS            = 18000

PROGRESS_SCALE          = 2.0
SPEED_SCALE             = 0.1
MAX_X_VELOCITY          = 16.0

RING_GAIN_REWARD        = 1.0
RING_LOSS_PENALTY       = 0.0

DEATH_PENALTY           = 200.0
TIME_PENALTY            = 0.1

# Transformation rewards
SUPER_SONIC_REWARD      = 500.0   # one-time reward for going Super Sonic
HYPER_SONIC_REWARD      = 750.0   # one-time reward for going Hyper Sonic

# Boss rewards
BOSS_HIT_REWARD         = 200.0  # reward per boss hit (via screen_shake rising edge)
BOSS_DEFEAT_REWARD      = 1000.0  # reward when boss is defeated

# Shield reward
SHIELD_REWARD           = 50.0   # one-time reward for picking up any shield

# Environmental context — available in info dict for future use, no penalties applied
# water_flag, water_level, dual_collision: agent infers context visually from pixel frames

# Post-SS progress boost — multiplies progress reward for a window after SS exit
# to pull agent away from the empty ring location toward the goalpost
POST_SS_BOOST_STEPS     = 600       # ~10 seconds at 60fps
POST_SS_BOOST_SCALE     = 5.0       # progress reward multiplier during boost window
STUCK_MAX_PENALTY       = 2.0     # max penalty per frame when stuck
GAME_FPS                = 60.0    # Genesis runs at 60fps

NUM_FRAMES              = 4
MAX_EPISODE_STEPS       = 36000  # force reset if episode runs too long (matches in-game timer)

# Curriculum settings
CURRICULUM_STAGE_FILE   = os.path.expanduser("~/sonic-rl/curriculum_stage.json")
CURRICULUM_STATES       = {
    0: "AngelIslandZone.Act1",                                    # normal spawn
    1: os.path.expanduser("~/sonic-rl/states/curriculum_stage_1.state"),  # special stage spawn
    2: os.path.expanduser("~/sonic-rl/states/curriculum_stage_2.state"),  # x~6873 post-wall
}

LOG_FILE = os.path.expanduser("~/sonic-rl/curriculum_log.jsonl")


def _write_log(entry):
    """Write a single JSON line to the log. Small writes are atomic on Linux."""
    line = json.dumps(entry) + "\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except Exception:
        pass


def _load_curriculum_stage():
    """Read current curriculum stage from disk. Defaults to 0 if missing."""
    try:
        with open(CURRICULUM_STAGE_FILE, 'r') as f:
            return int(json.load(f).get('stage', 0))
    except Exception:
        return 0


def make_coordinator(**kwargs):
    return SonicCoordinator(**kwargs)


class SonicCoordinator(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, ss_model=None):
        super().__init__()
        self.render_mode = render_mode
        self.ss_model = ss_model

        self.base_env = stable_retro.make(
            "SonicAndKnuckles3-Genesis-v0",
            render_mode=render_mode if render_mode == "human" else "rgb_array",
        )

        # 4 stacked frames
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(NUM_FRAMES, 84, 84), dtype=np.uint8
        )

        # 9 actions — correct button mapping
        # Button order: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
        self.action_space = spaces.Discrete(9)

        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: no-op
            [1,0,0,0,0,0,0,0,0,0,0,0],  # 1: B (jump)
            [0,0,0,0,0,0,1,0,0,0,0,0],  # 2: left
            [0,0,0,0,0,0,0,1,0,0,0,0],  # 3: right
            [0,0,0,0,0,1,0,0,0,0,0,0],  # 4: down (spin dash charge)
            [0,0,0,0,1,0,0,0,0,0,0,0],  # 5: up
            [1,0,0,0,0,0,1,0,0,0,0,0],  # 6: B + left
            [1,0,0,0,0,0,0,1,0,0,0,0],  # 7: B + right
            [1,0,0,0,1,0,0,0,0,0,0,0],  # 8: B + down (spin dash charge)
        ]

        self.frame_buffer = deque(maxlen=NUM_FRAMES)
        self._reset_state()

    def _reset_state(self):
        self.prev_info = {}
        self.prev_x = None
        self.start_x = None
        self.act_start_step = 0
        self.ss_start_step = 0
        self.total_steps = 0
        self.in_special_stage = False
        self.game_done = False
        self.prev_zone = None
        self.prev_act = None
        self.best_x = None
        self.last_x = None
        self.best_screen_x = None
        self.frame_buffer.clear()
        # Stuck detection using game timer
        self.stuck_timer = 0          # frames spent without x progress
        self.stuck_x_anchor = None    # x position when stuck timer started
        # Boss state
        self.in_boss_fight = False
        self.prev_level_end_bonus = 0
        # Super Sonic state
        self.was_super = False
        self.episode_steps = 0
        # Curriculum state
        self.curriculum_stage = 0
        # Logging state
        self.episode_stuck_xs = []
        self.last_logged_stuck_x = None
        self.death_x = None
        self.episode_start_time = time.time()
        self.max_rings = 0
        self.prev_blue_spheres = None
        self.best_screen_x_log = 0
        self.max_chaos_emeralds = 0
        self.max_super_emeralds = 0
        self.act_completed = False
        self.boss_encountered = False
        self.was_super_log = False
        self.giant_rings_entered = 0
        self.ss_failures = 0
        self.ss_successes = 0
        self.max_spheres_collected = 0
        self.ss_best_streak = 0
        self.ss_current_streak = 0
        self.ss_speed_level = 0
        self.ss_prev_speed_timer = None
        self.post_ss_boost_remaining = 0  # steps remaining in post-SS progress boost

    def _log_episode(self, zone, act):
        entry = {
            "t": round(time.time()),
            "zone": int(zone),
            "act": int(act),
            "best_x": int(self.best_x) if self.best_x is not None else None,
            "best_screen_x": int(self.best_screen_x_log),
            "death_x": int(self.death_x) if self.death_x is not None else None,
            "stuck_xs": self.episode_stuck_xs,
            "steps": self.total_steps,
            "duration_s": round(time.time() - self.episode_start_time, 1),
            "max_rings": int(self.max_rings),
            "max_chaos_emeralds": int(self.max_chaos_emeralds),
            "max_super_emeralds": int(self.max_super_emeralds),
            "act_completed": self.act_completed,
            "boss_encountered": self.boss_encountered,
            "was_super": self.was_super_log,
            "giant_rings_entered": self.giant_rings_entered,
            "ss_failures": self.ss_failures,
            "ss_successes": self.ss_successes,
            "max_spheres_collected": self.max_spheres_collected,
            "ss_best_streak": self.ss_best_streak,
            "ss_speed_level": self.ss_speed_level,
        }
        _write_log(entry)

    def _preprocess(self, obs):
        frame = obs[32:208, :, :]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        return frame

    def _get_stacked_obs(self, frame):
        self.frame_buffer.append(frame)
        while len(self.frame_buffer) < NUM_FRAMES:
            self.frame_buffer.appendleft(frame)
        return np.array(self.frame_buffer, dtype=np.uint8)

    def _is_super_sonic_draining(self, super_sonic_flag):
        """Use direct RAM flag instead of heuristic ring drain detection."""
        return super_sonic_flag != 0

    def _compute_reward(self, info):
        reward = 0.0

        zone             = info.get('zone', 0)
        act              = info.get('act', 0)
        x                = info.get('x', 0)
        y                = info.get('y', 0)
        rings            = info.get('rings', 0)
        score            = info.get('score', 0)
        lives            = info.get('lives', 3)
        chaos_em         = info.get('chaos_emeralds', 0)
        super_em         = info.get('super_emeralds', 0)
        special          = info.get('special_stage', 0)
        x_end            = info.get('screen_x_end', 24576)
        boss_flag        = info.get('boss_flag', 0)
        level_end_bonus  = info.get('level_end_bonus', 0)
        level_started    = info.get('level_started', 0)
        super_sonic      = info.get('super_sonic', 0)
        shield           = info.get('shield', 0)
        screen_shake     = info.get('screen_shake', 0)

        prev_rings       = self.prev_info.get('rings', rings)
        prev_lives       = self.prev_info.get('lives', lives)
        prev_chaos       = self.prev_info.get('chaos_emeralds', 0)
        prev_super_em    = self.prev_info.get('super_emeralds', 0)
        prev_special     = self.prev_info.get('special_stage', 0)
        prev_zone        = self.prev_info.get('zone', zone)
        prev_act         = self.prev_info.get('act', act)
        prev_x           = self.prev_info.get('x', x)
        prev_y           = self.prev_info.get('y', y)
        prev_shield      = self.prev_info.get('shield', 0)
        prev_screen_shake = self.prev_info.get('screen_shake', 0)

        # ── Game completion ──────────────────────────────────────────────
        if zone == CREDITS_ZONE and prev_zone != CREDITS_ZONE:
            reward += GAME_COMPLETION_BONUS
            self.game_done = True
            return reward

        # ── Special stage ────────────────────────────────────────────────
        if special == 1:
            if not self.in_special_stage:
                self.in_special_stage = True
                self.ss_start_step = self.total_steps
                self.prev_blue_spheres = None
                self.giant_rings_entered += 1
                self._spheres_collected_this_stage = 0
                self.ss_best_streak = 0
                self.ss_current_streak = 0
                self.ss_speed_level = 0
                self.ss_prev_speed_timer = None
                # Only reward entry if emeralds still remain to collect
                if chaos_em < 7 or super_em < 7:
                    reward += GIANT_RING_REWARD

            blue_spheres = info.get('blue_spheres_remaining', None)

            # Reward each blue sphere collected
            if blue_spheres is not None:
                if self.prev_blue_spheres is not None and blue_spheres < self.prev_blue_spheres:
                    spheres_collected = self.prev_blue_spheres - blue_spheres
                    reward += spheres_collected * SS_SPHERE_REWARD
                    self._spheres_collected_this_stage = getattr(self, '_spheres_collected_this_stage', 0) + spheres_collected
                    if self._spheres_collected_this_stage > self.max_spheres_collected:
                        self.max_spheres_collected = self._spheres_collected_this_stage
                    # Update streak and apply multiplier bonus
                    self.ss_current_streak += spheres_collected
                    streak_bonus = self.ss_current_streak * SS_STREAK_MULTIPLIER * spheres_collected
                    reward += streak_bonus
                    if self.ss_current_streak > self.ss_best_streak:
                        self.ss_best_streak = self.ss_current_streak
                self.prev_blue_spheres = blue_spheres

            # Track speed level via ss_speed_timer resets
            ss_speed_timer = info.get('ss_speed_timer', None)
            if ss_speed_timer is not None and self.ss_prev_speed_timer is not None:
                if ss_speed_timer > self.ss_prev_speed_timer + 100:  # timer reset = speed increase
                    self.ss_speed_level += 1
            self.ss_prev_speed_timer = ss_speed_timer

            reward -= TIME_PENALTY
            return reward

        if self.in_special_stage and special == 0:
            self.in_special_stage = False
            ss_steps = self.total_steps - self.ss_start_step
            emerald_gained = False
            if chaos_em > prev_chaos:
                gained = chaos_em - prev_chaos
                reward += CHAOS_EMERALD_REWARD * gained
                emerald_gained = True
            if super_em > prev_super_em:
                gained = super_em - prev_super_em
                reward += SUPER_EMERALD_REWARD * gained
                emerald_gained = True
            if emerald_gained:
                time_bonus = max(0.0, (SS_MAX_STEPS - ss_steps) / SS_MAX_STEPS) * SS_TIME_BONUS_MAX
                reward += SS_COMPLETION_BASE + time_bonus
                self.ss_successes += 1
            else:
                self.ss_failures += 1
            # Start post-SS progress boost regardless of success/failure —
            # agent should move toward goalpost either way
            self.post_ss_boost_remaining = POST_SS_BOOST_STEPS

        # ── Boss mode ────────────────────────────────────────────────────
        if boss_flag and not self.in_boss_fight:
            self.in_boss_fight = True

        # Track chaos emeralds and boss
        if chaos_em > self.max_chaos_emeralds:
            self.max_chaos_emeralds = chaos_em
        if super_em > self.max_super_emeralds:
            self.max_super_emeralds = super_em
        if boss_flag:
            self.boss_encountered = True
        if super_sonic != 0:
            self.was_super_log = True

        if self.in_boss_fight:
            # Boss defeated — level_end_bonus transitions 0 → nonzero
            if level_end_bonus and not self.prev_level_end_bonus:
                reward += BOSS_DEFEAT_REWARD
                self.in_boss_fight = False
                self.act_completed = True
            # In boss fight: suspend progress reward, just apply time penalty
            # Ring and death rewards still apply below
            self.prev_level_end_bonus = level_end_bonus
            reward -= TIME_PENALTY

            # Still track rings and death during boss
            ring_delta = rings - prev_rings
            draining = self._is_super_sonic_draining(super_sonic)
            if not draining:
                if ring_delta > 0:
                    reward += ring_delta * RING_GAIN_REWARD
                elif ring_delta < 0 and rings > 0:
                    reward += ring_delta * RING_LOSS_PENALTY
            if lives < prev_lives:
                reward -= DEATH_PENALTY
                self.death_x = x

            return reward

        # ── Act completion (level_end_bonus is canonical — fires after score tally) ──
        if level_end_bonus and not self.prev_level_end_bonus and not self.in_boss_fight:
            act_steps = self.total_steps - self.act_start_step
            time_bonus = max(0.0, (ACT_MAX_STEPS - act_steps) / ACT_MAX_STEPS) * ACT_TIME_BONUS_MAX
            reward += ACT_COMPLETION_BASE + time_bonus
            self.act_start_step = self.total_steps
            self.act_completed = True
        # Reset spatial tracking on zone/act change
        if (zone != prev_zone or act != prev_act) and prev_zone is not None:
            self.start_x = x
            self.best_x = x
            self.best_screen_x = None
            self.stuck_timer = 0
            self.stuck_x_anchor = None

        # ── Progress reward (camera scroll) ──────────────────────────────
        screen_x = info.get('screen_x', 0)
        if self.best_screen_x is None:
            self.best_screen_x = screen_x
        if screen_x > self.best_screen_x:
            level_length = x_end - (self.start_x or 0)
            if level_length > 0:
                progress_gained = (screen_x - self.best_screen_x) / level_length
                boost = POST_SS_BOOST_SCALE if self.post_ss_boost_remaining > 0 else 1.0
                reward += progress_gained * PROGRESS_SCALE * boost
            self.best_screen_x = screen_x

        # Count down post-SS boost window
        if self.post_ss_boost_remaining > 0:
            self.post_ss_boost_remaining -= 1

        # Track character x and screen_x for curriculum logging
        if self.best_x is None:
            self.best_x = x
        if x > self.best_x:
            self.best_x = x
        if screen_x > self.best_screen_x_log:
            self.best_screen_x_log = screen_x

        # ── Speed reward (weak rightward bootstrap) ───────────────────────
        x_delta = x - prev_x
        if x_delta > 0:
            speed_ratio = min(x_delta / MAX_X_VELOCITY, 1.0)
            reward += speed_ratio * SPEED_SCALE

        # ── Super/Hyper Sonic transformation reward ─────────────────────
        is_super = (super_sonic != 0)
        if is_super and not self.was_super:
            if super_sonic == 1:
                reward += SUPER_SONIC_REWARD
            elif super_sonic == 255:
                reward += HYPER_SONIC_REWARD
        self.was_super = is_super

        # ── Stuck detection (game-timer based, only after level starts) ───
        if level_started:
            if self.stuck_x_anchor is None:
                self.stuck_x_anchor = x

            if abs(x - self.stuck_x_anchor) > 16:
                # Made meaningful progress — reset stuck timer
                self.stuck_timer = 0
                self.stuck_x_anchor = x
            else:
                self.stuck_timer += 1
                stuck_seconds = self.stuck_timer / GAME_FPS
                if stuck_seconds > STUCK_SECONDS:
                    # Escalate penalty the longer we're stuck
                    progress = min((stuck_seconds - STUCK_SECONDS) / STUCK_SECONDS, 1.0)
                    penalty = STUCK_MAX_PENALTY * progress
                    reward -= penalty
                    # Log stuck position
                    if (self.last_logged_stuck_x is None or
                            abs(x - self.last_logged_stuck_x) > 50):
                        self.episode_stuck_xs.append(int(x))
                        self.last_logged_stuck_x = x

        # ── Ring rewards ─────────────────────────────────────────────────
        ring_delta = rings - prev_rings
        draining = self._is_super_sonic_draining(super_sonic)
        doomsday = (zone == DOOMSDAY_ZONE)

        if not doomsday and not draining:
            if ring_delta > 0:
                reward += ring_delta * RING_GAIN_REWARD
            elif ring_delta < 0 and rings > 0:
                reward += ring_delta * RING_LOSS_PENALTY
        if rings > self.max_rings:
            self.max_rings = rings

        # ── Shield acquisition ───────────────────────────────────────────
        # Reward picking up any shield (fire, lightning, bubble, or basic)
        if shield != 0 and prev_shield == 0:
            reward += SHIELD_REWARD

        # ── Boss hit detection (screen shake) ────────────────────────────
        # screen_shake goes positive on boss hit/bomb impact, negative for earthquakes
        # Rising edge of positive shake = boss hit or explosion
        if screen_shake > 0 and prev_screen_shake == 0 and self.in_boss_fight:
            reward += BOSS_HIT_REWARD

        # ── Death ────────────────────────────────────────────────────────
        if lives < prev_lives:
            reward -= DEATH_PENALTY
            self.death_x = x

        # ── Time penalty ─────────────────────────────────────────────────
        reward -= TIME_PENALTY

        return reward

    def reset(self, seed=None, options=None):
        if self.prev_zone is not None and self.best_x is not None:
            self._log_episode(self.prev_zone, self.prev_act or 0)

        # Load curriculum stage and spawn at appropriate state
        stage = _load_curriculum_stage()
        state_path = CURRICULUM_STATES.get(stage, CURRICULUM_STATES[0])

        if stage > 0 and os.path.exists(state_path):
            # Load custom save state for this curriculum stage
            obs, info = self.base_env.reset()
            with open(state_path, 'rb') as f:
                self.base_env.em.set_state(f.read())
            obs = self.base_env.em.get_screen()
        else:
            obs, info = self.base_env.reset()

        self._reset_state()
        self.curriculum_stage = stage
        # If spawning inside special stage, mark as already entered
        if stage == 1:
            self.in_special_stage = True
            self.ss_start_step = 0
            self._spheres_collected_this_stage = 0
            self.ss_best_streak = 0
            self.ss_current_streak = 0
            self.ss_speed_level = 0
            self.ss_prev_speed_timer = None
        frame = self._preprocess(obs)
        self.frame_buffer.clear()
        stacked = self._get_stacked_obs(frame)
        return stacked, {}

    def step(self, action):
        buttons = self._actions[int(action)]
        obs, _, terminated, truncated, info = self.base_env.step(buttons)
        self.total_steps += 1
        self.episode_steps += 1
        if self.episode_steps >= MAX_EPISODE_STEPS:
            truncated = True

        if self.prev_x is None:
            self.prev_x = info.get('x', 0)
            self.start_x = info.get('x', 0)
            self.best_x = info.get('x', 0)
            self.act_start_step = self.total_steps
            self.prev_zone = info.get('zone', 0)
            self.prev_act = info.get('act', 0)

        reward = self._compute_reward(info)

        # Cache prev_chaos BEFORE overwriting prev_info — the termination check below
        # needs the value from the previous step, not the current one.
        prev_chaos_cached = self.prev_info.get('chaos_emeralds', 0)

        self.prev_x = info.get('x', self.prev_x)
        self.prev_zone = info.get('zone', self.prev_zone)
        self.prev_act = info.get('act', self.prev_act)
        self.prev_info = info

        frame = self._preprocess(obs)
        stacked = self._get_stacked_obs(frame)

        # Red sphere detection: terminated inside special stage without earning an emerald
        if terminated and self.in_special_stage:
            curr_chaos = info.get('chaos_emeralds', 0)
            if curr_chaos == prev_chaos_cached:  # no emerald earned = red sphere hit
                spheres_collected = getattr(self, '_spheres_collected_this_stage', 0)
                # Flat penalty that always makes dying net -2500 regardless of N
                # sphere rewards = N * 2000
                # penalty = 2000 + 2000*N + 500 → net = -2500
                dynamic_penalty = SS_RED_SPHERE_BASE + SS_RED_SPHERE_SCALE * spheres_collected + 500
                reward -= dynamic_penalty
                self.ss_current_streak = 0  # break streak on red sphere

        done = terminated or self.game_done
        return stacked, reward, done, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
