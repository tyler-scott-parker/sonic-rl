import gymnasium as gym
import numpy as np
import cv2
import stable_retro
import os
import json
import time
import glob
import random
import re
from gymnasium import spaces

# Button order: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z

# ─────────────────────────────────────────────────────────────────────────────
# v4 — wall-targeted curriculum, on top of v3:
#   - CAPTURE_AT extended past 90 (84/88/90/92/94/97/100) so the final
#     ~12 cells finally get snapshotted; the Go-Explore ratchet gets a top.
#   - restart sampling is depth-weighted (RESTART_DEPTH_WEIGHTS) instead of
#     uniform-over-files, which previously drew from the over-stocked shallow
#     buckets and pinned avg restart depth at ~52, never reaching the wall.
#     Weights renormalize over buckets that actually have files, so empty deep
#     buckets cost nothing and the mean depth self-advances as states bank.
#   - RESTART_PROB unchanged (0.25) — ~78% fresh starts preserves early-game
#     skill (forgetting guard) and restart volume at the wall is already ample.
#
# v3 — self-curriculum restart pool (Go-Explore-lite), on top of v2.x
#
# Verified facts (see v2 header history):
#   buffer $F000 row-major 32x32; raw RAM byte-swapped; row=(y+8)%32 col=x
#   codes 0=empty 1=red 2=blue 3=bumper 4=ring, 10=transitional(->red)
#   heading $E426 N/W/S/E quadrants; rot k=(4-(h>>6))&3; fade gate $E433
#
# Curriculum:
#   - Crossing 30/40/.../90 TOTAL spheres snapshots emulator state into
#     POOL_DIR (40 newest kept per bucket, shared across worker processes)
#   - RESTART_PROB of resets load a random pool state instead of stage start
#   - Death penalty scales with THIS-EPISODE collections only (keeps the
#     net-negative-death invariant fair for restart episodes)
#   - obs feature 8 = true total progress from RAM (102 - remaining)
#   - converted-cell set rebuilds via diff against the original layout
#   - episodes log restart/restart_from for analyzer separation
# ─────────────────────────────────────────────────────────────────────────────

SS_SPHERE_REWARD        = 2000.0
SS_RED_SPHERE_BASE      = 2000.0
SS_RED_SPHERE_SCALE     = 2000.0
SS_COMPLETION_BASE      = 2000.0
SS_TIME_BONUS_MAX       = 300.0
SS_MAX_STEPS            = 18000
CHAOS_EMERALD_REWARD    = 10000.0
SUPER_EMERALD_REWARD    = 15000.0
TIME_PENALTY            = 0.1

SS_LOG_FILE = os.path.expanduser("~/sonic-rl/ss_log.jsonl")

NUM_RAM_FEATURES = 15
IMG_SIZE         = 84

GRID_BASE     = 0xF000
PATCH_R       = 5
PATCH_SIZE    = 2 * PATCH_R + 1
GRID_CHANNELS = 4

ADDR_SPHERES  = 0xE438
ADDR_X        = 0xE422
ADDR_HEADING  = 0xE426
ADDR_JUMPING  = 0xE432
ADDR_FADE     = 0xE433

HEADING_DELTA = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}  # (dx, dy)

# Multi-stage / architecture-probe config (env-var controlled so the existing
# resume trainer is byte-for-byte unaffected when these are unset).
#   SS_DROP_POSITION=1   -> zero out ram features 2-5 (sin/cos x,y). Forces the
#                           policy onto the stage-agnostic local grid patch.
#   SS_STAGES=1,2        -> comma list of stage state files to sample on fresh
#                           starts (default: single stage_1, original behavior).
DROP_POSITION = os.environ.get('SS_DROP_POSITION', '0') == '1'
_STAGE_IDS    = [s for s in os.environ.get('SS_STAGES', '').split(',') if s.strip()]
STAGE_FILES   = [os.path.expanduser(f'~/sonic-rl/states/curriculum_stage_{s.strip()}.state')
                 for s in _STAGE_IDS]

# Curriculum config
RESTART_PROB    = float(os.environ.get('SS_RESTART_PROB', '0.25'))
POOL_DIR        = os.path.expanduser('~/sonic-rl/states/curriculum_pool')
CAPTURE_AT      = [30, 40, 50, 60, 70, 80, 84, 88, 90, 92, 94, 97, 100]
POOL_PER_BUCKET = 40

# Skew restart sampling toward the wall. Keys must cover every CAPTURE_AT
# bucket (legacy s90 states included). Renormalizes over buckets that actually
# have files, so empty deep buckets cost nothing — the realized mean depth
# climbs on its own as the agent banks deeper states.
RESTART_DEPTH_WEIGHTS = {
    30: 1, 40: 1, 50: 1, 60: 1, 70: 2,
    80: 4, 84: 6, 88: 9, 90: 11, 92: 13, 94: 14, 97: 14, 100: 14,
}
_BUCKET_RE = re.compile(r's(\d+)_')


class SpecialStageEnv(gym.Env):
    """Special stage environment v3 — grid obs + self-curriculum restarts."""

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, state_path=None):
        super().__init__()

        self.base_env = stable_retro.make(
            game='SonicAndKnuckles3-Genesis-v0',
            use_restricted_actions=stable_retro.Actions.FILTERED
        )

        self.state_path = state_path or os.path.expanduser(
            '~/sonic-rl/states/curriculum_stage_1.state'
        )

        self.action_space = spaces.Discrete(7)
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: no-op
            [1,0,0,0,0,0,0,0,0,0,0,0],  # 1: B (jump)
            [0,0,0,0,0,0,1,0,0,0,0,0],  # 2: left
            [0,0,0,0,0,0,0,1,0,0,0,0],  # 3: right
            [0,0,0,0,1,0,0,0,0,0,0,0],  # 4: up (advance / bumper recovery)
            [1,0,0,0,0,0,1,0,0,0,0,0],  # 5: B + left
            [1,0,0,0,0,0,0,1,0,0,0,0],  # 6: B + right
        ]

        self.observation_space = spaces.Dict({
            'pixels': spaces.Box(low=0, high=255,
                                 shape=(IMG_SIZE, IMG_SIZE, 3),
                                 dtype=np.uint8),
            'ram': spaces.Box(low=-np.inf, high=np.inf,
                              shape=(NUM_RAM_FEATURES,),
                              dtype=np.float32),
            'grid': spaces.Box(low=0.0, high=1.0,
                               shape=(PATCH_SIZE, PATCH_SIZE, GRID_CHANNELS),
                               dtype=np.float32),
        })

        self._ram_swapped = None
        self._original_grid = None
        self._reset_state()

    # ── RAM access ───────────────────────────────────────────────────────

    def _raw_ram(self):
        return np.frombuffer(bytes(self.base_env.get_ram()), dtype=np.uint8)

    @staticmethod
    def _deswap(arr):
        return arr.reshape(-1, 2)[:, ::-1].reshape(-1)

    def _detect_swap(self, info):
        raw = self._raw_ram()
        spheres = int(info.get('blue_spheres_remaining', -1))
        ssx = int(info.get('ss_x', -1))

        def word(a, addr):
            return (int(a[addr]) << 8) | int(a[addr + 1])

        if word(raw, ADDR_SPHERES) == spheres and word(raw, ADDR_X) == ssx:
            self._ram_swapped = False
        else:
            sw = self._deswap(raw)
            if word(sw, ADDR_SPHERES) == spheres and word(sw, ADDR_X) == ssx:
                self._ram_swapped = True
            else:
                self._ram_swapped = None

    def _ram(self):
        raw = self._raw_ram()
        return self._deswap(raw) if self._ram_swapped else raw

    def _read_grid(self):
        return self._ram()[GRID_BASE: GRID_BASE + 0x400].reshape(32, 32)

    # ── Curriculum pool ──────────────────────────────────────────────────

    def _save_snapshot(self, bucket, total):
        try:
            os.makedirs(POOL_DIR, exist_ok=True)
            name = f"s{bucket:02d}_{os.getpid()}_{int(time.time()*1000) % 10**9}"
            path = os.path.join(POOL_DIR, name + '.state')
            with open(path, 'wb') as f:
                f.write(self.base_env.em.get_state())
            with open(path + '.json', 'w') as f:
                json.dump({'spheres': int(total),
                           'speed_level': int(self.speed_level)}, f)
            files = sorted(
                glob.glob(os.path.join(POOL_DIR, f's{bucket:02d}_*.state')),
                key=os.path.getmtime)
            for old in files[:-POOL_PER_BUCKET]:
                try:
                    os.remove(old)
                    os.remove(old + '.json')
                except OSError:
                    pass
        except Exception:
            pass

    def _maybe_capture(self, info):
        total = 102 - int(info.get('blue_spheres_remaining', 102))
        while (self._capture_idx < len(CAPTURE_AT)
               and total >= CAPTURE_AT[self._capture_idx]):
            self._save_snapshot(CAPTURE_AT[self._capture_idx], total)
            self._capture_idx += 1

    def _rebuild_converted(self):
        """After a pool restore: cells originally blue that are now neither
        blue nor ring were converted to red by play before the snapshot."""
        if self._original_grid is None or self._ram_swapped is None:
            return
        grid = self._read_grid()
        mask = ((self._original_grid == 2) & (grid != 2) & (grid != 4))
        self._converted = set(
            (int(r), int(c)) for r, c in np.argwhere(mask))

    def _weighted_pick(self, files):
        """Pick a restart state, biased toward deep buckets. Groups files by
        their s<NN> bucket, samples a bucket by RESTART_DEPTH_WEIGHTS (over the
        buckets that exist), then a uniform file within it. Falls back to
        uniform if no filename parses."""
        by_bucket = {}
        for f in files:
            m = _BUCKET_RE.search(os.path.basename(f))
            if m:
                by_bucket.setdefault(int(m.group(1)), []).append(f)
        if not by_bucket:
            return random.choice(files)
        buckets = list(by_bucket)
        weights = [RESTART_DEPTH_WEIGHTS.get(b, 1) for b in buckets]
        bucket = random.choices(buckets, weights=weights, k=1)[0]
        return random.choice(by_bucket[bucket])

    # ── Grid telemetry ───────────────────────────────────────────────────

    def _track_grid(self, info):
        ram = self._ram()
        if int(ram[ADDR_FADE]) != 0:
            self._prev_grid = None
            return

        grid = ram[GRID_BASE: GRID_BASE + 0x400].reshape(32, 32).copy()
        self._cur_grid = grid

        x_raw = int(info.get('ss_x', 0))
        y_raw = int(info.get('ss_y', 0))
        br = ((y_raw >> 8) + 8) % 32
        bc = (x_raw >> 8) % 32
        self._visited.add((br, bc))

        heading = int(ram[ADDR_HEADING])
        speed = int(info.get('ss_move_speed', 0))
        self._last_pose = (x_raw, y_raw, heading, speed)
        self._last_rings = int(info.get('ss_rings_collected', 0))

        cur_spheres = info.get('blue_spheres_remaining', None)
        drop = 0
        if cur_spheres is not None and self.prev_blue_spheres is not None:
            drop = max(0, int(self.prev_blue_spheres) - int(cur_spheres))

        if self._prev_grid is not None:
            diff = np.where(self._prev_grid != grid)
            if diff[0].size:
                blue_to_ring = 0
                for r, c in zip(diff[0], diff[1]):
                    old, new = int(self._prev_grid[r, c]), int(grid[r, c])
                    if old == 2 and new != 4:
                        self._converted.add((int(r), int(c)))
                    if old == 2 and new == 4:
                        blue_to_ring += 1
                if blue_to_ring and drop:
                    self._loop_events += 1
                    self._loop_spheres += min(blue_to_ring, drop)
        self._prev_grid = grid

    def _death_on_trail(self):
        if self._last_pose is None:
            return None
        x_raw, y_raw, heading, speed = self._last_pose
        dx, dy = HEADING_DELTA[(heading >> 6) & 3]
        if speed < 0:
            dx, dy = -dx, -dy
        br = ((y_raw >> 8) + 8) % 32
        bc = (x_raw >> 8) % 32
        ahead = (((y_raw + 128 * dy) >> 8) % 32 + 8) % 32, \
                ((x_raw + 128 * dx) >> 8) % 32
        return ((br, bc) in self._converted) or (ahead in self._converted)

    # ── Grid observation ─────────────────────────────────────────────────

    def _grid_patch(self, info):
        grid = self._cur_grid if self._cur_grid is not None \
            else self._read_grid()

        cx = (int(info.get('ss_x', 0)) >> 8) % 32
        cr = ((int(info.get('ss_y', 0)) >> 8) + 8) % 32

        rows = np.arange(cr - PATCH_R, cr + PATCH_R + 1) % 32
        cols = np.arange(cx - PATCH_R, cx + PATCH_R + 1) % 32
        patch = grid[np.ix_(rows, cols)]

        heading = int(self._ram()[ADDR_HEADING])
        k = (4 - (heading >> 6)) & 3
        patch = np.rot90(patch, k)

        out = np.zeros((PATCH_SIZE, PATCH_SIZE, GRID_CHANNELS),
                       dtype=np.float32)
        out[..., 0] = (patch == 2)
        out[..., 2] = (patch == 3)
        out[..., 3] = (patch == 4)
        known = (patch == 0) | (patch == 2) | (patch == 3) | (patch == 4)
        out[..., 1] = ~known
        return out, heading

    # ── Episode state ────────────────────────────────────────────────────

    def _reset_state(self):
        self.steps = 0
        self.prev_blue_spheres = None
        self.prev_chaos_em = 0
        self.prev_super_em = 0
        self.ss_current_streak = 0
        self.ss_best_streak = 0
        self._spheres_collected = 0     # THIS episode only (reward accounting)
        self.prev_speed_timer = None
        self.speed_level = 0
        self.prev_bumper_lock = 0
        self.bumper_hit = False
        self.episode_start_time = time.time()
        self._prev_grid = None
        self._cur_grid = None
        self._converted = set()
        self._visited = set()
        self._loop_events = 0
        self._loop_spheres = 0
        self._last_pose = None
        self._last_rings = 0
        self._from_restart = False
        self._restart_from = 0
        self._capture_idx = 0
        self._board_cleared = False
        self._stage_total = 102.0        # per-stage starting sphere count;
                                         # set from RAM at reset (102/127/118…)
        self._cur_stage = 0              # which stage this fresh episode loaded;
                                         # 0 = restart/single-state (not stage-picked)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset()

        restart_file = None
        if self._original_grid is not None and random.random() < RESTART_PROB:
            files = glob.glob(os.path.join(POOL_DIR, '*.state'))
            if files:
                restart_file = self._weighted_pick(files)

        meta = {}
        if restart_file:
            try:
                with open(restart_file, 'rb') as f:
                    self.base_env.em.set_state(f.read())
                try:
                    with open(restart_file + '.json') as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
            except Exception:
                restart_file = None

        if not restart_file:
            # Multi-stage: pick a stage at random for this fresh episode.
            # Falls back to the single state_path when SS_STAGES is unset.
            load_path = random.choice(STAGE_FILES) if STAGE_FILES else self.state_path
            # remember which stage this fresh episode is on, for per-stage logging
            m = re.search(r'curriculum_stage_(\d+)', load_path)
            self._cur_stage = int(m.group(1)) if m else 0
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    self.base_env.em.set_state(f.read())

        self._reset_state()

        frame, _, _, _, info = self.base_env.step(self._actions[0])
        if self._ram_swapped is None:
            self._detect_swap(info)

        if restart_file:
            self._from_restart = True
            rem = int(info.get('blue_spheres_remaining', 102))
            self._restart_from = int(meta.get('spheres', max(0, 102 - rem)))
            self.speed_level = int(meta.get('speed_level', 0))
            self._capture_idx = sum(
                1 for t in CAPTURE_AT if t <= self._restart_from)
            self._rebuild_converted()
            # full board size = banked depth + what's left on the restored board
            self._stage_total = float(self._restart_from + rem)
        elif self._ram_swapped is not None and self._original_grid is None:
            self._original_grid = self._read_grid().copy()

        if not restart_file:
            # fresh start: the first observed remaining IS this stage's total
            self._stage_total = float(info.get('blue_spheres_remaining', 102))

        return self._get_obs(frame, info), {}

    def step(self, action):
        buttons = self._actions[int(action)]
        frame, _, game_terminated, game_truncated, info = \
            self.base_env.step(buttons)
        self.steps += 1

        if self._ram_swapped is None:
            self._detect_swap(info)
        in_stage = int(info.get('special_stage', 1)) != 0
        if self._ram_swapped is not None and in_stage:
            self._track_grid(info)

        reward, done = self._compute_reward(info, game_terminated)

        if self._ram_swapped is not None and in_stage and not done:
            self._maybe_capture(info)

        truncated = self.steps >= SS_MAX_STEPS and not done
        if truncated:
            self._log_episode(success=False, emerald_type=None,
                              death_at_sphere=None, own_trail=None,
                              timeout=True)

        return self._get_obs(frame, info), reward, done, truncated, info

    # ── Observation ──────────────────────────────────────────────────────

    def _get_obs(self, frame, info):
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE),
                                   interpolation=cv2.INTER_AREA)

        if self._ram_swapped is not None:
            grid_patch, heading = self._grid_patch(info)
            jumping = float(self._ram()[ADDR_JUMPING] != 0)
        else:
            grid_patch = np.zeros(
                (PATCH_SIZE, PATCH_SIZE, GRID_CHANNELS), dtype=np.float32)
            heading, jumping = 0, 0.0

        blue_remaining = float(info.get('blue_spheres_remaining', 102))
        ss_next_corner = float(info.get('ss_next_corner', 0))
        ss_move_speed  = float(info.get('ss_move_speed', 0))
        ss_speed_timer = float(info.get('ss_speed_timer', 1800))
        bumper_lock    = float(info.get('bumper_contact', 0))

        ss_x = float(info.get('ss_x', 0))
        ss_y = float(info.get('ss_y', 0))
        x_angle = (ss_x / 65536.0) * 2.0 * np.pi
        y_angle = (ss_y / 65536.0) * 2.0 * np.pi
        h_angle = (heading / 256.0) * 2.0 * np.pi

        # Per-stage normalization (stage_total is 102/127/118… set at reset),
        # so features 0 and 8 are correct on every stage, not just stage 1.
        total = self._stage_total if self._stage_total > 0 else 102.0
        total_progress = (total - blue_remaining) / total   # true, incl. restart

        # Position encoding: full-res sin/cos absolute position is the
        # memorization handle. SS_DROP_POSITION=1 zeroes it to force the policy
        # onto the stage-agnostic local grid patch. Length stays 15 either way.
        if DROP_POSITION:
            sx = cx = sy = cy = 0.0
        else:
            sx, cx = np.sin(x_angle), np.cos(x_angle)
            sy, cy = np.sin(y_angle), np.cos(y_angle)

        ram = np.array([
            blue_remaining / total,                       #  0
            np.clip(ss_move_speed / 4096.0, -1.0, 1.0),   #  1
            sx,                                           #  2
            cx,                                           #  3
            sy,                                           #  4
            cy,                                           #  5
            float(self.speed_level) / 3.0,                #  6
            min(ss_speed_timer / 1800.0, 1.0),            #  7
            total_progress,                               #  8
            np.sin(h_angle),                              #  9
            np.cos(h_angle),                              # 10
            ss_next_corner / 4.0,                         # 11
            bumper_lock,                                  # 12
            float(self.bumper_hit),                       # 13
            jumping,                                      # 14
        ], dtype=np.float32)

        return {'pixels': frame_resized, 'ram': ram, 'grid': grid_patch}

    # ── Logging / reward ─────────────────────────────────────────────────

    def _log_episode(self, success, emerald_type, death_at_sphere=None,
                     own_trail=None, timeout=False):
        entry = {
            "t":               round(time.time()),
            "steps":           self.steps,
            "duration":        round(time.time() - self.episode_start_time, 1),
            "spheres":         self._spheres_collected,
            "best_streak":     self.ss_best_streak,
            "speed_level":     self.speed_level,
            "bumper_hit":      self.bumper_hit,
            "success":         success,
            "emerald":         emerald_type,
            "death_at_sphere": death_at_sphere,
            "timeout":         timeout,
            "own_trail_death": own_trail,
            "loops":           self._loop_events,
            "loop_spheres":    self._loop_spheres,
            "coverage":        len(self._visited),
            "rings":           self._last_rings,
            "restart":         self._from_restart,
            "restart_from":    self._restart_from,
            "stage":           getattr(self, "_cur_stage", None),
            "stage_total":     int(self._stage_total),
        }
        try:
            with open(SS_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _compute_reward(self, info, game_terminated):
        reward = -TIME_PENALTY

        bumper_lock = info.get('bumper_contact', 0)
        if bumper_lock > 0 and self.prev_bumper_lock == 0:
            self.bumper_hit = True
        self.prev_bumper_lock = bumper_lock
        done = False

        blue_spheres = info.get('blue_spheres_remaining', None)
        chaos_em = info.get('chaos_emeralds', 0)
        super_em = info.get('super_emeralds', 0)
        special = info.get('special_stage', 1)
        speed_timer = info.get('ss_speed_timer', None)

        if speed_timer is not None and self.prev_speed_timer is not None:
            if speed_timer > self.prev_speed_timer + 100:
                self.speed_level += 1
        self.prev_speed_timer = speed_timer

        if blue_spheres is not None and self.prev_blue_spheres is not None:
            if blue_spheres < self.prev_blue_spheres:
                collected = self.prev_blue_spheres - blue_spheres
                reward += collected * SS_SPHERE_REWARD
                self._spheres_collected += collected
                self.ss_current_streak += collected
                if self.ss_current_streak > self.ss_best_streak:
                    self.ss_best_streak = self.ss_current_streak

        self.prev_blue_spheres = blue_spheres

        # Board cleared = remaining hit zero after collecting >=1 this episode.
        # Stage-agnostic, restart-safe win signal: the emerald award lags via
        # the fly-off animation and may not credit on warped-in restart states.
        if blue_spheres == 0 and self._spheres_collected > 0:
            self._board_cleared = True

        if special == 0 and self.steps > 10:
            emerald_gained = False
            emerald_type = None
            if chaos_em > self.prev_chaos_em:
                reward += CHAOS_EMERALD_REWARD * (chaos_em - self.prev_chaos_em)
                emerald_gained = True
                emerald_type = "chaos"
            if super_em > self.prev_super_em:
                reward += SUPER_EMERALD_REWARD * (super_em - self.prev_super_em)
                emerald_gained = True
                emerald_type = "super"

            if self._board_cleared or emerald_gained:
                time_bonus = max(0.0, (SS_MAX_STEPS - self.steps)
                                 / SS_MAX_STEPS) * SS_TIME_BONUS_MAX
                reward += SS_COMPLETION_BASE + time_bonus
                self._log_episode(success=True,
                                  emerald_type=emerald_type or "cleared")
            else:
                n = self._spheres_collected   # this-episode only
                penalty = SS_RED_SPHERE_BASE + SS_RED_SPHERE_SCALE * n + 500
                reward -= penalty
                self.ss_current_streak = 0
                self._log_episode(success=False, emerald_type=None,
                                  death_at_sphere=n,
                                  own_trail=self._death_on_trail())

            done = True

        self.prev_chaos_em = chaos_em
        self.prev_super_em = super_em

        return reward, done

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
