import gymnasium as gym
import numpy as np
import cv2
import stable_retro
import os
import sys
import json
import time
import glob
import random
import re
import zlib
from gymnasium import spaces

# Button order: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z

# The agent's actions as button masks. Module-level so record_demo.py plays with
# exactly these controls: every frame of a human demo is a move the agent has.
ACTION_BUTTONS = [
    [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: no-op
    [1,0,0,0,0,0,0,0,0,0,0,0],  # 1: B (jump)
    [0,0,0,0,0,0,1,0,0,0,0,0],  # 2: left
    [0,0,0,0,0,0,0,1,0,0,0,0],  # 3: right
    [0,0,0,0,1,0,0,0,0,0,0,0],  # 4: up (advance / bumper recovery)
    [1,0,0,0,0,0,1,0,0,0,0,0],  # 5: B + left
    [1,0,0,0,0,0,0,1,0,0,0,0],  # 6: B + right
]

# ─────────────────────────────────────────────────────────────────────────────
# env v2 (2026-09-29) — logic-error fixes, every one checked in the emulator.
# SS_ENV_VERSION picks them (see "Env version" below); v1 reproduces the old
# env exactly, for the checkpoints trained on it.
#   - grid read from the real layout at $F100. v1 read $F000 and added 8 to
#     the row: the same bytes for board rows 0-23, but rows 24-31 wrapped onto
#     zeroed RAM — 284 of the 1328 blue spheres on the 14 boards (and whole
#     red walls) were invisible, and boards 2, 10, 11, 14 start inside them
#   - patch centre = the NEAREST cell. A sphere is collected as Sonic crosses
#     its half-cell boundary (x = ..80); v1 floored, which put the centre a
#     cell behind on S/E headings and a cell ahead on N/W for half of every
#     cell. The heading quadrant is rounded too, so the patch turns halfway
#     through a turn instead of at the start of a right / end of a left one
#   - springs (code 5) get their own grid channel; v1 lumped them in with red
#   - the episode ends ON red contact ($E433 starts) or ON the board clear.
#     v1 waited for special_stage to drop, 651-678 frames later for a death
#     (~1000 for a clear): ~44% of all v4/v5 steps were dead animation, and
#     the death penalty reached the fatal action discounted by gamma^675
#   - a clear pays its emerald reward. v1 compared the emerald count only on
#     the final frame, ~660 frames after it had changed, so it never paid
#   - speed feature scaled by the max speed 8192 (v1: /4096, flat at base)
#   - total_progress dropped (v1 feature 4/8; it is exactly 1 - feature 0)
#
# Verified facts (re-checked 2026-09-29; the old "buffer $F000,
# row=(y+8)%32" only held for board rows 0-23):
#   layout $F100-$F4FF, 32x32 row-major; raw RAM byte-swapped
#   cell of a position: row=((y+$80)>>8)%32, col=((x+$80)>>8)%32
#   codes 0=empty 1=red 2=blue 3=bumper 4=ring 5=spring 10=transitional(->red)
#   heading $E426: $00=N $40=W $80=S $C0=E; a left turn raises it, and
#     ss_next_corner reads +4 for a queued left turn, -4 for a right
#   $E432 jump: $80 = jump, $81 = spring launch
#   $E433 exit-animation timer: starts on red contact, or ~347 frames after a
#     board clear (when the emerald is awarded); special_stage stays 1 until
#     ~650 (death) / ~1000 (clear) frames after the triggering event
#
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

# Paths are relative to this file. The repo moved ~/sonic-rl -> ~/Projects/
# sonic-rl, and every hardcoded ~/sonic-rl path broke without an error.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STATES_DIR  = os.path.join(PROJECT_DIR, 'states')
SS_LOG_FILE = os.path.expanduser(
    os.environ.get("SS_LOG", os.path.join(PROJECT_DIR, "ss_log.jsonl")))

# ── Env version ──────────────────────────────────────────────────────────────
#   SS_ENV_VERSION=2 (default) -> the 2026-09-29 fixes listed in the header.
#   SS_ENV_VERSION=1           -> the pre-fix env, reproduced exactly. Every
#                                 checkpoint trained before 2026-09-29
#                                 (specialist, multistage v1-v5) saw v1 and
#                                 must be run on v1: the shapes differ, and so
#                                 would what a matching shape contains.
#   SS_ENV_VERSION=3           -> v2 + the agent choosing every FRAME_SKIP
#                                 frames + the stall rule (see below). For
#                                 learning from demos: run d2 onward.
#   SS_ENV_VERSION=4           -> v3's stall rule, but ONE CHOICE PER CORNER
#                                 (straight / left / right / jump; see below).
#                                 Run d3 onward.
ENV_VERSION = int(os.environ.get('SS_ENV_VERSION', '2'))
if ENV_VERSION not in (1, 2, 3, 4):
    raise ValueError(f'SS_ENV_VERSION must be 1, 2, 3 or 4, got {ENV_VERSION}')
V2 = ENV_VERSION >= 2       # v3 keeps every v2 fix
V3 = ENV_VERSION >= 3       # v4 keeps the stall rule
V4 = ENV_VERSION >= 4

# ── env v3 (2026-09-30), found by run d1 (per-frame, env v2) ─────────────────
#   - FRAME_SKIP: the agent chooses every 4 frames and the action is held.
#     One tap of left/right anywhere in a cell queues the turn for the next
#     corner, so per-frame choice bought nothing — and it made a straight
#     cell ~12 turn-free choices in a row: d1's policy pressed a turn on 53%
#     of frames and never held a line (0 matches from its first start point
#     in ~2,500 episodes). Demos still replay frame by frame.
#   - STALL_FRAMES: this long without a sphere ends the episode as a death
#     (same penalty). d1 learned to hop around the red board to postpone
#     dying: a median 752 frames alive after its start point, where the demo
#     needed 75. Tyler's stage-1 clear never went more than 257 frames
#     between spheres.
FRAME_SKIP    = 4 if V3 else 1
STALL_FRAMES  = 600

# ── env v4 (2026-10-09), found by run d2 (env v3) ────────────────────────────
#   One choice per corner. On v3 a turn needed one press anywhere in the ~3
#   choices before a corner and straight needed none in all of them; d2's
#   policy never managed that (where only straight was right it went straight
#   6-9% of the time, P(no turn in the window) ~0), so it jumped to go
#   straight — airborne 46% of frames, where it can't collect or turn — and
#   flew past the last spheres (attic/one_off_tools/probe_steering.py).
#   Now each step is one corner: the env taps the chosen button once, then
#   plays on alone until the next decision point. Emulator facts behind it
#   (checked 2026-10-09 by tapping on every frame of two cells):
#   - a queued turn fires in a window at the corner: the first frame Sonic is
#     0-31 units past its centre, along his travel. He then stops and rotates
#     for 16 frames. A tap any time after the previous corner's window queues
#     the turn for the next corner ($E42A ss_next_corner = +4 left / -4 right)
#   - a jump takes off on the frame B is tapped and lasts 32 frames whatever
#     the hold (2 cells at base speed). A turn tapped in the air is kept and
#     fires at the first corner after landing; a jump tapped with or just after
#     a turn CANCELS the turn (so v3's jump+turn actions were plain jumps)
#   - Sonic runs 16 units/frame at base speed (a cell = 256, 16 frames); the
#     patch's cell flips 128 units before a corner — after the previous
#     window, before the next — so a choice made there always lands on the
#     corner the patch is centred on
#   - after a bumper bounce he stops and runs forward again on his own; Up
#     does nothing while running, and at the stage start it skips the ~99
#     frame wait (Tyler pressed it on frames 0-1, so fresh starts do too)
#   - a bumper pushes him back about a cell, and he runs into it again until
#     a turn fires — so the agent must get choices during the bounce, and his
#     motion there runs against heading x speed sign
#   Decision point = the first grounded, moving frame on a cell not yet
#     decided, while still BEFORE its centre along his actual motion (either
#     way, so bumper bounces offer choices too). A cell first reached past its
#     centre (a long landing) is skipped: it goes straight.
#   - the game sometimes skips a frame's logic (a lag frame: nothing moves,
#     input ignored) — and it tends to be the frame right after Sonic crosses
#     into a new cell, i.e. right after a decision point. A one-frame tap there
#     is silently dropped (found replaying Tyler's stage-3 clear: a left turn
#     lost at frame 3339). So the button is held until the game shows it took
#     it — the corner byte reads the turn, or the jump byte is set — up to
#     CORNER_TAP_MAX frames, which ends well before the corner
CORNER_ACTIONS     = ('straight', 'left', 'right', 'jump')
CORNER_TAP         = (None, 2, 3, 1)    # the ACTION_BUTTONS index tapped for each
CORNER_TAP_MAX     = 3                  # frames a tap is held at most (lag frames)
CORNER_MAX_FRAMES  = 240                # a step never runs longer (spring flights)

# ── Observation shape (env-var driven; see OBSERVATION PRESETS below) ───────
#   SS_USE_PIXELS=1  -> include the 84x84x3 pixel stream (pre-v4 obs).
#                       Default 0: dropped. Rationale — pixels are a rotating
#                       absolute view, so the net must LEARN heading invariance
#                       the grid patch already has BY CONSTRUCTION. ~21k values
#                       (99.99% of input volume) spent re-deriving invariance.
#   SS_PATCH_R=N     -> local grid patch radius. 5 => 11x11 (pre-v4),
#                       7 => 15x15 (default; clears the 5-cell jumper distance).
#   SS_DROP_POSITION=1 -> drop the sin/cos absolute-position features entirely
#                       (they are THE memorization handle). Default 1.
#                       Set 0 to restore them (specialist / pre-multistage obs).
#
# OBSERVATION PRESETS (see RUNS in train_ss_multistage.py):
#   specialist : SS_ENV_VERSION=1 SS_USE_PIXELS=1 SS_PATCH_R=5 SS_DROP_POSITION=0
#                -> 15 feats, 4-channel grid, pixels
#   v4 / v5    : SS_ENV_VERSION=1 SS_USE_PIXELS=0 SS_PATCH_R=7 SS_DROP_POSITION=1
#                -> 11 feats, 4-channel grid
#   v6, d1     : SS_ENV_VERSION=2 SS_USE_PIXELS=0 SS_PATCH_R=7 SS_DROP_POSITION=1
#                -> 10 feats, 5-channel grid
#   d2         : SS_ENV_VERSION=3, otherwise as v6 (same obs; 4-frame choices)
#
# NOT reproducible here (deliberate): the dead v1/v2/v3 multistage runs used
# DROP_POSITION=1 with the position slots still present as CONSTANT ZEROS, i.e.
# 15 features of which 4 were dead. That wart is gone — DROP_POSITION now
# removes the slots. Those checkpoints cannot be resumed against this env; they
# are superseded (v3 plateaued ~3.5%, which is why v4 dropped the dead inputs).
#
# The obs shape is part of the checkpoint, so a wrong preset raises a loud shape
# mismatch in RecurrentPPO.load. Feature ORDER is what could fail silently,
# which is why position keeps its historical slot (2-5) when enabled.

USE_PIXELS    = os.environ.get('SS_USE_PIXELS', '0') == '1'
IMG_SIZE      = 84

# The layout lives at $F100-$F4FF. v1 read $F000-$F3FF with the row shifted by
# 8: the same bytes for board rows 0-23, but rows 24-31 wrap onto $F000-$F0FF,
# which is always zero. Kept only so v1 checkpoints see what they were fed.
GRID_BASE     = 0xF100 if V2 else 0xF000
GRID_ROW_OFF  = 0 if V2 else 8
PATCH_R       = int(os.environ.get('SS_PATCH_R', '7'))
PATCH_SIZE    = 2 * PATCH_R + 1
GRID_CHANNELS = 5 if V2 else 4      # v2 adds ch4 = spring

ADDR_SPHERES    = 0xE438
ADDR_X          = 0xE422
ADDR_HEADING    = 0xE426
ADDR_JUMPING    = 0xE432
ADDR_EXIT_TIMER = 0xE433    # was ADDR_FADE: the death / clear exit animation

HEADING_DELTA = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}  # (dx, dy)

# ss_move_speed runs 4096 at the start to 8192 at top speed; v1 divided by
# 4096 and clipped, so the feature read 1.0 for all normal running.
SPEED_SCALE   = 8192.0 if V2 else 4096.0

#   SS_STAGES=1,2        -> comma list of stage state files to sample on fresh
#                           starts (default: single stage_1).
DROP_POSITION = os.environ.get('SS_DROP_POSITION', '1') == '1'
# 11 core features (10 in v2, which drops total_progress); +4 (sin/cos of x,y)
# at indices 2-5 only when position is kept.
NUM_RAM_FEATURES = (11 if DROP_POSITION else 15) - (1 if V2 else 0)
_STAGE_IDS    = [s for s in os.environ.get('SS_STAGES', '').split(',') if s.strip()]


def stage_file(stage):
    return os.path.join(STATES_DIR, f'curriculum_stage_{stage}.state')


STAGE_FILES   = [stage_file(s.strip()) for s in _STAGE_IDS]

# Curriculum config
RESTART_PROB    = float(os.environ.get('SS_RESTART_PROB', '0.25'))
POOL_DIR        = os.path.join(STATES_DIR, 'curriculum_pool')
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

# ── Demonstrations (env v2; recorded with record_demo.py) ────────────────────
# A share of resets start from a state some recorded human run passed through,
# at a checkpoint the trainer's DemoCurriculumCallback moves from the end of the
# demo back toward its start as the agent learns to match or beat the demo from
# there (the "backward" curriculum of Salimans & Chen 2018). The demo's ACTIONS
# are never imitated — only its states are borrowed — so the reward alone
# decides what is good play, and the agent is free to do better than the demo.
#   SS_DEMOS=a.npz,b.npz   -> demo files (the trainer sets this from RUNS)
#   SS_DEMO_PROB=p         -> share of resets that start from a demo
DEMO_FILES      = [p for p in os.environ.get('SS_DEMOS', '').split(',') if p.strip()]
DEMO_PROB       = float(os.environ.get('SS_DEMO_PROB', '0.5'))
DEMO_CKPT_EVERY = 30    # frames between start points (~2 cells at base speed)
DEMO_DECISIONS_EVERY = 2    # v4: decision points between start points (~2 cells)
DEMO_TAIL       = 60    # the latest start point is this many frames before the end
DEMO_BAND       = 4     # starts are drawn from checkpoints [pointer, pointer + band]
if DEMO_FILES and not V2:
    raise ValueError('demos need SS_ENV_VERSION=2')


class SpecialStageEnv(gym.Env):
    """Special stage environment — grid obs + self-curriculum restarts."""

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, state_path=None):
        super().__init__()

        self.state_path = state_path or stage_file(1)
        # A missing state file used to fall through to the game's default
        # start (Angel Island): 11-frame "deaths", no error, nothing learned.
        missing = [f for f in (STAGE_FILES or [self.state_path])
                   if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(
                f'special-stage state file(s) not found: {missing}')

        self.base_env = stable_retro.make(
            game='SonicAndKnuckles3-Genesis-v0',
            use_restricted_actions=stable_retro.Actions.FILTERED
        )

        self.action_space = spaces.Discrete(
            len(CORNER_ACTIONS) if V4 else len(ACTION_BUTTONS))
        self._actions = ACTION_BUTTONS

        obs_spaces = {
            'ram': spaces.Box(low=-np.inf, high=np.inf,
                              shape=(NUM_RAM_FEATURES,),
                              dtype=np.float32),
            'grid': spaces.Box(low=0.0, high=1.0,
                               shape=(PATCH_SIZE, PATCH_SIZE, GRID_CHANNELS),
                               dtype=np.float32),
        }
        if USE_PIXELS:
            obs_spaces['pixels'] = spaces.Box(low=0, high=255,
                                              shape=(IMG_SIZE, IMG_SIZE, 3),
                                              dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        self._ram_swapped = None
        self._original_grids = {}     # stage -> its layout at a fresh start
        self._cur_stage = 0
        self._demos = None            # built from DEMO_FILES at the first reset
        self._demo_ptr = {}           # demo id -> start checkpoint (trainer-set)
        self._replaying = False       # True while _build_demos replays a demo
        self.frame_hook = None        # f(frame) on every emulator frame (watchers)
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

    @staticmethod
    def _cell(x_raw, y_raw):
        """(row, col) of a special-stage position in the _read_grid() array.
        v2 rounds: a sphere is collected as Sonic crosses its half-cell
        boundary, so the nearest cell is the one he is on. v1 floored."""
        if V2:
            return ((y_raw + 0x80) >> 8) % 32, ((x_raw + 0x80) >> 8) % 32
        return ((y_raw >> 8) + GRID_ROW_OFF) % 32, (x_raw >> 8) % 32

    def _at_decision(self, info):
        """v4: is this frame a decision point? (See the env v4 header.) Marks
        the cell decided, so each corner is offered at most once."""
        if self._ram_swapped is None:
            return False
        x, y = int(info.get('ss_x', 0)), int(info.get('ss_y', 0))
        prev, self._prev_xy = self._prev_xy, (x, y)
        if prev is None or int(self._ram()[ADDR_JUMPING]) != 0:
            return False                    # airborne, spring, or a buffered jump
        # travel = the actual move this frame. Not heading x speed sign: in a
        # bumper's push-back the game moves him against both (seen 2026-10-09)
        mx = (x - prev[0] + 4096) % 8192 - 4096
        my = (y - prev[1] + 4096) % 8192 - 4096
        if mx == my == 0:
            return False                    # standing, or rotating at a corner
        cell = self._cell(x, y)
        if cell == self._decided:
            return False
        self._decided = cell
        # signed offset from the cell's centre along the travel, board-wrapped
        ox = (x - (cell[1] << 8) + 4096) % 8192 - 4096
        oy = (y - (cell[0] << 8) + 4096) % 8192 - 4096
        if abs(mx) >= abs(my):
            return ox * mx < 0              # still before the corner
        return oy * my < 0

    @staticmethod
    def _quadrant(heading):
        """0=N 1=W 2=S 3=E. v2 rounds to the nearest quadrant, so a turn
        flips it halfway through either way; v1 floored, which flipped at the
        start of a right turn and at the end of a left one."""
        if V2:
            return ((heading + 0x20) >> 6) & 3
        return (heading >> 6) & 3

    # ── Curriculum pool ──────────────────────────────────────────────────

    def _save_snapshot(self, bucket, total):
        try:
            os.makedirs(POOL_DIR, exist_ok=True)
            name = f"s{bucket:02d}_{os.getpid()}_{int(time.time()*1000) % 10**9}"
            path = os.path.join(POOL_DIR, name + '.state')
            # Meta first: restarts glob *.state, so a visible state always has
            # its meta. stage/stage_total let a restart on a multistage run
            # restore the right board, not assume stage 1's 102 spheres.
            with open(path + '.json', 'w') as f:
                json.dump({'spheres': int(total),
                           'speed_level': int(self.speed_level),
                           'stage': int(self._cur_stage),
                           'stage_total': int(self._stage_total)}, f)
            with open(path, 'wb') as f:
                f.write(self.base_env.em.get_state())
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
        if RESTART_PROB <= 0 or self._replaying:
            return      # nothing reads the pool with restarts off
        # Spheres collected on this board so far, restart-banked ones included.
        # Was `102 - remaining`: on the 37-sphere board that read 65 at sphere 0
        # and filed four junk snapshots at the start of every fresh episode.
        remaining = int(info.get('blue_spheres_remaining', self._stage_total))
        total = int(self._stage_total) - remaining
        while (self._capture_idx < len(CAPTURE_AT)
               and total >= CAPTURE_AT[self._capture_idx]):
            self._save_snapshot(CAPTURE_AT[self._capture_idx], total)
            self._capture_idx += 1

    def _rebuild_converted(self):
        """After a pool restore: cells originally blue that are now neither
        blue nor ring were converted to red by play before the snapshot."""
        original = self._original_grids.get(self._cur_stage)
        if original is None or self._ram_swapped is None:
            return
        grid = self._read_grid()
        mask = ((original == 2) & (grid != 2) & (grid != 4))
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

    # ── Demonstrations ───────────────────────────────────────────────────

    def _snapshot_fields(self):
        """Episode bookkeeping a demo start restores besides the emulator state
        (the rest is re-read from RAM, or is logging only)."""
        return dict(steps=self.steps, speed_level=self.speed_level,
                    prev_speed_timer=self.prev_speed_timer,
                    bumper_hit=self.bumper_hit,
                    prev_bumper_lock=self.prev_bumper_lock,
                    spheres=self._spheres_collected)

    def _build_demos(self):
        """Replay each demo once through this env's own step logic, keeping the
        emulator state before every DEMO_CKPT_EVERY-th frame (compressed, ~60 KB)
        and the bookkeeping after it. Lazy, at the first reset of each worker;
        nothing is written to disk. A replay that doesn't reproduce the recorded
        run raises: a silently different demo would teach wrong targets."""
        import demos as demo_io
        self._demos = []
        self._replaying = True
        try:
            for path in DEMO_FILES:
                actions, meta = demo_io.load_demo(path)
                demo_io.check_compatible(path, meta, sys.modules[__name__])
                self.reset(options={'stage_file': stage_file(meta['stage']),
                                    'replay': True})
                n = len(actions)
                ckpts = []
                decisions = 0
                for j, a in enumerate(actions):
                    if V4:
                        state = self.base_env.em.get_state()
                    else:
                        state = (zlib.compress(self.base_env.em.get_state(), 6)
                                 if j % DEMO_CKPT_EVERY == 0 else None)
                    _, _, done, trunc, info = self._step_frames(int(a), 1)
                    if (done or trunc) and j != n - 1:
                        raise ValueError(f'{path}: replay ended at frame {j}, '
                                         f'the recording ran to {n - 1}')
                    if V4:
                        # v4 starts on a decision point the demo passed, so the
                        # agent's first step chooses that corner. Skipped where
                        # the demo had already queued a turn: "straight" can't
                        # undo it
                        point = (not done and self._at_decision(info)
                                 and int(info.get('ss_next_corner', 0)) == 0)
                        decisions += point
                        state = (zlib.compress(state, 6) if point and
                                 decisions % DEMO_DECISIONS_EVERY == 0 else None)
                    if state is not None and j + 1 <= n - DEMO_TAIL:
                        ckpts.append(dict(state=state, action=int(a),
                                          fields=self._snapshot_fields()))
                final = self._spheres_collected
                if final != meta['spheres']:
                    raise ValueError(f'{path}: replay collected {final} spheres, '
                                     f'the recording {meta["spheres"]}')
                if not ckpts:
                    raise ValueError(f'{path}: too short to start from ({n} '
                                     f'frames; needs more than {DEMO_TAIL})')
                self._demos.append(dict(
                    id=demo_io.demo_id(path), stage=meta['stage'],
                    total=float(meta['stage_total']), final=final, frames=n,
                    ckpts=ckpts))
        finally:
            self._replaying = False

    def _reset_from_demo(self):
        demo = random.choice(self._demos)
        last = len(demo['ckpts']) - 1
        ptr = min(self._demo_ptr.get(demo['id'], last), last)
        k = random.randint(ptr, min(last, ptr + DEMO_BAND))
        ck = demo['ckpts'][k]
        self.base_env.em.set_state(zlib.decompress(ck['state']))
        self._cur_stage = demo['stage']
        self._reset_state()
        # the reset frame is the demo's own next frame, so the episode starts
        # exactly on a state the demo passed through
        frame, _, _, _, info = self.base_env.step(ACTION_BUTTONS[ck['action']])
        f = ck['fields']
        self.steps = f['steps']
        self._last_pickup = self.steps      # the stall clock starts here
        self.speed_level = f['speed_level']
        self.prev_speed_timer = f['prev_speed_timer']
        self.bumper_hit = f['bumper_hit']
        self.prev_bumper_lock = f['prev_bumper_lock']
        self.prev_blue_spheres = info.get('blue_spheres_remaining')
        self._stage_total = demo['total']
        self._from_restart = True
        self._restart_from = f['spheres']
        self._capture_idx = sum(1 for t in CAPTURE_AT if t <= self._restart_from)
        self._rebuild_converted()
        if V4:      # a v4 start point is a decision point: this corner is the agent's
            self._prev_xy = (int(info.get('ss_x', 0)), int(info.get('ss_y', 0)))
            self._decided = self._cell(*self._prev_xy)
        # success from here = collect at least what the demo collected from here
        self._demo_episode = dict(demo=demo['id'], start=k,
                                  frac=f['steps'] / demo['frames'],
                                  target=demo['final'] - f['spheres'])
        return self._get_obs(frame, info), {}

    def set_demo_pointers(self, pointers):
        """Called by the trainer (VecEnv.env_method) as the curriculum moves."""
        self._demo_ptr.update({k: int(v) for k, v in pointers.items()})

    def demo_info(self):
        """Per demo: id, stage, spheres, length, and where each start point sits
        in it (fraction of its frames) — for the trainer's curriculum."""
        if DEMO_FILES and self._demos is None:
            self._build_demos()
        return [dict(id=d['id'], stage=d['stage'], final=d['final'],
                     frames=d['frames'],
                     fracs=[c['fields']['steps'] / d['frames'] for c in d['ckpts']])
                for d in (self._demos or [])]

    # ── Grid telemetry ───────────────────────────────────────────────────

    def _track_grid(self, info):
        ram = self._ram()
        if int(ram[ADDR_EXIT_TIMER]) != 0:
            self._prev_grid = None
            return

        grid = ram[GRID_BASE: GRID_BASE + 0x400].reshape(32, 32).copy()
        self._cur_grid = grid

        x_raw = int(info.get('ss_x', 0))
        y_raw = int(info.get('ss_y', 0))
        self._visited.add(self._cell(x_raw, y_raw))

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
        dx, dy = HEADING_DELTA[self._quadrant(heading)]
        if speed < 0:
            dx, dy = -dx, -dy
        here = self._cell(x_raw, y_raw)
        if V2:
            ahead = ((here[0] + dy) % 32, (here[1] + dx) % 32)
        else:
            ahead = self._cell(x_raw + 128 * dx, y_raw + 128 * dy)
        return (here in self._converted) or (ahead in self._converted)

    # ── Grid observation ─────────────────────────────────────────────────

    def _grid_patch(self, info):
        grid = self._cur_grid if self._cur_grid is not None \
            else self._read_grid()

        cr, cx = self._cell(int(info.get('ss_x', 0)), int(info.get('ss_y', 0)))

        rows = np.arange(cr - PATCH_R, cr + PATCH_R + 1) % 32
        cols = np.arange(cx - PATCH_R, cx + PATCH_R + 1) % 32
        patch = grid[np.ix_(rows, cols)]

        heading = int(self._ram()[ADDR_HEADING])
        k = (4 - self._quadrant(heading)) & 3
        patch = np.rot90(patch, k)

        out = np.zeros((PATCH_SIZE, PATCH_SIZE, GRID_CHANNELS),
                       dtype=np.float32)
        out[..., 0] = (patch == 2)
        out[..., 2] = (patch == 3)
        out[..., 3] = (patch == 4)
        known = (patch == 0) | (patch == 2) | (patch == 3) | (patch == 4)
        if V2:
            # springs are safe — they launch Sonic over several cells — and v1
            # showed them in ch1 alongside red
            out[..., 4] = (patch == 5)
            known |= (patch == 5)
        out[..., 1] = ~known
        return out, heading

    # ── Episode state ────────────────────────────────────────────────────

    def _reset_state(self):
        self.steps = 0
        self._last_pickup = 0            # frame of the last sphere (v3 stall rule)
        self._stalled = False
        self._decided = None             # v4: the cell whose corner was last offered
        self._prev_xy = None             # v4: last frame's position, for the travel
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
        self._demo_episode = None        # set when the episode starts from a demo

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if DEMO_FILES and self._demos is None:
            self._build_demos()
        self.base_env.reset()

        self._cur_stage = 0   # 0 = unknown; the stage pick / restart meta set it

        # options={'stage_file': path}: a fresh start on that stage, no restart
        # or demo draw. 'replay': True also leaves it on the stage's first
        # frame in v4, for replaying a demo's frames
        forced = (options or {}).get('stage_file')
        replay = bool((options or {}).get('replay'))
        if forced is None and self._demos and random.random() < DEMO_PROB:
            return self._reset_from_demo()

        restart_file = None
        if forced is None and self._original_grids and random.random() < RESTART_PROB:
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
            load_path = forced or (random.choice(STAGE_FILES) if STAGE_FILES
                                   else self.state_path)
            # remember which stage this fresh episode is on, for per-stage logging
            m = re.search(r'curriculum_stage_(\d+)', load_path)
            self._cur_stage = int(m.group(1)) if m else 0
            # no exists() guard: a missing file must raise, not play Angel Island
            with open(load_path, 'rb') as f:
                self.base_env.em.set_state(f.read())

        self._reset_state()

        frame, _, _, _, info = self.base_env.step(self._actions[0])
        if self._ram_swapped is None:
            self._detect_swap(info)

        rem = int(info.get('blue_spheres_remaining', 102))
        if restart_file:
            self._from_restart = True
            self._cur_stage = int(meta.get('stage', 0))
            if 'stage_total' in meta:
                # board size and banked depth, as recorded with the snapshot
                self._stage_total = float(meta['stage_total'])
                self._restart_from = max(0, int(meta['stage_total']) - rem)
            else:
                # pre-2026-09-29 snapshot: stage 1 only, a 102-sphere board
                self._restart_from = int(meta.get('spheres', max(0, 102 - rem)))
                # full board size = banked depth + what's left on the restored board
                self._stage_total = float(self._restart_from + rem)
            self.speed_level = int(meta.get('speed_level', 0))
            self._capture_idx = sum(
                1 for t in CAPTURE_AT if t <= self._restart_from)
            self._rebuild_converted()
        else:
            # fresh start: the first observed remaining IS this stage's total
            self._stage_total = float(rem)
            if (self._ram_swapped is not None
                    and self._cur_stage not in self._original_grids):
                self._original_grids[self._cur_stage] = self._read_grid().copy()

        if V4 and not replay:
            frame, info = self._advance_to_decision(frame, info, restart_file is None)
        return self._get_obs(frame, info), {}

    def _advance_to_decision(self, frame, info, fresh):
        """v4: play on to the first decision point, tapping Up on the first
        frame of a fresh start (skips the ~99-frame wait, as Tyler did)."""
        for t in range(CORNER_MAX_FRAMES):
            if t and self._at_decision(info):
                return frame, info
            frame, _, done, trunc, info = self._frame(
                self._actions[4 if fresh and t == 0 else 0])
            if done or trunc:
                raise RuntimeError('episode ended before its first decision point')
        raise RuntimeError(f'no decision point within {CORNER_MAX_FRAMES} frames of reset')

    def _tap_taken(self, action, info):
        """v4: did the game register this corner action? (A lag frame drops it.)"""
        if CORNER_ACTIONS[action] == 'jump':
            return int(self._ram()[ADDR_JUMPING]) != 0
        return int(info.get('ss_next_corner', 0)) == (4 if CORNER_ACTIONS[action] == 'left' else -4)

    def step(self, action):
        if V4:
            return self._step_corner(int(action))
        return self._step_frames(int(action), FRAME_SKIP)

    def _step_corner(self, action):
        """v4: one corner. Press the chosen button until the game takes it (one
        frame, unless a lag frame drops it), then play on until the next
        decision point (or the episode ends, or CORNER_MAX_FRAMES)."""
        tap = CORNER_TAP[action]
        held = tap is not None
        total = 0.0
        for t in range(CORNER_MAX_FRAMES):
            frame, reward, done, truncated, info = self._frame(
                self._actions[tap if held else 0])
            total += reward
            if held and (t + 1 >= CORNER_TAP_MAX or self._tap_taken(action, info)):
                held = False
            if done or truncated or self._at_decision(info):
                break
        info['frames'] = t + 1
        return self._get_obs(frame, info), total, done, truncated, info

    def _step_frames(self, action, n):
        """Hold `action` for up to n frames — FRAME_SKIP for the agent, 1 for
        a demo replay — stopping on the frame the episode ends. With n == 1
        this is exactly the per-frame step of env v1/v2."""
        total = 0.0
        for _ in range(n):
            frame, reward, done, truncated, info = self._frame(self._actions[action])
            total += reward
            if done or truncated:
                break
        return self._get_obs(frame, info), total, done, truncated, info

    def _frame(self, buttons):
        frame, _, game_terminated, game_truncated, info = \
            self.base_env.step(buttons)
        self.steps += 1
        if self.frame_hook is not None:
            self.frame_hook(frame)

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

        if (done or truncated) and self._demo_episode is not None:
            d = self._demo_episode      # read by the trainer's demo curriculum
            info['demo_result'] = dict(
                demo=d['demo'], start=d['start'], target=d['target'],
                gained=self._spheres_collected,
                success=self._spheres_collected >= d['target'],
                cleared=self._board_cleared)

        return frame, reward, done, truncated, info

    # ── Observation ──────────────────────────────────────────────────────

    def _get_obs(self, frame, info):
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

        h_angle = (heading / 256.0) * 2.0 * np.pi

        # Per-stage normalization (stage_total is 102/127/118… set at reset),
        # so feature 0 is correct on every stage, not just stage 1.
        total = self._stage_total if self._stage_total > 0 else 102.0
        total_progress = (total - blue_remaining) / total   # v1 only: 1 - f0

        # Absolute sin/cos position. THIS IS THE MEMORIZATION HANDLE
        # (cross-stage transfer ratio 0.06 with it, 0.75 without), so it is
        # dropped by default and kept only for single-stage specialist runs.
        # NOTE the slot: when kept it occupies indices 2-5, exactly where the
        # pre-v4 specialist checkpoints expect it. Feature ORDER must stay
        # byte-compatible per preset or an old checkpoint loads (shapes match)
        # and reads permuted inputs — a silent failure, unlike a shape mismatch.
        if DROP_POSITION:
            position = []
        else:
            ss_x = float(info.get('ss_x', 0))
            ss_y = float(info.get('ss_y', 0))
            x_angle = (ss_x / 65536.0) * 2.0 * np.pi
            y_angle = (ss_y / 65536.0) * 2.0 * np.pi
            position = [np.sin(x_angle), np.cos(x_angle),   # 2, 3
                        np.sin(y_angle), np.cos(y_angle)]   # 4, 5

        feats = [
            blue_remaining / total,                             #  0
            np.clip(ss_move_speed / SPEED_SCALE, -1.0, 1.0),    #  1
        ] + position + [                                        # (2-5 if kept)
            float(self.speed_level) / 3.0,
            min(ss_speed_timer / 1800.0, 1.0),
        ] + ([] if V2 else [total_progress]) + [
            np.sin(h_angle),
            np.cos(h_angle),
            ss_next_corner / 4.0,
            bumper_lock,
            float(self.bumper_hit),
            jumping,
        ]

        obs = {'ram': np.array(feats, dtype=np.float32), 'grid': grid_patch}
        if USE_PIXELS:
            obs['pixels'] = cv2.resize(frame, (IMG_SIZE, IMG_SIZE),
                                       interpolation=cv2.INTER_AREA)
        return obs

    # ── Logging / reward ─────────────────────────────────────────────────

    def _log_episode(self, success, emerald_type, death_at_sphere=None,
                     own_trail=None, timeout=False, stall=False):
        if self._replaying:
            return          # a demo replay is not an episode
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
        if V3:
            entry["stall"] = stall          # ended by the stall rule, not red
        if self._demo_episode is not None:
            d = self._demo_episode
            entry.update(demo=d['demo'], demo_start=d['start'],
                         demo_frac=round(d['frac'], 4), demo_target=d['target'])
        try:
            with open(SS_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # say so once: a run with no episode log can't be analyzed
            if not getattr(self, '_log_warned', False):
                self._log_warned = True
                print(f'[special_stage_env] WARNING: cannot write episode log '
                      f'{SS_LOG_FILE}: {e!r}', flush=True)

    def _stage_emerald(self):
        """The emerald a clear of this board awards: stages 1-7 are the Sonic 3
        (chaos) boards, 8-14 the S&K (super) boards."""
        if 1 <= self._cur_stage <= 7:
            return 'chaos', CHAOS_EMERALD_REWARD
        if 8 <= self._cur_stage <= 14:
            return 'super', SUPER_EMERALD_REWARD
        return None, 0.0

    def _terminal_v2(self, reward, special):
        """End the episode on the event, not when special_stage drops ~650
        (death) or ~1000 (clear) frames later. The clear is checked first:
        $E433 also starts during the clear's own exit sequence."""
        exiting = (self._ram_swapped is not None
                   and int(self._ram()[ADDR_EXIT_TIMER]) != 0)
        left = special == 0 and self.steps > 10    # fallback, not seen first

        if self._board_cleared:
            emerald_type, emerald_reward = self._stage_emerald()
            time_bonus = max(0.0, (SS_MAX_STEPS - self.steps)
                             / SS_MAX_STEPS) * SS_TIME_BONUS_MAX
            reward += SS_COMPLETION_BASE + time_bonus + emerald_reward
            self._log_episode(success=True,
                              emerald_type=emerald_type or "cleared")
            return reward, True

        # v3: STALL_FRAMES without a sphere counts as touching red
        stalled = (V3 and not (exiting or left)
                   and self.steps - self._last_pickup >= STALL_FRAMES)
        if exiting or left or stalled:
            n = self._spheres_collected   # this-episode only
            penalty = SS_RED_SPHERE_BASE + SS_RED_SPHERE_SCALE * n + 500
            reward -= penalty
            self.ss_current_streak = 0
            self._stalled = stalled
            self._log_episode(success=False, emerald_type=None,
                              death_at_sphere=n,
                              own_trail=None if stalled else self._death_on_trail(),
                              stall=stalled)
            return reward, True

        return reward, False

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
                self._last_pickup = self.steps
                self.ss_current_streak += collected
                if self.ss_current_streak > self.ss_best_streak:
                    self.ss_best_streak = self.ss_current_streak

        self.prev_blue_spheres = blue_spheres

        # Board cleared = remaining hit zero after collecting >=1 this episode.
        # Stage-agnostic, restart-safe win signal: the emerald award lags via
        # the fly-off animation and may not credit on warped-in restart states.
        if blue_spheres == 0 and self._spheres_collected > 0:
            self._board_cleared = True

        if V2:
            reward, done = self._terminal_v2(reward, special)

        elif special == 0 and self.steps > 10:
            # v1: waits for the stage to exit, and compares the emerald count
            # against LAST frame's — it changed ~660 frames earlier, so the
            # emerald reward never paid. Kept verbatim for v1 checkpoints.
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
