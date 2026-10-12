"""
coordinator.py — the base game: Sonic 3 & Knuckles acts, learned from
Tyler's recorded play and improved on. Rebuilt 2026-09-30 (the context doc,
"Base game rebuild", has the why; the old SonicCoordinator could not run).

The name is historical and still the plan: this env is where the special-
stage agent will take over when Sonic jumps into a giant ring. For now a
special or bonus stage ends the episode as a neutral truncation.

EPISODE — one act of a recorded playthrough (record_play.py), started from a
state that run passed through, until:
  clear    the in-game timer stops with Sonic alive in the level: on the
           boss's last hit, act 1 (miniboss) and act 2 alike — seen at every
           clear of Tyler's full playthrough (2026-10-10); the signpost and
           capsule come after, on a stopped clock. The timer then holds the
           act's time. Terminated, + CLEAR_BONUS. (LBZ2's timer stops at its
           boss and runs again for the Big Arm fight: part_name makes that
           its own act, 'LBZ2b'.) Doomsday Zone never stops the clock: it
           clears on the first frame of the ending zone ($0D), ~240 frames
           after the final hit (Tyler's recording: hit 21,220, white-out
           21,421, ending 21,463).
  death    Sonic's routine becomes 6 (dying), 8 (restarting) or $C (drowned)
           — on that frame. The lives counter drops ~80 frames later and the
           level reloads ~290 frames later; waiting for either would repeat
           the special-stage env's old mistake (dead frames in every sample,
           a death penalty discounted to nothing). Terminated, - DEATH_PENALTY.
  stage    entered a special stage (giant ring) or a bonus stage (starpost
           stars). Truncated: neutral either way, and the in-game timer
           doesn't run in there. With MG_SS=pilot a special stage is played
           instead and the episode goes on (SPECIAL STAGES below).
  stuck    no new progress along the demo's route for STUCK_FRAMES (boss
           fights excepted, where progress holds on purpose). Truncated. The
           first real run spent 84% of its frames in 47-second timeouts,
           pressed against one wall; cutting them at 10 s buys ~4x the attempts.
  timeout  far slower than the demo from the same start. Truncated.
  (MG_ENV_VERSION=2: reaching new ground — a 128-px block of the level not
  yet visited this episode — counts as progress for `stuck` too, so a route
  the demo never took isn't cut for leaving it. Tyler, 2026-10-10: his play
  is a guide; don't hinder innovation.)

REWARD — frames faster than the demo. Progress is measured along the demo's
own route, in demo frames: each step pays the newest demo frame it has
reached (a ratchet, so going back and forth earns nothing) minus the frames
the step took. At the demo's pace that nets ~0; faster is positive; standing
still costs 1 per frame. For a clear the episode's undiscounted return is
exactly
    (demo frames from the start point to the clear) - (agent frames) + CLEAR_BONUS
so improving on the demo is literally the objective, and a route the demo
never took is free to win: off the route, progress holds until the agent
rejoins it, then jumps to where it rejoined. Boss fights are gated: while the
boss flag is up, progress holds at the demo's fight start; when the fight
ends it jumps to the demo's fight end, so a faster kill is worth exactly the
frames it saves.

MG_ENV_VERSION=2 changes WHEN the time is paid, never how much: off the
route (and outside boss fights) the step's frames are booked instead of
charged, and the booked frames are settled on the first step back on the
route — together with the progress that rejoining pays — or when the episode
ends. A detour's net (demo frames gained minus frames spent) then lands as
one number when it pays off, instead of a run of -1s now and a lump later
that discounting (gamma 0.995 per step, a ~13 s horizon) mostly erases.
Every episode's undiscounted return is unchanged (check_mg_v2.py).
v2 also breaks ties in the route match by the clock: of the route points
about equally near the agent (within ROUTE_TIE px of the nearest), the one
closest to the last match + the frames since. Where the demo stays in one
place — AIZ2's bombing run, boss fights the game never raises Boss_flag for
(MGZ1, SOZ1, SOZ2, SSZ1, DEZ3) — every point of the arena was visited many
times, and the plain nearest point (the earliest of the ties) froze progress,
so `stuck` cut even the demo's own inputs (check_play_acts.py, 2026-10-10).

ACTIONS — ACTION_BUTTONS, each held for FRAME_SKIP frames.
OBSERVATION — the last STACK agent frames: 84x84 grayscale, each the max of
its step's last two emulator frames (sprite flicker). MG_AIR_BAR=1 (RUNS
'air_bar'; Tyler 2026-10-12: "let's add the air bar") draws Sonic's air
left into each frame's top-right corner (obs_frame): the agent sees 4 frames
and hears nothing, so it can't otherwise tell how long it has been under
water before the drowning digits show (Tyler's own run got down to 9 s of
air in HCZ1 and LBZ2). MG_RING_COMPASS=1 (RUNS 'ring_compass'; Tyler
2026-10-12: "Let's try the compass") draws, under the air bar, an 11x11 box
with one dot where the nearest giant ring of Sonic's zone and internal act
is (from the ROM's object layouts) that he hasn't used this episode: the
needed rings lie 80-250 px off his route, off-screen from it (draw_compass).

SPECIAL STAGES (MG_SS=pilot; default 'truncate' = the line above). On the
entry frame a trained special-stage agent (MG_SS_RUN, default d9; ss_pilot.py)
plays the board through the same env it trained on, attached to this
emulator, starting exactly as it trained (ss_pilot's docstring: the speed-up
timer is matched to the board's captured state). When the game has ended the
stage, no input until it has shown the results, put Sonic back at the ring and
the act's clock runs again; then the episode goes on from there, one long
step. With MG_SS_ROUTES=1 (default) a board the route book has a route for
(route_book.py: the fastest clear found so far, checked to replay from the
main game) is played by replaying that route instead, a guaranteed emerald
(Tyler 2026-10-11: "the main game agent is guaranteed an emerald each time");
all 14 boards have one, Tyler's own recordings to begin with.
The agent pays for it in frames. MG_SS_TIME=refund (default; Tyler
2026-10-11: "Winning emeralds should give a reward. That reward should offset
the time penalty for entering the ring"): every frame from the ring to the act
clock running again is charged, and paid back in full when an emerald is won,
so a won stage costs nothing and a lost one its whole time. =real: charged,
never paid back; =game: only the frames outside the stage (~50 of the reload),
as the act clock counts. MG_EMERALD_BONUS (default 0) pays frames per emerald
on top: with 0 a ring can only break even, so the agent takes rings only where
the special-stage agent nearly always wins. MG_SS_DETOUR=refund (Tyler
2026-10-12: "I don't want to discourage it from finding big rings ... I'd
like the agent to get all 14 emeralds and beat the secret final stage"; the
default 'charge' is the old rule): when an emerald is won, the trip to the
ring and back is free too: the frames from the agent's last progress along
his route before the ring (refunded at the stage), and from the stage until
its next progress (refunded when that progress comes), each capped at
DETOUR_CAP. Time after the stage is only refunded on getting back: AIZ1's
ring sits in a pocket that has to be jumped out of, and a refund paid frame
by frame made standing there free (every mg1bc/mg2 stage episode, 82 of 82,
camped there until `stuck`; Tyler 2026-10-12: "sonic appears to be camping
on the spot where the big ring used to be"). (The AIZ1 ring is 241 px below his route, past the
96-px route radius, so the trip is off-route time: mg1's episodes with a
stage had a median 1,240 off-route frames against 296 without.) A lost
stage's trip is charged as before. Stage frames don't run the
episode's timeout clock (a stage isn't a hopeless attempt); they do count in
its frames. Logged per episode as 'stages'. Bonus stages are always truncated.
With all 7 Chaos Emeralds, a giant ring in a Sonic & Knuckles zone sends Sonic
to Hidden Palace instead (entry flag 2, zone $17): the pilot walks to a gray
Super Emerald and jumps on it, which starts that emerald's board. Which one:
MG_SS_SUPER=auto (default) takes the most Super Emeralds per frame
(ss_pilot.choose_super: the shortest route in the book; a board without one
by the special-stage agent's own results, learned as they come in), or a
list of boards in order of preference; afterwards the game returns to the act
at the ring. (Doomsday Zone, after DEZ3 with all 14 emeralds, has no recording
yet, so no act to train.)

RAM facts (checked against the S3K disassembly, skdisasm, and in the
emulator on 2026-09-30):
  Player_1 object at $B000: routine +5 (2 control, 4 hurt, 6 dying,
    8 restarting, $C drowned), x +$10, y +$14, status +$2A
  Game_mode $F600: $0C level, $8C level loading, $34 special stage
  Update_HUD_timer $FE1E: 0 = timer stopped; Timer $FE22 (min/sec/frame at
    $FE23/24/25)
  Special_bonus_entry_flag $FE48: 1 special stage, 2 bonus stage (the
    integration calls it special_stage)
  Boss_flag $F7AA; Apparent_act $EE4F (what the title card says);
    Current_zone/act $FE10/$FE11 (the loaded level; AIZ switches its internal
    act before the act-1 results, with a -$2F00 shift in x)
  data.json mislabels three variables — level_end_bonus ($FAA8) is an
  unnamed byte (End_of_level_flag is $FAAA), shield ($FE47) is the starpost
  copy of the status byte, screen_x_end ($EE16) is the camera's current right
  limit, not the level's end. None of them is used here.

Environment variables (the trainer sets them before workers import this):
  MG_DEMOS=a.npz,b.npz  recorded playthroughs (record_play.py)
  MG_ACTS=AIZ1,AIZ2     acts to train on, named as act_name / part_name make
                        them ('AIZ1', 'LRZ3', 'LBZ2b'); empty = every cleared act
  MG_DEMO_PROB=p        share of resets that start on the curriculum's start
                        point; the rest start at the beginning of the act
  MG_LOG=path           episode log (JSON lines)
  MG_TEST_GOAL_X=x      TEST FIXTURE ONLY (smoke_main.py): spawn a signpost
                        when Sonic first passes x, making a short fake act
"""
import os
import json
import time
import zlib
import gzip
import random
import hashlib
import math
import struct
import collections

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import stable_retro

GAME = 'SonicAndKnuckles3-Genesis-v0'
DEFAULT_START = 'AngelIslandZone.Act1'
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Button order: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
# The old table's action 8 was B + UP under a "B + down (spin dash charge)"
# comment, so its agent could never rev a spindash. Down + left/right is left
# out on purpose: the game ignores down while left or right is held (it
# blocks ducking and rolling), except during a spindash — see keys_to_action.
ACTION_BUTTONS = [
    [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: no-op
    [0,0,0,0,0,0,1,0,0,0,0,0],  # 1: left
    [0,0,0,0,0,0,0,1,0,0,0,0],  # 2: right
    [0,0,0,0,0,1,0,0,0,0,0,0],  # 3: down        (duck; roll when moving)
    [0,0,0,0,1,0,0,0,0,0,0,0],  # 4: up          (look up; CNZ barrels)
    [1,0,0,0,0,0,0,0,0,0,0,0],  # 5: jump        (shield move in the air)
    [1,0,0,0,0,0,1,0,0,0,0,0],  # 6: jump + left
    [1,0,0,0,0,0,0,1,0,0,0,0],  # 7: jump + right
    [1,0,0,0,0,1,0,0,0,0,0,0],  # 8: jump + down (rev a spindash)
]
ACTION_NAMES = ['noop', 'left', 'right', 'down', 'up',
                'jump', 'jump+left', 'jump+right', 'jump+down']

FRAME_SKIP = 4
STACK      = 4
OBS_SIZE   = 84

# Reward, in frames (one frame = 1/60 s of the demo's time), then scaled.
CLEAR_BONUS    = 600.0     # a clear is worth 10 s on top of the time saved
DEATH_PENALTY  = 1800.0    # ~what a death costs a player: a restart from the
                           # starpost. Risky shortcuts pay only if they save
                           # more than P(death) x 30 s.
REWARD_SCALE   = 0.01

# Matching the demo from a start point = clearing within this much time.
MATCH_SLACK    = 0.25      # x (demo frames from there) ...
MATCH_ADD      = 60        # ... + 1 s
TIMEOUT_MULT   = 2.0       # an episode is cut (truncated) at 2x the demo's
TIMEOUT_ADD    = 1200      # time from its start point + 20 s
STUCK_FRAMES   = 600       # ... or after 10 s with no new route progress. The
                           # demo's own inputs never lag its route by > 40 frames

# Route progress (see RouteProgress)
ROUTE_RADIUS   = 96        # px: within this of the demo's route = on it
ROUTE_BACK     = 300       # demo frames searched behind the current match
ROUTE_AHEAD    = 600       # ... and ahead of it
REACQUIRE      = 8         # steps off the route before searching all of it
FIGHT_GAP      = 60        # boss-flag gaps shorter than this are one fight

# Demo start points
CKPT_EVERY     = 60        # demo frames between start points (1 s)
CKPT_TAIL      = 30        # no start point this close to the clear
# Extra start points between the regular ones, where a hard spot falls inside one
# second (mg1, 2026-10-12: AIZ1's climb from the rock pile onto the collapsing ledge,
# demo t 4,320 -> ~4,362). "ACT:t,t,...;DEMO:ACT:t,...", in demo frames; the regular ones
# stay where they were, so start points before the first extra keep their numbers.
MG_EXTRA_STARTS = {name: sorted(int(t) for t in ts.split(','))
                   for name, ts in (part.rsplit(':', 1) for part in
                                    os.environ.get('MG_EXTRA_STARTS', '').split(';') if part)}
DEMO_BAND      = 4         # a start is drawn from [pointer, pointer + band]

DEMO_FILES  = [p for p in os.environ.get('MG_DEMOS', '').split(',') if p.strip()]
ACTS        = [a for a in os.environ.get('MG_ACTS', '').split(',') if a.strip()]
DEMO_PROB   = float(os.environ.get('MG_DEMO_PROB', '0.75'))
LOG_FILE    = os.path.expanduser(
    os.environ.get('MG_LOG', os.path.join(PROJECT_DIR, 'mg_log.jsonl')))
TEST_GOAL_X = int(os.environ.get('MG_TEST_GOAL_X', '0'))
# 1 = the 2026-09-30 env (run aiz1); 2 = new ground counts against `stuck`,
# and off-route time is settled on return (module docstring)
MG_ENV_VERSION = int(os.environ.get('MG_ENV_VERSION', '1'))
if MG_ENV_VERSION not in (1, 2):
    raise ValueError(f'MG_ENV_VERSION must be 1 or 2, got {MG_ENV_VERSION}')
MG_V2 = MG_ENV_VERSION >= 2
BLOCK_SHIFT = 7          # v2 new ground: 128x128-px blocks
ROUTE_TIE = 16           # v2: route points within this many px of the nearest
                         # count as tied; the clock picks among them

# Special stages (giant rings; module docstring, "SPECIAL STAGES"). Bonus
# stages (starpost stars) always end the episode: no agent plays them.
MG_SS        = os.environ.get('MG_SS', 'truncate')    # or 'pilot'
MG_SS_RUN    = os.environ.get('MG_SS_RUN', 'd9')      # the train_ss_multistage run that plays
MG_SS_MODEL  = os.environ.get('MG_SS_MODEL', '')      # its .zip (default: the run's latest save)
MG_SS_TIME   = os.environ.get('MG_SS_TIME', 'refund')  # or 'real', 'game'
MG_EMERALD_BONUS = float(os.environ.get('MG_EMERALD_BONUS', '0'))   # frames per emerald won, on top
MG_SS_DETOUR = os.environ.get('MG_SS_DETOUR', 'charge')    # or 'refund': a won stage's trip is free
DETOUR_CAP = 1200                                          # frames forgiven per leg at most (20 s)
# Hidden Palace: which gray Super Emerald to take: 'auto' (ss_pilot.choose_super:
# most emeralds per frame, learned from results) or boards, most preferred first
MG_SS_SUPER = os.environ.get('MG_SS_SUPER', 'auto')
MG_SS_ROUTES = os.environ.get('MG_SS_ROUTES', '1') == '1'   # replay the route book's routes
if MG_SS not in ('truncate', 'pilot') or MG_SS_TIME not in ('refund', 'real', 'game'):
    raise ValueError(f'MG_SS must be truncate|pilot and MG_SS_TIME refund|real|game, '
                     f'got {MG_SS!r} / {MG_SS_TIME!r}')
RETURN_CAP = 3000        # pilot: the stage's end -> back in the level (results + reload)

# ── RAM ──────────────────────────────────────────────────────────────────────
# Registered on top of the integration's variables (zone, act, x, y, rings,
# lives are the integration's own and correct). Names are prefixed so they
# can't collide with data.json.
RAM_VARS = {
    'mg_routine':      (0xFFB005, '|u1'),   # Player_1+routine
    'mg_status':       (0xFFB02A, '|u1'),   # Player_1+status (bit 1 = in air)
    'mg_spindash':     (0xFFB03D, '|u1'),   # Player_1+spin_dash_flag: 1 while revving
    'mg_game_mode':    (0xFFF600, '|u1'),   # Game_mode
    'mg_hud_timer':    (0xFFFE1E, '|u1'),   # Update_HUD_timer
    'mg_timer_min':    (0xFFFE23, '|u1'),   # Timer_minute
    'mg_timer_sec':    (0xFFFE24, '|u1'),   # Timer_second
    'mg_timer_frame':  (0xFFFE25, '|u1'),   # Timer_frame (0-59)
    'mg_entry':        (0xFFFE48, '|u1'),   # Special_bonus_entry_flag
    'mg_boss':         (0xFFF7AA, '|u1'),   # Boss_flag
    'mg_apparent_act': (0xFFEE4F, '|u1'),   # Apparent_act
    'mg_signpost':     (0xFFFAA6, '>u2'),   # Signpost_addr (test fixture)
    'mg_air':          (0xFFB02C, '|u1'),   # Player_1+air_left: s of air (30 out of water)
}
GM_LEVEL       = 0x0C
ZONE_DDZ       = 0x0C      # Doomsday: no timer stop; cleared when the game moves on
ZONE_ENDING    = 0x0D      # ... to the ending (Tyler's recording, 2026-10-11)
STAGE_MODES    = (0x34, 0x48)              # special stage, its results
DEAD_ROUTINES  = (6, 8, 0xC)

ZONE_NAMES = {0: 'AIZ', 1: 'HCZ', 2: 'MGZ', 3: 'CNZ', 4: 'FBZ', 5: 'ICZ',
              6: 'LBZ', 7: 'MHZ', 8: 'SOZ', 9: 'LRZ', 0xA: 'SSZ', 0xB: 'DEZ',
              0xC: 'DDZ', 0xD: 'END'}
# Internal zones $16 and $17 each hold unrelated stretches of the game (seen in
# Tyler's full playthrough, 2026-10-10; both used to be named 'HPZ'):
ACT_NAMES = {(0x16, 0): 'LRZ3',     # Lava Reef's boss arena; the timer carries on from LRZ2
             (0x16, 1): 'HPZ1',     # Hidden Palace (one act, no number on its title card)
             (0x17, 0): 'DEZ3'}     # the final Death Egg boss; the timer carries on from DEZ2


def act_name(zone, apparent_act):
    """'AIZ1', 'HCZ2', ... — the act the title card shows (internal zones $16
    and $17 by ACT_NAMES)."""
    zone, act = int(zone), int(apparent_act)
    return ACT_NAMES.get((zone, act), f"{ZONE_NAMES.get(zone, f'Z{zone:02X}')}{act + 1}")


def part_name(title, prev):
    """The name of an act whose title card says `title`, when the act cleared
    just before it was `prev`. The same title right after a clear means the
    timer stopped for a cutscene and then ran on in the same act: LBZ2's timer
    stops when its boss is beaten and runs again for the Big Arm fight after
    the Death Egg launches. That second stretch is its own act to learn
    ('LBZ2b'; a third would be 'LBZ2c'), so no two acts of a recording share
    a name — training keys acts by name."""
    if prev is None or prev.rstrip('abcdefghijklmnopqrstuvwxyz') != title:
        return title
    return title + ('b' if prev == title else chr(ord(prev[-1]) + 1))


def make_retro(render_mode=None):
    """The emulator, with the RAM variables above registered. stable_retro
    allows one emulator per process."""
    env = stable_retro.make(game=GAME, render_mode=render_mode,
                            use_restricted_actions=stable_retro.Actions.FILTERED)
    for name, (addr, typ) in RAM_VARS.items():
        env.data.set_variable(name, {'address': addr, 'type': typ})
    return env


def integration_state(name):
    """The bytes of one of the integration's save states ('HydrocityZone.Act1'),
    or of a raw em.get_state() file in the repo ('states/doomsday_start.state':
    Doomsday Zone, reached by granting all 14 emeralds at the start of Tyler's
    DEZ3, which nobody's playthrough reaches otherwise)."""
    if name.endswith('.state'):
        with open(os.path.join(PROJECT_DIR, name), 'rb') as f:
            return f.read()
    path = stable_retro.data.get_file_path(GAME, name + '.state')
    if path is None:
        raise FileNotFoundError(f'no integration state {name!r}')
    with gzip.open(path, 'rb') as f:
        return f.read()


def sha1(data):
    return hashlib.sha1(data).hexdigest()


def process_frame(prev_raw, raw):
    """One observation frame: max of two emulator frames (sprites flicker),
    grayscale, 84x84."""
    img = raw if prev_raw is None else np.maximum(prev_raw, raw)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_AREA)


MG_AIR_BAR = os.environ.get('MG_AIR_BAR', '0') == '1'
AIR_FULL = 30                  # air_left out of water; drowned at 0
AIR_ROWS, AIR_X0 = 2, OBS_SIZE - AIR_FULL      # the bar: top 2 rows, the rightmost 30 columns


def obs_frame(prev_raw, raw, info, used=frozenset()):
    """process_frame, plus (MG_AIR_BAR) Sonic's air as a bar: one white
    pixel per second of air left, black for the rest (the strip covers
    2 x 30 of the 84 x 84), and (MG_RING_COMPASS) the ring compass."""
    f = process_frame(prev_raw, raw)
    if MG_AIR_BAR:
        air = min(max(int(info['mg_air']), 0), AIR_FULL)
        f[:AIR_ROWS, AIR_X0:] = 0
        f[:AIR_ROWS, AIR_X0:AIR_X0 + air] = 255
    if MG_RING_COMPASS:
        draw_compass(f, info, used)
    return f


# ── The ring compass ─────────────────────────────────────────────────────────
MG_RING_COMPASS = os.environ.get('MG_RING_COMPASS', '0') == '1'
RING_TABLE, RING_ID = 0x1E3D98, 0x85   # object layout pointers; object $85 = giant ring
COMPASS_R = 5                          # an 11 x 11 box, top right, under the air bar
COMPASS_Y0, COMPASS_X0 = AIR_ROWS + 1, OBS_SIZE - (2 * COMPASS_R + 1)
COMPASS_RANGE = 768                    # px: rings this far or farther sit on the box's edge
RING_NEAR = 160                        # px: entering a stage this close to a ring uses it


def giant_rings():
    """{(zone, internal act): [(x, y), ...]} of every giant ring, from the ROM's
    object layouts (attic/one_off_tools/giant_rings.py: 77 in all)."""
    path = os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', GAME, 'rom.md')
    rom = open(path, 'rb').read()
    out = collections.defaultdict(list)
    for z in range(13):
        for a in (0, 1):
            p = struct.unpack('>I', rom[RING_TABLE + 4 * (2 * z + a):RING_TABLE + 4 * (2 * z + a) + 4])[0]
            while True:
                x, yf, oid, _ = struct.unpack('>HHBB', rom[p:p + 6])
                if x == 0xFFFF:
                    break
                if oid == RING_ID:
                    out[(z, a)].append((x, yf & 0x0FFF))
                p += 6
    return dict(out)


GIANT_RINGS = giant_rings() if MG_RING_COMPASS else {}


def nearest_ring(info, used=frozenset()):
    """(key, dx, dy) of the nearest giant ring in Sonic's zone and internal act
    not in `used` (keys (zone, act, x, y)), or None."""
    za = (int(info['zone']), int(info['act']))
    x, y = int(info['x']), int(info['y'])
    best = None
    for rx, ry in GIANT_RINGS.get(za, ()):
        key = za + (rx, ry)
        if key in used:
            continue
        d2 = (rx - x) ** 2 + (ry - y) ** 2
        if best is None or d2 < best[0]:
            best = (d2, key, rx - x, ry - y)
    return None if best is None else best[1:]


def draw_compass(f, info, used=frozenset()):
    """The box black with a dim centre (Sonic) and one white dot toward the
    nearest unused ring: exact direction, distance squeezed by a square root so
    the near rings (the ones worth a detour) get most of the box; none = empty."""
    c = (COMPASS_Y0 + COMPASS_R, COMPASS_X0 + COMPASS_R)
    f[COMPASS_Y0:COMPASS_Y0 + 2 * COMPASS_R + 1, COMPASS_X0:COMPASS_X0 + 2 * COMPASS_R + 1] = 0
    f[c] = 96
    near = nearest_ring(info, used)
    if near is None:
        return
    _, dx, dy = near
    dist = math.hypot(dx, dy)
    reach = COMPASS_R * math.sqrt(min(dist, COMPASS_RANGE) / COMPASS_RANGE)
    px, py = (dx / dist * reach, dy / dist * reach) if dist else (0.0, 0.0)
    f[c[0] + int(round(py)), c[1] + int(round(px))] = 255


# ── Test fixture: a fake act end ─────────────────────────────────────────────
# Obj_EndSign lives at $837B2 in this ROM (the disassembly's labels are ROM
# addresses; checked against the bytes). Spawned into a free object slot, it
# drops from the top of the screen and runs the real results sequence, so a
# scripted run to x = MG_TEST_GOAL_X exercises the real clear signal.

OBJ_ENDSIGN = 0x837B2
SLOTTED_OBJECT_BITS = 0xFA9A   # the signpost needs bit 0 free or it deletes
                               # itself (SetUp_ObjAttributesSlotted); another
                               # object may hold it — the fixture takes it


def _ram_words(env, info):
    raw = np.frombuffer(bytes(env.get_ram()), dtype=np.uint8)
    swapped = raw.reshape(-1, 2)[:, ::-1].reshape(-1)
    for ram in (raw, swapped):                     # x and y: an x like $1E1E reads the same both ways
        if ((int(ram[0xB010]) << 8 | int(ram[0xB011])) == int(info['x'])
                and (int(ram[0xB014]) << 8 | int(ram[0xB015])) == int(info['y'])):
            return ram
    raise RuntimeError('cannot locate Player_1 in RAM')


def _spawn_signpost(env, info, x):
    ram = _ram_words(env, info)
    for i in range(3, 110):                 # the dynamic object slots
        base = 0xB000 + i * 0x4A
        if not ram[base:base + 4].any():
            writes = ((0xFF0000 + SLOTTED_OBJECT_BITS, '|u1',
                       int(ram[SLOTTED_OBJECT_BITS]) & 0xFE),
                      (0xFF0000 + base, '>u4', OBJ_ENDSIGN),
                      (0xFF0000 + base + 0x10, '>u2', x))
            for n, (addr, typ, val) in enumerate(writes):
                env.data.set_variable(f't_{n}', {'address': addr, 'type': typ})
                env.data.set_value(f't_{n}', val)
            return
    raise RuntimeError('no free object slot for the test signpost')


def emu_frame(env, mask):
    """Step the emulator one frame. The env, the recorder and the demo replay
    all step through here, so a test fixture can't make them disagree."""
    raw, _, _, _, info = env.step(mask)
    if (TEST_GOAL_X and int(info['x']) >= TEST_GOAL_X
            and int(info['mg_signpost']) == 0
            and int(info['mg_game_mode']) == GM_LEVEL):
        _spawn_signpost(env, info, TEST_GOAL_X + 0x40)
    return raw, info


# ── What a frame means ───────────────────────────────────────────────────────

def frame_flags(info):
    """(in_level, in_stage, dead, timer_on) for the frame just stepped."""
    gm = int(info['mg_game_mode'])
    in_stage = int(info['mg_entry']) != 0 or (gm & 0x7F) in STAGE_MODES
    in_level = gm == GM_LEVEL and not in_stage
    dead = int(info['mg_routine']) in DEAD_ROUTINES
    timer_on = int(info['mg_hud_timer']) != 0
    return in_level, in_stage, dead, timer_on


def game_time(info):
    """The in-game timer, as (minutes, seconds, frames)."""
    return (int(info['mg_timer_min']), int(info['mg_timer_sec']),
            int(info['mg_timer_frame']))


def fmt_time(t):
    m, s, f = t
    return f'{m}:{s:02d}.{f * 100 // 60:02d}'


class ActSplitter:
    """Splits a run into acts, one frame at a time. The recorder, the demo
    replay and the env all use it, so they agree on where every act starts
    and ends.

    An act starts on the first frame the in-game timer runs with Sonic alive
    in the level, and clears on the first frame the timer stops with Sonic
    alive in the level (Doomsday: the first frame of the ending zone). Its
    demo time `t` counts every frame since the start except those in a
    special or bonus stage (the in-game timer doesn't run there either);
    deaths and the reloads after them do count."""

    def __init__(self):
        self.acts = []              # finished (cleared) acts
        self.cur = None             # the act in progress
        self._stage = False
        self._dead = False

    def feed(self, i, info):
        """Frame i (0-based) was just stepped. Returns 'start', 'clear',
        'death', 'stage' or None."""
        in_level, in_stage, dead, timer_on = frame_flags(info)
        event = None
        cur = self.cur
        if cur is None:
            if in_level and timer_on and not dead:
                self.cur = dict(
                    name=part_name(act_name(info['zone'], info['mg_apparent_act']),
                                   self.acts[-1]['name'] if self.acts else None),
                    zone=int(info['zone']), start=i, t=0, clear=None,
                    time=None, deaths=[], stages=[])
                event = 'start'
        else:
            if not in_stage:
                cur['t'] += 1
            if in_stage and not self._stage:
                cur['stages'].append(i)
                event = 'stage'
            if dead and not self._dead:
                cur['deaths'].append(i)
                event = 'death'
            if not dead and ((in_level and not timer_on) or
                             (cur['zone'] == ZONE_DDZ and int(info['zone']) == ZONE_ENDING)):
                cur['clear'] = i
                cur['time'] = fmt_time(game_time(info))
                self.acts.append(cur)
                self.cur = None
                event = 'clear'
        self._stage, self._dead = in_stage, dead
        return event

    def summary(self):
        """Every act so far, the unfinished one last (clear = None)."""
        return [dict(a) for a in self.acts + ([self.cur] if self.cur else [])]


# ── Progress along the demo's route ──────────────────────────────────────────

class RouteProgress:
    """How far along one act of the demo the agent is, in demo frames.

    Each step the agent's position is matched to the nearest point of the
    demo's route (same internal zone/act, within ROUTE_RADIUS) whose demo time
    is within [-ROUTE_BACK, +ROUTE_AHEAD] of the last match; after REACQUIRE
    misses in a row the whole remaining route is searched, so a shortcut is
    paid the moment it rejoins the route. `best` only ever rises.

    Boss fights: a boss arena is covered by the demo's route many times over,
    so matching positions there would pay the fight's time for walking around.
    Instead progress holds at the demo's fight start while the agent's boss
    flag is up, and jumps to the demo's fight end when it drops."""

    def __init__(self, route, t0, boss_on):
        self.r = route
        self.best = self.p = int(t0)
        self.floor, self.cap = 0, int(route['T'])
        self.fight = None
        self.miss = 0
        self.on = True       # matched the route this step
        self.since = 0       # frames since the last match (v2's clock)
        if boss_on:
            self._enter_fight()

    def _enter_fight(self):
        for s, e in self.r['fights']:
            if e > self.best:
                self.fight = (s, e)
                self.cap = max(s, self.best)
                return
        self.fight = (self.best, self.best)     # a fight the demo never had

    def _leave_fight(self):
        _, e = self.fight
        self.fight = None
        self.best = self.p = max(self.best, e)
        self.since = 0
        self.floor, self.cap = e, int(self.r['T'])

    def _nearest(self, lo, hi, zone, act, x, y, clock=None):
        t = self.r['t']
        i0 = int(np.searchsorted(t, lo, 'left'))
        i1 = int(np.searchsorted(t, hi, 'right'))
        if i1 <= i0:
            return None
        sl = slice(i0, i1)
        d2 = (self.r['x'][sl] - x) ** 2 + (self.r['y'][sl] - y) ** 2
        d2 = np.where((self.r['zone'][sl] == zone) & (self.r['act'][sl] == act),
                      d2, np.iinfo(np.int64).max)
        k = int(np.argmin(d2))
        if d2[k] > ROUTE_RADIUS ** 2:
            return None
        if clock is not None:               # v2: of the tied points, the one nearest the clock
            tied = np.flatnonzero(d2 <= (np.sqrt(d2[k]) + ROUTE_TIE) ** 2)
            k = int(tied[np.argmin(np.abs(t[i0 + tied] - clock))])
        return int(t[i0 + k])

    def update(self, zone, act, x, y, boss_on):
        """Returns the demo frames gained this step (>= 0)."""
        before = self.best
        if boss_on and self.fight is None:
            self._enter_fight()
        elif not boss_on and self.fight is not None:
            self._leave_fight()
        args = (int(zone), int(act), int(x), int(y))
        self.since += FRAME_SKIP            # one update per agent step
        clock = self.p + self.since if MG_V2 else None
        t = self._nearest(max(self.floor, self.p - ROUTE_BACK),
                          min(self.cap, self.p + ROUTE_AHEAD), *args, clock=clock)
        if t is None:
            self.miss += 1
            if self.miss >= REACQUIRE:
                t = self._nearest(self.floor, self.cap, *args, clock=clock)
        self.on = t is not None
        if t is not None:
            self.miss = 0
            self.p = t
            self.since = 0
            self.best = max(self.best, t)
        return self.best - before

    def finish(self):
        """The clear: the rest of the demo's time, all at once."""
        gained = int(self.r['T']) - self.best
        self.best = self.p = int(self.r['T'])
        return max(0, gained)


def _fights(t, boss):
    """[(start, end)] in demo time of the demo's boss fights."""
    out = []
    for ti, b in zip(t, boss):
        if b:
            if out and ti - out[-1][1] <= FIGHT_GAP:
                out[-1][1] = ti + 1
            else:
                out.append([ti, ti + 1])
    return [tuple(f) for f in out]


# ── The environment ──────────────────────────────────────────────────────────

class SonicCoordinator(gym.Env):
    """One act at a time, from states along recorded playthroughs."""

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, demo_files=None, acts=None, demo_prob=None):
        super().__init__()
        self.demo_files = list(demo_files if demo_files is not None else DEMO_FILES)
        self.acts = list(acts if acts is not None else ACTS)
        self.demo_prob = DEMO_PROB if demo_prob is None else demo_prob
        if not self.demo_files:
            raise ValueError('SonicCoordinator needs recorded playthroughs '
                             '(MG_DEMOS / demo_files): episodes start on them '
                             'and progress is measured along them')
        self.base_env = make_retro()
        self.action_space = spaces.Discrete(len(ACTION_BUTTONS))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_SIZE, OBS_SIZE, STACK), dtype=np.uint8)
        self._segs = None            # built from the demos at the first reset
        self._ptr = {}               # segment key -> start checkpoint (trainer-set)
        self._log_warned = False
        self._last_raw = None
        self._stack = collections.deque(maxlen=STACK)

    # ── Demos → start points and routes ──────────────────────────────────

    def _build(self):
        """Replay every demo once through emu_frame, keeping for each act to
        train on: its route (position + boss flag per demo frame) and a start
        point every CKPT_EVERY demo frames (compressed emulator state + the
        observation stack there). A replay that doesn't reproduce the
        recording raises: a silently different demo would teach wrong targets.
        Lazy (first reset of each worker); nothing is written to disk."""
        import demos as demo_io
        self._segs = []
        for path in self.demo_files:
            actions, meta = demo_io.load_play(path)
            table = meta['action_table']
            start = demo_io.play_start_state(meta)
            did = demo_io.demo_id(path)         # an act may be named 'DEMO:ACT': that recording's only
            want = [i for i, a in enumerate(meta['acts'])
                    if a['clear'] is not None and (not self.acts or a['name'] in self.acts
                                                   or f"{did}:{a['name']}" in self.acts)]
            if not want:
                continue
            last = meta['acts'][want[-1]]['clear']
            self.base_env.reset()
            self.base_env.em.set_state(start)
            split = ActSplitter()
            # observation frames i-12 .. i; the stack at frame i is every 4th
            procs = collections.deque(maxlen=(STACK - 1) * FRAME_SKIP + 1)
            prev_raw = None
            seg = None
            for i, a in enumerate(actions[:last + 1]):
                raw, info = emu_frame(self.base_env, table[a])
                procs.append(obs_frame(prev_raw, raw, info))       # his play: no ring used
                prev_raw = raw
                event = split.feed(i, info)
                if event == 'start' and len(split.acts) in want:
                    seg = dict(demo=demo_io.demo_id(path), name=split.cur['name'],
                               index=len(split.acts), ckpts=[],
                               path=collections.defaultdict(list), regular=None,
                               extra=sorted(set(MG_EXTRA_STARTS.get(f"{did}:{split.cur['name']}", []))
                                            | set(MG_EXTRA_STARTS.get(split.cur['name'], []))))
                if seg is None:
                    continue
                if event == 'clear':
                    self._finish_seg(seg, split.acts[-1], meta['acts'][seg['index']])
                    seg = None
                    continue
                t = split.cur['t']
                in_level, _, dead, timer_on = frame_flags(info)
                if not in_level or dead:
                    continue
                p = seg['path']
                p['t'].append(t)
                p['zone'].append(int(info['zone']))
                p['act'].append(int(info['act']))
                p['x'].append(int(info['x']))
                p['y'].append(int(info['y']))
                p['boss'].append(int(info['mg_boss']) != 0)
                regular = seg['regular'] is None or t - seg['regular'] >= CKPT_EVERY
                extra = bool(seg['extra']) and t >= seg['extra'][0]
                if timer_on and (regular or extra):
                    if regular:
                        seg['regular'] = t
                    while seg['extra'] and seg['extra'][0] <= t:
                        seg['extra'].pop(0)
                    # a recording's first frames have no history: repeat the oldest
                    frames = [procs[0]] * (procs.maxlen - len(procs)) + list(procs)
                    seg['ckpts'].append(dict(
                        state=zlib.compress(self.base_env.em.get_state(), 6),
                        stack=np.stack(frames[::FRAME_SKIP], -1),
                        t=t, boss=int(info['mg_boss']) != 0, frame=i))
        if not self._segs:
            raise ValueError(f'no cleared act in {self.demo_files} matches '
                             f'MG_ACTS={self.acts}')

    def _finish_seg(self, seg, act, recorded):
        if (act['start'], act['clear'], act['name']) != (
                recorded['start'], recorded['clear'], recorded['name']):
            raise ValueError(f"{seg['demo']}: replay found {act['name']} at frames "
                             f"{act['start']}-{act['clear']}, the recording "
                             f"{recorded['name']} at {recorded['start']}-{recorded['clear']}")
        T = act['t']
        seg['ckpts'] = [c for c in seg['ckpts'] if T - c['t'] >= CKPT_TAIL]
        if not seg['ckpts']:
            raise ValueError(f"{seg['demo']} {act['name']}: too short to start from")
        seg.pop('regular'), seg.pop('extra')
        route = {k: np.asarray(v, dtype=np.int64) for k, v in seg.pop('path').items()}
        route['fights'] = _fights(route['t'], route.pop('boss'))
        route['T'] = T
        seg.update(route=route, T=T, time=act['time'],
                   key=f"{seg['demo']}:{act['name']}")
        self._segs.append(seg)

    def demo_info(self):
        """For the trainer's curriculum: per act of each demo, its key, act
        time, and where each start point sits in it (fraction of demo time)."""
        if self._segs is None:
            self._build()
        return [dict(key=s['key'], demo=s['demo'], name=s['name'], T=s['T'],
                     time=s['time'], fracs=[c['t'] / s['T'] for c in s['ckpts']])
                for s in self._segs]

    def set_demo_pointers(self, pointers):
        """Called by the trainer (VecEnv.env_method) as the curriculum moves."""
        self._ptr.update({k: int(v) for k, v in pointers.items()})

    # ── Episodes ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._segs is None:
            self._build()
        opts = options or {}
        if 'key' in opts:
            seg = next(s for s in self._segs if s['key'] == opts['key'])
        else:
            seg = random.choice(self._segs)
        last = len(seg['ckpts']) - 1
        if 'ckpt' in opts:
            k = min(int(opts['ckpt']), last)
        elif random.random() < self.demo_prob:
            ptr = min(self._ptr.get(seg['key'], last), last)
            k = random.randint(ptr, min(last, ptr + DEMO_BAND))
        else:
            k = 0                                   # the start of the act
        ck = seg['ckpts'][k]

        self.base_env.reset()
        self.base_env.em.set_state(zlib.decompress(ck['state']))
        self._stack.clear()
        for j in range(STACK):
            self._stack.append(ck['stack'][..., j])
        self._last_raw = None

        self._seg, self._k, self._ck = seg, k, ck
        self._ddz = int(seg['route']['zone'][0]) == ZONE_DDZ
        self._route = RouteProgress(seg['route'], ck['t'], ck['boss'])
        self._target = seg['T'] - ck['t']
        self._max_frames = int(self._target * TIMEOUT_MULT + TIMEOUT_ADD)
        self.frames = 0
        self._moved_at = 0            # frames when route progress last rose
        self._gain_at = 0             # ... route progress alone (MG_SS_DETOUR)
        self._back = None             # MG_SS_DETOUR=refund: frames since a won stage, refunded on return
        self._owed = 0                # v2: off-route frames booked, not yet charged
        self._off = 0                 # v2: frames spent off the route this episode
        self._seen = set()            # v2: blocks of the level visited this episode
        self._stages = []             # pilot: the special stages played this episode
        self._used_rings = set()      # MG_RING_COMPASS: giant rings entered this episode
        self._stage_frames = 0        # pilot: frames charged for them (not on the timeout clock)
        self.ret = 0.0
        self._t0 = time.time()
        self._xmax = None
        return self._obs(), {}

    def _obs(self):
        return np.stack(self._stack, axis=-1)

    def step(self, action):
        mask = ACTION_BUTTONS[int(action)]
        event, n, raws = None, 0, []
        for _ in range(FRAME_SKIP):
            raw, info = emu_frame(self.base_env, mask)
            n += 1
            raws.append(raw)
            in_level, in_stage, dead, timer_on = frame_flags(info)
            if in_stage:
                event = 'stage'
            elif dead:
                event = 'death'
                # routine $C = drowned (Tyler 2026-10-12: "should we measure drowning as it's own type of death?")
                self._death_cause = 'drown' if int(info['mg_routine']) == 0xC else 'other'
            elif (in_level and not timer_on) or (self._ddz and int(info['zone']) == ZONE_ENDING):
                event = 'clear'
            if event:
                break
        stage = None
        if event == 'stage' and MG_SS == 'pilot' and int(info['mg_entry']) in (1, 2):
            stage = self._special_stage(info)       # None: it couldn't; truncate as before
            if stage is not None:
                info, event = stage.pop('info'), None
        if stage is None:
            prev = raws[-2] if len(raws) > 1 else self._last_raw
            self._stack.append(obs_frame(prev, raws[-1], info, self._used_rings))
            self._last_raw = raws[-1]
        self.frames += n
        x = int(info['x'])
        self._xmax = x if self._xmax is None else max(self._xmax, x)

        r = -float(n)
        won_trip = False
        if stage is not None:
            if MG_SS_DETOUR == 'refund' and stage['emeralds'] > 0:
                there = min(self.frames - self._gain_at, DETOUR_CAP)   # the trip to the ring: free
                booked = min(self._owed, there)     # off-route frames not yet charged: drop them
                self._owed -= booked
                r += there - booked                 # the rest were charged: pay them back
                self._stages[-1]['trip_there'] = there
                won_trip = True
            self.frames += stage['cost']
            self._stage_frames += stage['cost']
            r += stage['paid'] - stage['cost']
            self._moved_at = self.frames            # a fresh stuck clock at the ring
        if event == 'clear':
            r += self._route.finish() + CLEAR_BONUS
        elif event == 'death':
            r -= DEATH_PENALTY
        elif event is None:
            gain = self._route.update(info['zone'], info['act'], x, int(info['y']),
                                      int(info['mg_boss']) != 0)
            if gain > 0:
                self._gain_at = self.frames
            if self._back is not None:              # MG_SS_DETOUR: the way back from a won ring is
                if gain > 0:                        # refunded once he makes progress again; time spent
                    back = min(self._back, DETOUR_CAP)   # NOT getting back (camping at the ring) is not
                    r += back
                    self._stages[-1]['trip_back'] = back
                    self._back = None
                else:
                    self._back += n
            new_ground = False
            if MG_V2:
                blk = (int(info['zone']), int(info['act']), x >> BLOCK_SHIFT,
                       int(info['y']) >> BLOCK_SHIFT)
                new_ground = blk not in self._seen
                self._seen.add(blk)
            if MG_V2 and not self._route.on and self._route.fight is None:
                r += n                          # off the route: book it
                self._owed += n
                self._off += n
            else:
                r += gain - self._owed          # back on it: settle
                self._owed = 0
            if gain > 0 or self._route.fight is not None or new_ground:
                self._moved_at = self.frames
            elif self.frames - self._moved_at >= STUCK_FRAMES:
                event = 'stuck'
        if won_trip:
            self._back = 0                          # from the next step: the way back, refunded on return
        self.ret += r
        terminated = event in ('clear', 'death')
        truncated = event in ('stage', 'stuck') or (
            not terminated and self.frames - self._stage_frames >= self._max_frames)
        if (terminated or truncated) and self._owed:
            r -= self._owed                     # v2: the episode ends: settle
            self.ret -= self._owed
            self._owed = 0
        if terminated or truncated:
            outcome = event or 'timeout'
            info = dict(info)
            info['mg_result'] = self._result(outcome, info)
        return self._obs(), r * REWARD_SCALE, terminated, truncated, info

    def _special_stage(self, info):
        """MG_SS=pilot, on the frame Sonic enters a special stage (entry flag
        1), or a giant ring sends him to Hidden Palace (2, zone $17: the
        pilot first touches a gray Super Emerald, MG_SS_SUPER; flag 2
        elsewhere is a bonus stage: None): the
        special-stage agent plays it (ss_pilot.py), then no input while the
        game shows its results and puts Sonic back at the ring, until the act's
        clock runs again. Returns the stage's record (+ 'info': that frame's)
        with 'cost' = the frames charged for it and 'paid' = the frames paid
        back (MG_SS_TIME, MG_EMERALD_BONUS: module docstring). None if it
        couldn't (the episode is then truncated, as with MG_SS=truncate)."""
        em0 = int(info['chaos_emeralds']) + int(info['super_emeralds'])
        near = nearest_ring(info, self._used_rings)      # the ring he just touched: gone now
        if near is not None and math.hypot(near[1], near[2]) <= RING_NEAR:
            self._used_rings.add(near[0])
        shrine = None
        try:
            import ss_pilot
            if int(info['mg_entry']) == 2 and int(info['zone']) != ss_pilot.SHRINE_ZONE:
                return None                         # a bonus stage: nobody plays those
            if getattr(self, '_pilot', None) is None:
                self._pilot = ss_pilot.SSPilot(self.base_env, run=MG_SS_RUN,
                                               model=MG_SS_MODEL or None, routes=MG_SS_ROUTES)
            if int(info['mg_entry']) == 2:          # Hidden Palace: touch a Super Emerald first
                prefer = (ss_pilot.choose_super(self._pilot, LOG_FILE)
                          if MG_SS == 'pilot' and MG_SS_SUPER == 'auto'
                          else [int(b) for b in MG_SS_SUPER.split(',')])
                shrine = self._pilot.shrine(prefer, dict(zip(ACTION_NAMES, ACTION_BUTTONS)))
            res = self._pilot.play()
        except Exception as e:
            print(f'[coordinator] special stage not played ({e!r}): truncating', flush=True)
            return None
        n_stage, n_out = 1 + res['frames'] + (shrine[0] if shrine else 0), 0   # 1: the entry frame
        procs = collections.deque(maxlen=(STACK - 1) * FRAME_SKIP + 1)
        prev_raw = None
        for _ in range(RETURN_CAP):
            raw, info = emu_frame(self.base_env, ACTION_BUTTONS[0])
            in_level, in_stage, dead, timer_on = frame_flags(info)
            if in_stage:
                n_stage += 1
            else:
                n_out += 1
            procs.append(obs_frame(prev_raw, raw, info, self._used_rings))
            prev_raw = raw
            if in_level and timer_on and not dead:
                break
        else:
            print(f'[coordinator] not back in the level {RETURN_CAP} frames after the '
                  f'special stage: truncating', flush=True)
            return None
        frames = [procs[0]] * (procs.maxlen - len(procs)) + list(procs)
        self._stack.clear()
        for f in frames[::FRAME_SKIP]:
            self._stack.append(f)
        self._last_raw = prev_raw
        emeralds = int(info['chaos_emeralds']) + int(info['super_emeralds']) - em0
        cost = n_out + (0 if MG_SS_TIME == 'game' else n_stage)
        paid = (MG_EMERALD_BONUS * emeralds
                + (cost if MG_SS_TIME == 'refund' and emeralds > 0 else 0))
        rec = dict(board=res['board'], outcome=res['outcome'], spheres=res['spheres'],
                   total=res['total'], emeralds=emeralds, in_stage=n_stage, reload=n_out,
                   cost=cost, paid=paid, at=self.frames, route=res.get('route'))
        if res.get('route_broke'):
            rec['route_broke'] = True
        if shrine:
            rec.update(shrine=shrine[0], slot=shrine[1], expected_board=shrine[2])
        self._stages.append(dict(rec))
        rec['info'] = info
        return rec

    def _result(self, outcome, info):
        cleared = outcome == 'clear'
        res = dict(
            key=self._seg['key'], start=self._k, outcome=outcome,
            frames=self.frames, target=self._target,
            margin=(self._target - self.frames) if cleared else None,
            success=cleared and self.frames <= self._target * (1 + MATCH_SLACK) + MATCH_ADD,
            progress=round((self._route.best - self._ck['t']) / max(1, self._target), 4))
        if outcome == 'death':
            res['cause'] = getattr(self, '_death_cause', 'other')
        self._log(res, info)
        return res

    def _log(self, res, info):
        entry = dict(
            t=round(time.time()), act=self._seg['name'], demo=self._seg['demo'],
            start=self._k, start_frac=round(self._ck['t'] / self._seg['T'], 4),
            outcome=res['outcome'], frames=res['frames'], target=res['target'],
            margin=res['margin'], success=res['success'], progress=res['progress'],
            ret=round(self.ret, 1), game_time=fmt_time(game_time(info)),
            x=int(info['x']), y=int(info['y']), x_max=self._xmax,
            rings=int(info['rings']), duration=round(time.time() - self._t0, 1))
        if MG_V2:
            entry['off_route'] = self._off
        if 'cause' in res:
            entry['cause'] = res['cause']                # 'drown' or 'other'
        if self._stages:
            entry['stages'] = self._stages
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            if not self._log_warned:                # say so once: a run with no
                self._log_warned = True             # episode log can't be read
                print(f'[coordinator] WARNING: cannot write episode log '
                      f'{LOG_FILE}: {e!r}', flush=True)

    def render(self):
        return self.base_env.em.get_screen()

    def close(self):
        self.base_env.close()
