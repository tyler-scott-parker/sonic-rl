#!/usr/bin/env python3
"""
verify_win_fix.py -- deterministic unit test of the board-clear success fix.

Drives a synthetic board clear (blue_spheres_remaining -> 0) through the
patched _compute_reward and confirms it is detected as a WIN that pays the
completion reward, WITHOUT relying on the emerald counter incrementing.

No real lucky clear, no training, no log writes. Run AFTER applying the
three edits to special_stage_env.py.
"""
import special_stage_env as m

env = m.SpecialStageEnv()

# capture what _log_episode would record, instead of writing to disk
logged = {}
env._log_episode = lambda **k: logged.update(k)
env._death_on_trail = lambda: False

# internal state: mid special stage, 2 spheres left, none collected yet,
# emerald counters flat (so a win can ONLY come from the board-clear latch)
env.steps = 50                  # > 10 so the termination gate can fire
env.prev_bumper_lock = 0
env.bumper_hit = False
env.prev_speed_timer = 0
env.speed_level = 0
env.ss_current_streak = 0
env.ss_best_streak = 0
env._spheres_collected = 0
env._board_cleared = False
env.prev_chaos_em = 0
env.prev_super_em = 0
env.prev_blue_spheres = 2

base = dict(special_stage=1, chaos_emeralds=0, super_emeralds=0,
            ss_speed_timer=0, bumper_contact=0)

print("initial: _board_cleared=%s collected=%s" %
      (env._board_cleared, env._spheres_collected))

# frame 1: collect one sphere (2 -> 1), still in stage
r1, d1 = env._compute_reward(dict(base, blue_spheres_remaining=1), False)
print("frame1 blue=1 : reward=%8.1f done=%s cleared=%s collected=%s" %
      (r1, d1, env._board_cleared, env._spheres_collected))

# frame 2: collect last sphere (1 -> 0), STILL in stage (special=1)
r2, d2 = env._compute_reward(dict(base, blue_spheres_remaining=0), False)
print("frame2 blue=0 : reward=%8.1f done=%s cleared=%s collected=%s" %
      (r2, d2, env._board_cleared, env._spheres_collected))

# frame 3: stage exits (special -> 0), NO emerald increment (chaos stays 0)
r3, d3 = env._compute_reward(
    dict(base, blue_spheres_remaining=0, special_stage=0), False)
print("frame3 exit, no emerald : reward=%8.1f done=%s" % (r3, d3))
print("logged:", logged)

ok = (env._board_cleared
      and logged.get('success') is True
      and d3 is True
      and r3 > 0)                     # win pays positive; a death is strongly negative
print()
if ok:
    print(">>> PASS - board clear detected as WIN with completion reward, "
          "no emerald required.")
else:
    print(">>> FAIL - clear not detected correctly. Check the three edits.")
    print("    expected: _board_cleared=True, success=True, done=True, reward>0")
