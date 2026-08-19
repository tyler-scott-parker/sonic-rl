#!/usr/bin/env python3
"""
zone_probe.py -- find which zone byte = Mushroom Hill, empirically & safely.

For each candidate zone id: reset the emulator, write zone, write act=0, then
force a level (re)load by writing level_started=0 and stepping. Read back what
the game settled on. We CANNOT see the screen here, so this reports the
readback + whether the state looks alive (rings/lives sane, no obvious garbage).
Run this to narrow candidates; final confirmation is visual in the capture tool.
"""
import stable_retro

GAME = 'SonicAndKnuckles3-Genesis-v0'

def try_zone(env, z):
    env.reset()
    for _ in range(30):
        env.step([0]*12)
    # write zone/act, then nudge a reload
    try:
        env.data.set_value('zone', z)
        env.data.set_value('act', 0)
    except Exception as e:
        return f"set failed: {e}"
    for _ in range(90):
        env.step([0]*12)
    z_read = env.data.lookup_value('zone')
    a_read = env.data.lookup_value('act')
    try:
        rings = env.data.lookup_value('rings')
        lives = env.data.lookup_value('lives')
    except Exception:
        rings = lives = '?'
    return f"wrote zone={z:2d} -> reads zone={z_read} act={a_read} rings={rings} lives={lives}"

env = stable_retro.make(GAME, use_restricted_actions=stable_retro.Actions.FILTERED,
                        render_mode='rgb_array')
# S3&K final zone bytes span roughly 0x00..0x0D for main zones; scan generously
for z in range(0, 16):
    print(try_zone(env, z))
env.close()
