#!/usr/bin/env python3
"""Quick distinctness check: load each captured stage state, read its starting
blue-sphere count + a RAM fingerprint, and flag any duplicates."""
import os, sys, glob, hashlib
import stable_retro

STATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'states')
GAME = 'SonicAndKnuckles3-Genesis-v0'

env = stable_retro.make(game=GAME,
                        use_restricted_actions=stable_retro.Actions.FILTERED,
                        render_mode='rgb_array')
env.reset()

files = sorted(glob.glob(os.path.join(STATES, 'curriculum_stage_*.state')),
               key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0))

print(f"{'stage':<8}{'blue':<8}{'special':<10}{'ram_fingerprint':<18}note")
print("-" * 60)
seen_blue = {}
seen_fp = {}
for f in files:
    n = os.path.basename(f).replace('curriculum_stage_', '').replace('.state', '')
    with open(f, 'rb') as fh:
        env.em.set_state(fh.read())
    # step a few frames so RAM settles into the stage
    info = {}
    for _ in range(4):
        _, _, _, _, info = env.step([0]*12)
    try:
        blue = env.data.lookup_value('blue_spheres_remaining')
    except Exception:
        blue = info.get('blue_spheres_remaining', '?')
    try:
        special = env.data.lookup_value('special_stage')
    except Exception:
        special = '?'
    ram = bytes(env.get_ram())
    fp = hashlib.md5(ram).hexdigest()[:12]
    note = ""
    if blue in seen_blue:
        note += f"DUP blue w/ stage {seen_blue[blue]}  "
    if fp in seen_fp:
        note += f"IDENTICAL RAM to stage {seen_fp[fp]}"
    seen_blue.setdefault(blue, n)
    seen_fp.setdefault(fp, n)
    print(f"{n:<8}{str(blue):<8}{str(special):<10}{fp:<18}{note}")

env.close()
print("\nDistinct blue counts:", sorted(set(seen_blue.keys()), key=lambda x: (isinstance(x,str), x)))
