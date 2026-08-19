#!/usr/bin/env python3
"""
find_stage_byte.py — Locate the special-stage NUMBER address in RAM.

After capturing a few stages with capture_stages.py, this loads the
curriculum_stage_<N>.ram.bin snapshots and finds RAM offsets whose byte
tracks the stage index across every capture.

Strongest signal: an offset where byte == N for all stages (1-indexed), or
byte == N-1 (0-indexed), or byte == N + c for a fixed c (the emerald counter
may be offset from the SS number). Each is reported separately. More captures
= sharper filter; 3-4 usually narrows EXACT/LINEAR to a single offset.

Usage:
  python3 find_stage_byte.py            # all captures in the states dir
  python3 find_stage_byte.py 1 2 3 4    # restrict to these stage indices
"""

import os
import re
import sys
import glob

STATES_DIR = os.path.expanduser('~/sonic-rl/states')


def load():
    caps = {}
    for p in glob.glob(os.path.join(STATES_DIR, 'curriculum_stage_*.ram.bin')):
        m = re.search(r'curriculum_stage_(\d+)\.ram\.bin$', p)
        if m:
            with open(p, 'rb') as f:
                caps[int(m.group(1))] = f.read()
    return caps


def main():
    caps = load()
    if len(sys.argv) > 1:
        want = {int(a) for a in sys.argv[1:]}
        caps = {k: v for k, v in caps.items() if k in want}

    if len(caps) < 2:
        print(f"Need >=2 captures in {STATES_DIR}; found {sorted(caps)}.")
        print("Run capture_stages.py on a few more stages first.")
        return

    idxs = sorted(caps)
    blobs = [caps[i] for i in idxs]
    n = min(len(b) for b in blobs)
    print(f"Comparing stages {idxs} over {n} bytes each.\n")

    exact, linear, monotonic = [], [], []
    for off in range(n):
        vals = [blobs[k][off] for k in range(len(idxs))]
        if all(v == s for v, s in zip(vals, idxs)):                 # byte == stage#
            exact.append((off, vals))
            continue
        diffs = {v - s for v, s in zip(vals, idxs)}                 # byte == stage# + c
        if len(diffs) == 1:
            linear.append((off, next(iter(diffs)), vals))
        elif all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            monotonic.append((off, vals))                          # weak: just increasing

    def show(title, rows, fmt):
        print(f"== {title}: {len(rows)} hit(s) ==")
        for r in rows[:25]:
            print("   " + fmt(r))
        if len(rows) > 25:
            print(f"   ... (+{len(rows) - 25} more)")
        print()

    show("EXACT   byte == stage#", exact,
         lambda r: f"${r[0]:04X} (off {r[0]:>5})  vals={r[1]}")
    show("LINEAR  byte == stage# + c", linear,
         lambda r: f"${r[0]:04X} (off {r[0]:>5})  c={r[1]:+d}  vals={r[2]}")
    show("MONOTONIC (weak hint)", monotonic,
         lambda r: f"${r[0]:04X} (off {r[0]:>5})  vals={r[1]}")

    pick = exact or linear
    if pick:
        off = pick[0][0]
        print(f"Most likely stage-number byte: ${off:04X} (offset {off}).")
        print("Verify it reads what you expect on each state, then either feed")
        print("it to the policy as a stage-ID input, or poke it to drive")
        print("selection if it turns out the game reads it at SS entry.")


if __name__ == '__main__':
    main()
