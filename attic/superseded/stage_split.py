#!/usr/bin/env python3
"""
stage_split.py -- read the per-stage collection split for the multistage probe.

Filters ss_log.jsonl to recent LABELED fresh-start episodes (stage 1/2/…),
skipping the old specialist episodes and any pre-fix stage:0 lines, then prints
avg / median / max / distribution PER STAGE.

This is the number the Option-A probe exists to produce: do stage-1 and stage-2
collection climb TOGETHER (position-drop forced a general skill) or does one
pull away while the other stalls (still specializing)?

Usage:
  python3 stage_split.py                # last 2000 labeled episodes
  python3 stage_split.py 5000           # last 5000
  python3 stage_split.py 2000 --curve   # + per-1000 trend, to see if the gap
                                        #   is widening or closing over time
"""
import os, sys, json
from collections import defaultdict

LOG = os.path.expanduser("~/sonic-rl/ss_log.jsonl")
args = [a for a in sys.argv[1:]]
N = next((int(a) for a in args if a.isdigit()), 2000)
show_curve = "--curve" in args

# read tail of file cheaply-ish: load all lines, keep labeled fresh episodes
rows = []
with open(LOG) as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        st = d.get("stage")
        # labeled fresh-start multistage episodes only: stage in {1,2,3,...},
        # not restarts (stage 0 / None) and not the old specialist (no 'stage' key)
        if isinstance(st, int) and st >= 1 and not d.get("restart"):
            rows.append((st, d.get("spheres", 0), d.get("stage_total", 0),
                         bool(d.get("success"))))

rows = rows[-N:]
if not rows:
    sys.exit("No labeled multistage episodes yet. Let the run log a few, then retry.")

def stats(vals):
    v = sorted(vals)
    n = len(v)
    avg = sum(v) / n
    med = v[n // 2]
    p90 = v[int(n * 0.9)] if n > 1 else v[0]
    return n, avg, med, p90, max(v)

by_stage = defaultdict(list)
wins = defaultdict(int)
for st, sph, tot, suc in rows:
    by_stage[st].append(sph)
    wins[st] += int(suc)

print(f"\nlast {len(rows)} labeled fresh episodes\n" + "=" * 52)
for st in sorted(by_stage):
    n, avg, med, p90, mx = stats(by_stage[st])
    # normalize avg to % of that stage's board, so stages with different totals
    # are comparable (34/102 vs 34/127 are NOT the same achievement)
    tot = next(t for s, sp, t, su in rows if s == st and t) if any(
        t for s, sp, t, su in rows if s == st) else 0
    pct = f"{100*avg/tot:4.1f}% of {tot}" if tot else ""
    print(f"  STAGE {st}:  n={n:5d}  avg={avg:5.1f}  median={med:3d}  "
          f"p90={p90:3d}  max={mx:3d}  wins={wins[st]}   {pct}")

if len(by_stage) >= 2:
    s = sorted(by_stage)
    a = sum(by_stage[s[0]]) / len(by_stage[s[0]])
    b = sum(by_stage[s[1]]) / len(by_stage[s[1]])
    lo, hi = sorted((a, b))
    ratio = lo / hi if hi else 0
    print("=" * 52)
    print(f"  transfer ratio (weaker/stronger avg): {ratio:.2f}")
    print("   ~1.0 = both stages learned equally (generalizing)")
    print("   ~0.1 = one stage barely learned (still specializing)")

if show_curve:
    print("\n  per-1000-episode trend (avg spheres by stage):")
    CH = 1000
    for i in range(0, len(rows), CH):
        chunk = rows[i:i + CH]
        cs = defaultdict(list)
        for st, sph, tot, suc in chunk:
            cs[st].append(sph)
        line = "  ".join(f"S{st}={sum(v)/len(v):4.1f}(n{len(v)})"
                         for st, v in sorted(cs.items()))
        print(f"    ep {i:5d}-{i+len(chunk):5d}:  {line}")

print()
