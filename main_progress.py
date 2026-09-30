#!/usr/bin/env python3
"""
main_progress.py — how a base-game run (train_main.py) is doing against the
recording it learns from.

  python3 main_progress.py --run aiz1
  python3 main_progress.py --run aiz1 --last 5000 --chunks 5

Per act of each recording:
  curriculum   where episodes start now (fraction of the act, 0% = its
               beginning), and how often the recent ones from there matched
               you (cleared within 1.25x your time + 1 s)
  from start   the honest number: episodes that began at the start of the act.
               Clear rate, deaths, and frames vs your time (margin > 0 = the
               agent was faster than you), with the trend over the window
"""
import os
import json
import argparse
import collections

import numpy as np

import train_main

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--run', required=True, help='run name from train_main.RUNS')
ap.add_argument('--last', type=int, default=0, help='only the last N episodes (0 = all)')
ap.add_argument('--chunks', type=int, default=5, help='trend rows for fresh starts')
args = ap.parse_args()

P = train_main.paths(args.run)
if not os.path.exists(P['log']):
    raise SystemExit(f"no episode log yet at {P['log']}")
eps = [json.loads(line) for line in open(P['log']) if line.strip()]
if args.last:
    eps = eps[-args.last:]
ptrs = json.load(open(P['curriculum'])) if os.path.exists(P['curriculum']) else {}
BAND = 4    # coordinator.DEMO_BAND (not imported: that would build an emulator config)


def secs(frames):
    return f'{frames / 60:.2f}s'


def describe(rows):
    n = len(rows)
    out = collections.Counter(r['outcome'] for r in rows)
    clears = [r for r in rows if r['outcome'] == 'clear']
    s = (f"{n:6d} eps  clear {out['clear'] / n:5.1%}  death {out['death'] / n:5.1%}  "
         f"stuck {out['stuck'] / n:5.1%}  timeout {out['timeout'] / n:5.1%}  "
         f"stage {out['stage'] / n:4.1%}")
    if clears:
        m = np.array([r['margin'] for r in clears])
        s += (f"  | vs you: median {np.median(m):+5.0f}f  best {m.max():+5.0f}f "
              f"({secs(m.max())})  beat you in {np.mean(m > 0):5.1%} of clears")
    return s


by_key = collections.defaultdict(list)
for e in eps:
    by_key[f"{e['demo']}:{e['act']}"].append(e)

print(f"run {args.run}: {len(eps)} episodes in {P['log']}")
for key, rows in by_key.items():
    target = max((r['target'] for r in rows if r['start'] == 0), default=None)
    print(f"\n== {key}" + (f"   your time {secs(target)} ({target} frames)" if target else ''))
    p = ptrs.get(key)
    if p:
        at = [r for r in rows if p['ptr'] <= r['start'] <= p['ptr'] + BAND][-64:]
        rate = f"{np.mean([r['success'] for r in at]):.0%} of the last {len(at)} matched you" \
            if at else 'no episodes there yet'
        print(f"  curriculum : starting {p['frac']:.0%} into the act (start point {p['ptr']}); {rate}")
    fresh = [r for r in rows if r['start'] == 0]
    if not fresh:
        print('  from start : none yet')
        continue
    print(f"  from start : {describe(fresh)}")
    k = max(1, min(args.chunks, len(fresh) // 20))
    for i, chunk in enumerate(np.array_split(np.array(fresh, dtype=object), k)):
        print(f"    part {i + 1}/{k}  {describe(list(chunk))}")
    deaths = [r for r in fresh if r['outcome'] == 'death']
    if deaths:
        xs = collections.Counter(r['x'] // 256 * 256 for r in deaths).most_common(3)
        print(f"  deaths from start, most common x: "
              + ', '.join(f'{x}-{x + 255} ({n})' for x, n in xs))
