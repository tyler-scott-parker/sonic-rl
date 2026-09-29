#!/usr/bin/env python3
"""
demo_progress.py — is the agent learning from your recorded runs, and beating them?

  python3 demo_progress.py --run d1
  python3 demo_progress.py --run d1 --chunks 10

For each demo:
  start    how far into your run episodes begin. The curriculum walks it from
           near the end back to 0% as the agent catches up with you.
  match    how often the agent collected at least as many spheres as you did
           from its starting point (the curriculum moves on at 25%).
  margin   spheres more (+) or fewer (-) than you collected from there.
Then FRESH starts on the same stage, against your whole run — the goal is fresh
play that equals or beats yours.
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import demos

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--run', required=True, help='run name from train_ss_multistage RUNS')
ap.add_argument('--chunks', type=int, default=8, help='time slices per table')
args = ap.parse_args()

log_path = os.path.join(HERE, f'ss_log_{args.run}.jsonl')
state_path = os.path.join(HERE, f'sonic_ss_multistage_{args.run}_demos.json')
if not os.path.exists(log_path):
    sys.exit(f'no log yet: {log_path}')
rows = []
with open(log_path) as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
state = json.load(open(state_path)) if os.path.exists(state_path) else {}

on_demo, fresh = defaultdict(list), defaultdict(list)
for r in rows:
    if r.get('demo'):
        on_demo[r['demo']].append(r)
    elif not r.get('restart'):
        fresh[r.get('stage')].append(r)


def chunks(seq, n):
    size = max(1, -(-len(seq) // n))
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


if not on_demo:
    sys.exit(f'{log_path} has no demo episodes yet.')

for did, rs in sorted(on_demo.items()):
    path = os.path.join(demos.DEMO_DIR, f'{did}.npz')
    meta = demos.load_demo(path)[1] if os.path.exists(path) else {}
    stage = meta.get('stage', rs[0].get('stage'))
    print(f"\n{did} — stage {stage}; your run: {meta.get('spheres', '?')} spheres, "
          f"{meta.get('outcome', '?')} after {meta.get('frames', '?')} frames")
    if did in state:
        print(f"  curriculum start point now: {state[did]['frac']:.0%} into your run")
    print(f"  {'episodes':>14}  {'start':>6}  {'match':>6}  {'margin':>7}")
    for c in chunks(rs, args.chunks):
        match = mean([r['spheres'] >= r['demo_target'] for r in c])
        margin = mean([r['spheres'] - r['demo_target'] for r in c])
        print(f"  {len(c):>14}  {mean([r['demo_frac'] for r in c]):>6.0%}  "
              f"{match:>6.0%}  {margin:>+7.1f}")

    fs = fresh.get(stage, [])
    if fs:
        yours = meta.get('spheres')
        print(f"  fresh starts on stage {stage} (you: {yours} spheres, "
              f"{meta.get('outcome', '?')}):")
        print(f"  {'episodes':>14}  {'mean':>6}  {'best':>5}  {'clears':>6}  "
              f"{'>= you':>6}")
        for c in chunks(fs, args.chunks):
            sph = [r['spheres'] for r in c]
            beat = mean([s >= yours for s in sph]) if yours is not None else float('nan')
            print(f"  {len(c):>14}  {mean(sph):>6.1f}  {max(sph):>5}  "
                  f"{sum(bool(r.get('success')) for r in c):>6}  {beat:>6.0%}")
print()
