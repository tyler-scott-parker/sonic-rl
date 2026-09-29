#!/usr/bin/env python3
"""
ms_analyze.py -- analyzer built for the 14-board multistage run.

Why not analyze_ss.py: it pools all stages into one distribution. With boards
ranging 37..155 spheres that's meaningless. Only %-of-board is comparable.
Why not stage_split --curve: 14 stages x N chunks is an unreadable wall.

This gives ONE ROW PER STAGE, sorted hardest->easiest, with a sparkline trend
so you can see at a glance which boards are climbing, flat, or regressing.

Usage:
  python3 ms_analyze.py                 # last 20000 labeled fresh episodes
  python3 ms_analyze.py 40000           # bigger window
  python3 ms_analyze.py 20000 --chunks 8    # sparkline resolution
  python3 ms_analyze.py 20000 --run v5       # resolves ss_log_v5.jsonl in this repo
  python3 ms_analyze.py 20000 --log <path>  # explicit path
"""
import os, sys, json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]

# Every run writes its OWN log (SS_LOG, set by the trainer from the run name).
# Forgetting --log made analyzers read the GHOST of a previous run — it bit the
# v2 and v3 launches. `--run v5` resolves the path so there is nothing to forget.
LOG = os.path.join(HERE, "ss_log.jsonl")
if "--run" in args:
    LOG = os.path.join(HERE, f"ss_log_{args[args.index('--run') + 1]}.jsonl")
if "--log" in args:
    LOG = os.path.expanduser(args[args.index("--log") + 1])
if not os.path.exists(LOG):
    sys.exit(f"no such log: {LOG}\n"
             f"  pass --run <name> (e.g. --run v5) or --log <path>")
# the window is the first bare number that isn't a flag's value — `--chunks 8
# 20000` used to analyze the last 8 episodes
_flag_vals = {i + 1 for i, a in enumerate(args) if a in ("--run", "--log", "--chunks")}
N = next((int(a) for i, a in enumerate(args) if a.isdigit() and i not in _flag_vals),
         20000)
CH = 6
if "--chunks" in args:
    try:
        CH = int(args[args.index("--chunks") + 1])
    except Exception:
        pass

BLOCKS = "▁▂▃▄▅▆▇█"

rows = []
SPEED_KEY = [None]   # which log field held the speed level
with open(LOG) as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        st = d.get("stage")
        if isinstance(st, int) and st >= 1 and not d.get("restart"):
            # speed level at death — field name varies by env version, so probe
            spd = None
            for k in ("speed_level", "speed", "max_speed_level", "speed_lvl"):
                if k in d and isinstance(d[k], (int, float)):
                    spd = float(d[k]); SPEED_KEY[0] = k; break
            rows.append((st, d.get("spheres", 0) or 0,
                         d.get("stage_total", 0) or 0,
                         bool(d.get("success")), spd))
rows = rows[-N:]
if not rows:
    sys.exit("No labeled multistage episodes yet.")

by = defaultdict(list)
wins = defaultdict(int)
total_of = {}
speeds = defaultdict(list)
for st, sph, tot, suc, spd in rows:
    by[st].append(sph)
    wins[st] += int(suc)
    if tot:
        total_of[st] = tot
    if spd is not None and not suc:      # speed AT DEATH only
        speeds[st].append(spd)

def spark(vals):
    """tiny sparkline from a list of floats"""
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return BLOCKS[3] * len(vals)
    return "".join(BLOCKS[min(7, int((v - lo) / (hi - lo) * 7.99))] for v in vals)

def trend_of(seq):
    """compare first third vs last third -> arrow"""
    if len(seq) < 3:
        return "?"
    k = max(1, len(seq) // 3)
    a = sum(seq[:k]) / k
    b = sum(seq[-k:]) / k
    if b > a * 1.15:
        return "UP"
    if b < a * 0.85:
        return "DOWN"
    return "flat"

# per-stage chunked %board series (chronological within the window)
series = defaultdict(list)
per_stage_seq = defaultdict(list)
for st, sph, tot, suc, spd in rows:
    per_stage_seq[st].append(sph / (total_of.get(st) or 1))
for st, seq in per_stage_seq.items():
    if not seq:
        continue
    size = max(1, len(seq) // CH)
    chunks = [seq[i:i + size] for i in range(0, len(seq), size)][:CH]
    series[st] = [100 * sum(c) / len(c) for c in chunks if c]

print(f"\n14-BOARD RUN — last {len(rows)} labeled fresh episodes")
print("=" * 74)
print(f"{'stg':<5}{'size':<6}{'n':<7}{'avg':<7}{'%board':<9}{'max':<6}"
      f"{'spd':<6}{'win':<5}{'trend':<7}history")
print("-" * 80)

stats = []
for st in sorted(by):
    v = by[st]
    tot = total_of.get(st, 0) or 1
    avg = sum(v) / len(v)
    pct = 100 * avg / tot
    sp = speeds.get(st, [])
    aspd = sum(sp) / len(sp) if sp else None
    stats.append((pct, st, tot, len(v), avg, max(v), wins[st], aspd))

# hardest first
for pct, st, tot, n, avg, mx, w, aspd in sorted(stats):
    s = series.get(st, [])
    spd_s = f"{aspd:.1f}" if aspd is not None else "-"
    print(f"{st:<5}{tot:<6}{n:<7}{avg:<7.1f}{pct:<9.1f}{mx:<6}{spd_s:<6}{w:<5}"
          f"{trend_of(s):<7}{spark(s)}")

print("=" * 74)
pcts = [p for p, *_ in stats]
climbing = sum(1 for _, st, *_ in stats if trend_of(series.get(st, [])) == "UP")
flat = sum(1 for _, st, *_ in stats if trend_of(series.get(st, [])) == "flat")
down = sum(1 for _, st, *_ in stats if trend_of(series.get(st, [])) == "DOWN")
print(f"  mean %board: {sum(pcts)/len(pcts):5.1f}    "
      f"spread: {min(pcts):.1f}% – {max(pcts):.1f}%    "
      f"ratio(min/max): {min(pcts)/max(pcts) if max(pcts) > 0 else 0.0:.2f}")
print(f"  trend: {climbing} climbing / {flat} flat / {down} regressing "
      f"(of {len(stats)} stages)")
print(f"  total wins: {sum(wins.values())}    "
      f"stages covered: {len(by)}/14    "
      f"episodes/stage avg: {len(rows)//max(1,len(by))}")
print()
if SPEED_KEY[0]:
    print(f"  spd = avg speed level AT DEATH (log field '{SPEED_KEY[0]}'). HIGH spd on a")
    print(f"        stalled board = hitting the speed ramp (reaction wall, same as the")
    print(f"        specialist). LOW spd on a stalled board = dying early, i.e. a")
    print(f"        navigation/layout problem. Different problems, different fixes.")
else:
    print("  spd = '-' : no speed field found in the log (tried speed_level/speed/")
    print("        max_speed_level/speed_lvl). Add one to _log_episode to enable it.")
print()
print("  Read: all stages climbing + spread narrowing = learning a shared skill.")
print("        A few stages stuck near the floor = board-specific difficulty.")
print("        ratio(min/max) near 1.0 = uniform competence across boards.")
print()
