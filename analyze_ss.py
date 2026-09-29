#!/usr/bin/env python3
"""
SS log analyzer v2 — tracks special stage training progress.
Usage: python3 analyze_ss.py [logfile]
Default: ss_log.jsonl in this repo

New in v2 (sections appear only if the log contains the fields):
  - Recent form window (last 500 eps) vs all-time
  - Median / p90 spheres
  - Timeout vs death split
  - Steps-per-sphere efficiency trend
  - Self-trail death rate + trend   (requires env v2.1 logs)
  - Loop discovery tracking          (requires env v2.1 logs)
  - Coverage stats                   (requires env v2.1 logs)
  - Deaths by speed level cross-tab
"""

import json
import os
import sys

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ss_log.jsonl")
RECENT_N = 500


def load(path):
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return episodes


def bar(value, max_value, width=40):
    filled = int(width * value / max_value) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def pct_bar(p, width=30):
    filled = int(width * p / 100)
    return "█" * filled + "░" * (width - filled)


def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(int(q * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE
    if not os.path.exists(path):
        print(f"Log not found: {path}")
        print("SS training hasn't produced any episodes yet.")
        return

    episodes = load(path)
    if not episodes:
        print("No episodes found.")
        return

    # ── Curriculum split: fresh-start vs pool-restart episodes ───────────
    has_restart = any(ep.get("restart") for ep in episodes)
    all_episodes = episodes
    if has_restart:
        restarts = [ep for ep in episodes if ep.get("restart")]
        episodes = [ep for ep in episodes if not ep.get("restart")]
        print(f"\n[curriculum active: {len(restarts)} restart episodes "
              f"separated; stats below are FRESH-START episodes only]")
    else:
        restarts = []

    total = len(episodes)
    successes = [ep for ep in episodes if ep.get("success")]
    sphere_counts = [ep.get("spheres", 0) for ep in episodes]
    streak_counts = [ep.get("best_streak", 0) for ep in episodes]
    step_counts   = [ep.get("steps", 0) for ep in episodes]
    sorted_spheres = sorted(sphere_counts)
    deaths   = [ep for ep in episodes
                if not ep.get("success") and ep.get("death_at_sphere") is not None]
    timeouts = [ep for ep in episodes
                if not ep.get("success") and ep.get("death_at_sphere") is None]

    section("OVERVIEW")
    print(f"  Total episodes:   {total}")
    print(f"  Successes:        {len(successes)} ({len(successes)/total*100:.1f}%)")
    print(f"  Deaths:           {len(deaths)} ({len(deaths)/total*100:.1f}%)")
    print(f"  Timeouts:         {len(timeouts)} ({len(timeouts)/total*100:.1f}%)")
    print(f"  Best spheres:     {max(sphere_counts)}")
    print(f"  Avg spheres:      {sum(sphere_counts)/total:.1f}")
    print(f"  Median spheres:   {quantile(sorted_spheres, 0.5)}")
    print(f"  p90 spheres:      {quantile(sorted_spheres, 0.9)}")
    print(f"  Best streak:      {max(streak_counts)}")
    print(f"  Avg steps/ep:     {int(sum(step_counts)/total)}")

    if total > RECENT_N:
        recent = episodes[-RECENT_N:]
        rsp = sorted(ep.get("spheres", 0) for ep in recent)
        rwins = sum(1 for ep in recent if ep.get("success"))
        section(f"RECENT FORM  (last {RECENT_N} episodes)")
        print(f"  Avg spheres:      {sum(rsp)/len(rsp):.1f}")
        print(f"  Median spheres:   {quantile(rsp, 0.5)}")
        print(f"  p90 spheres:      {quantile(rsp, 0.9)}")
        print(f"  Max spheres:      {max(rsp)}")
        print(f"  Wins:             {rwins}")

    section("LEARNING CURVE  (grouped in 200s)")
    chunk = 200
    for i in range(0, total, chunk):
        group = episodes[i:i+chunk]
        sp    = [ep.get("spheres", 0) for ep in group]
        st    = [ep.get("best_streak", 0) for ep in group]
        wins  = sum(1 for ep in group if ep.get("success"))
        gsteps = sum(ep.get("steps", 0) for ep in group)
        gspheres = sum(sp)
        eff = f"{gsteps/gspheres:6.0f}" if gspheres > 0 else "   n/a"
        label = f"ep {i+1:>6}-{i+len(group):<6}"
        print(f"  {label}  spheres avg={sum(sp)/len(sp):5.1f} max={max(sp):3d}  "
              f"streak max={max(st):3d}  steps/sphere={eff}  wins={wins}/{len(group)}")

    section("SPHERE DISTRIBUTION")
    buckets = {}
    for s in sphere_counts:
        b = (s // 10) * 10
        buckets[b] = buckets.get(b, 0) + 1
    max_count = max(buckets.values())
    for b in sorted(buckets):
        count = buckets[b]
        label = f"{b:>3}-{b+9}"
        print(f"  {label} spheres  {bar(count, max_count, 30)} {count}")

    section("SPEED LEVEL DISTRIBUTION")
    speed_counts = {}
    for ep in episodes:
        s = ep.get("speed_level", 0)
        speed_counts[s] = speed_counts.get(s, 0) + 1
    max_count = max(speed_counts.values())
    for s in sorted(speed_counts):
        print(f"  Speed {s}  {bar(speed_counts[s], max_count, 30)} {speed_counts[s]}")

    # ── Bumper section (v1 fields) ───────────────────────────────────────
    bumper_episodes = [ep for ep in episodes if "bumper_hit" in ep]
    if bumper_episodes:
        section("BUMPER HIT RATE")
        hit = sum(1 for ep in bumper_episodes if ep.get("bumper_hit"))
        hit_pct = hit / len(bumper_episodes) * 100
        print(f"  Episodes with bumper hit:  {hit} / {len(bumper_episodes)} ({hit_pct:.1f}%)")
        print(f"\n  Hit rate over training (per 500 episodes):")
        for i in range(0, len(bumper_episodes), 500):
            group = bumper_episodes[i:i+500]
            gp = sum(1 for ep in group if ep.get("bumper_hit")) / len(group) * 100
            print(f"  ep {i+1:>6}-{i+len(group):<6}  {pct_bar(gp)}  {gp:.1f}%")
        hit_eps   = [ep for ep in bumper_episodes if ep.get("bumper_hit")]
        clean_eps = [ep for ep in bumper_episodes if not ep.get("bumper_hit")]
        if hit_eps and clean_eps:
            avg_hit   = sum(ep.get("spheres", 0) for ep in hit_eps) / len(hit_eps)
            avg_clean = sum(ep.get("spheres", 0) for ep in clean_eps) / len(clean_eps)
            print(f"\n  Avg spheres — bumper-hit episodes:  {avg_hit:.1f}")
            print(f"  Avg spheres — clean episodes:       {avg_clean:.1f}")

    # ── Self-trail deaths (v2.1 fields) ──────────────────────────────────
    trail_eps = [ep for ep in deaths if ep.get("own_trail_death") is not None]
    if trail_eps:
        section("SELF-TRAIL DEATHS  (died on own converted sphere)")
        trail = sum(1 for ep in trail_eps if ep.get("own_trail_death"))
        tp = trail / len(trail_eps) * 100
        print(f"  Trail deaths:   {trail} / {len(trail_eps)} deaths ({tp:.1f}%)")
        print(f"\n  Trail-death rate over training (per 500 deaths):")
        for i in range(0, len(trail_eps), 500):
            group = trail_eps[i:i+500]
            gp = sum(1 for ep in group if ep.get("own_trail_death")) / len(group) * 100
            print(f"  death {i+1:>5}-{i+len(group):<6} {pct_bar(gp)}  {gp:.1f}%")
        t_avg = sum(ep.get("spheres", 0) for ep in trail_eps
                    if ep.get("own_trail_death"))
        n_trail = max(trail, 1)
        o_eps = [ep for ep in trail_eps if not ep.get("own_trail_death")]
        if o_eps:
            print(f"\n  Avg spheres — trail deaths:     {t_avg/n_trail:.1f}")
            print(f"  Avg spheres — original-red deaths: "
                  f"{sum(ep.get('spheres',0) for ep in o_eps)/len(o_eps):.1f}")

    # ── Loop discovery (v2.1 fields) ─────────────────────────────────────
    # Sanity guard: episodes logged before the fade-timer fix can contain
    # corrupted loop counts (buffer reused as scratch during exit fades).
    # loop_spheres can never exceed the 102 spheres on the map.
    loop_logged = [ep for ep in episodes
                   if "loops" in ep and ep.get("loop_spheres", 0) <= 102]
    if loop_logged:
        section("LOOP DISCOVERY  (perimeter conversions to rings)")
        loop_eps = [ep for ep in loop_logged if ep.get("loops", 0) > 0]
        tot_loops = sum(ep.get("loops", 0) for ep in loop_logged)
        tot_lsph  = sum(ep.get("loop_spheres", 0) for ep in loop_logged)
        print(f"  Episodes with ≥1 loop:  {len(loop_eps)} / {len(loop_logged)} "
              f"({len(loop_eps)/len(loop_logged)*100:.1f}%)")
        print(f"  Total loops:            {tot_loops}")
        print(f"  Spheres via loops:      {tot_lsph}")
        if len(loop_logged) >= 500:
            print(f"\n  Loop rate over training (per 500 episodes):")
            for i in range(0, len(loop_logged), 500):
                group = loop_logged[i:i+500]
                gp = sum(1 for ep in group if ep.get("loops", 0) > 0) / len(group) * 100
                print(f"  ep {i+1:>6}-{i+len(group):<6}  {pct_bar(gp)}  {gp:.1f}%")
        if loop_eps:
            avg_l = sum(ep.get("spheres", 0) for ep in loop_eps) / len(loop_eps)
            non_l = [ep for ep in loop_logged if ep.get("loops", 0) == 0]
            avg_n = sum(ep.get("spheres", 0) for ep in non_l) / max(len(non_l), 1)
            print(f"\n  Avg spheres — loop episodes:     {avg_l:.1f}")
            print(f"  Avg spheres — non-loop episodes: {avg_n:.1f}")

    # ── Coverage (v2.1 fields) ───────────────────────────────────────────
    cov_eps = [ep for ep in episodes if "coverage" in ep]
    if cov_eps:
        section("COVERAGE  (unique cells visited, of 1024)")
        covs = [ep.get("coverage", 0) for ep in cov_eps]
        print(f"  Avg coverage:   {sum(covs)/len(covs):.0f} cells")
        print(f"  Max coverage:   {max(covs)} cells")
        ranked = sorted(cov_eps, key=lambda e: e.get("spheres", 0))
        lo = ranked[:max(len(ranked)//10, 1)]
        hi = ranked[-max(len(ranked)//10, 1):]
        print(f"  Coverage in worst-decile sphere eps: "
              f"{sum(e.get('coverage',0) for e in lo)/len(lo):.0f}")
        print(f"  Coverage in best-decile sphere eps:  "
              f"{sum(e.get('coverage',0) for e in hi)/len(hi):.0f}")

    # ── Curriculum restart episodes (v3 fields) ──────────────────────────
    if restarts:
        section("CURRICULUM  (pool-restart episodes)")
        gains = [ep.get("spheres", 0) for ep in restarts]
        froms = [ep.get("restart_from", 0) for ep in restarts]
        frontier = max(f + g for f, g in zip(froms, gains))
        all_frontier = max(
            ep.get("restart_from", 0) + ep.get("spheres", 0)
            for ep in all_episodes)
        rwins = sum(1 for ep in restarts if ep.get("success"))
        print(f"  Restart episodes:   {len(restarts)}")
        print(f"  Avg restart depth:  {sum(froms)/len(froms):.0f} spheres")
        print(f"  Avg gained:         {sum(gains)/len(gains):.1f} spheres")
        print(f"  Wins from restarts: {rwins}")
        print(f"  FRONTIER (deepest total progress ever): {all_frontier} / 102")
        chunk = 500
        if len(restarts) >= chunk:
            print(f"\n  Avg gained over training (per {chunk} restart eps):")
            for i in range(0, len(restarts), chunk):
                group = restarts[i:i+chunk]
                g = sum(ep.get("spheres", 0) for ep in group) / len(group)
                fr = max(ep.get("restart_from", 0) + ep.get("spheres", 0)
                         for ep in group)
                print(f"  ep {i+1:>6}-{i+len(group):<6}  "
                      f"gained avg={g:5.1f}  frontier={fr}")

    if successes:
        section("SUCCESSES")
        chaos = sum(1 for ep in successes if ep.get("emerald") == "chaos")
        super_ = sum(1 for ep in successes if ep.get("emerald") == "super")
        avg_steps = int(sum(ep.get("steps", 0) for ep in successes) / len(successes))
        print(f"  Chaos emeralds:  {chaos}")
        print(f"  Super emeralds:  {super_}")
        print(f"  Avg steps to win:{avg_steps}")

    death_counts = [ep.get("death_at_sphere") for ep in deaths]
    if death_counts:
        section("DEATH CLUSTERS  (sphere # when red sphere hit)")
        buckets = {}
        for d in death_counts:
            b = (d // 5) * 5
            buckets[b] = buckets.get(b, 0) + 1
        max_count = max(buckets.values())
        for b in sorted(buckets):
            label = f"{b:>3}-{b+4}"
            print(f"  sphere {label}  {bar(buckets[b], max_count, 30)} {buckets[b]}")

        section("DEATHS BY SPEED LEVEL")
        by_speed = {}
        for ep in deaths:
            s = ep.get("speed_level", 0)
            by_speed.setdefault(s, []).append(ep.get("death_at_sphere", 0))
        for s in sorted(by_speed):
            ds = by_speed[s]
            print(f"  Speed {s}:  {len(ds):5d} deaths, "
                  f"avg sphere at death {sum(ds)/len(ds):5.1f}")

    print()


if __name__ == "__main__":
    main()
