#!/usr/bin/env python3
"""
Curriculum log analyzer for Sonic RL agent.
Usage: python3 analyze_curriculum.py [logfile]
Default logfile: ~/sonic-rl/curriculum_log.jsonl
"""

import json
import os
import sys
from collections import Counter, defaultdict

LOG_FILE = os.path.expanduser("~/sonic-rl/curriculum_log.jsonl")


def load_episodes(path):
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


def cluster(values, bucket=100):
    return Counter(round(v, -2) for v in values)


def bar(count, max_count, width=40):
    filled = int(width * count / max_count) if max_count > 0 else 0
    return "█" * filled + "░" * (width - filled)


def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")

def _load_landmarks():
    path = os.path.expanduser('~/sonic-rl/landmarks.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return {int(k): v for k, v in data.get('landmarks', {}).items()}
        except Exception:
            pass
    return {}

LANDMARKS = _load_landmarks()

def _landmark_label(x, radius=150):
    """Return landmark label if any known landmark is within radius of x."""
    best = None
    best_dist = radius
    for lx, label in LANDMARKS.items():
        dist = abs(x - lx)
        if dist < best_dist:
            best_dist = dist
            best = label
    return f' [{best}]' if best else ''

    print(f"{'═' * 60}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE
    if not os.path.exists(path):
        print(f"Log file not found: {path}")
        return

    episodes = load_episodes(path)
    if not episodes:
        print("No episodes found.")
        return

    print(f"\nLoaded {len(episodes)} episodes from {path}")

    # ── Overview ─────────────────────────────────────────────────────────
    section("OVERVIEW")
    best_xs = [ep["best_x"] for ep in episodes if ep.get("best_x") is not None]
    steps   = [ep["steps"]  for ep in episodes if ep.get("steps")  is not None]
    deaths  = [ep["death_x"] for ep in episodes if ep.get("death_x") is not None]

    if best_xs:
        print(f"  Episodes:         {len(episodes)}")
        print(f"  Best x ever:      {max(best_xs)}")
        print(f"  Avg best x:       {int(sum(best_xs)/len(best_xs))}")
        print(f"  Median best x:    {sorted(best_xs)[len(best_xs)//2]}")
        print(f"  Avg episode len:  {int(sum(steps)/len(steps))} steps")
        print(f"  Death rate:       {len(deaths)/len(episodes)*100:.1f}% of episodes")

    # ── Best x over time (learning curve) ────────────────────────────────
    section("LEARNING CURVE  (best_x per episode, grouped in 50s)")
    if best_xs:
        chunk = 50
        for i in range(0, len(best_xs), chunk):
            group = best_xs[i:i+chunk]
            avg = int(sum(group) / len(group))
            mx  = max(group)
            ep_range = f"ep {i+1}-{i+len(group)}"
            print(f"  {ep_range:<14} avg={avg:<6} max={mx}")

    # ── Stuck event clusters ──────────────────────────────────────────────
    section("STUCK CLUSTERS  (obstacle map)")
    all_stuck = []
    for ep in episodes:
        # Filter out x~0 which is a reset artifact
        all_stuck.extend(x for x in ep.get("stuck_xs", []) if x > 100)

    if all_stuck:
        clusters = cluster(all_stuck)
        max_count = max(clusters.values())
        print(f"  Total stuck events: {len(all_stuck)}")
        print()
        for x in sorted(clusters):
            count = clusters[x]
            if count >= 2:
                print(f"  x~{x:<6} {bar(count, max_count)} {count}")
    else:
        print("  No stuck events recorded yet.")

    # ── Death clusters ────────────────────────────────────────────────────
    section("DEATH CLUSTERS  (where the agent dies)")
    if deaths:
        dclusters = cluster(deaths)
        max_count = max(dclusters.values())
        print(f"  Total deaths recorded: {len(deaths)}")
        print()
        for x in sorted(dclusters):
            count = dclusters[x]
            if count >= 2:
                print(f"  x~{x:<6} {bar(count, max_count)} {count}")
    else:
        print("  No deaths recorded yet.")

    # ── Progress vs death correlation ─────────────────────────────────────
    section("PROGRESS vs DEATH  (did it die or time out?)")
    died      = [ep for ep in episodes if ep.get("death_x") is not None]
    timed_out = [ep for ep in episodes if ep.get("death_x") is None]
    print(f"  Died:      {len(died)} episodes  ({len(died)/len(episodes)*100:.1f}%)")
    print(f"  Timed out: {len(timed_out)} episodes  ({len(timed_out)/len(episodes)*100:.1f}%)")
    if died:
        avg_death_x = int(sum(ep["death_x"] for ep in died) / len(died))
        print(f"  Avg death x: {avg_death_x}")
    if timed_out:
        valid = [ep for ep in timed_out if ep.get("best_x")]
        if valid:
            avg_timeout_best = int(sum(ep["best_x"] for ep in valid) / len(valid))
            print(f"  Avg best_x on timeout: {avg_timeout_best}")

    # ── Episode efficiency ────────────────────────────────────────────────
    section("EPISODE EFFICIENCY  (best_x per step)")
    efficient = [(ep["best_x"] / ep["steps"], ep) for ep in episodes
                 if ep.get("best_x") and ep.get("steps") and ep["steps"] > 0]
    if efficient:
        efficient.sort(key=lambda t: t[0], reverse=True)
        print("  Top 5 most efficient episodes:")
        for ratio, ep in efficient[:5]:
            print(f"    best_x={ep['best_x']}  steps={ep['steps']}  ratio={ratio:.3f}")
        print()
        avg_ratio = sum(r for r, _ in efficient) / len(efficient)
        print(f"  Avg efficiency: {avg_ratio:.4f} x/step")

    # ── Ring collection ──────────────────────────────────────────────────
    section("RING COLLECTION")
    max_rings_list = [ep["max_rings"] for ep in episodes if ep.get("max_rings") is not None]
    if max_rings_list:
        print(f"  Avg max rings per episode: {int(sum(max_rings_list)/len(max_rings_list))}")
        print(f"  Best max rings:            {max(max_rings_list)}")
        zero_ring_eps = sum(1 for r in max_rings_list if r == 0)
        print(f"  Episodes with 0 rings:     {zero_ring_eps} ({zero_ring_eps/len(max_rings_list)*100:.1f}%)")

    # ── Milestones ────────────────────────────────────────────────────────
    section("MILESTONES")
    act_completions = sum(1 for ep in episodes if ep.get("act_completed"))
    boss_encounters = sum(1 for ep in episodes if ep.get("boss_encountered"))
    super_episodes  = sum(1 for ep in episodes if ep.get("was_super"))
    emerald_episodes = [ep for ep in episodes if ep.get("max_chaos_emeralds", 0) > 0]
    ring_entries = sum(ep.get("giant_rings_entered", 0) for ep in episodes)
    ring_episodes = sum(1 for ep in episodes if ep.get("giant_rings_entered", 0) > 0)
    ss_failures = sum(ep.get("ss_failures", 0) for ep in episodes)
    ss_successes = sum(ep.get("ss_successes", 0) for ep in episodes)
    ss_total = ss_failures + ss_successes
    ss_rate = f"{ss_successes/ss_total*100:.1f}%" if ss_total > 0 else "N/A"
    print(f"  Act completions:    {act_completions}")
    print(f"  Boss encounters:    {boss_encounters}")
    print(f"  Giant rings entered:{ring_entries} times across {ring_episodes} episodes")
    print(f"  SS attempts:        {ss_total} ({ss_successes} success, {ss_failures} failed, {ss_rate} success rate)")
    sphere_counts = [ep.get("max_spheres_collected", 0) for ep in episodes]
    streak_counts = [ep.get("ss_best_streak", 0) for ep in episodes]
    speed_levels = [ep.get("ss_speed_level", 0) for ep in episodes]
    if any(s > 0 for s in sphere_counts):
        avg_spheres = sum(sphere_counts) / len(sphere_counts)
        max_spheres = max(sphere_counts)
        episodes_with_spheres = sum(1 for s in sphere_counts if s > 0)
        avg_streak = sum(streak_counts) / len(streak_counts)
        max_streak = max(streak_counts)
        max_speed = max(speed_levels)
        episodes_with_speed = sum(1 for s in speed_levels if s > 0)
        print(f"  Blue spheres:       avg={avg_spheres:.1f} max={max_spheres} ({episodes_with_spheres} episodes with at least 1)")
        print(f"  Best streak:        avg={avg_streak:.1f} max={max_streak} consecutive spheres")
        print(f"  Speed triggers:     max={max_speed} ({episodes_with_speed} episodes with at least 1 speed up)")
        # Learning curve for sphere collection
        if len(sphere_counts) >= 50:
            groups = []
            for i in range(0, len(sphere_counts), 50):
                group = sphere_counts[i:i+50]
                streak_group = streak_counts[i:i+50]
                groups.append(f"ep {i+1}-{i+len(group)}: spheres avg={sum(group)/len(group):.1f} max={max(group)} streak max={max(streak_group)}")
            print(f"  Progression:")
            for g in groups:
                print(f"    {g}")
    print(f"  Super Sonic:        {super_episodes} episodes")
    print(f"  Chaos emeralds:     {len(emerald_episodes)} episodes with at least 1")
    if emerald_episodes:
        max_em = max(ep["max_chaos_emeralds"] for ep in emerald_episodes)
        print(f"  Max emeralds ever:  {max_em}")

    # ── Camera vs character x ─────────────────────────────────────────────
    section("CAMERA vs CHARACTER X")
    screen_xs = [ep["best_screen_x"] for ep in episodes if ep.get("best_screen_x") is not None]
    if screen_xs and best_xs:
        avg_screen = int(sum(screen_xs)/len(screen_xs))
        avg_char   = int(sum(best_xs)/len(best_xs))
        print(f"  Avg best character x: {avg_char}")
        print(f"  Avg best camera x:    {avg_screen}")
        print(f"  Avg divergence:       {avg_char - avg_screen} px")
        print(f"  Max camera x ever:    {max(screen_xs)}")

    # ── Suggested curriculum checkpoints ─────────────────────────────────
    section("SUGGESTED CURRICULUM CHECKPOINTS")
    if all_stuck:
        clusters_sorted = sorted(cluster(all_stuck).items())  # (x, count) ascending by x

        # Find the spawn x — minimum best_x across episodes as a proxy
        spawn_x = min((x for x in all_stuck if x > 100), default=0)

        # Identify obstacle peaks: x positions where stuck count is
        # significantly higher than local neighbors
        xs     = [x for x, _ in clusters_sorted]
        counts = [c for _, c in clusters_sorted]
        max_c  = max(counts)
        threshold = max_c * 0.15  # must be at least 15% of the peak

        # Find local maxima above threshold
        obstacles = []
        for i, (x, c) in enumerate(clusters_sorted):
            if c < threshold:
                continue
            left  = counts[i-1] if i > 0 else 0
            right = counts[i+1] if i < len(counts)-1 else 0
            if c >= left and c >= right:
                obstacles.append(x)

        # Merge obstacles within 300px of each other
        merged = []
        for x in sorted(obstacles):
            if not merged or x - merged[-1] > 300:
                merged.append(x)

        if merged:
            print("  Based on stuck event clusters:\n")
            prev_spawn = spawn_x
            for i, obs_x in enumerate(merged):
                clear_x = obs_x + 200  # spawn past the obstacle with margin
                print(f"  Stage {i}: spawn at x~{int(prev_spawn):<6}  →  obstacle at x~{int(obs_x)}")
                prev_spawn = clear_x
            print(f"  Stage {len(merged)}: spawn at x~{int(prev_spawn):<6}  →  end of act")
        else:
            print("  Not enough data yet for checkpoint suggestions.")
    else:
        print("  Need stuck event data to suggest checkpoints.")

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()
