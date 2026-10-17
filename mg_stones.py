#!/usr/bin/env python3
"""mg_stones.py — automatic stepping stones: where an act's start point has stalled, add start points
between it and the next one (Tyler 2026-10-12: "This game is riddled with precise jumps"; hand-placed
stones broke AIZ1's climb and its path-swapper loop-back).

    python3 mg_stones.py RUN [--dry] [--force ACT] [--state logs/RUN_watch_state.json]
    exit 0 = stones added (restart the run: train_main reads them), 2 = nothing to add, 1 = error

A start point has stalled when mg_status.py's "stepping stones" flag would fire: >= 200 episodes began
AT it since it arrived (since_arrival below) and < 8 of its last 32 matched him, while the next
start point (one second later in his run) matches him at that same 8-of-32 bar (next_point_works:
otherwise the whole stretch is hard, like mg2's MGZ2 at 284-288, and stones only cost restarts; Tyler
2026-10-15: add the guard). Then the hard part lies between the two: stones go at the quarter points of that gap (>= 16 demo frames), at its
middle (8-15), or nowhere (< 8, about two agent steps: that's for the imitation pull, not stones).
They are added to main_<run>_stones.json (train_main merges them into MG_EXTRA_STARTS at the next
start, so the env builds them like RUNS 'extra_starts'; check_extra_starts.py covers both), and the
act's pointer moves onto the last new stone (the one nearest the start point that works), shifting
by the number of stones added; other acts' pointers don't move. --dry prints, writes nothing.
Start-point times come from main_<run>_starts.json (the trainer writes it at each start); for a run
older than that, the run's segments are built here once to make it (CPU, a minute or more).
"""
import argparse
import json
import os
import sys

HOME = os.path.dirname(os.path.abspath(__file__))
STONES_AT, WINDOW, MATCHES = 200, 32, 8


def since_arrival(rows, p, floor=0):
    """The episodes of one act (rows, in log order) that began AT start point p since p became the
    act's pointer. Starts are drawn from [pointer, pointer + band] and the pointer only moves back,
    so an episode at p came after the move, except one from an earlier visit to p before a rollback
    moved the pointer forward again: an episode that began below p marks that, so only those after
    the last one below p count. floor: none earlier than this (unix time; mg_status.py's look before
    the move, which also covers stones renumbering the start points). Until 2026-10-15 this was the
    curriculum file's last write, which any act's move rewrites: in a six-act run it hardly fired.
    Boss-, ring- and front-share starts begin below p on purpose and don't mark a rollback."""
    lo = max([floor] + [r['t'] for r in rows if 0 < r['start'] < p
                        and not r.get('boss_share') and not r.get('ring_share') and not r.get('front_share')])
    return [r for r in rows if r['start'] == p and r['t'] >= lo]


def next_point_works(rows, p, at):
    """(works, rate, n): how start point p + 1 matched him in the episodes since p became the pointer
    (at = since_arrival's list); works = at the bar the pointer moves on, MATCHES of WINDOW."""
    t0 = at[0]['t'] if at else float('inf')
    nxt = [r for r in rows if r['start'] == p + 1 and r['t'] >= t0]
    rate = sum(r['success'] for r in nxt) / len(nxt) if nxt else 0.0
    return rate >= MATCHES / WINDOW, rate, len(nxt)


def start_times(run, P):
    try:
        return json.load(open(P['starts']))
    except (OSError, ValueError):
        pass
    import train_main
    train_main.configure_env(run)
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    env = C.SonicCoordinator(demo_prob=0.0)
    starts = {d['key']: [round(f * d['T']) for f in d['fracs']] for d in env.demo_info()}
    json.dump(starts, open(P['starts'], 'w'))
    return starts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--force', metavar='ACT', help='treat this act as stalled (testing)')
    ap.add_argument('--state', help="mg_status.py's state, for when each pointer arrived "
                                    "(default logs/RUN_watch_state.json)")
    a = ap.parse_args()
    sys.path.insert(0, HOME)
    import train_main
    P = train_main.paths(a.run)
    cur = json.load(open(P['curriculum']))
    try:
        st = json.load(open(a.state or os.path.join(HOME, 'logs', f'{a.run}_watch_state.json')))
    except (OSError, ValueError):
        st = {}
    rows = []
    try:
        for line in open(P['log']):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    starts = start_times(a.run, P)
    try:
        stones = json.load(open(P['stones']))
    except (OSError, ValueError):
        stones = {}
    added = {}
    for key, c in cur.items():
        act = key.split(':')[-1]
        p = c['ptr']
        if p == 0:      # the act's own start: a stall there is beating his time from the start, not a hard
            continue    # stretch, and stones would move the pointer forward (mg2 AIZ1, 2026-10-17 03:14)
        s = st.get(key, {})
        rs = [r for r in rows if r['act'] == act]
        at = since_arrival(rs, p, s.get('floor', 0) if s.get('ptr') == p else 0)
        matched = sum(r['success'] for r in at[-WINDOW:])
        stalled = len(at) >= STONES_AT and matched < MATCHES
        works, rate, n = next_point_works(rs, p, at)
        if stalled and not works and not a.force:
            print(f"{key}: start point {p} stalled ({len(at)} starts there, {matched} of the last "
                  f"{min(len(at), WINDOW)} matched), but start point {p + 1} matched only {rate:.0%} "
                  f"of {n}: hard throughout, not stones' job")
            continue
        if a.force:
            stalled = a.force == act
        if not stalled:
            continue
        ts = starts.get(key)
        if not ts or p + 1 >= len(ts):
            print(f'{key}: start point {p} has no next start point to split toward')
            continue
        t0, t1 = ts[p], ts[p + 1]
        gap = t1 - t0
        new = ([t0 + gap * i // 4 for i in (1, 2, 3)] if gap >= 16 else [t0 + gap // 2] if gap >= 8 else [])
        new = [t for t in new if t not in ts]
        print(f"{key}: start point {p} (t {t0}) stalled: {len(at)} starts there, {matched} of the last "
              f"{min(len(at), WINDOW)} matched; next start point at t {t1} (gap {gap} frames) -> "
              + (f'stones at t {new}' if new else 'too fine for stones: the imitation pull\'s job'))
        if not new:
            continue
        added[key] = (p, new)
    if not added:
        sys.exit(2)
    if a.dry:
        print('(dry run: nothing written)')
        return
    for key, (p, new) in added.items():
        stones[key] = sorted(set(stones.get(key, [])) | set(new))
        T_est = starts[key][p] / cur[key]['frac'] if cur[key]['frac'] else None
        cur[key]['ptr'] = p + len(new)                   # onto the last new stone
        if T_est:
            cur[key]['frac'] = round(max(new) / T_est, 4)
    with open(P['stones'] + '.tmp', 'w') as f:
        json.dump(stones, f, indent=1)
    os.replace(P['stones'] + '.tmp', P['stones'])
    with open(P['curriculum'] + '.tmp', 'w') as f:
        json.dump(cur, f, indent=1)
    os.replace(P['curriculum'] + '.tmp', P['curriculum'])
    os.remove(P['starts'])                               # stale now; the trainer rewrites it at start
    print(f"wrote {P['stones']} and moved the pointers: restart {a.run} (--resume) to build the stones")


if __name__ == '__main__':
    main()
