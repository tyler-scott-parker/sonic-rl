"""Is any blue inside the agent's 15x15 view (toroidal Chebyshev distance <= 7, as probe_visibility.py)
during the 600-frame stall window? Yardstick: frames while collecting should be ~100%.
Usage: stall_view.py <steer_episodes json>"""
import json, sys, bisect
from statistics import median
R, STALL = 7, 600
cell = lambda fr: (((fr['y'] + 0x80) >> 8) % 32, ((fr['x'] + 0x80) >> 8) % 32)
def nearest(grid, r0, c0):
    best = 99
    for r in range(32):
        for c in range(32):
            if grid[r][c] == 2:
                dr = abs(r - r0); dr = min(dr, 32 - dr)
                dc = abs(c - c0); dc = min(dc, 32 - dc)
                best = min(best, max(dr, dc))
    return best
eps = json.load(open(sys.argv[1]))
stall_in, stall_d, coll_in, n_st = [], [], [], 0
for e in eps:
    boards = sorted(e['boards'], key=lambda b: b['i']); idx = [b['i'] for b in boards]
    fr = e['frames']
    def d_at(k):
        g = boards[max(0, bisect.bisect_right(idx, k) - 1)]['grid']
        return nearest(g, *cell(fr[k]))
    if e['stalled']:
        n_st += 1
        ks = range(max(0, len(fr) - STALL), len(fr), 10)        # every 10th frame of the window
        ds = [d_at(k) for k in ks]
        stall_in.append(sum(d <= R for d in ds) / len(ds)); stall_d.append(median(ds))
    ks = range(0, max(1, len(fr) - STALL), 50)                   # before any stall window
    coll_in += [d_at(k) <= R for k in ks]
print(f'{len(eps)} episodes, {n_st} stalled')
print(f'  yardstick, frames before the stall window: blue in view {100 * sum(coll_in) / len(coll_in):.0f}% of {len(coll_in)}')
if stall_in:
    print(f'  stall windows: blue in view median {100 * median(stall_in):.0f}% of the window; '
          f'windows with it in view >= half the time: {sum(x >= 0.5 for x in stall_in)} of {n_st}')
    print(f'  nearest blue (Chebyshev cells) in stall windows: median of window medians {median(stall_d)}')
