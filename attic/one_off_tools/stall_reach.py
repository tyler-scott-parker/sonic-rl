"""In stall windows: is the nearest blue reachable along grid lines without touching red/bumpers
(walk_dist copied from probe_visibility.py), and how far by that route vs straight-line?
Usage: stall_reach.py <steer_episodes json>"""
import json, sys, bisect
from collections import deque, Counter
from statistics import median
cell = lambda fr: (((fr['y'] + 0x80) >> 8) % 32, ((fr['x'] + 0x80) >> 8) % 32)
def walk_dist(grid, r0, c0):
    seen = {(r0, c0)}; q = deque([(r0, c0, 0)])
    while q:
        r, c, d = q.popleft()
        if d and grid[r][c] == 2:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = (r + dr) % 32, (c + dc) % 32
            if (nr, nc) not in seen and grid[nr][nc] not in (1, 3, 10):
                seen.add((nr, nc)); q.append((nr, nc, d + 1))
    return None
eps = json.load(open(sys.argv[1]))
res = Counter(); dists = []
for e in eps:
    if not e['stalled']:
        continue
    boards = sorted(e['boards'], key=lambda b: b['i']); idx = [b['i'] for b in boards]
    fr = e['frames']; k = max(0, len(fr) - 600)          # start of the stall window
    g = boards[max(0, bisect.bisect_right(idx, k) - 1)]['grid']
    d = walk_dist(g, *cell(fr[k]))
    res['reachable' if d is not None else 'walled off (jump only)'] += 1
    if d is not None:
        dists.append(d)
print(f'stall windows: {dict(res)}; red-free route length to nearest blue: median {median(dists) if dists else None}')
