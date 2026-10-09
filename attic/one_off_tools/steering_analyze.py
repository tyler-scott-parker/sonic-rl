"""Summarize probe_steering.py: route decisions at intersections.

At every intersection Sonic reaches, the move he made (straight / left /
right) is compared with the moves that start a SHORTEST route to a blue
sphere, found by a heading-aware search on the board as it was then (no
reversing, no crossing red or bumpers, no jumps). Tyler's run is the
yardstick for the search itself.

usage: steering_analyze.py [dir]   (default: this script's directory)
"""
import os, sys, json
from collections import Counter, deque
from statistics import median
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
DIRS = [(-1, 0), (0, -1), (1, 0), (0, 1)]        # N W S E, as (drow, dcol) — the quadrant order
BLOCKED = (1, 3, 10)                             # red, bumper, turning red
MOVES = {0: 'S', 1: 'L', 3: 'R', 2: 'rev'}       # (out - in) % 4; a left turn raises the heading
TURN_ACTS = {'L': (2, 5), 'R': (3, 6)}


def cell(fr):
    return ((fr['y'] + 0x80) >> 8) % 32, ((fr['x'] + 0x80) >> 8) % 32


def step_dir(a, b):
    dr = (b[0] - a[0] + 16) % 32 - 16
    dc = (b[1] - a[1] + 16) % 32 - 16
    return DIRS.index((dr, dc)) if (dr, dc) in DIRS else None


def route(grid, r0, c0, q0):
    """Shortest no-jump route from intersection (r0, c0), arriving heading q0,
    to any blue sphere. Returns (cells, set of first moves that achieve it,
    set of legal first moves); cells is None if no blue is reachable."""
    legal, frontier = set(), []
    for m, q in (('S', q0), ('L', (q0 + 1) % 4), ('R', (q0 - 1) % 4)):
        r, c = (r0 + DIRS[q][0]) % 32, (c0 + DIRS[q][1]) % 32
        if grid[r][c] not in BLOCKED:
            legal.add(m)
            frontier.append((r, c, q, m))
    seen = {}                                     # (r, c, q) -> first moves at the shortest depth
    d = 1
    while frontier:
        hit = {m for r, c, q, m in frontier if grid[r][c] == 2}
        if hit:
            return d, hit, legal
        nxt = {}
        for r, c, q, m in frontier:
            for q2 in (q, (q + 1) % 4, (q - 1) % 4):
                r2, c2 = (r + DIRS[q2][0]) % 32, (c + DIRS[q2][1]) % 32
                if grid[r2][c2] in BLOCKED:
                    continue
                k = (r2, c2, q2)
                if k in seen:
                    continue
                nxt.setdefault(k, set()).add(m)
        for k, ms in nxt.items():
            seen[k] = ms
        frontier = [(r, c, q, m) for (r, c, q), ms in nxt.items() for m in ms]
        d += 1
        if d > 200:
            break
    return None, set(), legal


def board_at(boards, i):
    g = boards[0]['grid']
    for b in boards:
        if b['i'] > i:
            break
        g = b['grid']
    return g


def intersections(rec):
    """One row per intersection Sonic passed through: arrival frame index,
    cell, heading in/out, the move, and the route search at arrival."""
    fr = rec['frames']
    seq = []                                       # (first frame index on the cell, cell)
    for i, f in enumerate(fr):
        c = cell(f)
        if not seq or seq[-1][1] != c:
            seq.append((i, c))
    rows = []
    for (ia, a), (ib, b), (ic, c) in zip(seq, seq[1:], seq[2:]):
        qin, qout = step_dir(a, b), step_dir(b, c)
        if qin is None or qout is None:
            continue                               # airborne skip (spring) — not a decision
        # the centre: the frame on b closest to its middle
        span = range(ib, ic)
        centre = min(span, key=lambda i: abs(((fr[i]['x'] - (b[1] << 8) + 4096) % 8192) - 4096)
                     + abs(((fr[i]['y'] - (b[0] << 8) + 4096) % 8192) - 4096))
        grid = board_at(rec['boards'], ib + 1)     # +1: b's own sphere is gone by then
        dist, opt, legal = route(grid, b[0], b[1], qin)
        rows.append(dict(i=ib, centre=centre, cell=b, qin=qin, move=MOVES[(qout - qin) % 4],
                         dist=dist, opt=opt, legal=legal, got=fr[ib]['got'],
                         air=bool(fr[centre]['jump'] & 0x80), corner=[fr[k]['corner'] for k in span]))
    for prev, r in zip([None] + rows, rows):
        r['from'] = prev['centre'] if prev else max(0, r['i'] - 16)
    return rows


def window_decisions(rec, r):
    """Decisions that could queue the turn at this intersection: made after
    the previous centre, up to this one."""
    return [d for d in rec['decisions'] if r['from'] < d['i'] + 1 <= r['centre']]


def pct(n, d):
    return f'{100 * n / d:.0f}%' if d else '-'


def agreement(rows, label):
    rows = [r for r in rows if r['dist'] is not None and not r['air']]
    ok = sum(r['move'] in r['opt'] for r in rows)
    chance = sum(len(r['opt'] & r['legal']) / len(r['legal']) for r in rows if r['legal'])
    print(f'  {label}: {len(rows)} intersections, move on a shortest route {pct(ok, len(rows))}'
          f' (random legal move: {pct(chance, len(rows))})')
    return rows


def by_dist(rows, label):
    out = []
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 999)):
        rs = [r for r in rows if r['dist'] is not None and lo <= r['dist'] <= hi and not r['air']]
        if rs:
            ok = sum(r['move'] in r['opt'] for r in rs)
            ch = sum(len(r['opt'] & r['legal']) / len(r['legal']) for r in rs if r['legal'])
            out.append(f"{lo}{'' if lo == hi else '-' + str(hi) if hi < 999 else '+'}: {pct(ok, len(rs))} vs {pct(ch, len(rs))} (n={len(rs)})")
    print(f'  {label} by distance to nearest blue — agree vs chance:\n     ' + ' | '.join(out))


def errors(rows, label):
    """How the wrong moves were wrong; 'late' = went straight where the turn was
    due and made that turn at the next intersection; 'early' = turned one
    intersection before the turn was due."""
    c = Counter()
    for k, r in enumerate(rows):
        if r['dist'] is None or r['air'] or r['move'] in r['opt']:
            continue
        nxt = rows[k + 1] if k + 1 < len(rows) else None
        if r['move'] == 'S':
            due = r['opt'] - {'S'}
            late = nxt is not None and nxt['move'] in due and nxt['i'] - r['i'] < 40
            c['went straight, turn was due' + (' (made it 1 late)' if late else '')] += 1
        elif 'S' in r['opt']:
            c['turned, straight was right'] += 1
        elif r['move'] == 'rev':
            c['reversed (bumper)'] += 1
        else:
            c['turned the wrong way'] += 1
    n = sum(c.values())
    print(f'  {label} wrong moves ({n}): ' + ', '.join(f'{k} {pct(v, n)}' for k, v in c.most_common()))


# ── load ────────────────────────────────────────────────────────────────────
demo = json.load(open(os.path.join(D, 'steer_demo.json')))
eps = json.load(open(os.path.join(D, 'steer_episodes.json')))
demo['decisions'] = []

# ── yardsticks ──────────────────────────────────────────────────────────────
print('== yardsticks ==')
drows = intersections(demo)
pre = [r for r in drows if r['dist'] == 1]
took = sum(r['move'] in r['opt'] for r in pre)
print(f'  Tyler: {len(drows)} intersections; with a blue 1 step away he took it {pct(took, len(pre))} (n={len(pre)})')
# the queued-corner byte must agree with the inferred move: +4 left, -4 right
agree = tot = 0
for rows in [drows] + [intersections(e) for e in eps[:30]]:
    for r in rows:
        if r['move'] in ('L', 'R') and not r['air']:
            tot += 1
            want = 4 if r['move'] == 'L' else -4
            agree += want in r['corner']
print(f'  turn direction inferred from cells vs the queued-corner byte: {pct(agree, tot)} of {tot} turns')
# every pickup should land on a cell that was blue just before
bad = n = 0
for e in eps[:30]:
    fr = e['frames']
    for i in range(1, len(fr)):
        if fr[i]['got'] > fr[i - 1]['got']:
            n += 1
            g = board_at(e['boards'], i - 1)
            r, c = cell(fr[i])
            bad += g[r][c] != 2 and all(g[(r + dr) % 32][(c + dc) % 32] != 2 for dr, dc in DIRS)
print(f'  pickups on (or next to) a cell that was blue: {pct(n - bad, n)} of {n}')

# ── Tyler vs agent ──────────────────────────────────────────────────────────
print('\n== moves vs a shortest route to the nearest blue ==')
t_all = agreement(drows, 'Tyler, whole run')
t_end = agreement([r for r in drows if r['i'] >= demo['start_frame']], 'Tyler, from 79%')
by_dist(t_all, 'Tyler')

out = Counter('cleared' if e['cleared'] else 'stall' if e['stalled']
              else 'timeout' if e['timeout'] else 'red' for e in eps)
print(f'\n  agent: {len(eps)} episodes from 79%: {dict(out)}')
stall_rows, other_rows, ep_rows = [], [], []
for e in eps:
    rows = intersections(e)
    final = e['frames'][-1]['got']
    last_i = max([i for i, f in enumerate(e['frames']) if f['got'] < final] + [-1]) + 1
    for r in rows:
        r['ep'] = e
        r['stall'] = e['stalled'] and r['i'] >= last_i
        (stall_rows if r['stall'] else other_rows).append(r)
    ep_rows.append(rows)
a_other = agreement(other_rows, 'agent, while collecting')
a_stall = agreement(stall_rows, 'agent, stall windows')
by_dist(a_other, 'agent collecting')
by_dist(a_stall, 'agent stalled')

print('\n== how the wrong moves were wrong ==')
errors(drows, 'Tyler')
for label, flag in (('agent collecting', False), ('agent stalled', True)):
    errors([r for rows in ep_rows for r in rows if r['stall'] == flag], label)

# ── what the policy wanted at intersections where a turn was due ────────────
print('\n== where a turn was due (shortest routes all turn) ==')
for label, rows in (('collecting', a_other), ('stalled', a_stall)):
    due = [r for r in rows if r['opt'] and 'S' not in r['opt']]
    pmax, pany, made = [], [], 0
    for r in due:
        ws = window_decisions(r['ep'], r)
        acts = [a for m in r['opt'] for a in TURN_ACTS.get(m, ())]
        ps = [sum(d['p'][a] for a in acts) for d in ws]
        if ps:
            pmax.append(max(ps)); pany.append(1 - np.prod([1 - p for p in ps]))
        made += r['move'] in r['opt']
    if due:
        print(f'  {label}: {len(due)} | made the turn {pct(made, len(due))} | P(that turn) per decision, max in the window:'
              f' median {median(pmax):.2f} | chance it is pressed at least once in the window: median {median(pany):.2f}'
              f' | decisions per window: median {median(len(window_decisions(r["ep"], r)) for r in due)}')

# ── the stall windows themselves ────────────────────────────────────────────
print('\n== stall windows ==')
st = [e for e in eps if e['stalled']]
for e in st:
    fr = e['frames']; final = fr[-1]['got']
    e['_win0'] = max([i for i, f in enumerate(fr) if f['got'] < final] + [-1]) + 1
moves = Counter(r['move'] for r in stall_rows)
cmoves = Counter(r['move'] for r in other_rows)
tmoves = Counter(r['move'] for r in t_end)
fmt = lambda c: ', '.join(f'{k} {pct(v, sum(c.values()))}' for k, v in sorted(c.items()))
print(f'  moves at intersections — stalled: {fmt(moves)} | collecting: {fmt(cmoves)} | Tyler from 79%: {fmt(tmoves)}')
dec_s = [d for e in st for d in e['decisions'] if d['i'] >= e['_win0']]
dec_c = [d for e in eps for d in e['decisions'] if not (e['stalled'] and d['i'] >= e.get('_win0', 10**9))]
for label, ds in (('stalled', dec_s), ('collecting', dec_c)):
    P = np.array([d['p'] for d in ds])
    ent = -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)
    acts = Counter(d['a'] for d in ds)
    print(f'  {label}: {len(ds)} decisions | mean P: noop {P[:,0].mean():.2f} jump {P[:,1].mean():.2f} L {P[:,2].mean():.2f}'
          f' R {P[:,3].mean():.2f} up {P[:,4].mean():.2f} jL {P[:,5].mean():.2f} jR {P[:,6].mean():.2f}'
          f' | entropy median {np.median(ent):.2f} (max {np.log(7):.2f}) | airborne {pct(sum(1 for d in ds if d["a"] in (1,5,6)), len(ds))} of picks')
# does the distance to the nearest blue shrink over the window?
trend = Counter()
dists = []
for e in st:
    rs = [r for r in stall_rows if r['ep'] is e and r['dist'] is not None]
    if len(rs) >= 2:
        d0, d1, dmin = rs[0]['dist'], rs[-1]['dist'], min(r['dist'] for r in rs)
        dists.append((d0, dmin, d1, len(rs)))
        trend['got closer' if d1 < d0 else 'same' if d1 == d0 else 'farther'] += 1
if dists:
    print(f'  route distance, first -> closest -> last intersection of the window: medians'
          f' {median(x[0] for x in dists)} -> {median(x[1] for x in dists)} -> {median(x[2] for x in dists)};'
          f' intersections per window median {median(x[3] for x in dists)}; {dict(trend)}')
cells = [len({r['cell'] for r in stall_rows if r['ep'] is e}) / max(1, sum(r['ep'] is e for r in stall_rows)) for e in st]
print(f'  distinct cells / intersections in a window: median {median(cells):.2f} (1.0 = never revisits)')

# ── jumping over the spheres it needs ───────────────────────────────────────
print('\n== airborne ==')


def flyovers(rec, i0, i1):
    """Blue cells Sonic passed over in the air between frames i0 and i1 —
    cells that were blue as he crossed them and that he did not collect."""
    fr, n, seen = rec['frames'], 0, set()
    for i in range(max(i0, 1), i1):
        c = cell(fr[i])
        if fr[i]['jump'] & 0x80 and c not in seen:
            g = board_at(rec['boards'], i)
            if g[c[0]][c[1]] == 2 and fr[i]['got'] == fr[i - 1]['got']:
                n += 1
                seen.add(c)
    return n


air = lambda rs: pct(sum(r['air'] for r in rs), len(rs))
fair = lambda recs: pct(sum(f['jump'] >> 7 for e, i0 in recs for f in e['frames'][i0:]),
                        sum(len(e['frames'][i0:]) for e, i0 in recs))
print(f'  frames airborne ($E432 bit 7) — Tyler from 79%: {fair([(demo, demo["start_frame"])])}'
      f' | agent, whole episodes: {fair([(e, 0) for e in eps])} | stall windows: {fair([(e, e["_win0"]) for e in st])}')
tm = Counter(r['move'] for r in stall_rows + other_rows if r['air'])
print(f'  moves made at intersections crossed in the air: {dict(tm)}')
print(f'  intersections crossed in the air — Tyler from 79%: {air([r for r in drows if r["i"] >= demo["start_frame"]])}'
      f' | agent collecting: {air(other_rows)} | stalled: {air(stall_rows)}')
print(f'  blues jumped over — Tyler from 79%: {flyovers(demo, demo["start_frame"], len(demo["frames"]))}')
fo = [flyovers(e, e['_win0'], len(e['frames'])) for e in st]
print(f'  blues jumped over inside stall windows: {sum(fo)} in {len(fo)} windows; windows with at least one: {pct(sum(1 for x in fo if x), len(fo))}')
