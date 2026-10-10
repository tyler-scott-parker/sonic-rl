"""Summarize probe_steering_v4.py (run d3, env v4): d3's gate check.

The question from d2: where only straight was right, does it go straight on
the ground? (d2 on v3: 6-9%; it jumped instead and flew past the stragglers.)
On v4 each decision is one corner, so the choice is scored directly: each
decision is paired with the intersection it chose for, and compared with the
moves that start a shortest route to a blue (heading-aware, no reversing, no
red/bumpers, no jumps). The intersection scoring below is copied UNCHANGED
from steering_analyze.py, so the numbers compare with d2's.

usage: steering_analyze_v4.py [tag] [dir]   (tag: v4_ptr, v4_79, ...;
       reads steer_demo_<tag>.json + steer_episodes_<tag>.json)
"""
import os, sys, json
from collections import Counter, deque
from statistics import median
import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else 'v4_ptr'
D = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
CHOICES = ('straight', 'left', 'right', 'jump')   # env v4's CORNER_ACTIONS
JUMP = 3
MOVE_OF = {0: 'S', 1: 'L', 2: 'R'}
CHOICE_OF = {'S': 0, 'L': 1, 'R': 2}

DIRS = [(-1, 0), (0, -1), (1, 0), (0, 1)]        # N W S E, as (drow, dcol) — the quadrant order
BLOCKED = (1, 3, 10)                             # red, bumper, turning red
MOVES = {0: 'S', 1: 'L', 3: 'R', 2: 'rev'}       # (out - in) % 4; a left turn raises the heading


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


def decision_rows(rec, rows):
    """Pair each v4 decision with the intersection it chose for: the row on
    the decision's cell, entered at or before the decision frame (the patch
    flips to a cell 128 units before its corner, so the choice lands there)."""
    out = []
    for d in rec['decisions']:
        if d['cell'] is None:
            continue
        i = max(d['i'] - 1, 0)
        cand = [r for r in rows if r['cell'] == tuple(d['cell']) and r['i'] <= i < r['i'] + 40]
        out.append((d, cand[-1] if cand else None))
    return out


# ── load ────────────────────────────────────────────────────────────────────
demo = json.load(open(os.path.join(D, f'steer_demo_{TAG}.json')))
eps = json.load(open(os.path.join(D, f'steer_episodes_{TAG}.json')))
demo['decisions'] = []
s0 = demo['start_frame']
start_pct = f"{s0 / len(demo['frames']):.0%}"

# ── yardsticks ──────────────────────────────────────────────────────────────
print('== yardsticks ==')
drows = intersections(demo)
pre = [r for r in drows if r['dist'] == 1]
took = sum(r['move'] in r['opt'] for r in pre)
print(f'  Tyler: {len(drows)} intersections; with a blue 1 step away he took it {pct(took, len(pre))} (n={len(pre)})')
agree = tot = 0
for rows in [drows] + [intersections(e) for e in eps[:30]]:
    for r in rows:
        if r['move'] in ('L', 'R') and not r['air']:
            tot += 1
            agree += (4 if r['move'] == 'L' else -4) in r['corner']
print(f'  turn direction inferred from cells vs the queued-corner byte: {pct(agree, tot)} of {tot} turns')
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

# v4: the env must do what the agent chose. Every decision should pair with an
# intersection, and a straight/left/right choice should be the move made there
ep_rows, pairs = [], []
for e in eps:
    rows = intersections(e)
    final = e['frames'][-1]['got']
    last_i = max([i for i, f in enumerate(e['frames']) if f['got'] < final] + [-1]) + 1
    for r in rows:
        r['ep'] = e
        r['stall'] = e['stalled'] and r['i'] >= last_i
    ep_rows.append(rows)
    pairs += [(d, r, e) for d, r in decision_rows(e, rows)]
paired = [(d, r, e) for d, r, e in pairs if r is not None]
ground = [(d, r, e) for d, r, e in paired if d['a'] != JUMP and r['move'] != 'rev']
same = sum(MOVE_OF[d['a']] == r['move'] for d, r, e in ground)
print(f'  decisions paired with an intersection: {pct(len(paired), len(pairs))} of {len(pairs)}')
print(f'  straight/left/right choice == the move made there: {pct(same, len(ground))} of {len(ground)}'
      f' (bumper reversals excluded: {sum(r["move"] == "rev" for d, r, e in paired)})')
odd = Counter((MOVE_OF[d['a']], r['move']) for d, r, e in ground if MOVE_OF[d['a']] != r['move'])
if odd:
    print(f'     mismatches (chose, made): {dict(odd)}')

# ── the gate question ───────────────────────────────────────────────────────
print('\n== where ONLY straight was on a shortest route (d2 on v3 went straight 6-9%) ==')
for label, flag in (('collecting', False), ('stalled', True)):
    ps = [(d, r) for d, r, e in paired if r['dist'] is not None and r['opt'] == {'S'} and r['stall'] == flag]
    if not ps:
        continue
    acts = Counter(CHOICES[d['a']] for d, r in ps)
    print(f'  {label}: {len(ps)} decisions | chose straight {pct(acts["straight"], len(ps))}'
          f' | chose: {", ".join(f"{k} {pct(v, len(ps))}" for k, v in acts.most_common())}'
          f' | P(straight) median {median(d["p"][0] for d, r in ps):.2f}')
print('\n== where a turn was due (shortest routes all turn) ==')
for label, flag in (('collecting', False), ('stalled', True)):
    ps = [(d, r) for d, r, e in paired if r['dist'] is not None and r['opt'] and 'S' not in r['opt']
          and r['stall'] == flag]
    if not ps:
        continue
    ok = sum(MOVE_OF.get(d['a']) in r['opt'] for d, r in ps)
    pt = [sum(d['p'][CHOICE_OF[m]] for m in r['opt'] if m in CHOICE_OF) for d, r in ps]
    print(f'  {label}: {len(ps)} decisions | chose a right turn {pct(ok, len(ps))} | P(a right turn) median {median(pt):.2f}')

# ── Tyler vs agent ──────────────────────────────────────────────────────────
print(f'\n== moves vs a shortest route to the nearest blue (grounded intersections) ==')
t_all = agreement(drows, 'Tyler, whole run')
t_end = agreement([r for r in drows if r['i'] >= s0], f'Tyler, from {start_pct}')
by_dist(t_all, 'Tyler')
out = Counter('cleared' if e['cleared'] else 'stall' if e['stalled']
              else 'timeout' if e['timeout'] else 'red' for e in eps)
print(f'\n  agent: {len(eps)} episodes from {start_pct}: {dict(out)}'
      f' | reached its target {pct(sum(e["got"] >= e["target"] for e in eps), len(eps))}')
stall_rows = [r for rows in ep_rows for r in rows if r['stall']]
other_rows = [r for rows in ep_rows for r in rows if not r['stall']]
a_other = agreement(other_rows, 'agent, while collecting')
a_stall = agreement(stall_rows, 'agent, stall windows')
by_dist(a_other, 'agent collecting')
by_dist(a_stall, 'agent stalled')
print('\n== how the wrong moves were wrong ==')
errors(drows, 'Tyler')
for label, flag in (('agent collecting', False), ('agent stalled', True)):
    errors([r for rows in ep_rows for r in rows if r['stall'] == flag], label)

# ── the policy's choices ────────────────────────────────────────────────────
print('\n== choices ==')
st = [e for e in eps if e['stalled']]
for e in eps:
    fr = e['frames']; final = fr[-1]['got']
    e['_win0'] = max([i for i, f in enumerate(fr) if f['got'] < final] + [-1]) + 1
for label, ds in (('stalled', [d for e in st for d in e['decisions'] if d['i'] >= e['_win0']]),
                  ('collecting', [d for e in eps for d in e['decisions']
                                  if not (e['stalled'] and d['i'] >= e['_win0'])])):
    if not ds:
        continue
    P = np.array([d['p'] for d in ds])
    ent = -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)
    acts = Counter(CHOICES[d['a']] for d in ds)
    print(f'  {label}: {len(ds)} decisions | picked: {", ".join(f"{k} {pct(v, len(ds))}" for k, v in acts.most_common())}'
          f' | mean P: ' + ' '.join(f'{c} {P[:, k].mean():.2f}' for k, c in enumerate(CHOICES))
          + f' | entropy median {np.median(ent):.2f} (max {np.log(len(CHOICES)):.2f})')

print('\n== stall windows ==')
fmt = lambda c: ', '.join(f'{k} {pct(v, sum(c.values()))}' for k, v in sorted(c.items()))
print(f'  moves at intersections — stalled: {fmt(Counter(r["move"] for r in stall_rows))}'
      f' | collecting: {fmt(Counter(r["move"] for r in other_rows))}'
      f' | Tyler from {start_pct}: {fmt(Counter(r["move"] for r in t_end))}')
trend, dists = Counter(), []
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
    left = [e['target'] - e['got'] for e in st]
    print(f'  spheres short of the target when it stalled: median {median(left)}, {dict(sorted(Counter(left).items()))}')

print('\n== airborne ==')


def flyovers(rec, i0, i1):
    """Blue cells Sonic passed over in the air between frames i0 and i1."""
    fr, n, seen = rec['frames'], 0, set()
    for i in range(max(i0, 1), i1):
        c = cell(fr[i])
        if fr[i]['jump'] & 0x80 and c not in seen:
            g = board_at(rec['boards'], i)
            if g[c[0]][c[1]] == 2 and fr[i]['got'] == fr[i - 1]['got']:
                n += 1
                seen.add(c)
    return n


fair = lambda recs: pct(sum(f['jump'] >> 7 for e, i0 in recs for f in e['frames'][i0:]),
                        sum(len(e['frames'][i0:]) for e, i0 in recs))
print(f'  frames airborne ($E432 bit 7) — Tyler from {start_pct}: {fair([(demo, s0)])}'
      f' | agent, whole episodes: {fair([(e, 0) for e in eps])} (d2: 46%) | stall windows: {fair([(e, e["_win0"]) for e in st])}')
print(f'  blues jumped over — Tyler from {start_pct}: {flyovers(demo, s0, len(demo["frames"]))}')
fo = [flyovers(e, e['_win0'], len(e['frames'])) for e in st]
if fo:
    print(f'  blues jumped over inside stall windows: {sum(fo)} in {len(fo)} windows; windows with at least one: {pct(sum(1 for x in fo if x), len(fo))}')
