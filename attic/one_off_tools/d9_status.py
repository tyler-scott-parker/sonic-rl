"""Compact status for a multi-board demo run (d9, d10). Usage:
    d9_status.py [run] [--state FILE]
Per board: frames (steps - demo_frac x recording length; fresh: steps), the
curriculum start point (last training table in logs/ss_multistage_<run>.log),
fresh starts, fresh clears, best fresh spheres / board total, median fresh
clear frames vs his recording. Totals, and the run's agreed rules: gate at
70M total frames (every board off its recording's end, all < 90%, median
< 75%), stall stop (no board's start point moves for 15M total frames, tracked
in --state across calls), main test (a fresh clear on all 14 within 280M).
Prints ALERT lines for: stall stop reached, gate reached, and a possible collapse
(demo-start success below a quarter of its recent level; on d8c it fires right after
the collapse, on d8 once at its 8-10M dip, which recovered: a cue to look, not a stop).
d10 (demos='book', 2026-10-11): demos are the route book's routes (routes/curriculum/),
the gate alert is d9's only, and it prints the book (route_book.py) with d10's
proposed stop: ALERT when 50M frames pass without a book improvement."""
import json, os, re, statistics as st, sys
REPO = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, REPO)
import demos                                     # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith('--')]
RUN = args[0] if args else 'd9'
STATE = sys.argv[sys.argv.index('--state') + 1] if '--state' in sys.argv else None
if STATE in args:
    args.remove(STATE)
GATE, STALL, MAIN = 70e6, 15e6, 280e6

meta = {}


def demo_meta(did):
    if did not in meta:
        p = os.path.join(demos.DEMO_DIR, f'{did}.npz')
        if not os.path.exists(p):
            p = os.path.join(REPO, 'routes', 'curriculum', f'{did}.npz')
        meta[did] = demos.load_demo(p)[1]
    return meta[did]


total = 0.0
per = {}
demo_ok = []                                     # success of each demo-start episode, in order
for line in open(os.path.join(REPO, f'ss_log_{RUN}.jsonl')):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    s = int(r['stage'])
    p = per.setdefault(s, dict(frames=0.0, fresh=0, clears=[], best=0, total=r.get('stage_total')))
    did = r.get('demo')
    f = r['steps'] - r['demo_frac'] * demo_meta(did)['frames'] if did else r['steps']
    p['frames'] += f
    total += f
    if did:
        demo_ok.append(bool(r['success']))
    if not did:
        p['fresh'] += 1
        p['best'] = max(p['best'], r['spheres'])
        if r['success']:
            p['clears'].append(r['steps'])

log = open(os.path.join(REPO, 'logs', f'ss_multistage_{RUN}.log')).read()
tables = log.split('| demo/')
fracs = {int(m.group(1)): float(m.group(2))
         for m in re.finditer(r'(?:stage|board)0*(\d+)_\S*/start_\S*\s*\|\s*([0-9.]+)', tables[-1])} if len(tables) > 1 else {}
steps = re.findall(r'total_timesteps\s*\|\s*(\d+)', log)

import route_book                                # noqa: E402
his = {b: len(demos.load_demo(p)[0]) for b, p in route_book.his_recordings().items()}
book = route_book.book_frames()

state = {}
if STATE and os.path.exists(STATE):
    state = json.load(open(STATE))
moved = state.get('moved', {})
last = state.get('fracs', {})
init = state.get('init', {})          # each board's highest start point seen = its recording's end
for s, fr in fracs.items():
    if str(s) not in last or last[str(s)] != fr:
        moved[str(s)] = total
    init[str(s)] = max(init.get(str(s), 0.0), fr)
gate_alert = RUN == 'd9' and total >= GATE and not state.get('gate_reported')
book_total = sum(book.values())
book_at = state.get('book_at', total) if state.get('book_total') == book_total else total
if STATE:
    json.dump(dict(fracs={str(k): v for k, v in fracs.items()}, moved=moved, init=init, total=total,
                   gate_reported=state.get('gate_reported') or gate_alert,
                   book_total=book_total, book_at=book_at), open(STATE, 'w'))

print(f'{RUN}: {total / 1e6:.2f}M frames, {int(steps[-1]) if steps else 0:,} steps')
print(f'  {"board":>5s} {"frames":>7s} {"start":>6s} {"fresh":>6s} {"clears":>6s} {"best":>8s} {"med clear":>9s} {"his":>5s}')
for s in sorted(per):
    p = per[s]
    med = f'{st.median(p["clears"]):.0f}' if p['clears'] else '-'
    print(f'  {s:5d} {p["frames"] / 1e6:6.2f}M {fracs.get(s, float("nan")):6.1%} {p["fresh"]:6d} '
          f'{len(p["clears"]):6d} {p["best"]:4d}/{p["total"] or 0:<3d} {med:>9s} {his.get(s, 0):5d}')
fs = [fracs[s] for s in sorted(fracs)]
if fs:
    if STATE:                                    # needs the state's record of each board's first start point
        print(f'  boards off their recording\'s end: {sum(fracs[s] < init.get(str(s), 1) for s in fracs)}/{len(fracs)}')
    print(f'  start points: median {st.median(fs):.1%}, max {max(fs):.1%}, min {min(fs):.1%}; '
          f'boards with a fresh clear: {sum(1 for p in per.values() if p["clears"])}/{len(per)}')
if moved:
    since = total - max(moved.values())
    print(f'  last start-point move: {since / 1e6:.2f}M frames ago (stall stop at {STALL / 1e6:.0f}M)')
    if since >= STALL:
        print('ALERT stall stop: no board\'s start point moved for 15M frames')
# collapse check (d8c lost every clear in one update): demo-start success, last 400 vs the 2,000 before
if len(demo_ok) >= 2400:
    now, before = sum(demo_ok[-400:]) / 400, sum(demo_ok[-2400:-400]) / 2000
    print(f'  demo-start success: last 400 {now:.1%}, the 2,000 before {before:.1%}')
    if before >= 0.10 and now < 0.25 * before:
        print('ALERT collapse? demo-start success fell below a quarter of its recent level')
beat = [b for b in sorted(book) if b in his and book[b] < his[b]]
print(f'  route book: faster than Tyler on {len(beat)}/14 {beat}; '
      f'{sum(his.values()) - book_total:+,d} frames over all 14; last improvement '
      f'{(total - book_at) / 1e6:.2f}M frames ago')
if RUN != 'd9' and total - book_at >= 50e6:
    print('ALERT d10 stop rule: 50M frames without a route-book improvement')
if gate_alert:
    print(f'ALERT gate reached ({total / 1e6:.1f}M >= 70M): all off their end? '
          f'{all(fracs[s] < init.get(str(s), 1) for s in fracs)}; every board < 90%? {all(f < 0.9 for f in fs)}; '
          f'median < 75%? {st.median(fs) < 0.75}')
