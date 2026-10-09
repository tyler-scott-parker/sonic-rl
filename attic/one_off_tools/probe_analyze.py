"""Summarize probe_visibility.py's output."""
import os, json
from collections import Counter
from statistics import median
OUT = os.path.dirname(os.path.abspath(__file__))
demo = json.load(open(os.path.join(OUT, 'probe_demo.json')))
eps = json.load(open(os.path.join(OUT, 'probe_episodes.json')))


def longest_blind(rows, frames_per_row):
    best = cur = 0
    for r in rows:
        cur = 0 if r['vis'] else cur + 1
        best = max(best, cur)
    return best * frames_per_row


def pct(x):
    return f'{100 * x:.0f}%'


print('== Tyler, from 79% to the clear ==')
print(f"  frames: {len(demo)}, a blue sphere in the 15x15 view on {pct(sum(r['vis'] for r in demo) / len(demo))} of them")
print(f"  longest stretch with none in view: {longest_blind(demo, 1)} frames")
print(f"  nearest blue, median {median(r['near'] for r in demo)} cells, max {max(r['near'] for r in demo)}")

print(f'\n== d2 from the 79-83% starts: {len(eps)} episodes ==')
out = Counter('cleared' if e['cleared'] else 'stall' if e['stalled']
              else 'timeout' if e['timeout'] else 'red' for e in eps)
print('  ', dict(out), f"| reached your total: {sum(e['got'] >= e['target'] for e in eps)}")
prod = [r for e in eps for r in e['rows'] if r['f'] <= r['last'] + 40]   # within 40 frames of a pickup
print(f"  while collecting (<=40 frames since a pickup): sphere in view {pct(sum(r['vis'] for r in prod) / len(prod))} of steps")

st = [e for e in eps if e['stalled']]
cls = Counter()
print(f'\n== the {len(st)} stall endings (600 frames without a sphere) ==')
details = []
for e in st:
    last_pick = e['rows'][-1]['last']
    win = [r for r in e['rows'] if r['f'] > last_pick]
    frac = sum(r['vis'] for r in win) / len(win)
    left = e['rows'][-1]['nblue']
    first_near = win[0]['near']; min_near = min(r['near'] for r in win)
    kind = 'never in view' if frac == 0 else 'in view, not taken'
    cls[kind] += 1
    details.append(dict(kind=kind, frac=frac, left=left, gap=e['target'] - e['got'],
                        first_near=first_near, min_near=min_near))
for k, n in cls.most_common():
    d = [x for x in details if x['kind'] == k]
    print(f"  {k}: {n} ({pct(n / len(st))})"
          f" | blues left on board median {median(x['left'] for x in d)}"
          f" | nearest when the clock started, median {median(x['first_near'] for x in d)} cells"
          f" | closest it ever got, median {median(x['min_near'] for x in d)}"
          + (f" | in view {pct(median(x['frac'] for x in d))} of the window (median)" if k != 'never in view' else ''))
print('  blues left at a stall:', sorted(Counter(x['left'] for x in details).items()))
print('  nearest blue when the stall clock started (cells):', sorted(Counter(x['first_near'] for x in details).items()))

# ── walk reachability (blues reachable without crossing red/bumpers) ───────
if 'walk' in demo[0]:
    print('\n== reachability ==')
    pre = []   # the step right before each pickup: the yardstick check
    for e in eps:
        rs = e['rows']
        for a, b in zip(rs, rs[1:]):
            if b['got'] > a['got']:
                pre.append(a['walk'])
    print('  yardstick - clear-path distance one step before a pickup:',
          sorted(Counter('walled' if w is None else min(w, 6) for w in pre).items(), key=str))
    print(f"  Tyler from 79%: every blue walled off (jump needed) on {pct(sum(r['walk'] is None for r in demo) / len(demo))} of frames;"
          f" clear-path distance median {median(r['walk'] for r in demo if r['walk'] is not None)}")
    cls2 = Counter()
    for e in st:
        last_pick = e['rows'][-1]['last']
        win = [r for r in e['rows'] if r['f'] > last_pick]
        seen = any(r['vis'] for r in win)
        walkable = any(r['walk'] is not None for r in win)
        cls2[('in view' if seen else 'never in view') + ', ' +
             ('a clear path existed' if walkable else 'walled off by red - needed a jump')] += 1
    for k, n in cls2.most_common():
        print(f'  {k}: {n} ({pct(n / len(st))})')
